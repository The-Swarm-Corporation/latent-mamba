"""
main.py: adaptive continuous ("latent") reasoning on a Mamba-2 backbone.

Overview
--------
A Mamba-2 language model is extended with three small modules so it can think
silently, in continuous space, between reading a prompt and writing an answer::

    prompt ... <bot>                                                 <eot> answer
         |                                                             |
         v                                                             v
    token embedding --> Mamba-2 backbone --> final hidden h --> halting head
                          ^  (fixed-size conv + SSM state)          |      |
                          |                                     continue  halt
                          |                                         |      |
                          +----------- thought projector <----------+      v
                                                                        LM head

* ThoughtProjector: maps the final hidden state back into input-embedding
  space and adds a learned "thinking-mode" embedding. Fed back as the next
  input, this is a Coconut-style continuous thought (Hao et al., 2024).
* HaltingHead: emits lambda_n = P(halt at step n | not halted before n).
  It is trained PonderNet-style (Banino et al., 2021).
* Every thought step only updates the fixed-size per-layer state (causal-conv
  window + SSM state), so memory is O(1) in the number of thoughts.

Training recipe (all stages are implemented here)
-------------------------------------------------
1. Language CoT       ``latent_loss(..., n_thoughts=0)`` with the full chain of
                      thought as the target (Coconut "stage 0").
2. Coconut curriculum ``latent_loss(..., n_thoughts=c * k)``: at stage k the
                      first k CoT steps are replaced by c*k continuous
                      thoughts and removed from the target.
3. Adaptive halting   ``ponder_loss(...)``: expected answer loss under the
                      halting distribution, plus KL to a geometric prior.

Inference: ``generate(...)`` with ``halt_rule="learned"`` (halting head),
``"convergence"`` (stop when h stops changing), or ``"fixed"``.
``ablate="zeros" | "noise"`` replaces thought content to check that the model
actually uses its thoughts rather than treating them as pause tokens.

Implementation notes
--------------------
* Pure PyTorch: sequences use the chunked SSD algorithm of Mamba-2 (Dao & Gu,
  2024) with an initial state; single tokens use the exact recurrent step.
  Both are differentiable, so gradients flow through every thought step.
* All state updates are functional (they return new tensors), so a cache can
  be branched, for example to score the answer after every thought step in
  ``ponder_loss``.
* The mixer's parameterization follows Mamba-2 (in_proj -> [z, xBC, dt],
  depthwise causal conv over xBC, per-head A/dt/D, gated RMSNorm). Porting
  pretrained Mamba-2 weights is mostly a matter of renaming keys (not
  included here).
* Prompts within a batch must have equal length. Left padding would push pad
  tokens through the recurrent state, so group prompts by length instead.

Quick start
-----------
>>> cfg = LatentMambaConfig(vocab_size=1000, bot_id=997, eot_id=998, pad_id=999)
>>> model = LatentMamba(cfg)
>>> prompt = torch.randint(0, 997, (2, 12))
>>> prompt[:, -1] = cfg.bot_id
>>> target = torch.randint(0, 997, (2, 3))
>>> out = model.ponder_loss(prompt, target, max_thoughts=6)
>>> out.loss.backward()
>>> gen = model.generate(prompt, max_new_tokens=3, max_thoughts=6)
>>> gen.tokens.shape, gen.n_thoughts
(torch.Size([2, 3]), tensor([...]))

Call ``self_test()`` for numerical checks, or ``run_demo()`` for an end-to-end
run of the training recipe on a synthetic pointer-chasing task. Both take plain
keyword arguments (device, step counts, seed). Running the file directly calls
both with their defaults.

References: Gu & Dao (2023) Mamba; Dao & Gu (2024) Mamba-2 / SSD; Hao et al.
(2024) Coconut; Banino et al. (2021) PonderNet; Graves (2016) ACT; Geiping et
al. (2025) recurrent-depth latent reasoning.

Requires Python >= 3.10 and PyTorch >= 2.1.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "IGNORE_INDEX",
    "LatentMambaConfig",
    "RMSNorm",
    "LayerState",
    "MambaCache",
    "ssd_chunked_scan",
    "ssd_step",
    "Mamba2Mixer",
    "ResidualBlock",
    "MambaBackbone",
    "ThoughtProjector",
    "HaltingHead",
    "LatentLossOutput",
    "PonderLossOutput",
    "GenerationOutput",
    "LatentMamba",
    "PointerChasingTask",
    "self_test",
    "run_demo",
]

IGNORE_INDEX: int = -100
"""Target value that is excluded from every loss (padding in target tensors)."""

HaltRule = Literal["learned", "convergence", "fixed"]
Ablation = Literal["zeros", "noise"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class LatentMambaConfig:
    """Hyperparameters for :class:`LatentMamba`.

    Attributes:
        vocab_size: Number of token ids, including the special tokens.
        bot_id: Token that switches the model into latent mode. It must be
            the last token of every prompt.
        eot_id: Token fed after the last thought to switch back to language mode.
        pad_id: Token used as model *input* where targets are padding.
        d_model: Residual stream width.
        n_layers: Number of Mamba-2 residual blocks.
        d_state: SSM state size N per head.
        d_conv: Causal depthwise convolution width (must be at least 2).
        expand: Inner width multiplier (d_inner = expand * d_model).
        headdim: SSM head dimension P (d_inner must be divisible by it).
        chunk_size: Chunk length for the chunked SSD scan.
        dt_min, dt_max, dt_init_floor: Range for the initial step size dt.
        a_init_range: Range for the initial decay magnitude -A.
        tie_embeddings: Share weights between the embedding and the LM head.
        init_std: Std of the embedding init. It also sets the initial output
            scale of the thought projector so thoughts start at token scale.
        norm_eps: Epsilon for every RMSNorm.
        halt_hidden: Hidden width of the halting MLP (default d_model // 4).
        halt_prior: lambda_p of the geometric prior. The halting head is
            initialised to output exactly this probability at every step.
        halt_detach_input: If True, halting-head gradients do not reach the
            backbone (halting is learned as a pure probe).
    """

    vocab_size: int
    bot_id: int
    eot_id: int
    pad_id: int
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 32
    chunk_size: int = 64
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    a_init_range: tuple[float, float] = (1.0, 16.0)
    tie_embeddings: bool = True
    init_std: float = 0.02
    norm_eps: float = 1e-5
    halt_hidden: int | None = None
    halt_prior: float = 0.2
    halt_detach_input: bool = False

    def __post_init__(self) -> None:
        if self.d_inner % self.headdim != 0:
            raise ValueError(f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}")
        if self.d_conv < 2:
            raise ValueError("d_conv must be >= 2")
        if not 0.0 < self.halt_prior < 1.0:
            raise ValueError("halt_prior must be in (0, 1)")
        for name in ("bot_id", "eot_id", "pad_id"):
            value = getattr(self, name)
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"{name}={value} is outside the vocabulary")

    @property
    def d_inner(self) -> int:
        """Width of the SSM branch."""
        return self.expand * self.d_model

    @property
    def n_heads(self) -> int:
        """Number of SSM heads."""
        return self.d_inner // self.headdim

    @property
    def conv_dim(self) -> int:
        """Channels that pass through the causal conv (x, B and C)."""
        return self.d_inner + 2 * self.d_state


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """RMS normalisation with an optional SiLU gate, as in Mamba-2.

    ``forward(x, gate)`` computes ``rmsnorm(x * silu(gate)) * weight``.
    """

    def __init__(self, dim: int, eps: float = 1e-5, init: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: Tensor, gate: Tensor | None = None) -> Tensor:
        if gate is not None:
            x = x * F.silu(gate)
        compute_dtype = torch.promote_types(x.dtype, torch.float32)  # upcast half precision only
        x_f = x.to(compute_dtype)
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f * self.weight.to(compute_dtype)).to(x.dtype)


def _bmask(mask: Tensor, like: Tensor) -> Tensor:
    """Reshape a (B,) boolean mask so it broadcasts against ``like`` (B, ...)."""
    return mask.view(-1, *([1] * (like.dim() - 1)))


@dataclass
class LayerState:
    """Recurrent state of one Mamba-2 layer.

    Attributes:
        conv: Last ``d_conv - 1`` inputs of the causal conv, shape (B, conv_dim, d_conv - 1).
        ssm: SSM hidden state, shape (B, n_heads, headdim, d_state).
    """

    conv: Tensor
    ssm: Tensor

    @classmethod
    def zeros(
        cls,
        config: LatentMambaConfig,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> LayerState:
        return cls(
            conv=torch.zeros(batch_size, config.conv_dim, config.d_conv - 1, device=device, dtype=dtype),
            ssm=torch.zeros(
                batch_size, config.n_heads, config.headdim, config.d_state, device=device, dtype=dtype
            ),
        )


@dataclass
class MambaCache:
    """Per-layer recurrent state of the whole backbone.

    Its size is independent of how many tokens or thoughts have been
    processed. Every method returns a new cache, so caches can be branched
    and recombined freely.
    """

    layers: list[LayerState]

    @classmethod
    def zeros(
        cls,
        config: LatentMambaConfig,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> MambaCache:
        return cls([LayerState.zeros(config, batch_size, device, dtype) for _ in range(config.n_layers)])

    @property
    def batch_size(self) -> int:
        return self.layers[0].ssm.shape[0]

    def where(self, mask: Tensor, other: MambaCache) -> MambaCache:
        """Per example, take ``self`` where ``mask`` (B,) is True, else ``other``."""
        return MambaCache(
            [
                LayerState(
                    conv=torch.where(_bmask(mask, a.conv), a.conv, b.conv),
                    ssm=torch.where(_bmask(mask, a.ssm), a.ssm, b.ssm),
                )
                for a, b in zip(self.layers, other.layers)
            ]
        )

    @staticmethod
    def cat(caches: Sequence[MambaCache]) -> MambaCache:
        """Concatenate caches along the batch dimension."""
        n_layers = len(caches[0].layers)
        return MambaCache(
            [
                LayerState(
                    conv=torch.cat([c.layers[i].conv for c in caches], dim=0),
                    ssm=torch.cat([c.layers[i].ssm for c in caches], dim=0),
                )
                for i in range(n_layers)
            ]
        )

    def detach(self) -> MambaCache:
        """Cut the autograd graph (for truncated backprop through long loops)."""
        return MambaCache([LayerState(s.conv.detach(), s.ssm.detach()) for s in self.layers])

    def num_bytes(self) -> int:
        """Memory held by the cache. Constant in sequence and thought length."""
        return sum(s.conv.numel() * s.conv.element_size() + s.ssm.numel() * s.ssm.element_size() for s in self.layers)


# --------------------------------------------------------------------------- #
# Selective state-space kernels (pure PyTorch)
# --------------------------------------------------------------------------- #


def _segsum(x: Tensor) -> Tensor:
    """Stable segment sum over the last dim.

    Returns ``out[..., i, j] = sum_{k=j+1..i} x[..., k]`` for ``j <= i`` and
    ``-inf`` above the diagonal, so ``exp(out)`` is the causal decay matrix.
    """
    t = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, t)
    strict = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~strict, 0.0)
    out = torch.cumsum(x, dim=-2)
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~causal, -torch.inf)


def ssd_chunked_scan(
    x: Tensor,
    dt: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    chunk_size: int,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Chunked SSD scan of Mamba-2 (single B/C group), with an initial state.

    Computes, for every position t,
    ``h_t = exp(dt_t * a) * h_{t-1} + dt_t * x_t (outer) b_t`` and ``y_t = h_t @ c_t``.

    Args:
        x: Inputs, shape (B, L, H, P).
        dt: Positive step sizes, shape (B, L, H).
        a: Negative per-head decay rates, shape (H,).
        b: Input projections, shape (B, L, N).
        c: Output projections, shape (B, L, N).
        chunk_size: Maximum chunk length.
        initial_state: State before position 0, shape (B, H, P, N). Defaults to zeros.

    Returns:
        ``(y, final_state)`` with shapes (B, L, H, P) and (B, H, P, N).
    """
    bsz, seqlen, n_heads, headdim = x.shape
    d_state = b.shape[-1]
    if initial_state is None:
        initial_state = x.new_zeros(bsz, n_heads, headdim, d_state)

    xs = x * dt.unsqueeze(-1)
    a_dt = dt * a
    bs = b.unsqueeze(2).expand(bsz, seqlen, n_heads, d_state)
    cs = c.unsqueeze(2).expand(bsz, seqlen, n_heads, d_state)

    chunk = min(chunk_size, seqlen)
    pad = (-seqlen) % chunk
    if pad:
        # Zero padding at the end is exact: A=0 means no decay and X=0 means no input.
        xs = F.pad(xs, (0, 0, 0, 0, 0, pad))
        a_dt = F.pad(a_dt, (0, 0, 0, pad))
        bs = F.pad(bs, (0, 0, 0, 0, 0, pad))
        cs = F.pad(cs, (0, 0, 0, 0, 0, pad))
    n_chunks = (seqlen + pad) // chunk

    xs = xs.reshape(bsz, n_chunks, chunk, n_heads, headdim)
    bs = bs.reshape(bsz, n_chunks, chunk, n_heads, d_state)
    cs = cs.reshape(bsz, n_chunks, chunk, n_heads, d_state)
    a_dt = a_dt.reshape(bsz, n_chunks, chunk, n_heads).permute(0, 3, 1, 2)  # (b, h, c, l)
    a_cumsum = torch.cumsum(a_dt, dim=-1)

    # 1. Intra-chunk outputs (diagonal blocks).
    decay = torch.exp(_segsum(a_dt))
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", cs, bs, decay, xs)

    # 2. State contributed by each chunk, measured at its end.
    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", bs, decay_states, xs)

    # 3. Inter-chunk recurrence: states at every chunk boundary.
    states = torch.cat([initial_state.unsqueeze(1), states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(a_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Contribution of the carried-in state to each position.
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", cs, states, torch.exp(a_cumsum))

    y = (y_diag + y_off).reshape(bsz, n_chunks * chunk, n_heads, headdim)[:, :seqlen]
    return y, final_state


def ssd_step(x: Tensor, dt: Tensor, a: Tensor, b: Tensor, c: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Exact single-token recurrence, equivalent to ``ssd_chunked_scan`` with L=1.

    Args:
        x: (B, H, P)
        dt: (B, H)
        a: (H,)
        b: (B, N)
        c: (B, N)
        state: (B, H, P, N)

    Returns:
        ``(y, new_state)`` with shapes (B, H, P) and (B, H, P, N).
    """
    decay = torch.exp(dt * a)[..., None, None]
    update = (dt.unsqueeze(-1) * x).unsqueeze(-1) * b[:, None, None, :]
    new_state = state * decay + update
    y = torch.einsum("bhpn,bn->bhp", new_state, c)
    return y, new_state


# --------------------------------------------------------------------------- #
# Mamba-2 backbone
# --------------------------------------------------------------------------- #


class Mamba2Mixer(nn.Module):
    """Mamba-2 token mixer with explicit, functional state handling.

    ``forward(u, state)`` processes ``u`` (B, L, d_model) starting from
    ``state``. It returns ``(out, new_state)``. L can be any length, including
    1 (a single thought step).
    """

    def __init__(self, config: LatentMambaConfig) -> None:
        super().__init__()
        self.config = config
        d_in_proj = 2 * config.d_inner + 2 * config.d_state + config.n_heads
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(
            config.conv_dim, config.conv_dim, kernel_size=config.d_conv, groups=config.conv_dim, bias=True
        )

        dt = torch.exp(
            torch.rand(config.n_heads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        a = torch.empty(config.n_heads).uniform_(*config.a_init_range)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(config.n_heads))

        self.norm = RMSNorm(config.d_inner, eps=config.norm_eps)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def forward(self, u: Tensor, state: LayerState | None = None) -> tuple[Tensor, LayerState]:
        cfg = self.config
        bsz, seqlen, _ = u.shape
        if state is None:
            state = LayerState.zeros(cfg, bsz, u.device, u.dtype)

        z, xbc, dt = torch.split(self.in_proj(u), [cfg.d_inner, cfg.conv_dim, cfg.n_heads], dim=-1)

        # Causal depthwise conv, continuing from the stored window.
        window = torch.cat([state.conv, xbc.transpose(1, 2)], dim=-1)  # (B, C, d_conv - 1 + L)
        new_conv = window[..., -(cfg.d_conv - 1) :]
        xbc = F.conv1d(window, self.conv1d.weight, self.conv1d.bias, groups=cfg.conv_dim)
        xbc = F.silu(xbc).transpose(1, 2)  # (B, L, C)

        x, b, c = torch.split(xbc, [cfg.d_inner, cfg.d_state, cfg.d_state], dim=-1)
        x = x.reshape(bsz, seqlen, cfg.n_heads, cfg.headdim)
        dt = F.softplus(dt + self.dt_bias)
        a = -torch.exp(self.A_log)

        if seqlen == 1:
            y, new_ssm = ssd_step(x[:, 0], dt[:, 0], a, b[:, 0], c[:, 0], state.ssm)
            y = y.unsqueeze(1)
        else:
            y, new_ssm = ssd_chunked_scan(x, dt, a, b, c, cfg.chunk_size, state.ssm)

        y = y + x * self.D[:, None]
        y = self.norm(y.reshape(bsz, seqlen, cfg.d_inner), gate=z)
        return self.out_proj(y), LayerState(conv=new_conv, ssm=new_ssm)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: ``x + mixer(norm(x))``."""

    def __init__(self, config: LatentMambaConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mixer = Mamba2Mixer(config)

    def forward(self, x: Tensor, state: LayerState | None = None) -> tuple[Tensor, LayerState]:
        out, new_state = self.mixer(self.norm(x), state)
        return x + out, new_state


class MambaBackbone(nn.Module):
    """Embedding, a stack of Mamba-2 residual blocks, and a final RMSNorm.

    ``forward`` takes input *embeddings* rather than token ids, so tokens and
    continuous thoughts enter through the same path.
    """

    def __init__(self, config: LatentMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, std=config.init_std)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs_embeds: Tensor, cache: MambaCache | None = None) -> tuple[Tensor, MambaCache]:
        """Run the stack.

        Args:
            inputs_embeds: (B, L, d_model).
            cache: State to continue from. None starts from zeros.

        Returns:
            ``(hidden, new_cache)``: normalised final hidden states (B, L, d_model)
            and the state after the last position.
        """
        h = inputs_embeds
        new_layers: list[LayerState] = []
        for i, layer in enumerate(self.layers):
            h, layer_state = layer(h, None if cache is None else cache.layers[i])
            new_layers.append(layer_state)
        return self.norm_f(h), MambaCache(new_layers)


# --------------------------------------------------------------------------- #
# Latent-reasoning modules
# --------------------------------------------------------------------------- #


class ThoughtProjector(nn.Module):
    """Maps a final hidden state to the next input embedding (a continuous thought).

    ``thought = Linear(RMSNorm(h)) + mode_embedding``

    At initialisation the linear layer is the identity and the norm scale
    equals the embedding RMS. The model therefore starts as vanilla Coconut
    (feed back the last hidden state), at the scale of a token embedding.
    The mode embedding (initially zero) tells the backbone it is thinking
    rather than reading. It lets the model learn, for example, a small dt in
    thought mode, so the prompt is not decayed out of the state.
    """

    def __init__(self, config: LatentMambaConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps, init=config.init_std)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.eye_(self.proj.weight)
        self.mode_embedding = nn.Parameter(torch.zeros(config.d_model))

    def content(self, h: Tensor) -> Tensor:
        """Thought content without the mode embedding, shape (..., d_model)."""
        return self.proj(self.norm(h))

    def forward(self, h: Tensor) -> Tensor:
        return self.content(h) + self.mode_embedding

    @torch.no_grad()
    def match_embedding_scale(self, embedding_weight: Tensor) -> None:
        """Re-scale thought outputs to the RMS of an embedding matrix.

        Call this after loading pretrained backbone weights.
        """
        self.norm.weight.fill_(embedding_weight.float().pow(2).mean().sqrt().item())


class HaltingHead(nn.Module):
    """Small MLP producing the halting logit for each thought step.

    ``sigmoid(logit)`` is lambda_n = P(halt now | not halted before). The last
    layer is initialised to output ``logit(halt_prior)`` everywhere. The
    initial halting distribution therefore equals the geometric prior.
    """

    def __init__(self, config: LatentMambaConfig) -> None:
        super().__init__()
        hidden = config.halt_hidden or max(16, config.d_model // 4)
        self.fc1 = nn.Linear(config.d_model, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, math.log(config.halt_prior / (1.0 - config.halt_prior)))

    def forward(self, h: Tensor) -> Tensor:
        """Map (B, d_model) to halting logits (B,)."""
        return self.fc2(F.silu(self.fc1(h))).squeeze(-1)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


@dataclass
class LatentLossOutput:
    """Result of :meth:`LatentMamba.latent_loss`.

    Attributes:
        loss: Scalar cross-entropy over target tokens.
        n_thoughts: Number of thoughts used per example, shape (B,).
    """

    loss: Tensor
    n_thoughts: Tensor


@dataclass
class PonderLossOutput:
    """Result of :meth:`LatentMamba.ponder_loss`.

    Attributes:
        loss: ``reconstruction + beta * kl`` (scalar).
        reconstruction: Expected answer loss under the halting distribution.
        kl: KL(halting distribution || truncated geometric prior).
        halting_distribution: p_n for n = 1..N, shape (B, N). Rows sum to 1.
        per_step_loss: Answer cross-entropy after n thoughts, shape (B, N).
        expected_thoughts: Sum over n of n * p_n, shape (B,).
    """

    loss: Tensor
    reconstruction: Tensor
    kl: Tensor
    halting_distribution: Tensor
    per_step_loss: Tensor
    expected_thoughts: Tensor


@dataclass
class GenerationOutput:
    """Result of :meth:`LatentMamba.generate`.

    Attributes:
        tokens: Generated answer tokens (B, T), excluding the prompt and <eot>.
        n_thoughts: Thought steps taken per example, shape (B,).
    """

    tokens: Tensor
    n_thoughts: Tensor


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class LatentMamba(nn.Module):
    """Mamba-2 language model with adaptive continuous thinking.

    Sequence layout in latent mode::

        prompt ... <bot> [thought_1 ... thought_n] <eot> target_1 ... target_T

    The prompt is prefilled with the parallel scan. Thoughts are recurrent
    single steps that feed the projected hidden state back in. The answer is
    teacher-forced (training) or decoded (inference) from the resulting
    state. Loss is applied only to target tokens.
    """

    def __init__(self, config: LatentMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = MambaBackbone(config)
        self.thought_projector = ThoughtProjector(config)
        self.halting_head = HaltingHead(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    # ----- basic passes -------------------------------------------------- #

    def embed(self, input_ids: Tensor) -> Tensor:
        """Token embeddings, shape (B, L, d_model)."""
        return self.backbone.embedding(input_ids)

    def forward(self, input_ids: Tensor, cache: MambaCache | None = None) -> tuple[Tensor, MambaCache]:
        """Plain language-mode pass that returns ``(logits (B, L, V), cache)``."""
        hidden, cache = self.backbone(self.embed(input_ids), cache)
        return self.lm_head(hidden), cache

    def encode_prompt(self, prompt_ids: Tensor) -> tuple[Tensor, MambaCache]:
        """Prefill a prompt that ends with <bot>.

        Returns:
            ``(h, cache)``: the final hidden state at <bot> (B, d_model), which
            is the source of the first thought, and the state after the prompt.
        """
        if not bool((prompt_ids[:, -1] == self.config.bot_id).all()):
            raise ValueError("every prompt must end with bot_id")
        hidden, cache = self.backbone(self.embed(prompt_ids))
        return hidden[:, -1], cache

    def thought_step(
        self, h: Tensor, cache: MambaCache, ablate: Ablation | None = None
    ) -> tuple[Tensor, MambaCache, Tensor]:
        """Take one continuous thought.

        Args:
            h: Final hidden state from the previous step, shape (B, d_model).
            cache: Current recurrent state.
            ablate: Diagnostic only. ``"zeros"`` feeds just the mode embedding
                (equivalent to a pause token). ``"noise"`` feeds Gaussian
                noise at the thought's RMS.

        Returns:
            ``(h_next, new_cache, halt_logit)`` where halt_logit has shape (B,).
        """
        if ablate is None:
            thought = self.thought_projector(h)
        else:
            content = self.thought_projector.content(h)
            if ablate == "zeros":
                content = torch.zeros_like(content)
            elif ablate == "noise":
                content = torch.randn_like(content) * content.pow(2).mean().sqrt()
            else:
                raise ValueError(f"unknown ablation {ablate!r}")
            thought = content + self.thought_projector.mode_embedding
        hidden, cache = self.backbone(thought.unsqueeze(1), cache)
        h_next = hidden[:, 0]
        halt_input = h_next.detach() if self.config.halt_detach_input else h_next
        return h_next, cache, self.halting_head(halt_input)

    # ----- losses -------------------------------------------------------- #

    def lm_loss(self, input_ids: Tensor, loss_mask: Tensor | None = None) -> Tensor:
        """Standard next-token loss, for pretraining or plain CoT fine-tuning.

        Args:
            input_ids: (B, L) token ids.
            loss_mask: Optional (B, L) bool mask over *target* positions 1..L-1.
                Its first column is ignored.
        """
        logits, _ = self(input_ids[:, :-1])
        targets = input_ids[:, 1:].clone()
        if loss_mask is not None:
            targets = targets.masked_fill(~loss_mask[:, 1:], IGNORE_INDEX)
        return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX)

    def _answer_inputs(self, target_ids: Tensor) -> Tensor:
        """Teacher-forcing inputs ``[<eot>, target_1 .. target_{T-1}]``, with padding mapped to pad_id."""
        eot = torch.full_like(target_ids[:, :1], self.config.eot_id)
        shifted = torch.cat([eot, target_ids[:, :-1]], dim=1)
        return shifted.masked_fill(shifted == IGNORE_INDEX, self.config.pad_id)

    def _per_example_answer_loss(self, cache: MambaCache, target_ids: Tensor) -> Tensor:
        """Mean target cross-entropy per example after ``cache``, shape (B,)."""
        hidden, _ = self.backbone(self.embed(self._answer_inputs(target_ids)), cache)
        logits = self.lm_head(hidden)
        token_loss = F.cross_entropy(
            logits.transpose(1, 2), target_ids, ignore_index=IGNORE_INDEX, reduction="none"
        )
        mask = (target_ids != IGNORE_INDEX).float()
        return (token_loss * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def _run_thoughts(
        self, h: Tensor, cache: MambaCache, n_thoughts: Tensor
    ) -> tuple[Tensor, MambaCache]:
        """Run ``n_thoughts[i]`` thoughts for example i. Finished examples are frozen."""
        for step in range(int(n_thoughts.max().item()) if n_thoughts.numel() else 0):
            active = step < n_thoughts
            h_new, cache_new, _ = self.thought_step(h, cache)
            h = torch.where(active[:, None], h_new, h)
            cache = cache_new.where(active, cache)
        return h, cache

    def latent_loss(self, prompt_ids: Tensor, target_ids: Tensor, n_thoughts: int | Tensor) -> LatentLossOutput:
        """Coconut-curriculum loss with a fixed (per-example) number of thoughts.

        Args:
            prompt_ids: (B, Lp), each ending with <bot>.
            target_ids: (B, T): remaining language reasoning plus the answer,
                padded with ``IGNORE_INDEX``.
            n_thoughts: Thoughts per example, as an int or a (B,) tensor. Zero
                gives the language-CoT stage.
        """
        h, cache = self.encode_prompt(prompt_ids)
        n = torch.as_tensor(n_thoughts, device=prompt_ids.device).expand(prompt_ids.shape[0]).long()
        h, cache = self._run_thoughts(h, cache, n)
        hidden, _ = self.backbone(self.embed(self._answer_inputs(target_ids)), cache)
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits.transpose(1, 2), target_ids, ignore_index=IGNORE_INDEX)
        return LatentLossOutput(loss=loss, n_thoughts=n)

    def ponder_loss(
        self,
        prompt_ids: Tensor,
        target_ids: Tensor,
        max_thoughts: int,
        beta: float = 0.01,
        prior: float | None = None,
    ) -> PonderLossOutput:
        """PonderNet objective for learning when to stop thinking.

        The model takes ``max_thoughts`` thoughts, scoring the answer from the
        state after each one. All N branches run in one batched pass over the
        answer tokens, so the cost is about N times one answer pass. The
        halting distribution is
        ``p_n = lambda_n * prod_{j<n} (1 - lambda_j)``, with lambda_N forced to 1.

        Args:
            prompt_ids: (B, Lp), each ending with <bot>.
            target_ids: (B, T), padded with ``IGNORE_INDEX``.
            max_thoughts: N, the maximum number of thoughts (>= 1).
            beta: Weight of the KL regulariser.
            prior: lambda_p of the geometric prior (defaults to config.halt_prior).
        """
        if max_thoughts < 1:
            raise ValueError("max_thoughts must be >= 1")
        prior = self.config.halt_prior if prior is None else prior
        bsz = prompt_ids.shape[0]

        h, cache = self.encode_prompt(prompt_ids)
        caches: list[MambaCache] = []
        lambdas: list[Tensor] = []
        for _ in range(max_thoughts):
            h, cache, halt_logit = self.thought_step(h, cache)
            caches.append(cache)
            lambdas.append(torch.sigmoid(halt_logit))

        # Score the answer from every step's state in one batched pass.
        per_step = self._per_example_answer_loss(MambaCache.cat(caches), target_ids.repeat(max_thoughts, 1))
        per_step = per_step.view(max_thoughts, bsz).transpose(0, 1)  # (B, N)

        lam = torch.stack(lambdas, dim=1)  # (B, N)
        lam = torch.cat([lam[:, :-1], torch.ones_like(lam[:, :1])], dim=1)
        log_continue = torch.log1p(-lam[:, :-1].clamp(max=1.0 - 1e-6))
        log_survive = F.pad(torch.cumsum(log_continue, dim=1), (1, 0))  # log prod_{j<n} (1 - lambda_j)
        p = lam * torch.exp(log_survive)

        steps = torch.arange(1, max_thoughts + 1, device=p.device, dtype=p.dtype)
        geometric = prior * (1.0 - prior) ** (steps - 1)
        geometric = geometric / geometric.sum()
        kl = (p * (torch.log(p.clamp(min=1e-12)) - torch.log(geometric))).sum(1).mean()

        reconstruction = (p * per_step).sum(1).mean()
        return PonderLossOutput(
            loss=reconstruction + beta * kl,
            reconstruction=reconstruction,
            kl=kl,
            halting_distribution=p,
            per_step_loss=per_step,
            expected_thoughts=(p * steps).sum(1),
        )

    # ----- inference ------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 1,
        max_thoughts: int = 16,
        halt_rule: HaltRule = "learned",
        halt_threshold: float = 0.5,
        convergence_tol: float = 1e-2,
        ablate: Ablation | None = None,
        temperature: float = 0.0,
        eos_id: int | None = None,
    ) -> GenerationOutput:
        """Think adaptively, then decode the answer.

        Halting rules:
            * ``"learned"``: halt once the cumulative halting probability
              ``1 - prod (1 - lambda_j)`` reaches ``halt_threshold``. The
              default 0.5 is the median of the learned halting distribution.
              At least one thought is always taken.
            * ``"convergence"``: halt when ``||h_n - h_{n-1}|| / ||h_{n-1}||``
              falls below ``convergence_tol``. This needs no training.
            * ``"fixed"``: always take ``max_thoughts`` thoughts (0 allowed).

        Halted examples are frozen while the rest of the batch keeps thinking.
        Memory stays constant throughout.

        Args:
            prompt_ids: (B, Lp), each ending with <bot>.
            max_new_tokens: Answer tokens to decode after <eot>.
            max_thoughts: Upper bound on thought steps.
            halt_rule: One of the rules above.
            halt_threshold: Threshold for the ``"learned"`` rule.
            convergence_tol: Tolerance for the ``"convergence"`` rule.
            ablate: Optional thought ablation, see :meth:`thought_step`.
            temperature: 0 for greedy decoding, otherwise softmax sampling.
            eos_id: Optional end token. Finished rows then emit pad_id.
        """
        was_training = self.training
        self.eval()
        bsz, device = prompt_ids.shape[0], prompt_ids.device

        h, cache = self.encode_prompt(prompt_ids)
        halted = torch.zeros(bsz, dtype=torch.bool, device=device)
        n_used = torch.zeros(bsz, dtype=torch.long, device=device)
        survive = torch.ones(bsz, device=device, dtype=h.dtype)

        for _ in range(max_thoughts):
            if bool(halted.all()):
                break
            active = ~halted
            h_new, cache_new, halt_logit = self.thought_step(h, cache, ablate=ablate)
            delta = (h_new - h).norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-6)
            h = torch.where(active[:, None], h_new, h)
            cache = cache_new.where(active, cache)
            n_used += active.long()
            if halt_rule == "learned":
                survive = torch.where(active, survive * (1.0 - torch.sigmoid(halt_logit)), survive)
                halted |= active & (1.0 - survive >= halt_threshold)
            elif halt_rule == "convergence":
                halted |= active & (delta < convergence_tol)
            elif halt_rule != "fixed":
                raise ValueError(f"unknown halt_rule {halt_rule!r}")

        token = torch.full((bsz, 1), self.config.eot_id, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        generated: list[Tensor] = []
        for _ in range(max_new_tokens):
            logits, cache = self(token, cache)
            logits = logits[:, -1]
            if temperature > 0:
                next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.config.pad_id), next_token)
            generated.append(next_token)
            if eos_id is not None:
                finished |= next_token == eos_id
                if bool(finished.all()):
                    break
            token = next_token[:, None]

        self.train(was_training)
        tokens = torch.stack(generated, dim=1) if generated else prompt_ids.new_zeros(bsz, 0)
        return GenerationOutput(tokens=tokens, n_thoughts=n_used)


# --------------------------------------------------------------------------- #
# Synthetic task for the demo
# --------------------------------------------------------------------------- #


class PointerChasingTask:
    """k-hop pointer chasing over a permutation given in context.

    Prompt: ``a_1 f(a_1) a_2 f(a_2) ... <sep> x <hop_k> <bot>`` with a fresh
    random permutation f per example. The chain of thought is
    ``f(x), f^2(x), ..., f^{k-1}(x)`` and the answer is ``f^k(x)``. More hops
    need more sequential computation, which makes it a simple probe of
    adaptive thinking.
    """

    def __init__(self, n_symbols: int = 6, max_hops: int = 3, seed: int = 0) -> None:
        self.n_symbols = n_symbols
        self.max_hops = max_hops
        self.sep_id = n_symbols
        self.bot_id = n_symbols + 1
        self.eot_id = n_symbols + 2
        self.pad_id = n_symbols + 3
        self.hop_base = n_symbols + 4
        self.vocab_size = n_symbols + 4 + max_hops
        self.rng = random.Random(seed)

    def sample(self, batch_size: int, hops: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Draw a batch.

        Returns:
            ``(prompt_ids (B, 2S+4), chains (B, max_hops), hops (B,))``. Here
            ``chains[i, :k]`` holds f(x) .. f^k(x) and the rest is ``IGNORE_INDEX``.
        """
        prompts: list[list[int]] = []
        chains: list[list[int]] = []
        hop_list: list[int] = []
        for _ in range(batch_size):
            perm = list(range(self.n_symbols))
            self.rng.shuffle(perm)
            order = list(range(self.n_symbols))
            self.rng.shuffle(order)
            x = self.rng.randrange(self.n_symbols)
            k = hops if hops is not None else self.rng.randint(1, self.max_hops)
            chain, v = [], x
            for _ in range(k):
                v = perm[v]
                chain.append(v)
            pairs = [tok for a in order for tok in (a, perm[a])]
            prompts.append(pairs + [self.sep_id, x, self.hop_base + k - 1, self.bot_id])
            chains.append(chain + [IGNORE_INDEX] * (self.max_hops - k))
            hop_list.append(k)
        return torch.tensor(prompts), torch.tensor(chains), torch.tensor(hop_list)

    def stage_targets(self, chains: Tensor, hops: Tensor, stage: int) -> tuple[Tensor, Tensor]:
        """Coconut stage ``stage``: drop the first min(stage, k-1) CoT steps.

        Returns:
            ``(target_ids (B, max_hops), removed (B,))``. The caller turns
            ``removed`` into ``c * removed`` thoughts.
        """
        removed = torch.clamp(hops - 1, max=stage).clamp(min=0)
        targets = torch.full_like(chains, IGNORE_INDEX)
        for i in range(chains.shape[0]):
            k, r = int(hops[i]), int(removed[i])
            targets[i, : k - r] = chains[i, r:k]
        return targets, removed


# --------------------------------------------------------------------------- #
# Self-test and demo
# --------------------------------------------------------------------------- #


def self_test(device: str = "cpu", seed: int = 0) -> None:
    """Numerical and functional checks. Raises AssertionError on failure."""
    torch.manual_seed(seed)
    cfg = LatentMambaConfig(
        vocab_size=50, bot_id=47, eot_id=48, pad_id=49,
        d_model=64, n_layers=2, d_state=16, headdim=16, chunk_size=8,
    )
    model = LatentMamba(cfg).to(device).double()

    # 1. Chunked scan (with padding and an initial state) matches the recurrence.
    bsz, seqlen = 3, 21
    x = torch.randn(bsz, seqlen, cfg.n_heads, cfg.headdim, device=device, dtype=torch.double)
    dt = F.softplus(torch.randn(bsz, seqlen, cfg.n_heads, device=device, dtype=torch.double))
    a = -torch.rand(cfg.n_heads, device=device, dtype=torch.double) * 2
    b = torch.randn(bsz, seqlen, cfg.d_state, device=device, dtype=torch.double)
    c = torch.randn(bsz, seqlen, cfg.d_state, device=device, dtype=torch.double)
    h0 = torch.randn(bsz, cfg.n_heads, cfg.headdim, cfg.d_state, device=device, dtype=torch.double)
    y_scan, h_scan = ssd_chunked_scan(x, dt, a, b, c, cfg.chunk_size, h0)
    state, ys = h0, []
    for t in range(seqlen):
        y_t, state = ssd_step(x[:, t], dt[:, t], a, b[:, t], c[:, t], state)
        ys.append(y_t)
    assert torch.allclose(y_scan, torch.stack(ys, 1), atol=1e-9), "scan output != recurrence"
    assert torch.allclose(h_scan, state, atol=1e-9), "scan final state != recurrence"

    # 2. Prefill + token-by-token continuation matches one full forward pass.
    ids = torch.randint(0, 47, (2, 19), device=device)
    full_logits, _ = model(ids)
    logits_a, cache = model(ids[:, :11])
    step_logits = [logits_a]
    for t in range(11, 19):
        lg, cache = model(ids[:, t : t + 1], cache)
        step_logits.append(lg)
    assert torch.allclose(full_logits, torch.cat(step_logits, 1), atol=1e-8), "cached decoding mismatch"
    size_before = cache.num_bytes()

    # 3. Thoughts leave the cache size unchanged and losses backpropagate.
    prompt = torch.randint(0, 47, (4, 10), device=device)
    prompt[:, -1] = cfg.bot_id
    target = torch.randint(0, 47, (4, 3), device=device)
    target[0, 2] = IGNORE_INDEX
    h, cache = model.encode_prompt(prompt)
    for _ in range(5):
        h, cache, _ = model.thought_step(h, cache)
    assert cache.num_bytes() == size_before * 2, "cache size should depend only on batch size"

    out = model.latent_loss(prompt, target, n_thoughts=torch.tensor([0, 1, 2, 3], device=device))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert model.thought_projector.proj.weight.grad is not None, "no gradient through thoughts"
    model.zero_grad()

    pout = model.ponder_loss(prompt, target, max_thoughts=4)
    pout.loss.backward()
    assert torch.allclose(pout.halting_distribution.sum(1), torch.ones(4, device=device, dtype=torch.double))
    assert model.halting_head.fc2.weight.grad is not None, "no gradient to halting head"

    # 4. Per-example thought counts: an example that stopped is frozen while others continue.
    h2, c2 = model.encode_prompt(prompt[:2])
    _, c_single, _ = model.thought_step(h2, c2)
    _, c_mixed = model._run_thoughts(h2, c2, torch.tensor([1, 3], device=device))
    assert torch.allclose(c_mixed.layers[-1].ssm[0], c_single.layers[-1].ssm[0]), "stopped example changed"
    assert not torch.allclose(c_mixed.layers[-1].ssm[1], c_single.layers[-1].ssm[1]), "active example frozen"

    # 5. Generation with every halting rule.
    for rule in ("learned", "convergence", "fixed"):
        gen = model.generate(prompt, max_new_tokens=3, max_thoughts=4, halt_rule=rule)
        assert gen.tokens.shape == (4, 3) and int(gen.n_thoughts.max()) <= 4
    gen = model.generate(prompt, max_new_tokens=2, max_thoughts=3, ablate="zeros")
    assert gen.tokens.shape == (4, 2)
    print("self-test passed: scan == recurrence, cached decoding exact, O(1) state, gradients OK")


def _accuracy_by_hops(
    model: LatentMamba,
    task: PointerChasingTask,
    n_eval: int,
    device: torch.device,
    **generate_kwargs: object,
) -> dict[int, tuple[float, float]]:
    """Return {hops: (accuracy, mean thoughts)} on fresh k-hop problems."""
    results: dict[int, tuple[float, float]] = {}
    for k in range(1, task.max_hops + 1):
        prompts, chains, _ = task.sample(n_eval, hops=k)
        prompts, chains = prompts.to(device), chains.to(device)
        gen = model.generate(prompts, max_new_tokens=1, **generate_kwargs)  # type: ignore[arg-type]
        acc = (gen.tokens[:, 0] == chains[:, k - 1]).float().mean().item()
        results[k] = (acc, gen.n_thoughts.float().mean().item())
    return results


def run_demo(
    steps_per_stage: int = 600,
    ponder_steps: int = 800,
    batch_size: int = 64,
    thoughts_per_step: int = 2,
    lr: float = 2e-3,
    device: str | None = None,
    seed: int = 0,
) -> LatentMamba:
    """Train a tiny Latent Mamba through every stage of the recipe.

    The stages are language CoT, the Coconut curriculum, and PonderNet
    halting. The task is pointer chasing, and the run reports accuracy and
    thoughts used per hop count. This is a small-scale smoke test of the
    pipeline, not a benchmark.
    """
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    task = PointerChasingTask(n_symbols=6, max_hops=3, seed=seed)
    cfg = LatentMambaConfig(
        vocab_size=task.vocab_size, bot_id=task.bot_id, eot_id=task.eot_id, pad_id=task.pad_id,
        d_model=128, n_layers=4, d_state=32, headdim=32, chunk_size=32,
    )
    model = LatentMamba(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={dev}  params={n_params / 1e6:.2f}M  task: {task.max_hops}-hop pointer chasing, "
          f"{task.n_symbols} symbols")

    def train(label: str, steps: int, step_fn: Callable[[], tuple[Tensor, str]]) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # reset per stage, as in Coconut
        start = time.time()
        for step in range(1, steps + 1):
            loss, info = step_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % max(1, steps // 4) == 0 or step == steps:
                print(f"  [{label}] step {step:4d}/{steps}  loss {loss.item():.3f}  {info}  "
                      f"({time.time() - start:.0f}s)")

    c = thoughts_per_step
    for stage in range(task.max_hops):
        def coconut_step(stage: int = stage) -> tuple[Tensor, str]:
            prompts, chains, hops = task.sample(batch_size)
            targets, removed = task.stage_targets(chains, hops, stage)
            out = model.latent_loss(prompts.to(dev), targets.to(dev), (c * removed).to(dev))
            return out.loss, f"mean thoughts {out.n_thoughts.float().mean().item():.2f}"

        name = "language CoT" if stage == 0 else f"coconut stage {stage}"
        train(name, steps_per_stage, coconut_step)

    max_thoughts = c * (task.max_hops - 1) + 2

    def ponder_step() -> tuple[Tensor, str]:
        prompts, chains, hops = task.sample(batch_size)
        targets, _ = task.stage_targets(chains, hops, task.max_hops)
        out = model.ponder_loss(prompts.to(dev), targets.to(dev), max_thoughts=max_thoughts, beta=0.01)
        means = [out.expected_thoughts[hops.to(dev) == k].mean().item() for k in range(1, task.max_hops + 1)]
        return out.loss, "E[thoughts] by hops " + " / ".join(f"{m:.2f}" for m in means)

    train("ponder halting", ponder_steps, ponder_step)

    print("\nevaluation (500 fresh problems per hop count, chance = "
          f"{1 / task.n_symbols:.2f})")
    evals = {
        "learned halting": dict(max_thoughts=max_thoughts, halt_rule="learned"),
        "fixed max thoughts": dict(max_thoughts=max_thoughts, halt_rule="fixed"),
        "no thoughts": dict(max_thoughts=0, halt_rule="fixed"),
        "ablated (zeros)": dict(max_thoughts=max_thoughts, halt_rule="learned", ablate="zeros"),
    }
    header = "  ".join(f"{k}-hop acc / thoughts" for k in range(1, task.max_hops + 1))
    print(f"  {'mode':<20}{header}")
    for label, kwargs in evals.items():
        res = _accuracy_by_hops(model, task, 500, dev, **kwargs)
        row = "  ".join(f"{res[k][0]:>8.2f} / {res[k][1]:<9.2f}" for k in range(1, task.max_hops + 1))
        print(f"  {label:<20}{row}")
    return model


if __name__ == "__main__":
    self_test()
    run_demo()
