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
* Prompts of different lengths are batched with left padding plus a
  ``prompt_mask`` (True for real tokens). Pad positions get dt = 0 and a zero
  conv input, so they leave the state exactly as if they were absent.

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

import random
import time
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IGNORE_INDEX: int = -100
"""Target value that is excluded from every loss (padding in target tensors)."""

HaltRule = Literal["learned", "convergence", "fixed"]
Ablation = Literal["zeros", "noise"]

from latent_mamba.main import LatentMamba, LatentMambaConfig, ssd_chunked_scan, ssd_step

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

    def sample(
        self, batch_size: int, hops: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
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

    def stage_targets(
        self, chains: Tensor, hops: Tensor, stage: int
    ) -> tuple[Tensor, Tensor]:
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
        vocab_size=50,
        bot_id=47,
        eot_id=48,
        pad_id=49,
        d_model=64,
        n_layers=2,
        d_state=16,
        headdim=16,
        chunk_size=8,
    )
    model = LatentMamba(cfg).to(device).double()

    # 1. Chunked scan (with padding and an initial state) matches the recurrence.
    bsz, seqlen = 3, 21
    x = torch.randn(
        bsz, seqlen, cfg.n_heads, cfg.headdim, device=device, dtype=torch.double
    )
    dt = F.softplus(
        torch.randn(bsz, seqlen, cfg.n_heads, device=device, dtype=torch.double)
    )
    a = -torch.rand(cfg.n_heads, device=device, dtype=torch.double) * 2
    b = torch.randn(bsz, seqlen, cfg.d_state, device=device, dtype=torch.double)
    c = torch.randn(bsz, seqlen, cfg.d_state, device=device, dtype=torch.double)
    h0 = torch.randn(
        bsz, cfg.n_heads, cfg.headdim, cfg.d_state, device=device, dtype=torch.double
    )
    y_scan, h_scan = ssd_chunked_scan(x, dt, a, b, c, cfg.chunk_size, h0)
    state, ys = h0, []
    for t in range(seqlen):
        y_t, state = ssd_step(x[:, t], dt[:, t], a, b[:, t], c[:, t], state)
        ys.append(y_t)
    assert torch.allclose(
        y_scan, torch.stack(ys, 1), atol=1e-9
    ), "scan output != recurrence"
    assert torch.allclose(h_scan, state, atol=1e-9), "scan final state != recurrence"

    # 2. Prefill + token-by-token continuation matches one full forward pass.
    ids = torch.randint(0, 47, (2, 19), device=device)
    full_logits, _ = model(ids)
    logits_a, cache = model(ids[:, :11])
    step_logits = [logits_a]
    for t in range(11, 19):
        lg, cache = model(ids[:, t : t + 1], cache)
        step_logits.append(lg)
    assert torch.allclose(
        full_logits, torch.cat(step_logits, 1), atol=1e-8
    ), "cached decoding mismatch"
    size_before = cache.num_bytes()

    # 3. Thoughts leave the cache size unchanged and losses backpropagate.
    prompt = torch.randint(0, 47, (4, 10), device=device)
    prompt[:, -1] = cfg.bot_id
    target = torch.randint(0, 47, (4, 3), device=device)
    target[0, 2] = IGNORE_INDEX
    h, cache = model.encode_prompt(prompt)
    for _ in range(5):
        h, cache, _ = model.thought_step(h, cache)
    assert (
        cache.num_bytes() == size_before * 2
    ), "cache size should depend only on batch size"

    out = model.latent_loss(
        prompt, target, n_thoughts=torch.tensor([0, 1, 2, 3], device=device)
    )
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert (
        model.thought_projector.proj.weight.grad is not None
    ), "no gradient through thoughts"
    model.zero_grad()

    pout = model.ponder_loss(prompt, target, max_thoughts=4)
    pout.loss.backward()
    assert torch.allclose(
        pout.halting_distribution.sum(1),
        torch.ones(4, device=device, dtype=torch.double),
    )
    assert model.halting_head.fc2.weight.grad is not None, "no gradient to halting head"

    # 4. Per-example thought counts: an example that stopped is frozen while others continue.
    h2, c2 = model.encode_prompt(prompt[:2])
    _, c_single, _ = model.thought_step(h2, c2)
    _, c_mixed = model._run_thoughts(h2, c2, torch.tensor([1, 3], device=device))
    assert torch.allclose(
        c_mixed.layers[-1].ssm[0], c_single.layers[-1].ssm[0]
    ), "stopped example changed"
    assert not torch.allclose(
        c_mixed.layers[-1].ssm[1], c_single.layers[-1].ssm[1]
    ), "active example frozen"

    # 5. Generation with every halting rule.
    for rule in ("learned", "convergence", "fixed"):
        gen = model.generate(prompt, max_new_tokens=3, max_thoughts=4, halt_rule=rule)
        assert gen.tokens.shape == (4, 3) and int(gen.n_thoughts.max()) <= 4
    gen = model.generate(prompt, max_new_tokens=2, max_thoughts=3, ablate="zeros")
    assert gen.tokens.shape == (4, 2)

    # 6. Left padding with a mask is exact: a padded row matches the same prompt run alone.
    short = prompt[:, 4:]  # (4, 6), still ends with <bot>
    padded = torch.cat([torch.full_like(prompt[:, :4], cfg.pad_id), short], dim=1)
    pmask = torch.ones_like(padded, dtype=torch.bool)
    pmask[:, :4] = False
    padded[1:], pmask[1:] = prompt[1:], True  # mixed batch: only row 0 is padded
    h_pad, c_pad = model.encode_prompt(padded, pmask)
    h_ref, c_ref = model.encode_prompt(short[:1])
    assert torch.allclose(
        h_pad[:1], h_ref, atol=1e-9
    ), "left padding changed the hidden state"
    for lp, lr in zip(c_pad.layers, c_ref.layers):
        assert torch.allclose(lp.ssm[:1], lr.ssm, atol=1e-9) and torch.allclose(
            lp.conv[:1], lr.conv, atol=1e-9
        )
    h_full, _ = model.encode_prompt(prompt[1:])
    assert torch.allclose(h_pad[1:], h_full, atol=1e-9), "mask changed unpadded rows"
    print(
        "self-test passed: scan == recurrence, cached decoding exact, O(1) state, gradients OK, padding exact"
    )


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
        vocab_size=task.vocab_size,
        bot_id=task.bot_id,
        eot_id=task.eot_id,
        pad_id=task.pad_id,
        d_model=128,
        n_layers=4,
        d_state=32,
        headdim=32,
        chunk_size=32,
    )
    model = LatentMamba(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={dev}  params={n_params / 1e6:.2f}M  task: {task.max_hops}-hop pointer chasing, "
        f"{task.n_symbols} symbols"
    )

    def train(
        label: str, steps: int, step_fn: Callable[[], tuple[Tensor, str]]
    ) -> None:
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01
        )  # reset per stage, as in Coconut
        start = time.time()
        for step in range(1, steps + 1):
            loss, info = step_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % max(1, steps // 4) == 0 or step == steps:
                print(
                    f"  [{label}] step {step:4d}/{steps}  loss {loss.item():.3f}  {info}  "
                    f"({time.time() - start:.0f}s)"
                )

    c = thoughts_per_step
    for stage in range(task.max_hops):

        def coconut_step(stage: int = stage) -> tuple[Tensor, str]:
            prompts, chains, hops = task.sample(batch_size)
            targets, removed = task.stage_targets(chains, hops, stage)
            out = model.latent_loss(
                prompts.to(dev), targets.to(dev), (c * removed).to(dev)
            )
            return out.loss, f"mean thoughts {out.n_thoughts.float().mean().item():.2f}"

        name = "language CoT" if stage == 0 else f"coconut stage {stage}"
        train(name, steps_per_stage, coconut_step)

    max_thoughts = c * (task.max_hops - 1) + 2

    def ponder_step() -> tuple[Tensor, str]:
        prompts, chains, hops = task.sample(batch_size)
        targets, _ = task.stage_targets(chains, hops, task.max_hops)
        out = model.ponder_loss(
            prompts.to(dev), targets.to(dev), max_thoughts=max_thoughts, beta=0.01
        )
        means = [
            out.expected_thoughts[hops.to(dev) == k].mean().item()
            for k in range(1, task.max_hops + 1)
        ]
        return out.loss, "E[thoughts] by hops " + " / ".join(f"{m:.2f}" for m in means)

    train("ponder halting", ponder_steps, ponder_step)

    print(
        "\nevaluation (500 fresh problems per hop count, chance = "
        f"{1 / task.n_symbols:.2f})"
    )
    evals = {
        "learned halting": dict(max_thoughts=max_thoughts, halt_rule="learned"),
        "fixed max thoughts": dict(max_thoughts=max_thoughts, halt_rule="fixed"),
        "no thoughts": dict(max_thoughts=0, halt_rule="fixed"),
        "ablated (zeros)": dict(
            max_thoughts=max_thoughts, halt_rule="learned", ablate="zeros"
        ),
    }
    header = "  ".join(f"{k}-hop acc / thoughts" for k in range(1, task.max_hops + 1))
    print(f"  {'mode':<20}{header}")
    for label, kwargs in evals.items():
        res = _accuracy_by_hops(model, task, 500, dev, **kwargs)
        row = "  ".join(
            f"{res[k][0]:>8.2f} / {res[k][1]:<9.2f}"
            for k in range(1, task.max_hops + 1)
        )
        print(f"  {label:<20}{row}")
    return model


if __name__ == "__main__":
    self_test()
    run_demo()
