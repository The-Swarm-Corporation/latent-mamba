"""
train.py: Latent Mamba on real reasoning benchmarks (Step 2 of the evaluation).

Fine-tunes a pretrained Mamba-2 with the Latent Mamba recipe on Coconut's
datasets and compares it against

* the same Mamba answering directly (no chain of thought),
* the same Mamba writing its chain of thought out,
* a Coconut GPT-2 of similar size trained with the same curriculum,

and measures state size, peak memory and latency against thinking budget.

Subcommands
-----------
data     Download GSM8k-Aug (iCoT) and ProsQA, convert a locally generated
         ProntoQA file, and write Coconut's JSON format
         ({"question", "steps", "answer"}) to ``data/{dataset}_{split}.json``.
train    Fine-tune one (model, dataset, method). Methods:
           cot     question <bot><eot> step_1 .. step_s ### answer
           nocot   question <bot><eot> ### answer
           latent  CoT stage, Coconut stages 1..K (stage k replaces the first
                   k steps with c*k thoughts), a final stage (all language
                   reasoning replaced by c*K thoughts) and, for Mamba only, a
                   PonderNet halting stage.
eval     Accuracy on a split under several inference modes (fixed, learned
         halting, convergence halting, no thoughts, zero/noise ablations) and
         an accuracy-vs-thought-budget sweep.
bench    State size, peak memory and latency against thinking budget for
         Mamba and the Coconut Transformer.
report   Collect every eval and bench JSON into markdown tables.
upload   Upload a run folder or result file to the Hugging Face Hub.

Results go to https://huggingface.co/kye/latent-mamba by default, one folder
per run (``{dataset}-{model}-{method}/`` with checkpoints, run.json,
metrics.jsonl and eval JSON). ``train`` checks the token before training,
uploads after every phase and again at the end; pass ``--no-hf-upload`` to
skip. ``--wandb`` logs to Weights & Biases. Training resumes by default:
rerunning the same command skips phases already finished in ``--out``.

Full recipe (GSM8k shown; ProsQA/ProntoQA run the CoT stage inside the latent
run, as Coconut does, so they need no --init-from)::

    python train.py data --dataset gsm8k prosqa
    torchrun --nproc_per_node 4 train.py train --model mamba2-130m --dataset gsm8k --method cot
    torchrun --nproc_per_node 4 train.py train --model mamba2-130m --dataset gsm8k --method nocot
    torchrun --nproc_per_node 4 train.py train --model mamba2-130m --dataset gsm8k --method latent \\
        --init-from runs/gsm8k-mamba2-130m-cot/cot.pt --start-stage 1
    # the same three runs with --model gpt2 (the latent run has no ponder stage)
    python train.py eval --ckpt runs/gsm8k-mamba2-130m-latent/ponder.pt --split test
    python train.py bench --models mamba2-130m gpt2
    python train.py report

Curriculum defaults follow Coconut's published configs (4 GPUs x 32 examples,
AdamW at a constant lr 1e-4, weight decay 0.01, optimiser reset at every stage,
checkpoint chosen by validation accuracy). The halting stage, gradient
clipping and bf16 autocast are additions.

Notes
-----
* Pretrained weights come from ``state-spaces/mamba2-*``. Their parameter
  names already match ``main.py``, so they load without renaming. The three
  special tokens use spare rows of the padded vocabulary and start from the
  embedding of "<<", as Coconut's do.
* The selective scan runs in float32 even under bf16 autocast. On CUDA, the
  Triton kernel from ``mamba_ssm`` replaces it for prompts and answers by
  default. It is checked against the reference scan (outputs and gradients) on
  startup; if ``mamba_ssm`` is missing or disagrees, the reference scan is used
  instead. ``--fused-scan`` makes the kernel mandatory, ``--no-fused-scan``
  turns it off. Inference thoughts and decoding also use mamba_ssm's fused
  single-step kernel (and causal-conv1d's, if installed) and are replayed as
  CUDA graphs (``--no-cuda-graphs`` turns that off). Training thoughts are
  compiled with torch.compile (``--compile-thoughts``). Every fast path is
  checked against the reference at startup and falls back to it on mismatch.
* Multi-GPU runs use ``torchrun``. Gradients are all-reduced by hand rather
  than through DDP, because the losses are not ``forward`` calls and some
  parameters get no gradient in some stages.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import random
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

with contextlib.suppress(ImportError):
    __import__("setuptools")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import main as lm
from main import (
    IGNORE_INDEX,
    GenerationOutput,
    LatentLossOutput,
    LatentMamba,
    LatentMambaConfig,
)

# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Preset:
    """Curriculum and decoding settings for one dataset.

    Attributes:
        c_thought: Continuous thoughts per removed reasoning step (c).
        max_latent_stage: Number of latent stages (K).
        cot_epochs: Epochs of the language-CoT stage. Also used for the no-CoT baseline.
        epochs_per_stage: Epochs of each latent stage 1..K.
        final_epochs: Epochs of stage K+1, where all language reasoning is
            replaced by c*K thoughts.
        ponder_epochs: Epochs of PonderNet halting (Mamba only; not in Coconut).
        max_new_tokens: Decoding budget at evaluation.
    """

    c_thought: int
    max_latent_stage: int
    cot_epochs: int
    epochs_per_stage: int
    final_epochs: int
    ponder_epochs: int
    max_new_tokens: int


PRESETS: dict[str, Preset] = {
    # args/gsm_cot.yaml: 25 epochs. args/gsm_coconut.yaml: c=2, K=3, 3 epochs per
    # stage, resumes at epoch 3 of 25, so the final stage gets 13 epochs.
    "gsm8k": Preset(
        c_thought=2,
        max_latent_stage=3,
        cot_epochs=25,
        epochs_per_stage=3,
        final_epochs=13,
        ponder_epochs=3,
        max_new_tokens=64,
    ),
    # args/prosqa_coconut.yaml and args/prontoqa_coconut.yaml: c=1, K=6, 5 epochs
    # per stage (stage 0 included), 50 epochs, so the final stage gets 15.
    "prosqa": Preset(
        c_thought=1,
        max_latent_stage=6,
        cot_epochs=5,
        epochs_per_stage=5,
        final_epochs=15,
        ponder_epochs=5,
        max_new_tokens=128,
    ),
    "prontoqa": Preset(
        c_thought=1,
        max_latent_stage=6,
        cot_epochs=5,
        epochs_per_stage=5,
        final_epochs=15,
        ponder_epochs=5,
        max_new_tokens=128,
    ),
}

DATASETS = tuple(PRESETS)
SPLITS = ("train", "valid", "test")

MAMBA_REPOS = {
    f"mamba2-{s}": f"state-spaces/mamba2-{s}"
    for s in ("130m", "370m", "780m", "1.3b", "2.7b")
}
GPT2_REPOS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
}
TINY_MODELS = ("tiny-mamba", "tiny-gpt2")  # random init, for smoke tests
MODEL_CHOICES = (*MAMBA_REPOS, *GPT2_REPOS, *TINY_MODELS)


def is_mamba(name: str) -> bool:
    return name in MAMBA_REPOS or name == "tiny-mamba"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

_ICOT = "da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/{split}.txt"
ICOT_URLS = {  # train.txt is stored in Git LFS, so it is served from a different host
    "train": "https://media.githubusercontent.com/media/" + _ICOT.format(split="train"),
    "valid": "https://raw.githubusercontent.com/" + _ICOT.format(split="valid"),
    "test": "https://raw.githubusercontent.com/" + _ICOT.format(split="test"),
}
PROSQA_URL = (
    "https://raw.githubusercontent.com/facebookresearch/coconut/"
    "27273cb8cca4bb763c041a63b036d0c3b7cbbb48/data/prosqa_{split}.json"
)
PRONTOQA_HELP = """\
ProntoQA must be generated locally (this is Coconut's procedure):

    git clone https://github.com/asaparov/prontoqa && cd prontoqa
    git checkout f0145b867b3c106285ec9ea1941a3f6eb7c6162d
    python run_experiment.py --model-name json --model-size dummy --ordering random \\
        --num-trials 10000 --few-shot-examples 0 --ontology fictional \\
        --min-hops 5 --max-hops 5 --hops-skip 1

then run:  python train.py data --dataset prontoqa --prontoqa-raw path/to/5hop_0shot_random.json
"""


def _download(url: str) -> bytes:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "latent-mamba-train"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()


def _write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    os.replace(tmp, path)
    print(f"  wrote {path} ({len(rows)} examples)")


def convert_icot(text: str) -> list[dict]:
    """iCoT ``question||<<a>> <<b>> #### answer`` lines, parsed as in Coconut's gsm_icot.py."""
    return [
        {
            "question": line.split("||")[0],
            "steps": line.split("||")[1].split("##")[0].strip().split(" "),
            "answer": line.split("##")[-1].strip(),
        }
        for line in text.split(
            "\n"
        )  # only "\n", like readlines(): questions contain "\r" and "\u2028"
        if line.strip()
    ]


def convert_prontoqa(raw: dict) -> dict[str, list[dict]]:
    """Coconut's preprocessing/prontoqa.py: 9000 / 200 / rest split."""
    rows = []
    for value in raw.values():
        ex = value["test_example"]
        cot = ex["chain_of_thought"]
        rows.append(
            {
                "question": ex["question"] + " " + ex["query"],
                "steps": [" ".join(cot[i : i + 2]) for i in range(0, len(cot), 2)],
                "answer": ex["answer"],
            }
        )
    return {"train": rows[:9000], "valid": rows[9000:9200], "test": rows[9200:]}


def data_cmd(args: argparse.Namespace) -> None:
    out = Path(args.data_dir)
    for name in args.dataset:
        print(f"[{name}]")
        if name == "gsm8k":
            for split in SPLITS:
                _write_json(
                    convert_icot(_download(ICOT_URLS[split]).decode()),
                    out / f"gsm8k_{split}.json",
                )
        elif name == "prosqa":
            for split in SPLITS:
                raw = json.loads(_download(PROSQA_URL.format(split=split)))
                rows = [
                    {
                        "question": r["question"],
                        "steps": r["steps"],
                        "answer": r["answer"],
                    }
                    for r in raw
                ]
                _write_json(rows, out / f"prosqa_{split}.json")
        elif name == "prontoqa":
            if not args.prontoqa_raw:
                print(PRONTOQA_HELP)
                continue
            for split, rows in convert_prontoqa(
                json.loads(Path(args.prontoqa_raw).read_text())
            ).items():
                _write_json(rows, out / f"prontoqa_{split}.json")


def load_rows(
    data_dir: str, dataset: str, split: str, limit: int | None = None
) -> list[dict]:
    path = Path(data_dir) / f"{dataset}_{split}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run: python train.py data --dataset {dataset}"
        )
    rows = json.loads(path.read_text())
    return rows[:limit] if limit else rows


@dataclass
class Example:
    """One tokenised example, in Coconut's layout.

    question: "<question>\\n"; steps[i]: "<step>\\n"; answer: "### <answer>" + eos.
    """

    question: list[int]
    steps: list[list[int]]
    answer: list[int]
    answer_text: str
    n_steps: int

    def n_tokens(self) -> int:
        return len(self.question) + sum(map(len, self.steps)) + len(self.answer)


def normalise_answer(text: str) -> str:
    return text.replace(",", "").strip()


def parse_answer(generated: str) -> str:
    """Coconut's rule: the text after the last '#', without commas."""
    return normalise_answer(generated.split("#")[-1])


def tokenize(rows: Sequence[dict], tokenizer: Any, eos_id: int) -> list[Example]:
    def enc(texts: list[str]) -> list[list[int]]:
        return tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []

    questions = enc([r["question"] + "\n" for r in rows])
    flat_steps = enc([s + "\n" for r in rows for s in r["steps"]])
    answers = enc(["### " + r["answer"] for r in rows])
    examples, i = [], 0
    for row, q, a in zip(rows, questions, answers):
        n = len(row["steps"])
        examples.append(
            Example(
                q,
                flat_steps[i : i + n],
                a + [eos_id],
                normalise_answer(row["answer"]),
                n,
            )
        )
        i += n
    return examples


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class CoconutGPT2(nn.Module):
    """Coconut (Hao et al., 2024) on GPT-2: the Transformer baseline.

    Exposes the same ``latent_loss`` / ``generate`` interface as
    :class:`main.LatentMamba`. The last hidden state (after ``ln_f``) is fed
    back as the next input embedding, and every thought adds a KV-cache entry.
    Left padding uses the attention mask. When examples in a batch take
    different numbers of thoughts, the cache slot of an example that has
    already stopped is masked out, which is exact.
    """

    def __init__(self, model: nn.Module, bot_id: int, eot_id: int, pad_id: int) -> None:
        super().__init__()
        self.lm = model
        self.bot_id, self.eot_id, self.pad_id = bot_id, eot_id, pad_id

    @property
    def max_positions(self) -> int:
        return int(self.lm.config.n_positions)

    def embed(self, ids: Tensor) -> Tensor:
        return self.lm.transformer.wte(ids)

    def _forward(
        self, embeds: Tensor, attn: Tensor, pos: Tensor, cache: Any
    ) -> tuple[Tensor, Any]:
        if int(pos.max()) >= self.max_positions:
            raise ValueError(
                f"sequence exceeds GPT-2's context of {self.max_positions} positions"
            )
        out = self.lm.transformer(
            inputs_embeds=embeds,
            attention_mask=attn,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        return out.last_hidden_state, out.past_key_values

    def think(
        self, prompt_ids: Tensor, n_thoughts: Tensor, prompt_mask: Tensor | None = None
    ) -> tuple[Tensor, Any, Tensor, Tensor]:
        """Prefill the prompt and run ``n_thoughts[i]`` thoughts for example i.

        Returns ``(h, cache, attention_mask, next_position)``.
        """
        if not bool((prompt_ids[:, -1] == self.bot_id).all()):
            raise ValueError("every prompt must end with bot_id")
        mask = (
            torch.ones_like(prompt_ids, dtype=torch.bool)
            if prompt_mask is None
            else prompt_mask.bool()
        )
        attn = mask.long()
        hidden, cache = self._forward(
            self.embed(prompt_ids), attn, (attn.cumsum(1) - 1).clamp(min=0), None
        )
        h, next_pos = hidden[:, -1], attn.sum(1)
        for step in range(int(n_thoughts.max()) if n_thoughts.numel() else 0):
            active = step < n_thoughts
            attn = torch.cat([attn, active.long()[:, None]], dim=1)
            hidden, cache = self._forward(h[:, None], attn, next_pos[:, None], cache)
            h = torch.where(active[:, None], hidden[:, 0], h)
            next_pos = next_pos + active.long()
        return h, cache, attn, next_pos

    def _answer_inputs(self, target_ids: Tensor) -> Tensor:
        eot = torch.full_like(target_ids[:, :1], self.eot_id)
        shifted = torch.cat([eot, target_ids[:, :-1]], dim=1)
        return shifted.masked_fill(shifted == IGNORE_INDEX, self.pad_id)

    def latent_loss(
        self,
        prompt_ids: Tensor,
        target_ids: Tensor,
        n_thoughts: int | Tensor,
        prompt_mask: Tensor | None = None,
    ) -> LatentLossOutput:
        n = (
            torch.as_tensor(n_thoughts, device=prompt_ids.device)
            .expand(prompt_ids.shape[0])
            .long()
        )
        _, cache, attn, next_pos = self.think(prompt_ids, n, prompt_mask)
        inputs = self._answer_inputs(target_ids)
        length = inputs.shape[1]
        attn = torch.cat([attn, torch.ones_like(inputs)], dim=1)
        pos = next_pos[:, None] + torch.arange(length, device=inputs.device)
        hidden, _ = self._forward(self.embed(inputs), attn, pos, cache)
        logits = self.lm.lm_head(hidden)
        loss = F.cross_entropy(
            logits.transpose(1, 2).float(), target_ids, ignore_index=IGNORE_INDEX
        )
        return LatentLossOutput(loss=loss, n_thoughts=n)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 1,
        max_thoughts: int = 16,
        halt_rule: str = "fixed",
        ablate: str | None = None,
        eos_id: int | None = None,
        prompt_mask: Tensor | None = None,
        **_: Any,
    ) -> GenerationOutput:
        """Greedy decoding after a fixed number of thoughts (Coconut has no halting)."""
        if halt_rule != "fixed" or ablate is not None:
            raise ValueError(
                "the Coconut GPT-2 baseline supports only halt_rule='fixed' without ablation"
            )
        was_training = self.training
        self.eval()
        bsz, device = prompt_ids.shape[0], prompt_ids.device
        n = torch.full((bsz,), max_thoughts, dtype=torch.long, device=device)
        _, cache, attn, next_pos = self.think(prompt_ids, n, prompt_mask)
        token = torch.full((bsz, 1), self.eot_id, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        generated = []
        for _ in range(max_new_tokens):
            attn = torch.cat([attn, torch.ones_like(token)], dim=1)
            hidden, cache = self._forward(
                self.embed(token), attn, next_pos[:, None], cache
            )
            next_pos = next_pos + 1
            next_token = self.lm.lm_head(hidden[:, -1]).argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, self.pad_id), next_token
            )
            generated.append(next_token)
            if eos_id is not None:
                finished |= next_token == eos_id
                if bool(finished.all()):
                    break
            token = next_token[:, None]
        self.train(was_training)
        return GenerationOutput(tokens=torch.stack(generated, dim=1), n_thoughts=n)

    @staticmethod
    def cache_bytes(cache: Any) -> int:
        return sum(
            t.numel() * t.element_size()
            for layer in cache.layers
            for t in (layer.keys, layer.values)
            if t is not None
        )


@dataclass
class Bundle:
    """A model with its tokenizer and special-token ids."""

    name: str
    kind: str  # "mamba" or "gpt2"
    model: nn.Module
    tokenizer: Any
    bot_id: int
    eot_id: int
    pad_id: int
    eos_id: int


def _copy_rows(weight: Tensor, rows: Sequence[int], tokenizer: Any) -> None:
    """Initialise new special-token rows from the embedding of "<<", as Coconut does."""
    source = tokenizer.encode("<<", add_special_tokens=False)[0]
    with torch.no_grad():
        for row in rows:
            weight[row] = weight[source]


def build_mamba(
    name: str, pretrained: bool, chunk_size: int, halt_prior: float
) -> Bundle:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    n_tok = len(tokenizer)
    if name == "tiny-mamba":
        dims = dict(d_model=128, n_layers=2, d_state=32, headdim=32, expand=2, d_conv=4)
        ckpt_vocab = 0
    else:
        repo = MAMBA_REPOS[name]
        hf = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        ssm = hf.get("ssm_cfg", {})
        if (
            ssm.get("layer") != "Mamba2"
            or hf.get("attn_layer_idx")
            or hf.get("d_intermediate", 0)
        ):
            raise ValueError(f"{repo} is not a pure Mamba-2 model")
        if ssm.get("ngroups", 1) != 1:
            raise ValueError("only ngroups=1 checkpoints are supported")
        dims = dict(
            d_model=hf["d_model"],
            n_layers=hf["n_layer"],
            d_state=ssm.get("d_state", 128),
            headdim=ssm.get("headdim", 64),
            expand=ssm.get("expand", 2),
            d_conv=ssm.get("d_conv", 4),
        )
        multiple = hf.get("pad_vocab_size_multiple", 1)
        ckpt_vocab = -(-hf["vocab_size"] // multiple) * multiple
    vocab = max(ckpt_vocab, -(-(n_tok + 3) // 16) * 16)
    bot_id, eot_id, pad_id = (
        n_tok,
        n_tok + 1,
        n_tok + 2,
    )  # spare rows of the padded vocabulary
    cfg = LatentMambaConfig(
        vocab_size=vocab,
        bot_id=bot_id,
        eot_id=eot_id,
        pad_id=pad_id,
        chunk_size=chunk_size,
        halt_prior=halt_prior,
        **dims,
    )
    model = LatentMamba(cfg)
    if pretrained and name != "tiny-mamba":
        state = torch.load(
            hf_hub_download(repo, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        state = {k: v.float() for k, v in state.items()}
        for key in ("backbone.embedding.weight", "lm_head.weight"):
            if key in state and state[key].shape[0] < vocab:
                grown = model.state_dict()[key].clone()
                grown[: state[key].shape[0]] = state[key]
                state[key] = grown
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = [
            k
            for k in missing
            if not k.startswith(("thought_projector.", "halting_head."))
        ]
        if bad or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={bad} unexpected={unexpected}"
            )
        _copy_rows(model.backbone.embedding.weight, [bot_id, eot_id, pad_id], tokenizer)
        model.thought_projector.match_embedding_scale(
            model.backbone.embedding.weight[:n_tok]
        )
    return Bundle(
        name, "mamba", model, tokenizer, bot_id, eot_id, pad_id, tokenizer.eos_token_id
    )


def build_gpt2(name: str, pretrained: bool) -> Bundle:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
    )

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>"])
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    if name == "tiny-gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(n_layer=2, n_embd=128, n_head=4, vocab_size=len(tokenizer))
        )
    elif pretrained:
        model = AutoModelForCausalLM.from_pretrained(GPT2_REPOS[name])
        model.resize_token_embeddings(len(tokenizer))
        _copy_rows(model.transformer.wte.weight, [bot_id, eot_id], tokenizer)
    else:
        config = AutoConfig.from_pretrained(GPT2_REPOS[name])
        config.vocab_size = len(tokenizer)
        model = GPT2LMHeadModel(config)
    pad_id = tokenizer.eos_token_id
    return Bundle(
        name,
        "gpt2",
        CoconutGPT2(model, bot_id, eot_id, pad_id),
        tokenizer,
        bot_id,
        eot_id,
        pad_id,
        tokenizer.eos_token_id,
    )


def build(
    name: str, pretrained: bool = True, chunk_size: int = 64, halt_prior: float = 0.2
) -> Bundle:
    if is_mamba(name):
        return build_mamba(name, pretrained, chunk_size, halt_prior)
    return build_gpt2(name, pretrained)


def load_checkpoint(
    path: str | Path, chunk_size: int | None = None
) -> tuple[Bundle, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    bundle = build(
        ckpt["model_name"],
        pretrained=False,
        chunk_size=chunk_size or ckpt["chunk_size"],
        halt_prior=ckpt["halt_prior"],
    )
    bundle.model.load_state_dict(ckpt["model"])
    return bundle, ckpt


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Selective scan: float32 reference, or the fused mamba_ssm kernel
# --------------------------------------------------------------------------- #

_REF_SCAN, _REF_STEP, _REF_CONV = lm.ssd_chunked_scan, lm.ssd_step, lm.conv_step


def _scan_fp32(x, dt, a, b, c, chunk_size, initial_state=None):  # type: ignore[no-untyped-def]
    with torch.autocast(x.device.type, enabled=False):
        y, state = _REF_SCAN(
            x.float(),
            dt.float(),
            a.float(),
            b.float(),
            c.float(),
            chunk_size,
            None if initial_state is None else initial_state.float(),
        )
    return y.to(x.dtype), state


def _step_fp32(x, dt, a, b, c, state):  # type: ignore[no-untyped-def]
    """Reference single step in float32, with as few dtype conversions as possible.

    The state is kept in float32 and stays there. x and b are not cast: they
    meet float32 dt and state in elementwise products, which promote to float32
    inside the same kernel. Only c is cast, because einsum needs one dtype.
    """
    with torch.autocast(x.device.type, enabled=False):
        dt, state = dt.float(), state.float()  # no-ops when already float32
        decay = torch.exp(dt * a.float())[..., None, None]
        update = (dt.unsqueeze(-1) * x).unsqueeze(-1) * b[:, None, None, :]
        new_state = state * decay + update
        y = torch.einsum("bhpn,bn->bhp", new_state, c.to(new_state.dtype))
    return y.to(x.dtype), new_state


def _scan_fused(x, dt, a, b, c, chunk_size, initial_state=None):  # type: ignore[no-untyped-def]
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    y, state = mamba_chunk_scan_combined(
        x.contiguous(),
        dt.contiguous(),
        a.float(),
        b.unsqueeze(2).contiguous(),
        c.unsqueeze(2).contiguous(),
        chunk_size,
        initial_states=(
            None if initial_state is None else initial_state.float().contiguous()
        ),
        return_final_states=True,
    )
    return y, state


def _check_fused(device: torch.device) -> None:
    """Compare the fused kernel with the reference scan, forward and backward."""
    g = torch.Generator(device="cpu").manual_seed(0)
    bsz, seqlen, heads, headdim, d_state = 2, 300, 4, 64, 128

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g).to(device)

    x, b, c = (
        rand(bsz, seqlen, heads, headdim),
        rand(bsz, seqlen, d_state),
        rand(bsz, seqlen, d_state),
    )
    dt = F.softplus(rand(bsz, seqlen, heads) - 2)
    a = -torch.rand(heads, generator=g).to(device) * 4
    h0 = rand(bsz, heads, headdim, d_state) * 0.1
    outs = []
    for fn in (_scan_fp32, _scan_fused):
        xi, h0i = x.clone().requires_grad_(), h0.clone().requires_grad_()
        y, s = fn(xi, dt, a, b, c, 64, h0i)
        (y.square().mean() + s.square().mean()).backward()
        outs.append((y.detach(), s.detach(), xi.grad, h0i.grad))
    for name, ref, got in zip(("y", "state", "dx", "dh0"), outs[0], outs[1]):
        err = (
            (got.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
        ).item()
        if not err < 1e-2:
            raise RuntimeError(
                f"mamba_ssm kernel disagrees with the reference ({name} rel err {err:.2e})"
            )
    print("fused scan matches the reference (outputs and gradients)")


def enable_grad_checkpointing(bundle: Bundle) -> None:
    """Recompute each Mamba layer in the backward pass instead of storing its activations.

    Activation memory drops to one (B, L, d_model) tensor per layer, at the
    cost of about one extra forward pass. GPT-2 is left alone: Hugging Face's
    checkpointing turns off the KV cache that Coconut's thought loop needs.
    """
    if bundle.kind != "mamba":
        print("note: --grad-checkpoint only applies to Mamba models; ignored")
        return
    from torch.utils.checkpoint import checkpoint

    for layer in bundle.model.backbone.layers:
        forward = layer.forward

        def wrapped(
            x: Tensor,
            state: Any = None,
            mask: Tensor | None = None,
            _forward: Any = forward,
        ) -> Any:
            if torch.is_grad_enabled():
                return checkpoint(_forward, x, state, mask, use_reentrant=False)
            return _forward(x, state, mask)

        layer.forward = wrapped


def _step_fused(x, dt, a, b, c, state):  # type: ignore[no-untyped-def]
    """mamba_ssm's fused single-step kernel. It updates the state in place, so it
    works on a copy and the old state stays valid (caches are functional)."""
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    heads, headdim, d_state = state.shape[1:]
    zero = torch.zeros(heads, device=x.device)[:, None].expand(heads, headdim)
    new_state = state.float().clone()
    y = selective_state_update(
        new_state,
        x.contiguous(),
        dt.float()[..., None].expand(-1, -1, headdim),
        a.float()[:, None, None].expand(heads, headdim, d_state),
        b[:, None, :].contiguous(),
        c[:, None, :].contiguous(),
        # mamba-ssm 2.3.1 reads D.stride() and dt_bias.stride() without checking
        # for None, so both are passed as zeros (stride 0 keeps the fast path).
        # They add nothing: dt is already biased and softplus-ed, and the D skip
        # connection is added by the mixer.
        D=zero,
        dt_bias=zero,
        dt_softplus=False,
    )
    return y, new_state


def _conv_fused(x, conv_state, weight, bias):  # type: ignore[no-untyped-def]
    """causal_conv1d's fused single-token conv + SiLU, on a copy of the window."""
    from causal_conv1d import causal_conv1d_update

    new_state = conv_state.to(x.dtype).clone()
    y = causal_conv1d_update(
        x.contiguous(), new_state, weight[:, 0, :].to(x.dtype), bias.to(x.dtype), "silu"
    )
    return y, new_state


def _step_dispatch(x, dt, a, b, c, state):  # type: ignore[no-untyped-def]
    # the fused kernel has no backward pass: training thoughts use the reference
    if torch.is_grad_enabled() or not x.is_cuda:
        return _step_fp32(x, dt, a, b, c, state)
    return _step_fused(x, dt, a, b, c, state)


def _conv_dispatch(x, conv_state, weight, bias):  # type: ignore[no-untyped-def]
    if torch.is_grad_enabled() or not x.is_cuda:
        return _REF_CONV(x, conv_state, weight, bias)
    return _conv_fused(x, conv_state, weight, bias)


def _rel_err(got: Tensor, ref: Tensor) -> float:
    return (
        (got.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    ).item()


def _check_fused_step(device: torch.device) -> None:
    g = torch.Generator(device="cpu").manual_seed(1)
    bsz, heads, headdim, d_state = 3, 4, 64, 128
    rand = lambda *shape: torch.randn(*shape, generator=g).to(device)  # noqa: E731
    x, b, c = rand(bsz, heads, headdim), rand(bsz, d_state), rand(bsz, d_state)
    dt = F.softplus(rand(bsz, heads) - 2)
    a = -torch.rand(heads, generator=g).to(device) * 4
    state = rand(bsz, heads, headdim, d_state)
    before = state.clone()
    with torch.no_grad():
        ref = _step_fp32(x, dt, a, b, c, state)
        got = _step_fused(x, dt, a, b, c, state)
    for name, r, o in zip(("y", "state"), ref, got):
        err = _rel_err(o, r)
        if not err < 1e-3:
            raise RuntimeError(
                f"fused step disagrees with the reference ({name} rel err {err:.2e})"
            )
    if not torch.equal(state, before):
        raise RuntimeError("fused step modified its input state")


def _check_fused_conv(device: torch.device) -> None:
    g = torch.Generator(device="cpu").manual_seed(2)
    bsz, channels, width = 3, 96, 4
    rand = lambda *shape: torch.randn(*shape, generator=g).to(device)  # noqa: E731
    x, conv_state = rand(bsz, channels), rand(bsz, channels, width - 1)
    weight, bias = rand(channels, 1, width), rand(channels)
    before = conv_state.clone()
    with torch.no_grad():
        ref = _REF_CONV(x, conv_state, weight, bias)
        got = _conv_fused(x, conv_state, weight, bias)
    for name, r, o in zip(("y", "state"), ref, got):
        err = _rel_err(o, r)
        if not err < 1e-3:
            raise RuntimeError(
                f"fused conv disagrees with the reference ({name} rel err {err:.2e})"
            )
    if not torch.equal(conv_state, before):
        raise RuntimeError("fused conv modified its input state")


def install_scan(fused: bool | None, device: torch.device) -> dict[str, str]:
    """Choose the kernels. Returns which one is in use for each of scan, step and conv.

    ``fused=None`` (the default) uses mamba_ssm's kernels on CUDA when they are
    installed and pass the startup check: the chunked scan for prompts and
    answers, and the single-step kernels (``selective_state_update``, and
    ``causal_conv1d_update`` if causal-conv1d is installed) for inference
    thoughts and decoding. Training thoughts always use the reference step,
    because the fused step has no backward pass. ``True`` requires the fused
    scan; ``False`` never uses fused kernels.
    """
    lm.ssd_step = _step_fp32
    lm.ssd_chunked_scan = _scan_fp32
    lm.conv_step = _REF_CONV
    used = {"scan": "reference", "step": "reference", "conv": "reference"}
    if fused is False:
        return used
    if device.type != "cuda":
        if fused:
            raise SystemExit("--fused-scan needs a CUDA GPU")
        return used
    try:
        __import__("mamba_ssm")
    except ImportError as exc:
        if fused:
            raise SystemExit(
                "--fused-scan needs `pip install mamba-ssm --no-build-isolation`"
            ) from exc
        print(
            "note: mamba_ssm is not installed, using the reference kernels "
            "(uv sync --extra cuda, or pip install mamba-ssm --no-build-isolation)"
        )
        return used
    try:
        _check_fused(device)
    except Exception as exc:  # a broken build or kernel must not change results
        if fused:
            raise SystemExit(f"--fused-scan: {exc}") from exc
        print(f"warning: {exc}; using the reference scan")
    else:
        lm.ssd_chunked_scan = _scan_fused
        used["scan"] = "fused"
    try:
        _check_fused_step(device)
    except Exception as exc:
        print(
            f"warning: fused single step unavailable ({exc}); using the reference step"
        )
    else:
        lm.ssd_step = _step_dispatch
        used["step"] = "fused (inference)"
        print("fused single step matches the reference")
    try:
        __import__("causal_conv1d")
        _check_fused_conv(device)
    except Exception as exc:
        print(
            f"note: fused single-token conv unavailable ({exc}); using the reference conv"
        )
    else:
        lm.conv_step = _conv_dispatch
        used["conv"] = "fused (inference)"
        print("fused single-token conv matches the reference")
    return used


def enable_cuda_graphs(bundle: Bundle, device: torch.device, flag: bool | None) -> bool:
    """CUDA graphs for inference thought and decode steps (default: on for CUDA)."""
    on = device.type == "cuda" if flag is None else flag
    if on and device.type != "cuda":
        raise SystemExit("--cuda-graphs needs a CUDA GPU")
    if bundle.kind == "mamba":
        bundle.model.enable_cuda_graphs(on)
        return on
    return False


def enable_compiled_thoughts(
    bundle: Bundle, device: torch.device, mode: str, dtype: str
) -> str:
    """Compile the training thought step with torch.compile.

    A thought is one token through every layer: dozens of small kernels, so
    training time is dominated by launches. torch.compile fuses them. Mode
    ``"on"`` uses inductor's default; ``"cudagraphs"`` also records CUDA graphs
    over forward and backward (``mode="reduce-overhead"``, experimental). The
    compiled step is checked against the eager one, forward and backward, before
    use, and any failure falls back to eager. Only grad-enabled calls use it;
    inference keeps the eager step (and its CUDA graphs).
    """
    if mode == "auto":
        mode = "on" if device.type == "cuda" else "off"
    if mode == "off" or bundle.kind != "mamba":
        return "off"
    model = bundle.model
    eager = model._thought_step
    compiled = torch.compile(
        eager, mode="reduce-overhead" if mode == "cudagraphs" else None
    )

    # Inputs with realistic magnitudes: the state after a short random prompt.
    # (On random states, bf16 itself is unstable, so nothing can be compared.)
    g = torch.Generator(device="cpu").manual_seed(3)
    ids = torch.randint(0, bundle.bot_id, (2, 24), generator=g)
    ids[:, -1] = bundle.bot_id
    with torch.no_grad():
        h0, c0 = model.encode_prompt(ids.to(device))

    def run(fn, dt):  # type: ignore[no-untyped-def]
        h = h0.detach().clone().requires_grad_()
        # as in training, where the cache carries gradients
        cache = lm._map_tensors(c0, lambda t: t.detach().clone().requires_grad_())
        with autocast(device, dt):
            h1, c1, logit = fn(h, cache)
            h2, c2, logit2 = fn(h1, c1)
        loss = (
            h2.float().square().mean()
            + logit2.float().square().mean()
            + c2.layers[-1].ssm.float().square().mean()
        )
        loss.backward()
        out = (
            h2.detach().float(),
            logit2.detach().float(),
            h.grad.float(),
            model.thought_projector.proj.weight.grad.float(),
        )
        model.zero_grad(set_to_none=True)
        return out

    try:
        # Judge both against float32: in bf16, compiled and eager round
        # differently, so the compiled step passes if it is about as accurate.
        exact = run(eager, "fp32")
        ref = run(eager, dtype)
        if mode == "cudagraphs":
            torch.compiler.cudagraph_mark_step_begin()
        got = run(compiled, dtype)
        for name, x, r, o in zip(("h", "halt", "dh", "dW"), exact, ref, got):
            err, base = _rel_err(o, x), _rel_err(r, x)
            if not err <= 3 * base + 1e-3:
                raise RuntimeError(
                    f"compiled thought step is less accurate than eager ({name}: rel err "
                    f"{err:.2e} vs {base:.2e} against float32)"
                )
    except Exception as exc:
        model.zero_grad(set_to_none=True)
        print(
            f"warning: compiled thoughts unavailable, using eager ({type(exc).__name__}: {str(exc)[:300]})"
        )
        return "off"

    def step(h, cache, ablate=None):  # type: ignore[no-untyped-def]
        if ablate is None and torch.is_grad_enabled():
            return compiled(h, cache)
        return eager(h, cache, ablate)

    model._thought_step = step
    print(f"compiled thought step matches eager (mode {mode})")
    return mode


# --------------------------------------------------------------------------- #
# Batching, distributed helpers, logging
# --------------------------------------------------------------------------- #


def layout(
    ex: Example, method: str, stage: int, preset: Preset
) -> tuple[list[int], int]:
    """Target tokens and thought count for one example at one stage (Coconut's rule).

    Stage 0 is language CoT. Stage k in 1..K drops the first k steps and uses
    c*k thoughts (even when the example has fewer steps: Coconut's
    pad_latent_to_max). Stage K+1 drops every step and uses c*K thoughts.
    """
    if method == "nocot":
        return ex.answer, 0
    if stage == 0:
        return [t for s in ex.steps for t in s] + ex.answer, 0
    k_max, c = preset.max_latent_stage, preset.c_thought
    skip = stage if stage <= k_max else len(ex.steps)
    return [t for s in ex.steps[skip:] for t in s] + ex.answer, c * min(stage, k_max)


@dataclass
class Batch:
    prompt: Tensor  # (B, Lp), left padded, ends with <bot>
    mask: Tensor  # (B, Lp) bool
    target: Tensor  # (B, T), right padded with IGNORE_INDEX
    n_thoughts: Tensor  # (B,)


def prompt_batch(
    examples: Sequence[Example], bundle: Bundle, device: torch.device
) -> tuple[Tensor, Tensor]:
    prompts = [ex.question + [bundle.bot_id] for ex in examples]
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), bundle.pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, width - len(p) :] = torch.tensor(p)
        mask[i, width - len(p) :] = True
    return ids.to(device), mask.to(device)


def collate(
    examples: Sequence[Example],
    method: str,
    stage: int,
    preset: Preset,
    bundle: Bundle,
    device: torch.device,
) -> Batch:
    prompt, mask = prompt_batch(examples, bundle, device)
    pairs = [layout(ex, method, stage, preset) for ex in examples]
    target = torch.full(
        (len(pairs), max(len(t) for t, _ in pairs)), IGNORE_INDEX, dtype=torch.long
    )
    for i, (t, _) in enumerate(pairs):
        target[i, : len(t)] = torch.tensor(t)
    n = torch.tensor([k for _, k in pairs], dtype=torch.long)
    return Batch(prompt, mask, target.to(device), n.to(device))


@dataclass
class DistInfo:
    rank: int
    world: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_dist(device_arg: str | None) -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return DistInfo(0, 1, pick_device(device_arg))
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available() and device_arg != "cpu":
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local)
    else:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    return DistInfo(rank, world, device)


def broadcast_params(model: nn.Module, d: DistInfo) -> None:
    if d.world > 1:
        for t in [*model.parameters(), *model.buffers()]:
            dist.broadcast(t.data, src=0)


def allreduce_grads(params: list[nn.Parameter], d: DistInfo) -> None:
    """Average gradients across ranks. Missing gradients count as zero."""
    if d.world == 1:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= d.world
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n


def autocast(device: torch.device, dtype: str) -> contextlib.AbstractContextManager:
    if dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(
        device.type, dtype=torch.bfloat16 if dtype == "bf16" else torch.float16
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def add_wandb_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("weights & biases")
    g.add_argument(
        "--wandb",
        action="store_true",
        help="log to Weights & Biases (needs `wandb login` or WANDB_API_KEY)",
    )
    g.add_argument("--wandb-project", default="latent-mamba")
    g.add_argument("--wandb-entity", help="team or user (default: your default entity)")
    g.add_argument("--wandb-name", help="run name (default: derived from the run)")
    g.add_argument(
        "--wandb-group", help="group runs together (default: the run directory name)"
    )
    g.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )


def wandb_start(
    args: argparse.Namespace,
    d_is_main: bool,
    *,
    job_type: str,
    name: str,
    group: str,
    config: dict,
    tags: Sequence[str] = (),
    run_id: str | None = None,
) -> Any | None:
    """Start a W&B run on the main process when --wandb is set, else return None.

    Passing ``run_id`` continues an existing run (used when training resumes).
    """
    if not getattr(args, "wandb", False) or not d_is_main:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("--wandb needs `pip install wandb`") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or name,
        group=args.wandb_group or group,
        job_type=job_type,
        config=config,
        tags=[t for t in tags if t],
        mode=args.wandb_mode,
        id=run_id,
        resume="allow" if run_id else None,
    )
    run.define_metric("step")
    for prefix in ("train", "val", "phase"):
        run.define_metric(f"{prefix}/*", step_metric="step")
    return run


def jsonable_args(args: argparse.Namespace) -> dict:
    return {k: v for k, v in vars(args).items() if k != "fn"}


DEFAULT_HF_REPO = "kye/latent-mamba"


def add_hf_args(p: argparse.ArgumentParser, default_upload: bool) -> None:
    g = p.add_argument_group("hugging face upload")
    g.add_argument(
        "--hf-repo", default=DEFAULT_HF_REPO, help="model repo to upload results to"
    )
    g.add_argument(
        "--hf-upload",
        action=argparse.BooleanOptionalAction,
        default=default_upload,
        help="upload to --hf-repo (needs HF_TOKEN or `hf auth login`)",
    )
    g.add_argument(
        "--hf-private",
        action="store_true",
        help="create the repo as private if it does not exist",
    )


def hf_prepare(repo: str, private: bool) -> None:
    """Check the token and create the repo, so a bad token fails before training, not after."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:
        raise SystemExit(
            f"uploading to {repo} needs a Hugging Face write token: set HF_TOKEN or run `hf auth login` "
            "(or pass --no-hf-upload)"
        ) from exc
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    print(f"results will be uploaded to https://huggingface.co/{repo} as {user}")


def hf_upload(
    repo: str,
    local: Path,
    path_in_repo: str,
    message: str,
    retries: int = 3,
    strict: bool = False,
) -> bool:
    """Upload a file or folder. Unchanged large files are not re-sent.

    Failures are retried, then reported as warnings unless ``strict``: a
    transient Hub error should not kill a multi-day run, and everything stays
    on disk for `python train.py upload`.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    for attempt in range(1, retries + 1):
        try:
            if local.is_dir():
                api.upload_folder(
                    repo_id=repo,
                    folder_path=str(local),
                    path_in_repo=path_in_repo,
                    commit_message=message,
                    ignore_patterns=["*.tmp", "wandb/*"],
                )
            else:
                api.upload_file(
                    repo_id=repo,
                    path_or_fileobj=str(local),
                    path_in_repo=path_in_repo,
                    commit_message=message,
                )
            print(
                f"uploaded {local} -> https://huggingface.co/{repo}/tree/main/{path_in_repo.rsplit('/', 1)[0] if not local.is_dir() else path_in_repo}"
            )
            return True
        except Exception as exc:  # network errors, rate limits, Hub outages
            print(
                f"warning: upload of {local} failed (attempt {attempt}/{retries}): {exc}"
            )
            time.sleep(30 * attempt)
    if strict:
        raise SystemExit(f"could not upload {local} to {repo}")
    print(
        f"warning: giving up on uploading {local}; retry later with: python train.py upload --path {local}"
    )
    return False


def upload_cmd(args: argparse.Namespace) -> None:
    hf_prepare(args.hf_repo, args.hf_private)
    local = Path(args.path)
    if not local.exists():
        raise SystemExit(f"{local} does not exist")
    hf_upload(
        args.hf_repo,
        local,
        args.path_in_repo or local.name,
        f"Upload {local.name}",
        strict=True,
    )


class MetricsLog:
    """Appends records to a JSONL file, prints them, and mirrors them to W&B.

    In W&B, numeric fields appear as ``{namespace}/{key}`` against ``step``.
    """

    def __init__(self, path: Path, enabled: bool, run: Any | None = None) -> None:
        self.path, self.enabled, self.run = path, enabled, run

    def __call__(
        self, record: dict, echo: str | None = None, namespace: str = "train"
    ) -> None:
        if not self.enabled:
            return
        with self.path.open("a") as f:
            f.write(json.dumps({"time": time.time(), **record}) + "\n")
        if echo:
            print(echo, flush=True)
        if self.run is not None:
            metrics = {
                f"{namespace}/{k}": v
                for k, v in record.items()
                if isinstance(v, (int, float)) and k != "step"
            }
            self.run.log({"step": record["step"], **metrics})


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def decode_answer(tokens: list[int], bundle: Bundle) -> str:
    if bundle.eos_id in tokens:
        tokens = tokens[: tokens.index(bundle.eos_id)]
    return bundle.tokenizer.decode(tokens)


@torch.no_grad()
def evaluate(
    bundle: Bundle,
    examples: Sequence[Example],
    d: DistInfo,
    *,
    max_thoughts: int,
    halt_rule: str = "fixed",
    ablate: str | None = None,
    max_new_tokens: int,
    batch_size: int,
    dtype: str = "fp32",
    samples: int = 0,
) -> dict:
    """Accuracy and thoughts used, overall and by number of gold reasoning steps.

    Examples are sharded across ranks and the counts are gathered.
    """
    model = bundle.model
    model.eval()
    shard = list(examples)[d.rank :: d.world]
    stats: dict[int, list[float]] = {}  # n_steps -> [correct, total, thoughts]
    shown: list[dict] = []
    for start in range(0, len(shard), batch_size):
        chunk = shard[start : start + batch_size]
        prompt, mask = prompt_batch(chunk, bundle, d.device)
        with autocast(d.device, dtype):
            gen = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                max_thoughts=max_thoughts,
                halt_rule=halt_rule,
                ablate=ablate,
                eos_id=bundle.eos_id,
                prompt_mask=mask,
            )
        for ex, row, n in zip(chunk, gen.tokens.tolist(), gen.n_thoughts.tolist()):
            text = decode_answer(row, bundle)
            ok = parse_answer(text) == ex.answer_text
            s = stats.setdefault(ex.n_steps, [0.0, 0.0, 0.0])
            s[0] += ok
            s[1] += 1
            s[2] += n
            if len(shown) < samples:
                shown.append(
                    {
                        "output": text,
                        "gold": ex.answer_text,
                        "correct": ok,
                        "thoughts": n,
                    }
                )
    if d.world > 1:
        gathered: list[Any] = [None] * d.world
        dist.all_gather_object(gathered, stats)
        stats = {}
        for part in gathered:
            for k, (c, t, n) in part.items():
                s = stats.setdefault(k, [0.0, 0.0, 0.0])
                s[0], s[1], s[2] = s[0] + c, s[1] + t, s[2] + n
    correct = sum(s[0] for s in stats.values())
    total = sum(s[1] for s in stats.values())
    thoughts = sum(s[2] for s in stats.values())
    return {
        "acc": correct / max(total, 1),
        "mean_thoughts": thoughts / max(total, 1),
        "n": int(total),
        "by_steps": {
            str(k): {"acc": c / t, "mean_thoughts": n / t, "n": int(t)}
            for k, (c, t, n) in sorted(stats.items())
        },
        "samples": shown,
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Phase:
    name: str
    stage: int
    epochs: int
    ponder: bool = False


def plan_phases(
    method: str, preset: Preset, start_stage: int, kind: str
) -> list[Phase]:
    if method in ("cot", "nocot"):
        return [Phase(method, 0, preset.cot_epochs)]
    k_max = preset.max_latent_stage
    phases = []
    if start_stage <= 0:
        phases.append(Phase("cot", 0, preset.cot_epochs))
    phases += [
        Phase(f"stage{k}", k, preset.epochs_per_stage)
        for k in range(max(1, start_stage), k_max + 1)
    ]
    if start_stage <= k_max + 1:
        phases.append(Phase("final", k_max + 1, preset.final_epochs))
    if kind == "mamba":
        phases.append(Phase("ponder", k_max + 1, preset.ponder_epochs, ponder=True))
    return [p for p in phases if p.epochs > 0]


def phase_eval(phase: Phase, preset: Preset, ponder_max: int) -> dict:
    """Inference settings that match what a phase trains."""
    if phase.ponder:
        return {"max_thoughts": ponder_max, "halt_rule": "learned"}
    return {
        "max_thoughts": preset.c_thought * min(phase.stage, preset.max_latent_stage),
        "halt_rule": "fixed",
    }


def resolve_preset(args: argparse.Namespace) -> Preset:
    preset = PRESETS[args.dataset]
    overrides = {
        f: getattr(args, f)
        for f in asdict(preset)
        if getattr(args, f, None) is not None
    }
    return replace(preset, **overrides)


def save_checkpoint(
    path: Path,
    bundle: Bundle,
    args: argparse.Namespace,
    preset: Preset,
    phase: Phase,
    epoch: int,
    val: dict | None,
    ponder_max: int,
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": {
                k: v.detach().cpu() for k, v in bundle.model.state_dict().items()
            },
            "model_name": args.model,
            "dataset": args.dataset,
            "method": args.method,
            "phase": phase.name,
            "stage": phase.stage,
            "ponder_trained": phase.ponder,
            "epoch": epoch,
            "val_acc": None if val is None else val["acc"],
            "preset": asdict(preset),
            "chunk_size": args.chunk_size,
            "halt_prior": args.halt_prior,
            "ponder_max_thoughts": ponder_max,
        },
        tmp,
    )
    os.replace(tmp, path)


def train_cmd(args: argparse.Namespace) -> None:
    d = setup_dist(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    kernels = install_scan(args.fused_scan, d.device)
    preset = resolve_preset(args)
    out = Path(args.out or f"runs/{args.dataset}-{args.model}-{args.method}")
    if args.hf_upload and d.is_main:
        hf_prepare(args.hf_repo, args.hf_private)
    progress_path = out / "progress.json"
    progress: dict[str, Any] = {
        "completed": [],
        "step": 0,
        "epochs": 0,
        "examples_seen": 0,
        "wandb_id": None,
    }
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
    bundle = build(
        args.model,
        pretrained=not args.no_pretrained,
        chunk_size=args.chunk_size,
        halt_prior=args.halt_prior,
    )
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        if ckpt["model_name"] != args.model:
            raise SystemExit(
                f"--init-from was trained with {ckpt['model_name']}, not {args.model}"
            )
        bundle.model.load_state_dict(ckpt["model"])
    if progress[
        "completed"
    ]:  # resume from the weights at the end of the last finished phase
        bundle.model.load_state_dict(
            torch.load(out / "last.pt", map_location="cpu", weights_only=True)["model"]
        )
        if d.is_main:
            print(f"resuming {out}: finished phases {', '.join(progress['completed'])}")
    model = bundle.model.to(d.device)
    if args.grad_checkpoint:
        enable_grad_checkpointing(bundle)
    kernels["thoughts_compiled"] = enable_compiled_thoughts(
        bundle, d.device, args.compile_thoughts, args.dtype
    )
    kernels["cuda_graphs"] = enable_cuda_graphs(bundle, d.device, args.cuda_graphs)
    broadcast_params(model, d)
    params = list(model.parameters())

    ponder_max = args.ponder_max_thoughts or preset.c_thought * preset.max_latent_stage
    train = tokenize(
        load_rows(args.data_dir, args.dataset, "train", args.max_train_examples),
        bundle.tokenizer,
        bundle.eos_id,
    )
    valid = tokenize(
        load_rows(args.data_dir, args.dataset, "valid", args.max_val_examples),
        bundle.tokenizer,
        bundle.eos_id,
    )
    limit = args.max_tokens
    if (
        bundle.kind == "gpt2"
    ):  # learned positions: prompt + <bot> + thoughts + <eot> + target must fit
        room = (
            bundle.model.max_positions - 2 - preset.c_thought * preset.max_latent_stage
        )
        limit = min(limit or room, room)
    if limit:
        before = len(train)
        train = [ex for ex in train if ex.n_tokens() <= limit]
        valid = [
            ex for ex in valid if len(ex.question) + preset.max_new_tokens <= limit
        ]
        if d.is_main and before != len(train):
            print(
                f"dropped {before - len(train)} training examples longer than {limit} tokens"
            )

    phases = plan_phases(args.method, preset, args.start_stage, bundle.kind)
    run_info = {
        "args": jsonable_args(args),
        "preset": asdict(preset),
        "phases": [asdict(p) for p in phases],
        "params": count_params(model),
        "world_size": d.world,
        "kernels": kernels,
        "train_examples": len(train),
        "valid_examples": len(valid),
    }
    if d.is_main:
        out.mkdir(parents=True, exist_ok=True)
        print(
            f"{args.model}: {count_params(model) / 1e6:.1f}M params on {d.device} x{d.world}, "
            f"{len(train)} train / {len(valid)} valid examples"
        )
        print("phases: " + ", ".join(f"{p.name} ({p.epochs} ep)" for p in phases))
    wb = wandb_start(
        args,
        d.is_main,
        job_type="train",
        name=out.name,
        group=out.name,
        config=run_info,
        tags=(args.model, args.dataset, args.method, bundle.kind),
        run_id=progress.get("wandb_id"),
    )
    if d.is_main:
        run_info["wandb_url"] = wb.url if wb is not None else None
        (out / "run.json").write_text(json.dumps(run_info, indent=2))
        if wb is not None:
            progress["wandb_id"] = wb.id
    log = MetricsLog(out / "metrics.jsonl", d.is_main, wb)

    per_step = args.batch_size * d.world
    n_batches = len(train) // per_step
    if n_batches == 0:
        raise SystemExit(
            f"need at least {per_step} training examples, have {len(train)}"
        )
    epoch_counter = progress["epochs"]
    step = progress["step"]
    examples_seen = progress["examples_seen"]
    for phase_index, phase in enumerate(phases):
        if phase.name in progress["completed"]:
            continue
        opt = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )  # reset per stage, as Coconut
        eval_kw = phase_eval(phase, preset, ponder_max)
        best = -1.0
        log(
            {
                "phase": phase.name,
                "step": step,
                "index": phase_index,
                "stage": phase.stage,
            },
            namespace="phase",
        )
        for epoch in range(phase.epochs):
            model.train()
            order = torch.randperm(
                len(train),
                generator=torch.Generator().manual_seed(
                    args.seed * 7919 + epoch_counter
                ),
            )
            epoch_counter += 1
            start = time.time()
            opt.zero_grad(set_to_none=True)
            for bi in range(n_batches):
                lo = bi * per_step + d.rank * args.batch_size
                batch = collate(
                    [train[i] for i in order[lo : lo + args.batch_size].tolist()],
                    args.method,
                    phase.stage,
                    preset,
                    bundle,
                    d.device,
                )
                if kernels["thoughts_compiled"] == "cudagraphs":
                    torch.compiler.cudagraph_mark_step_begin()
                with autocast(d.device, args.dtype):
                    if phase.ponder:
                        out_p = model.ponder_loss(
                            batch.prompt,
                            batch.target,
                            max_thoughts=ponder_max,
                            beta=args.beta,
                            prompt_mask=batch.mask,
                        )
                        loss = out_p.loss
                        info = {
                            "expected_thoughts": out_p.expected_thoughts.mean().item(),
                            "kl": out_p.kl.item(),
                        }
                    else:
                        out_l = model.latent_loss(
                            batch.prompt,
                            batch.target,
                            batch.n_thoughts,
                            prompt_mask=batch.mask,
                        )
                        loss = out_l.loss
                        info = {"thoughts": out_l.n_thoughts.float().mean().item()}
                (loss / args.grad_accum).backward()
                examples_seen += per_step
                if (bi + 1) % args.grad_accum == 0 or bi + 1 == n_batches:
                    allreduce_grads(params, d)
                    grad_norm = nn.utils.clip_grad_norm_(
                        params, args.grad_clip if args.grad_clip > 0 else float("inf")
                    )
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0:
                        rate = (bi + 1) / (time.time() - start)
                        log(
                            {
                                "phase": phase.name,
                                "epoch": epoch_counter - 1 + (bi + 1) / n_batches,
                                "step": step,
                                "loss": loss.item(),
                                "grad_norm": grad_norm.item(),
                                "lr": args.lr,
                                "stage": phase.stage,
                                "batches_per_s": rate,
                                "examples_seen": examples_seen,
                                **info,
                            },
                            echo=f"[{phase.name} ep {epoch + 1}/{phase.epochs}] batch {bi + 1}/{n_batches} "
                            f"loss {loss.item():.4f} "
                            + " ".join(f"{k} {v:.3f}" for k, v in info.items())
                            + f" ({rate:.2f} it/s)",
                        )
            val = None
            if valid:
                val = evaluate(
                    bundle,
                    valid,
                    d,
                    max_new_tokens=preset.max_new_tokens,
                    batch_size=args.eval_batch_size,
                    dtype=args.dtype,
                    samples=args.wandb_samples if wb is not None else 0,
                    **eval_kw,
                )
                by_steps = {
                    f"acc_{k}_steps": v["acc"] for k, v in val["by_steps"].items()
                }
                by_steps |= {
                    f"thoughts_{k}_steps": v["mean_thoughts"]
                    for k, v in val["by_steps"].items()
                }
                log(
                    {
                        "phase": phase.name,
                        "epoch": epoch_counter,
                        "step": step,
                        "acc": val["acc"],
                        "mean_thoughts": val["mean_thoughts"],
                        **by_steps,
                    },
                    echo=f"[{phase.name} ep {epoch + 1}/{phase.epochs}] val acc {val['acc']:.4f} "
                    f"thoughts {val['mean_thoughts']:.2f} ({time.time() - start:.0f}s)",
                    namespace="val",
                )
                if wb is not None and val["samples"]:
                    import wandb

                    table = wandb.Table(
                        columns=[
                            "phase",
                            "epoch",
                            "output",
                            "gold",
                            "correct",
                            "thoughts",
                        ]
                    )
                    for s in val["samples"]:
                        table.add_data(
                            phase.name,
                            epoch_counter,
                            s["output"],
                            s["gold"],
                            s["correct"],
                            s["thoughts"],
                        )
                    wb.log({"step": step, f"val/samples_{phase.name}": table})
            score = val["acc"] if val else float(epoch)
            if (
                d.is_main and score > best
            ):  # keep the best epoch of each phase, as Coconut selects by validation
                best = score
                save_checkpoint(
                    out / f"{phase.name}.pt",
                    bundle,
                    args,
                    preset,
                    phase,
                    epoch,
                    val,
                    ponder_max,
                )
                if wb is not None and val is not None:
                    wb.summary[f"best_val_acc/{phase.name}"] = val["acc"]
                    wb.summary[f"best_val_thoughts/{phase.name}"] = val["mean_thoughts"]
        if wb is not None and args.wandb_checkpoints:
            import wandb

            artifact = wandb.Artifact(
                f"{out.name}-{phase.name}".replace("/", "-"),
                type="model",
                metadata={"phase": phase.name, "best_val_acc": best},
            )
            artifact.add_file(str(out / f"{phase.name}.pt"))
            wb.log_artifact(artifact)
        if d.is_main:  # end of phase: weights to resume from, progress, and an upload
            save_checkpoint(
                out / "last.pt",
                bundle,
                args,
                preset,
                phase,
                phase.epochs - 1,
                None,
                ponder_max,
            )
            progress.update(
                completed=[*progress["completed"], phase.name],
                step=step,
                epochs=epoch_counter,
                examples_seen=examples_seen,
            )
            progress_path.write_text(json.dumps(progress, indent=2))
            if args.hf_upload:
                hf_upload(
                    args.hf_repo,
                    out,
                    out.name,
                    f"{out.name}: finished phase {phase.name}",
                )
    if d.is_main:
        print(f"done. checkpoints in {out}/")
        if args.hf_upload:
            hf_upload(args.hf_repo, out, out.name, f"{out.name}: training finished")
    if wb is not None:
        wb.finish()
    if d.world > 1:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Eval command
# --------------------------------------------------------------------------- #


def eval_modes(ckpt: dict, preset: Preset, kind: str) -> list[tuple[str, dict]]:
    c_k = preset.c_thought * preset.max_latent_stage
    phase = ckpt["phase"]
    if phase in ("cot", "nocot"):
        return [("language" if phase == "cot" else "direct", {"max_thoughts": 0})]
    if phase.startswith("stage"):
        return [("fixed", {"max_thoughts": preset.c_thought * ckpt["stage"]})]
    modes = [("fixed", {"max_thoughts": c_k}), ("no-thoughts", {"max_thoughts": 0})]
    if kind == "mamba":
        ponder_max = ckpt["ponder_max_thoughts"]
        if ckpt.get("ponder_trained"):
            modes.insert(
                0, ("learned", {"max_thoughts": ponder_max, "halt_rule": "learned"})
            )
        modes += [
            ("convergence", {"max_thoughts": ponder_max, "halt_rule": "convergence"}),
            ("ablate-zeros", {"max_thoughts": c_k, "ablate": "zeros"}),
            ("ablate-noise", {"max_thoughts": c_k, "ablate": "noise"}),
        ]
    return modes


def eval_cmd(args: argparse.Namespace) -> None:
    d = setup_dist(args.device)
    torch.manual_seed(args.seed)
    kernels = install_scan(args.fused_scan, d.device)
    bundle, ckpt = load_checkpoint(args.ckpt, args.chunk_size)
    bundle.model.to(d.device)
    kernels["cuda_graphs"] = enable_cuda_graphs(bundle, d.device, args.cuda_graphs)
    preset = Preset(**ckpt["preset"])
    dataset = args.dataset or ckpt["dataset"]
    max_new = args.max_new_tokens or preset.max_new_tokens
    examples = tokenize(
        load_rows(args.data_dir, dataset, args.split, args.max_examples),
        bundle.tokenizer,
        bundle.eos_id,
    )
    if bundle.kind == "gpt2":
        room = (
            bundle.model.max_positions
            - 2
            - max(args.budgets or [2 * preset.c_thought * preset.max_latent_stage])
        )
        kept = [ex for ex in examples if len(ex.question) + max_new <= room]
        if len(kept) != len(examples) and d.is_main:
            print(
                f"warning: {len(examples) - len(kept)} examples do not fit GPT-2's context and are scored wrong"
            )
        n_dropped = len(examples) - len(kept)
        examples = kept
    else:
        n_dropped = 0

    common = dict(max_new_tokens=max_new, batch_size=args.batch_size, dtype=args.dtype)
    result: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "kernels": kernels,
        "model": ckpt["model_name"],
        "method": ckpt["method"],
        "phase": ckpt["phase"],
        "dataset": dataset,
        "split": args.split,
        "n_examples": len(examples) + n_dropped,
        "modes": {},
        "sweep": {},
    }

    def scored(res: dict) -> dict:  # examples that did not fit count as wrong
        if n_dropped:
            res["acc"] = res["acc"] * res["n"] / (res["n"] + n_dropped)
            res["n"] += n_dropped
        return res

    for label, kw in eval_modes(ckpt, preset, bundle.kind):
        t0 = time.time()
        res = scored(
            evaluate(bundle, examples, d, samples=args.samples, **common, **kw)
        )
        result["modes"][label] = res
        if d.is_main:
            print(
                f"{label:<14} acc {res['acc']:.4f}  thoughts {res['mean_thoughts']:.2f}  "
                f"n {res['n']}  ({time.time() - t0:.0f}s)"
            )
    if ckpt["phase"] in ("final", "ponder") and not args.no_sweep:
        budgets = args.budgets or list(
            range(0, 2 * preset.c_thought * preset.max_latent_stage + 1)
        )
        for n in budgets:
            res = scored(evaluate(bundle, examples, d, max_thoughts=n, **common))
            result["sweep"][str(n)] = {"acc": res["acc"], "n": res["n"]}
            if d.is_main:
                print(f"budget {n:<4} acc {res['acc']:.4f}")
    if d.is_main:
        path = (
            Path(args.out)
            if args.out
            else Path(args.ckpt).with_name(
                f"eval_{args.split}_{Path(args.ckpt).stem}.json"
            )
        )
        path.write_text(json.dumps(result, indent=2))
        print(f"wrote {path}")
        run_dir = Path(args.ckpt).parent.name
        wb = wandb_start(
            args,
            True,
            job_type="eval",
            name=f"eval-{run_dir}-{ckpt['phase']}-{args.split}",
            group=run_dir,
            config={
                **{k: v for k, v in ckpt.items() if k != "model"},
                "eval_args": jsonable_args(args),
            },
            tags=(ckpt["model_name"], dataset, ckpt["method"], "eval"),
        )
        if wb is not None:
            log_eval_to_wandb(wb, result)
            wb.finish()
        if args.hf_upload:
            hf_prepare(args.hf_repo, args.hf_private)
            hf_upload(
                args.hf_repo, path, f"{run_dir}/{path.name}", f"{run_dir}: {path.name}"
            )
    if d.world > 1:
        dist.destroy_process_group()


def log_eval_to_wandb(wb: Any, result: dict) -> None:
    """Summary metrics plus tables for modes, accuracy-vs-budget and accuracy by difficulty."""
    import wandb

    modes = wandb.Table(columns=["mode", "acc", "mean_thoughts", "n"])
    by_steps = wandb.Table(columns=["mode", "gold_steps", "acc", "mean_thoughts", "n"])
    samples = wandb.Table(columns=["mode", "output", "gold", "correct", "thoughts"])
    for mode, r in result["modes"].items():
        wb.summary[f"eval/{mode}/acc"] = r["acc"]
        wb.summary[f"eval/{mode}/mean_thoughts"] = r["mean_thoughts"]
        modes.add_data(mode, r["acc"], r["mean_thoughts"], r["n"])
        for k, v in r["by_steps"].items():
            by_steps.add_data(mode, int(k), v["acc"], v["mean_thoughts"], v["n"])
        for smp in r["samples"]:
            samples.add_data(
                mode, smp["output"], smp["gold"], smp["correct"], smp["thoughts"]
            )
    payload: dict[str, Any] = {
        "eval/modes": modes,
        "eval/by_steps": by_steps,
        "eval/samples": samples,
    }
    if result["sweep"]:
        sweep = wandb.Table(
            columns=["budget", "acc"],
            data=[[int(n), r["acc"]] for n, r in result["sweep"].items()],
        )
        payload["eval/sweep"] = sweep
        payload["eval/acc_vs_budget"] = wandb.plot.line(
            sweep, "budget", "acc", title="accuracy vs. thought budget"
        )
    wb.log(payload)


# --------------------------------------------------------------------------- #
# Bench command
# --------------------------------------------------------------------------- #


@torch.no_grad()
def state_bytes(bundle: Bundle, prompt: Tensor, n: int) -> int:
    model = bundle.model
    if bundle.kind == "mamba":
        h, cache = model.encode_prompt(prompt)
        for _ in range(n):
            h, cache, _ = model.thought_step(h, cache)
        return cache.num_bytes()
    _, cache, _, _ = model.think(
        prompt, torch.full((prompt.shape[0],), n, device=prompt.device)
    )
    return CoconutGPT2.cache_bytes(cache)


def bench_cmd(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    kernels = install_scan(args.fused_scan, device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.dtype
    ]
    rows: list[dict] = []
    for name in args.models:
        bundle = build(name, pretrained=False, chunk_size=args.chunk_size)
        model = bundle.model.to(device=device, dtype=dtype).eval()
        graphs = enable_cuda_graphs(bundle, device, args.cuda_graphs)
        n_params = count_params(model)
        torch.manual_seed(0)
        prompt = torch.randint(
            0, 1000, (args.batch_size, args.prompt_len), device=device
        )
        prompt[:, -1] = bundle.bot_id
        for n in args.budgets:
            row: dict[str, Any] = {
                "model": name,
                "params_m": round(n_params / 1e6, 1),
                "budget": n,
                "prompt_len": args.prompt_len,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "device": str(device),
                "kernels": (
                    {**kernels, "cuda_graphs": graphs}
                    if bundle.kind == "mamba"
                    else "transformers"
                ),
            }
            if bundle.kind == "gpt2" and args.prompt_len + n + 1 >= model.max_positions:
                row["note"] = f"exceeds context ({model.max_positions})"
                rows.append(row)
                print(f"{name:<12} n={n:<6} {row['note']}")
                continue

            def run() -> None:
                model.generate(
                    prompt, max_new_tokens=1, max_thoughts=n, halt_rule="fixed"
                )

            run()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                base = torch.cuda.memory_allocated(device)
            times = []
            for _ in range(args.repeats):
                synchronize(device)
                t0 = time.perf_counter()
                run()
                synchronize(device)
                times.append(time.perf_counter() - t0)
            row["latency_ms"] = 1000 * statistics.median(times)
            row["state_mib"] = state_bytes(bundle, prompt, n) / 2**20
            if device.type == "cuda":
                row["peak_extra_mib"] = (
                    torch.cuda.max_memory_allocated(device) - base
                ) / 2**20
            rows.append(row)
            print(
                f"{name:<12} n={n:<6} state {row['state_mib']:9.2f} MiB  latency {row['latency_ms']:9.1f} ms"
                + (
                    f"  peak {row['peak_extra_mib']:9.1f} MiB"
                    if "peak_extra_mib" in row
                    else ""
                )
            )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {path}")
    wb = wandb_start(
        args,
        True,
        job_type="bench",
        name=f"bench-{'-'.join(args.models)}",
        group="bench",
        config=jsonable_args(args),
        tags=("bench", *args.models),
    )
    if wb is not None:
        import wandb

        columns = [
            "model",
            "params_m",
            "budget",
            "state_mib",
            "peak_extra_mib",
            "latency_ms",
            "note",
        ]
        table = wandb.Table(
            columns=columns, data=[[r.get(c) for c in columns] for r in rows]
        )
        payload: dict[str, Any] = {"bench/results": table}
        measured = [r for r in rows if "note" not in r]
        for metric in ("state_mib", "latency_ms", "peak_extra_mib"):
            if any(metric in r for r in measured):
                models = sorted({r["model"] for r in measured})
                xs = [
                    [r["budget"] for r in measured if r["model"] == m and metric in r]
                    for m in models
                ]
                ys = [
                    [r[metric] for r in measured if r["model"] == m and metric in r]
                    for m in models
                ]
                payload[f"bench/{metric}_vs_budget"] = wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=models,
                    title=f"{metric} vs. thinking budget",
                    xname="thoughts",
                )
        wb.log(payload)
        wb.finish()


# --------------------------------------------------------------------------- #
# Report command
# --------------------------------------------------------------------------- #


def report_cmd(args: argparse.Namespace) -> None:
    lines: list[str] = []
    evals = [
        json.loads(Path(p).read_text())
        for p in sorted(glob.glob(f"{args.runs}/**/eval_*.json", recursive=True))
    ]
    if evals:
        lines += [
            "## Accuracy",
            "",
            "| dataset | split | model | method | checkpoint | mode | accuracy | mean thoughts | n |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for e in sorted(
            evals, key=lambda e: (e["dataset"], e["model"], e["method"], e["phase"])
        ):
            for mode, r in e["modes"].items():
                lines.append(
                    f"| {e['dataset']} | {e['split']} | {e['model']} | {e['method']} | {e['phase']} | {mode} "
                    f"| {r['acc']:.4f} | {r['mean_thoughts']:.2f} | {r['n']} |"
                )
        sweeps = [e for e in evals if e["sweep"]]
        if sweeps:
            budgets = sorted({int(b) for e in sweeps for b in e["sweep"]})
            lines += [
                "",
                "## Accuracy vs. thought budget",
                "",
                "| dataset | model | checkpoint | "
                + " | ".join(f"n={b}" for b in budgets)
                + " |",
                "|---|---|---|" + "---|" * len(budgets),
            ]
            for e in sweeps:
                cells = [
                    f"{e['sweep'][str(b)]['acc']:.3f}" if str(b) in e["sweep"] else ""
                    for b in budgets
                ]
                lines.append(
                    f"| {e['dataset']} | {e['model']} | {e['phase']} | "
                    + " | ".join(cells)
                    + " |"
                )
        steps = [e for e in evals if any("by_steps" in r for r in e["modes"].values())]
        if steps:
            lines += [
                "",
                "## By number of gold reasoning steps (accuracy / mean thoughts)",
                "",
            ]
            for e in steps:
                keys = sorted(
                    {int(k) for r in e["modes"].values() for k in r["by_steps"]}
                )
                lines += [
                    f"**{e['dataset']} · {e['model']} · {e['method']} · {e['phase']}**",
                    "",
                    "| mode | " + " | ".join(f"{k} steps" for k in keys) + " |",
                    "|---|" + "---|" * len(keys),
                ]
                for mode, r in e["modes"].items():
                    cells = [
                        (
                            f"{r['by_steps'][str(k)]['acc']:.3f} / {r['by_steps'][str(k)]['mean_thoughts']:.1f}"
                            if str(k) in r["by_steps"]
                            else ""
                        )
                        for k in keys
                    ]
                    lines.append(f"| {mode} | " + " | ".join(cells) + " |")
                lines.append("")
    benches = [
        (p, json.loads(Path(p).read_text()))
        for p in sorted(glob.glob(f"{args.runs}/**/bench*.json", recursive=True))
    ]
    for path, rows in benches:
        lines += [
            "",
            f"## Memory and latency vs. thinking budget ({path})",
            "",
            "| model | params (M) | budget | state / KV cache (MiB) | peak extra (MiB) | latency (ms) |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            if "note" in r:
                lines.append(
                    f"| {r['model']} | {r['params_m']} | {r['budget']} | {r['note']} | | |"
                )
            else:
                peak = f"{r['peak_extra_mib']:.1f}" if "peak_extra_mib" in r else "n/a"
                lines.append(
                    f"| {r['model']} | {r['params_m']} | {r['budget']} | {r['state_mib']:.2f} | {peak} "
                    f"| {r['latency_ms']:.1f} |"
                )
    if not lines:
        raise SystemExit(f"no eval_*.json or bench*.json files under {args.runs}/")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)
    print(f"wrote {args.out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="download and convert datasets")
    p.add_argument(
        "--dataset", nargs="+", choices=DATASETS, default=["gsm8k", "prosqa"]
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument(
        "--prontoqa-raw",
        help="path to 5hop_0shot_random.json generated with the ProntoQA repo",
    )
    p.set_defaults(fn=data_cmd)

    p = sub.add_parser("train", help="fine-tune one model with one method")
    p.add_argument("--model", required=True, choices=MODEL_CHOICES)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--method", choices=("latent", "cot", "nocot"), default="latent")
    p.add_argument(
        "--out", help="run directory (default runs/{dataset}-{model}-{method})"
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument(
        "--init-from", help="checkpoint to start from, e.g. a CoT run's cot.pt"
    )
    p.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="first curriculum stage of a latent run (1 skips the CoT stage; K+2 runs only ponder)",
    )
    g = p.add_argument_group("curriculum (defaults from the dataset preset)")
    for field in (
        "c_thought",
        "max_latent_stage",
        "cot_epochs",
        "epochs_per_stage",
        "final_epochs",
        "ponder_epochs",
        "max_new_tokens",
    ):
        g.add_argument("--" + field.replace("_", "-"), type=int, dest=field)
    g = p.add_argument_group("optimisation")
    g.add_argument("--batch-size", type=int, default=32, help="per GPU")
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    g.add_argument(
        "--dtype",
        choices=("fp32", "bf16", "fp16"),
        default=None,
        help="autocast dtype (default bf16 on CUDA, else fp32)",
    )
    g = p.add_argument_group("halting")
    g.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="weight of the KL to the geometric prior",
    )
    g.add_argument("--halt-prior", type=float, default=0.2)
    g.add_argument("--ponder-max-thoughts", type=int, help="default c * K")
    g = p.add_argument_group("misc")
    g.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="SSD chunk length (exact for any value)",
    )
    g.add_argument(
        "--fused-scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mamba_ssm's Triton scan for prompts and answers (default: on for CUDA if installed)",
    )
    g.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute Mamba layers in the backward pass to save memory",
    )
    g.add_argument(
        "--compile-thoughts",
        choices=("auto", "off", "on", "cudagraphs"),
        default="auto",
        help="torch.compile the training thought step; cudagraphs also records CUDA graphs "
        "over forward and backward (experimental). auto = on for CUDA",
    )
    g.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replay inference thought and decode steps as CUDA graphs (default: on for CUDA)",
    )
    g.add_argument("--max-train-examples", type=int)
    g.add_argument("--max-val-examples", type=int)
    g.add_argument(
        "--max-tokens", type=int, help="drop training examples longer than this"
    )
    g.add_argument("--eval-batch-size", type=int, default=64)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device")
    g.add_argument(
        "--no-pretrained", action="store_true", help="random init (smoke tests)"
    )
    add_wandb_args(p)
    p.add_argument(
        "--wandb-samples",
        type=int,
        default=4,
        help="decoded validation outputs logged per epoch",
    )
    p.add_argument(
        "--wandb-checkpoints",
        action="store_true",
        help="upload each phase's best checkpoint as an artifact",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip phases already finished in --out (progress.json + last.pt)",
    )
    add_hf_args(p, default_upload=True)
    p.set_defaults(fn=train_cmd)

    p = sub.add_parser("eval", help="evaluate a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument(
        "--dataset", choices=DATASETS, help="default: the checkpoint's dataset"
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-examples", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        help="thought budgets for the sweep (default 0..2cK)",
    )
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument(
        "--samples", type=int, default=3, help="decoded outputs to store per mode"
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default=None)
    p.add_argument("--chunk-size", type=int)
    p.add_argument(
        "--fused-scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mamba_ssm's Triton scan for prompts and answers (default: on for CUDA if installed)",
    )
    p.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replay inference thought and decode steps as CUDA graphs (default: on for CUDA)",
    )
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    add_wandb_args(p)
    add_hf_args(p, default_upload=True)
    p.set_defaults(fn=eval_cmd)

    p = sub.add_parser(
        "bench", help="state size, peak memory and latency vs. thinking budget"
    )
    p.add_argument(
        "--models", nargs="+", choices=MODEL_CHOICES, default=["mamba2-130m", "gpt2"]
    )
    p.add_argument(
        "--budgets", type=int, nargs="+", default=[0, 16, 64, 256, 480, 1024, 4096]
    )
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--out", default="runs/bench.json")
    p.add_argument("--device")
    p.add_argument(
        "--fused-scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mamba_ssm's Triton scan for prompts and answers (default: on for CUDA if installed)",
    )
    p.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replay inference thought and decode steps as CUDA graphs (default: on for CUDA)",
    )
    add_wandb_args(p)
    p.set_defaults(fn=bench_cmd)

    p = sub.add_parser("report", help="collect eval and bench results into markdown")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/report.md")
    p.set_defaults(fn=report_cmd)

    p = sub.add_parser(
        "upload", help="upload a run folder or result file to the Hugging Face Hub"
    )
    p.add_argument(
        "--path",
        required=True,
        help="file or folder, e.g. runs/gsm8k-mamba2-130m-latent",
    )
    p.add_argument("--path-in-repo", help="default: the file or folder name")
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    p.add_argument("--hf-private", action="store_true")
    p.set_defaults(fn=upload_cmd)
    return parser


def main_cli(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "dtype", "unset") is None:
        args.dtype = (
            "bf16"
            if torch.cuda.is_available() and getattr(args, "device", None) != "cpu"
            else "fp32"
        )
    args.fn(args)


if __name__ == "__main__":
    main_cli()
