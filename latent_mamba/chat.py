"""Chat with a trained Latent Mamba checkpoint in the terminal.

    uv run latent_mamba/chat.py
    uv run latent_mamba/chat.py --trace
    uv run latent_mamba/chat.py --ckpt prosqa-mamba2-130m-latent/final.pt
    uv run latent_mamba/chat.py --question "Every wumpus is a ... Is Bob a ... or ...?"

By default this downloads the final ProsQA model (``ponder.pt``) from
https://huggingface.co/kye/latent-mamba. ``--ckpt`` takes a local path or a
file in that repo. The checkpoints were trained only on ProsQA, so they answer
questions of the form "<facts>. Is <name> a <concept> or <concept>?" and
nothing else.

Type a question at the prompt, or a command:

    /example            a random ProsQA test question (from data/, if present)
    /thoughts N         maximum number of thoughts (fixed rule: exactly N)
    /halt RULE          learned | fixed | convergence
    /trace on|off       print every thought: halting probability and the
                        tokens the thought is closest to through the LM head
    /ablate MODE        zeros | noise | off (replace thought content)
    /settings           show the current settings
    /help, /quit
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from train import (
    decode_answer,
    enable_cuda_graphs,
    install_scan,
    load_checkpoint,
    parse_answer,
)

DEFAULT_REPO = "kye/latent-mamba"
DEFAULT_CKPT = "prosqa-mamba2-130m-latent/ponder.pt"
HALT_RULES = ("learned", "fixed", "convergence")
ABLATIONS = ("zeros", "noise")

EXAMPLE = {
    "question": (
        "Every grimpus is a jelpus. Every jelpus is a shumpus. Bob is a storpus. "
        "Every storpus is a grimpus. Tom is a bompus. Every bompus is a gerpus. "
        "Is Bob a shumpus or gerpus?"
    ),
    "answer": "Bob is a shumpus.",
}

try:  # line editing and history for input()
    __import__("readline")
except ImportError:
    pass

_COLOR = sys.stdout.isatty()


def style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(text: str) -> str:
    return style(text, "2")


def bold(text: str) -> str:
    return style(text, "1")


def green(text: str) -> str:
    return style(text, "32")


def red(text: str) -> str:
    return style(text, "31")


def cyan(text: str) -> str:
    return style(text, "36")


@dataclass
class Settings:
    max_thoughts: int
    halt_rule: str
    halt_threshold: float = 0.5
    convergence_tol: float = 1e-2
    max_new_tokens: int = 64
    trace: bool = False
    ablate: str | None = None
    top_k: int = 5


@dataclass
class Thought:
    p_halt: float  # halting probability of this step, lambda_n
    p_stopped: float  # cumulative probability of having halted by this step
    change: float  # relative change of the hidden state
    top_tokens: list[str]


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple GPU: ~100x faster than CPU here
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint(ckpt: str, repo: str) -> Path:
    path = Path(ckpt)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    print(dim(f"loading {repo}/{ckpt} (downloads 642 MiB the first time) ..."))
    return Path(hf_hub_download(repo, ckpt))


def default_settings(ckpt: dict) -> Settings:
    """Inference settings matching how the checkpoint was trained."""
    phase = ckpt.get("phase", "ponder")
    if phase in ("cot", "nocot"):  # baselines: no thoughts, the answer is written out
        return Settings(max_thoughts=0, halt_rule="fixed", max_new_tokens=128)
    if phase.startswith("stage"):
        return Settings(max_thoughts=int(ckpt.get("stage", 0)), halt_rule="fixed")
    n = int(ckpt.get("ponder_max_thoughts") or 6)
    rule = "learned" if ckpt.get("ponder_trained") or phase == "ponder" else "fixed"
    return Settings(max_thoughts=n, halt_rule=rule)


@torch.no_grad()
def answer(
    model, bundle, question: str, s: Settings
) -> tuple[str, list[Thought], float]:
    """Think, then decode the answer. Same rules as ``LatentMamba.generate``."""
    device = next(model.parameters()).device
    ids = bundle.tokenizer(question.strip() + "\n", add_special_tokens=False)[
        "input_ids"
    ]
    prompt = torch.tensor([ids + [bundle.bot_id]], device=device)
    t0 = time.perf_counter()

    h, cache = model.encode_prompt(prompt)
    thoughts: list[Thought] = []
    survive = 1.0
    for _ in range(s.max_thoughts):
        h_new, cache, halt_logit = model.thought_step(h, cache, ablate=s.ablate)
        change = ((h_new - h).norm() / h.norm().clamp(min=1e-6)).item()
        h = h_new
        lam = torch.sigmoid(halt_logit).item()
        survive *= 1.0 - lam
        top = model.lm_head(h).topk(s.top_k, dim=-1).indices[0].tolist()
        thoughts.append(
            Thought(
                lam, 1.0 - survive, change, [bundle.tokenizer.decode([t]) for t in top]
            )
        )
        if s.halt_rule == "learned" and 1.0 - survive >= s.halt_threshold:
            break
        if s.halt_rule == "convergence" and change < s.convergence_tol:
            break

    token = torch.tensor([[bundle.eot_id]], device=device)
    out: list[int] = []
    for _ in range(s.max_new_tokens):
        logits, cache = model.decode_step(token, cache)
        nxt = int(logits[0].argmax())
        out.append(nxt)
        if nxt == bundle.eos_id:
            break
        token = torch.tensor([[nxt]], device=device)
    return decode_answer(out, bundle), thoughts, time.perf_counter() - t0


def show_trace(thoughts: list[Thought]) -> None:
    for i, t in enumerate(thoughts, 1):
        tokens = " ".join(repr(x.strip() or x) for x in t.top_tokens)
        print(
            dim(
                f"  thought {i}:  halt {t.p_halt:5.1%}  stopped {t.p_stopped:5.1%}  "
                f"change {t.change:6.3f}  nearest tokens "
            )
            + cyan(tokens)
        )


def load_examples(data_dir: str) -> list[dict]:
    path = Path(data_dir) / "prosqa_test.json"
    if path.exists():
        return json.loads(path.read_text())
    return [EXAMPLE]


def describe(s: Settings) -> str:
    rule = s.halt_rule
    if rule == "learned":
        rule += f" (stop at {s.halt_threshold:.0%})"
    elif rule == "convergence":
        rule += f" (tol {s.convergence_tol:g})"
    budget = "exactly" if s.halt_rule == "fixed" else "at most"
    return (
        f"thoughts: {budget} {s.max_thoughts}  |  halting: {rule}  |  "
        f"trace: {'on' if s.trace else 'off'}  |  ablation: {s.ablate or 'off'}"
    )


def run_question(
    model, bundle, question: str, s: Settings, gold: str | None = None
) -> None:
    text, thoughts, secs = answer(model, bundle, question, s)
    if s.trace and thoughts:
        show_trace(thoughts)
    reply = text.split("###")[-1].strip() if "###" in text else text.strip()
    if (
        "###" in text and text.split("###")[0].strip()
    ):  # CoT baseline: show its reasoning
        print(dim(text.split("###")[0].rstrip()))
    print(bold(reply or "(empty answer)"))
    meta = f"{len(thoughts)} thought{'s' if len(thoughts) != 1 else ''}, {secs:.2f}s"
    if gold is not None:
        ok = parse_answer(text) == parse_answer(gold)
        meta += "  " + (green("correct") if ok else red(f"wrong, expected: {gold}"))
    print(dim(meta))


def handle_command(line: str, s: Settings, examples: list[dict], model, bundle) -> bool:
    """Apply a /command. Returns False to quit."""
    cmd, _, arg = line[1:].strip().partition(" ")
    arg = arg.strip()
    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        print(
            __doc__.split("Type a question at the prompt, or a command:")[-1].rstrip()
        )
    elif cmd == "example":
        ex = random.choice(examples)
        print(dim("question: ") + ex["question"])
        run_question(model, bundle, ex["question"], s, gold=ex["answer"])
    elif cmd == "thoughts" and arg.isdigit():
        s.max_thoughts = int(arg)
    elif cmd == "halt" and arg in HALT_RULES:
        s.halt_rule = arg
    elif cmd == "trace" and arg in ("on", "off"):
        s.trace = arg == "on"
    elif cmd == "ablate" and arg in (*ABLATIONS, "off"):
        s.ablate = None if arg == "off" else arg
    elif cmd == "settings":
        pass
    else:
        print(
            red(f"unknown command or bad argument: {line.strip()}")
            + dim("  (try /help)")
        )
        return True
    if cmd in ("thoughts", "halt", "trace", "ablate", "settings"):
        print(dim(describe(s)))
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with a Latent Mamba checkpoint.")
    p.add_argument(
        "--ckpt", default=DEFAULT_CKPT, help="local path, or a file in --repo"
    )
    p.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo for --ckpt")
    p.add_argument(
        "--device", help="cpu, cuda or mps (default: cuda, then mps, then cpu)"
    )
    p.add_argument("--max-thoughts", type=int, help="default: as trained")
    p.add_argument("--halt", choices=HALT_RULES, help="default: as trained")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--trace", action="store_true", help="print every thought")
    p.add_argument("--data-dir", default="data", help="for /example (prosqa_test.json)")
    p.add_argument("--question", help="answer one question and exit")
    args = p.parse_args()

    device = pick_device(args.device)
    install_scan(None, device)  # fused mamba_ssm kernels on CUDA when installed
    path = resolve_checkpoint(args.ckpt, args.repo)
    bundle, ckpt = load_checkpoint(path)
    model = bundle.model.to(device).eval()
    enable_cuda_graphs(bundle, device, None)  # thought and decode steps as CUDA graphs on CUDA

    s = default_settings(ckpt)
    if args.max_thoughts is not None:
        s.max_thoughts = args.max_thoughts
    if args.halt:
        s.halt_rule = args.halt
    if args.max_new_tokens:
        s.max_new_tokens = args.max_new_tokens
    s.trace = args.trace

    if args.question:
        run_question(model, bundle, args.question, s)
        return

    examples = load_examples(args.data_dir)
    print(
        bold("Latent Mamba")
        + dim(f"  {ckpt.get('model_name')}, phase {ckpt.get('phase')}, on {device}")
    )
    print(dim(describe(s)))
    print(dim('Ask a ProsQA-style question: facts, then "Is <name> a <X> or <Y>?"'))
    print(dim("/example for a test question, /help for commands, /quit to leave.\n"))

    while True:
        try:
            line = input(cyan("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(line, s, examples, model, bundle):
                break
            continue
        run_question(model, bundle, line, s)


if __name__ == "__main__":
    main()
