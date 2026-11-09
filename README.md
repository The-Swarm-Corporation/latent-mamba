## Latent Mamba

Implementation of <a href="docs/paper.pdf">Latent Mamba</a>, adaptive continuous reasoning in constant memory, in Pytorch


A Mamba-2 language model is taught to think silently in continuous space between reading a prompt and writing an answer. The final normalised hidden state is projected back into input-embedding space by a learned *thought projector* (plus a thinking-mode embedding) and fed in as the next input, Coconut-style. A small *halting head*, trained with the PonderNet objective, decides when to stop thinking.

The key observation is a *time-depth collapse*: in a selective state-space model, depth recurrence with a shared state is the same computation as interleaving self-generated latent inputs along the sequence. So the thoughts are placed in time rather than in depth. Every thought only updates the fixed-size recurrent state (conv window + SSM state), so **memory stays O(1) in the number of thoughts**, unlike a Transformer where every continuous thought still occupies a KV cache slot. For a 2.7B configuration the added modules come to about 0.3% of parameters.

```
prompt ... <bot>                                                 <eot> answer
     |                                                             |
     v                                                             v
token embedding --> Mamba-2 backbone --> final hidden h --> halting head
                      ^  (fixed-size conv + SSM state)          |      |
                      |                                     continue  halt
                      |                                         |      |
                      +----------- thought projector <----------+      v
                                                                    LM head
```

The design rationale (why thoughts go in time rather than depth, pitfalls, related work) is written up in [docs/architecture.md](docs/architecture.md), with the original [interactive architecture diagram](https://claude.ai/chat/a327398f-fd14-42ac-928a-e84e1de7229e?artifact=6098850e-24e4-4ba1-80aa-fee723e678e0).

The model lives in a single file, [`latent_mamba/main.py`](latent_mamba/main.py). It is pure Pytorch with no custom kernels: sequences use the chunked SSD scan of Mamba-2 (with an initial state), single tokens use the exact recurrence, and both are differentiable so gradients flow through every thought step.

## Install

```bash
$ git clone https://github.com/The-Swarm-Corporation/latent-mamba
$ cd latent-mamba
$ uv sync                  # model only (PyTorch)
$ uv sync --extra train    # plus the training pipeline
```

Uses [uv](https://docs.astral.sh/uv/) with a locked environment (`uv.lock`). Requires Python >= 3.10 and PyTorch >= 2.4. Run code with `uv run`, e.g. `uv run latent_mamba/train.py --help`.

## Usage

```python
import torch
from latent_mamba import LatentMamba, LatentMambaConfig

config = LatentMambaConfig(
    vocab_size = 1000,
    bot_id = 997,          # begin-of-thought, must end every prompt
    eot_id = 998,          # end-of-thought, fed before the answer
    pad_id = 999,
    d_model = 256,
    n_layers = 4,
    d_state = 64,
    headdim = 32,
    halt_prior = 0.2       # lambda_p of the geometric prior over halting steps
)

model = LatentMamba(config)

prompt = torch.randint(0, 997, (2, 12))
prompt[:, -1] = config.bot_id

target = torch.randint(0, 997, (2, 3))

# adaptive halting (PonderNet) loss - answer is scored after every thought in one batched pass

out = model.ponder_loss(prompt, target, max_thoughts = 6)
out.loss.backward()

# think adaptively, then decode

gen = model.generate(prompt, max_new_tokens = 3, max_thoughts = 6)

gen.tokens      # (2, 3)
gen.n_thoughts  # (2,) - thoughts taken per example
```

### Training recipe

All three stages from the paper are implemented.

```python
# 1. language chain of thought - zero thoughts, full CoT as the target (Coconut stage 0)

loss = model.latent_loss(prompt, cot_and_answer, n_thoughts = 0).loss

# 2. Coconut curriculum - at stage k, the first k CoT steps are replaced by c * k continuous thoughts
#    n_thoughts can also be a (batch,) tensor, finished examples are frozen while the rest keep thinking

loss = model.latent_loss(prompt, remaining_cot_and_answer, n_thoughts = c * k).loss

# 3. learning to halt - expected answer loss under the halting distribution + beta * KL to a geometric prior

out = model.ponder_loss(prompt, answer, max_thoughts = 8, beta = 0.01)

out.loss
out.halting_distribution  # (batch, max_thoughts), rows sum to 1
out.expected_thoughts     # (batch,)
```

Targets are padded with `IGNORE_INDEX` (`-100`). Prompts of different lengths can share a batch: left-pad them and pass `prompt_mask` (True for real tokens) to `latent_loss`, `ponder_loss` and `generate`. Pad positions get Δ = 0 and a zero conv input, so they leave the state exactly as if they were absent.

### Inference

```python
# learned halting - stop at the median of the predicted halting distribution

model.generate(prompt, halt_rule = 'learned', halt_threshold = 0.5)

# convergence - stop when ||h_n - h_{n-1}|| / ||h_{n-1}|| < tol, needs no halting training

model.generate(prompt, halt_rule = 'convergence', convergence_tol = 1e-2)

# fixed - always take max_thoughts

model.generate(prompt, halt_rule = 'fixed', max_thoughts = 4)

# ablations - check the model actually uses its thoughts rather than treating them as pause tokens

model.generate(prompt, ablate = 'zeros')
model.generate(prompt, ablate = 'noise')
```

### Constant memory

```python
h, cache = model.encode_prompt(prompt)

before = cache.num_bytes()

for _ in range(100):
    h, cache, halt_logit = model.thought_step(h, cache)

assert cache.num_bytes() == before
```

Caches are functional (every update returns new tensors), so they can be branched, frozen per example with `cache.where(mask, other)`, concatenated with `MambaCache.cat`, or truncated with `cache.detach()` for long thinking budgets.

## Test and demo

```bash
$ uv run latent_mamba/main.py
```

This runs `self_test()`, which checks that the chunked scan matches the exact recurrence, that prefill + cached decoding matches a single full pass, that state size is independent of thought count, that gradients reach both the thought projector and halting head, and that left padding is exact. It then runs `run_demo()`, which trains a tiny model through the full recipe (language CoT, Coconut curriculum, PonderNet halting) on k-hop pointer chasing and reports accuracy and thoughts used per hop count, against fixed-budget, no-thought and zero-ablated baselines.

```python
from latent_mamba import self_test, run_demo

self_test(device = 'cpu')
model = run_demo(steps_per_stage = 600, ponder_steps = 800, device = 'cuda')
```

The demo is a smoke test of the pipeline, not a benchmark. Real benchmark results are below.


## Links

| Resource | Link |
|---|---|
| Paper | [docs/paper.pdf](docs/paper.pdf) · LaTeX source [docs/latent_mamba.tex](docs/latent_mamba.tex) |
| Model | [latent_mamba/main.py](latent_mamba/main.py) |
| Training & evaluation | [latent_mamba/train.py](latent_mamba/train.py) · [scripts/run_pipeline.sh](scripts/run_pipeline.sh) · [Dockerfile](Dockerfile) |
| Chat | [latent_mamba/chat.py](latent_mamba/chat.py) |
| Environment | [pyproject.toml](pyproject.toml) · [uv.lock](uv.lock) |
| Guides | [docs/RUNPOD.md](docs/RUNPOD.md) (GPUs, RunPod) · [docs/architecture.md](docs/architecture.md) (design notes) |
| Checkpoints | [huggingface.co/kye/latent-mamba](https://huggingface.co/kye/latent-mamba) |
| Training logs | [wandb.ai/apacai/latent-mamba](https://wandb.ai/apacai/latent-mamba) |


## Training

[`latent_mamba/train.py`](latent_mamba/train.py) fine-tunes pretrained Mamba-2 (`state-spaces/mamba2-130m`, `-370m`, `-780m`, `-1.3b`, `-2.7b`) on Coconut's GSM8k-Aug, ProsQA and ProntoQA with the full recipe: language CoT, the Coconut curriculum, then learning to halt. It also trains the baselines (the same Mamba with no CoT and with written CoT, and a Coconut GPT-2 of matched size), evaluates on the test set and benchmarks memory and latency against thinking budget.

Install the training dependencies and download the data

```bash
$ uv sync --extra train
$ uv run latent_mamba/train.py data --dataset prosqa gsm8k
```

Train Latent Mamba on ProsQA on one GPU (about 14 hours for 130M on an H100)

```bash
$ uv run latent_mamba/train.py train \
    --model mamba2-130m \
    --dataset prosqa \
    --method latent \
    --no-hf-upload
```

The baselines use the same command with `--method cot` (written chain of thought) or `--method nocot` (answer directly). `--model gpt2` trains the Coconut Transformer baseline.

Quick smoke test on a laptop, a tiny randomly initialised model on a slice of the data (seconds)

```bash
$ uv run latent_mamba/train.py train --model tiny-mamba --dataset prosqa --method latent \
    --max-train-examples 64 --max-val-examples 8 \
    --cot-epochs 1 --epochs-per-stage 1 --final-epochs 1 --ponder-epochs 1 \
    --no-hf-upload
```

Several GPUs, with Weights & Biases logging

```bash
$ uv run torchrun --nproc_per_node 4 latent_mamba/train.py train \
    --model mamba2-370m --dataset prosqa --method latent --wandb
```

GSM8k, where the latent curriculum starts from the written-CoT checkpoint, as in Coconut

```bash
$ uv run latent_mamba/train.py train --model mamba2-130m --dataset gsm8k --method cot
$ uv run latent_mamba/train.py train --model mamba2-130m --dataset gsm8k --method latent \
    --init-from runs/gsm8k-mamba2-130m-cot/cot.pt --start-stage 1
```

Evaluate a checkpoint on the test set (every halting rule, the thought ablations and a 0 to 12 thought sweep), and benchmark memory and latency

```bash
$ uv run latent_mamba/train.py eval --ckpt runs/prosqa-mamba2-130m-latent/ponder.pt --no-hf-upload
$ uv run latent_mamba/train.py bench --models mamba2-130m gpt2
```

Checkpoints go to `runs/{dataset}-{model}-{method}/`, one per phase (`cot.pt`, `stage1.pt` ... `final.pt`, `ponder.pt`), each the best validation epoch of that phase. Runs resume from where they stopped. By default results also upload to [kye/latent-mamba](https://huggingface.co/kye/latent-mamba), which needs a write token (`hf auth login`); `--no-hf-upload` turns that off.

Useful flags

- `--batch-size 16 --grad-accum 2` if a GPU runs out of memory
- `--wandb` logs to [Weights & Biases](https://wandb.ai/apacai/latent-mamba)
- on CUDA the fused `mamba_ssm` scan is used automatically when installed (`pip install mamba-ssm --no-build-isolation`); `--no-fused-scan` turns it off
- `uv run latent_mamba/train.py train --help` lists everything

[`scripts/run_pipeline.sh`](scripts/run_pipeline.sh) runs the whole thing for one model and dataset (data, the three runs, test evaluation, benchmark, upload), and the [`Dockerfile`](Dockerfile) wraps it.

```bash
$ MODEL=mamba2-130m DATASET=prosqa WORKDIR=$PWD uv run bash scripts/run_pipeline.sh
```

See [docs/RUNPOD.md](docs/RUNPOD.md) for GPU choices and step-by-step cloud instructions.

## Chat

[`latent_mamba/chat.py`](latent_mamba/chat.py) loads the trained model from [Hugging Face](https://huggingface.co/kye/latent-mamba) (642 MiB, downloaded once) and answers questions in the terminal, showing how many thoughts it took.

```bash
$ uv run --extra train latent_mamba/chat.py
```

```
Latent Mamba  mamba2-130m, phase ponder, on mps
thoughts: at most 6  |  halting: learned (stop at 50%)  |  trace: off  |  ablation: off

> Every grimpus is a jelpus. Every jelpus is a shumpus. Bob is a storpus. Every storpus is a grimpus. Tom is a bompus. Every bompus is a gerpus. Is Bob a shumpus or gerpus?
Bob is a shumpus.
3 thoughts, 0.31s
```

The model was trained only on ProsQA, so it answers questions of that form: a list of "every X is a Y" facts, then "Is <name> a <X> or <Y>?". Anything else gets a ProsQA-shaped answer.

| Command | Effect |
|---|---|
| `/example` | ask a random ProsQA test question and grade the answer (needs `data/prosqa_test.json`) |
| `/trace on` | print every thought: its halting probability and the tokens it is closest to |
| `/thoughts N` | maximum number of thoughts (exactly N with the fixed rule) |
| `/halt learned`, `/halt fixed`, `/halt convergence` | how the model decides to stop thinking |
| `/ablate zeros`, `/ablate noise`, `/ablate off` | replace what the thoughts contain |
| `/settings`, `/help`, `/quit` | |

```bash
$ uv run --extra train latent_mamba/chat.py --trace                       # trace from the start
$ uv run --extra train latent_mamba/chat.py --question "..."              # answer one question and exit
$ uv run --extra train latent_mamba/chat.py --ckpt prosqa-mamba2-130m-cot/cot.pt   # any checkpoint in the repo, or a local path
```

It picks CUDA, then Apple's GPU, then the CPU. On the CPU it works but takes several seconds per question.

## Results

ProsQA test set (500 questions, chance 50%), Mamba-2 130M fine-tuned from the pretrained checkpoint on one H100, one seed. Details in Section 6 of the [paper](docs/paper.pdf).

| Model | Thoughts | Test accuracy |
|---|---|---|
| No CoT | 0 | 74.6% |
| Written CoT | 0 | 70.8% |
| **Latent Mamba**, learned halting | **3.1 (mean)** | **98.6%** |
| **Latent Mamba**, fixed budget | **6** | **99.0%** |
| ablation: no thoughts | 0 | 73.6% |
| ablation: thoughts set to zero | 6 | 76.8% |
| ablation: thoughts replaced by noise | 6 | 75.8% |

Removing or scrambling the thoughts drops accuracy back to the baseline, so the content of the thoughts carries the reasoning. Learned halting uses half the thoughts for 0.4 points, matching a fixed budget of three, but adapts only weakly to question difficulty. The recurrent state stayed at 18.2 MiB for budgets up to 4,096 thoughts.


## Citations

```bibtex
@techreport{gomez2026latentmamba,
    title   = {Latent Mamba: Adaptive Continuous Reasoning in Constant Memory},
    author  = {Kye Gomez},
    year    = {2026},
    month   = {November},
    institution = {Swarms Corp},
    url     = {https://github.com/The-Swarm-Corporation/latent-mamba}
}
```

```bibtex
@article{Gu2023MambaLS,
    title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
    author  = {Albert Gu and Tri Dao},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.00752}
}
```

```bibtex
@inproceedings{Dao2024TransformersAS,
    title   = {Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
    author  = {Tri Dao and Albert Gu},
    booktitle = {International Conference on Machine Learning},
    year    = {2024}
}
```

```bibtex
@article{Hao2024TrainingLL,
    title   = {Training Large Language Models to Reason in a Continuous Latent Space},
    author  = {Shibo Hao and Sainbayar Sukhbaatar and DiJia Su and Xian Li and Zhiting Hu and Jason Weston and Yuandong Tian},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2412.06769}
}
```

```bibtex
@article{Banino2021PonderNetLT,
    title   = {PonderNet: Learning to Ponder},
    author  = {Andrea Banino and Jan Balaguer and Charles Blundell},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2107.05407}
}
```

```bibtex
@article{Graves2016AdaptiveCT,
    title   = {Adaptive Computation Time for Recurrent Neural Networks},
    author  = {Alex Graves},
    journal = {ArXiv},
    year    = {2016},
    volume  = {abs/1603.08983}
}
```

```bibtex
@inproceedings{Geiping2025ScalingUT,
    title   = {Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach},
    author  = {Jonas Geiping and Sean McLeish and Neel Jain and John Kirchenbauer and Siddharth Singh and Brian R. Bartoldson and Bhavya Kailkhura and Abhinav Bhatele and Tom Goldstein},
    booktitle = {NeurIPS},
    year    = {2025}
}
```

```bibtex
@article{Deng2024FromEC,
    title   = {From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step},
    author  = {Yuntian Deng and Yejin Choi and Stuart Shieber},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2405.14838}
}
```

*Thinking is just time, if you share the state.*
