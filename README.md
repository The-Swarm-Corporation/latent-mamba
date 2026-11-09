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
$ pip install torch
```

Requires Python >= 3.10 and PyTorch >= 2.1. The training pipeline also needs `pip install -r requirements.txt`.

## Usage

```python
import torch
from latent_mamba.main import LatentMamba, LatentMambaConfig

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
$ python latent_mamba/main.py
```

This runs `self_test()`, which checks that the chunked scan matches the exact recurrence, that prefill + cached decoding matches a single full pass, that state size is independent of thought count, that gradients reach both the thought projector and halting head, and that left padding is exact. It then runs `run_demo()`, which trains a tiny model through the full recipe (language CoT, Coconut curriculum, PonderNet halting) on k-hop pointer chasing and reports accuracy and thoughts used per hop count, against fixed-budget, no-thought and zero-ablated baselines.

```python
from latent_mamba.main import self_test, run_demo

self_test(device = 'cpu')
model = run_demo(steps_per_stage = 600, ponder_steps = 800, device = 'cuda')
```

The demo is a smoke test of the pipeline, not a benchmark. Real benchmark results are below.


## Links

|---|---|
| Paper | [docs/paper.pdf](docs/paper.pdf) · LaTeX source [docs/latent_mamba.tex](docs/latent_mamba.tex) |
| Model | [latent_mamba/main.py](latent_mamba/main.py) |
| Training & evaluation | [latent_mamba/train.py](latent_mamba/train.py) · [scripts/run_pipeline.sh](scripts/run_pipeline.sh) · [Dockerfile](Dockerfile) |
| Guides | [docs/RUNPOD.md](docs/RUNPOD.md) (GPUs, RunPod) · [docs/architecture.md](docs/architecture.md) (design notes) |
| Checkpoints | [huggingface.co/kye/latent-mamba](https://huggingface.co/kye/latent-mamba) |
| Training logs | [wandb.ai/apacai/latent-mamba](https://wandb.ai/apacai/latent-mamba) |


## Training on reasoning benchmarks

[`latent_mamba/train.py`](latent_mamba/train.py) fine-tunes pretrained Mamba-2 (`state-spaces/mamba2-130m`, `-370m`, ...) on Coconut's GSM8k-Aug, ProsQA and ProntoQA with the full recipe. It compares against the same Mamba with no CoT, with written CoT, and against a Coconut GPT-2 of matched size, then benchmarks memory and latency against thinking budget. Runs log to [Weights & Biases](https://wandb.ai/apacai/latent-mamba) and upload to [kye/latent-mamba](https://huggingface.co/kye/latent-mamba).

```bash
$ pip install -r requirements.txt
$ python latent_mamba/train.py data --dataset gsm8k prosqa
$ torchrun --nproc_per_node 4 latent_mamba/train.py train --model mamba2-130m --dataset prosqa --method latent --wandb
$ python latent_mamba/train.py eval --ckpt runs/prosqa-mamba2-130m-latent/ponder.pt
```

[`scripts/run_pipeline.sh`](scripts/run_pipeline.sh) runs everything for one model and dataset, and the [`Dockerfile`](Dockerfile) wraps it. See [docs/RUNPOD.md](docs/RUNPOD.md) for GPUs and step-by-step instructions.

## Results

ProsQA, Mamba-2 130M fine-tuned from the pretrained checkpoint on one H100, validation accuracy (300 questions, chance 50%). Details in Section 6 of the [paper](docs/paper.pdf).

| Model | Thoughts | Written reasoning | Val. accuracy |
|---|---|---|---|
| No CoT | 0 | none | 80.0% |
| Written CoT | 0 | all steps | 74.3% |
| **Latent Mamba** (final stage) | **6** | **none** | **98.7%** |

These are validation results from a single run. Test-set accuracy, thought ablations, learned halting, the Coconut GPT-2 baseline and measured memory are still running.

## Todo

- [x] load pretrained Mamba-2 weights (the parameter names already match)
- [ ] run the proposed evaluation (H1 - H3) on pointer chasing
- [x] ProsQA following the Coconut settings (validation: 98.7% latent vs 74.3% written CoT vs 80.0% no CoT)
- [ ] ProsQA test set, thought ablations and learned halting
- [ ] ProntoQA and GSM8k
- [ ] memory and latency comparison against a Coconut Transformer of matched size (`latent_mamba/train.py bench`)

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
