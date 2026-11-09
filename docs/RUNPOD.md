# Running Latent Mamba on RunPod

One command, `scripts/run_pipeline.sh`, runs the whole Step 2 experiment for one model on one dataset:

1. Download the data.
2. Train the CoT, no-CoT and latent runs.
3. Evaluate on the test split.
4. Benchmark memory and latency against thinking budget.
5. Write a report.

Everything logs to Weights & Biases. Every run folder (checkpoints, `run.json`, `metrics.jsonl`, eval results) is uploaded to <https://huggingface.co/kye/latent-mamba>, one folder per run, e.g. `gsm8k-mamba2-130m-latent/`. The upload happens after every training phase and again at the end.

If a pod restarts, rerunning the script continues where it stopped. Finished phases are skipped, the model reloads from `last.pt`, and the same W&B run continues.

## GPUs

The models are small (130M and 370M parameters), so memory is not the limit. Speed is limited by the pure-PyTorch scan and by the thought steps, which run one at a time.

| Job | GPUs | Why |
|---|---|---|
| Smoke test (first, always) | 1× H100 80GB (or A100 80GB) | About 20 minutes. Gives you the real throughput number for budgeting |
| ProsQA or ProntoQA, full recipe | 1× H100 80GB | 18k / 9k training examples |
| GSM8k, full recipe | 4× H100 SXM 80GB | 385k examples × ~75 epochs across the three runs. Matches Coconut's 4 GPUs × 32 batch. Pick SXM (NVLink), since gradients are all-reduced every step |
| GSM8k on a budget | 4× A100 80GB SXM | Same setup, slower |

I have not measured GPU throughput yet. Budget from the smoke test, not from guesses. W&B logs `train/batches_per_s`, and the time for one epoch is `examples / (32 × GPUs) / batches_per_s`.

Run the Mamba and GPT-2 baselines as separate pods at the same time if you want results sooner.

## One-time setup

1. **Hugging Face:** create a token with *write* access at <https://huggingface.co/settings/tokens>.
2. **W&B:** copy your API key from <https://wandb.ai/authorize>.
3. **RunPod secrets** (Settings → Secrets): add `hf_token` and `wandb_api_key`. In a pod's environment variables you reference them as `{{ RUNPOD_SECRET_hf_token }}` and `{{ RUNPOD_SECRET_wandb_api_key }}`, so the keys never sit in a template.
4. **Network volume** (Storage → New Network Volume): 200 GB, in a datacenter that has the GPUs you want. Pods mount it at `/workspace`. It holds data, the HF model cache, checkpoints and logs, and survives pod termination.

## Option A: clone the repo (quickest to start)

Deploy a pod with:
- **GPUs:** from the table above
- **Template:** RunPod's PyTorch 2.x template
- **Volume:** your network volume
- **Environment variables:** `HF_TOKEN={{ RUNPOD_SECRET_hf_token }}` and `WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}`

Then open its web terminal or SSH in and run:

```bash
cd /workspace
git clone https://github.com/The-Swarm-Corporation/latent-mamba && cd latent-mamba
pip install uv && uv pip install --system -r pyproject.toml --extra train   # reuses the pod's PyTorch
export HF_HOME=/workspace/hf-cache WANDB_DIR=/workspace

# smoke test: about 20 minutes, small slices of ProsQA, everything end to end
MODEL=mamba2-130m DATASET=prosqa KEEP_ALIVE=0 HF_REPO=kye/latent-mamba-smoke \
  TRAIN_ARGS="--max-train-examples 2000 --cot-epochs 1 --epochs-per-stage 1 --final-epochs 1 --ponder-epochs 1" \
  EVAL_ARGS="--max-examples 100" BENCH=0 \
  bash scripts/run_pipeline.sh

# the real runs: inside tmux, so closing the terminal does not stop them
tmux new -s train
MODEL=mamba2-130m DATASET=gsm8k KEEP_ALIVE=0 bash scripts/run_pipeline.sh
```

The smoke test uploads to `kye/latent-mamba-smoke` so the main repo stays clean. Detach from tmux with `Ctrl-b d` and reattach with `tmux attach -t train`.

## Option B: Docker image (repeatable, hands-off)

Build and push once, from the repo root:

```bash
docker build --platform linux/amd64 -t <dockerhub-user>/latent-mamba:0.1 .
docker push <dockerhub-user>/latent-mamba:0.1
```

Create a RunPod template:
- **Container image:** `<dockerhub-user>/latent-mamba:0.1`
- **Container disk:** 30 GB
- **Volume mount path:** `/workspace` (attach the network volume when deploying)
- **Environment variables:**

```
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}
MODEL=mamba2-130m
DATASET=gsm8k
```

Deploy a pod from the template. The container runs the pipeline on start. When the pipeline finishes or fails, it sleeps instead of exiting. RunPod restarts containers that exit, so this stops the pipeline from looping. **Stop the pod yourself when the W&B run is done.**

## Run matrix

The comparison needs each dataset run with these models:

| `MODEL` | `METHODS` | Provides |
|---|---|---|
| `mamba2-130m` | `cot nocot latent` (default) | Latent Mamba plus both Mamba baselines |
| `gpt2` | `cot latent` | Coconut Transformer baseline (GSM8k's latent run starts from its CoT checkpoint) |
| `mamba2-370m` / `gpt2-medium` | same as above | The larger pair |

Useful `TRAIN_ARGS`:
- `--batch-size 16 --grad-accum 2`: if a GPU runs out of memory (keeps the global batch at 128 on 4 GPUs)
- Fast paths, all on by default on a CUDA GPU and each checked against the reference at startup (the reference is used if they disagree):
    - fused `mamba_ssm` scan for prompts and answers (`--no-fused-scan` turns it off)
    - training thoughts compiled with `torch.compile` (`--compile-thoughts off`; `cudagraphs` adds CUDA graphs over forward and backward, experimental)
    - inference thoughts and decoding through `mamba_ssm`'s fused single step, `causal-conv1d`'s fused conv, and CUDA graphs (`--no-cuda-graphs`)

  The pipeline installs `mamba_ssm` and `causal-conv1d` if they are missing (`FUSED=0` skips that); the Docker image already includes them
- `--wandb-checkpoints`: also stores checkpoints as W&B artifacts
- `--max-tokens N`: drops overly long training examples

**ProntoQA** has to be generated first with the official generator (`python latent_mamba/train.py data --dataset prontoqa` prints the commands). Put the output in `/workspace/data/` by running `python latent_mamba/train.py data --dataset prontoqa --prontoqa-raw 5hop_0shot_random.json --data-dir /workspace/data`.

## Where results go

- **W&B project `latent-mamba`:**
  - one run per training run, grouped by run folder
  - curves: `train/loss`, `train/batches_per_s`, `val/acc` per phase, `val/acc_{k}_steps`, sample outputs
  - one `eval-…` run per evaluation, with tables and the accuracy-vs-budget plot
  - a `bench` run
- **Hugging Face `kye/latent-mamba`:**
  - one folder per run: phase checkpoints (`cot.pt`, `stage*.pt`, `final.pt`, `ponder.pt`), `last.pt`, `run.json`, `progress.json`, `metrics.jsonl`, `eval_test_*.json`
  - plus `report.md` and `bench/`
- **Network volume:** `/workspace/runs`, `/workspace/logs/pipeline-*.log`

To load a checkpoint anywhere:

```python
from huggingface_hub import hf_hub_download
import sys; sys.path.insert(0, "latent_mamba")
from train import load_checkpoint

bundle, meta = load_checkpoint(hf_hub_download("kye/latent-mamba", "gsm8k-mamba2-130m-latent/ponder.pt"))
```

To upload something by hand (for example after an interrupted upload): `python latent_mamba/train.py upload --path /workspace/runs/<run> --hf-repo kye/latent-mamba`.
