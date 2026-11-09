#!/usr/bin/env bash
# Full Step 2 pipeline for one model on one dataset:
#   data -> train (cot, nocot, latent) -> eval on test -> memory/latency bench -> report -> upload
#
# Every step can be rerun safely: finished training phases and finished
# evaluations are skipped, so a restarted pod continues where it stopped.
#
# Environment (all optional):
#   MODEL          mamba2-130m | mamba2-370m | gpt2 | gpt2-medium | tiny-mamba | tiny-gpt2  (default mamba2-130m)
#   DATASET        gsm8k | prosqa | prontoqa                                      (default gsm8k)
#   METHODS        runs to do, in order                                           (default "cot nocot latent")
#   NGPU           GPUs to use                                                    (default: all visible)
#   WORKDIR        persistent directory, e.g. a RunPod network volume             (default /workspace)
#   HF_TOKEN       Hugging Face write token. Without it nothing is uploaded.
#   HF_REPO        upload target                                                  (default kye/latent-mamba)
#   WANDB_API_KEY  turns on Weights & Biases logging (also WANDB_PROJECT, WANDB_ENTITY)
#   TRAIN_ARGS     extra arguments for every train call, e.g. "--batch-size 16 --grad-accum 2"
#   EVAL_ARGS      extra arguments for every eval call
#   BENCH          1 runs the memory/latency benchmark                            (default 1)
#   KEEP_ALIVE     1 sleeps when finished or failed, so the container is not restarted (default 1)
#   FUSED          1 installs mamba_ssm and causal-conv1d when a CUDA GPU is present and they
#                  are missing, for the fused kernels train.py then uses automatically (default 1)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-mamba2-130m}
DATASET=${DATASET:-gsm8k}
METHODS=${METHODS:-"cot nocot latent"}
WORKDIR=${WORKDIR:-/workspace}
HF_REPO=${HF_REPO:-kye/latent-mamba}
BENCH=${BENCH:-1}
KEEP_ALIVE=${KEEP_ALIVE:-1}
TRAIN_ARGS=${TRAIN_ARGS:-}
EVAL_ARGS=${EVAL_ARGS:-}
if [ -z "${NGPU:-}" ]; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  [ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1
fi

DATA="$WORKDIR/data"
RUNS="$WORKDIR/runs"
mkdir -p "$DATA" "$RUNS" "$WORKDIR/logs"
exec > >(tee -a "$WORKDIR/logs/pipeline-$DATASET-$MODEL.log") 2>&1

finish() {
  echo "$1"
  if [ "$KEEP_ALIVE" = 1 ]; then
    echo "sleeping so the container stays up; stop the pod when you are done (KEEP_ALIVE=0 to exit instead)"
    sleep infinity
  fi
}
trap 'finish "pipeline FAILED at line $LINENO (log: $WORKDIR/logs/pipeline-$DATASET-$MODEL.log)"' ERR

# Fused kernels: train.py uses mamba_ssm (scan, single step) and causal-conv1d (single-token
# conv) automatically on CUDA when installed, each checked against the reference at startup.
# Versions match the `cuda` extra in pyproject.toml.
FUSED=${FUSED:-1}
if [ "$FUSED" = 1 ] && python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if python -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
    echo "mamba_ssm and causal_conv1d found: fused kernels on"
  else
    echo "CUDA detected, installing mamba_ssm and causal-conv1d for the fused kernels (can take ~20 min if they compile)"
    if [ -n "${VIRTUAL_ENV:-}" ] && command -v uv >/dev/null 2>&1; then PIP=(uv pip install); else PIP=(python -m pip install); fi
    if "${PIP[@]}" packaging ninja wheel setuptools && "${PIP[@]}" --no-build-isolation mamba-ssm==2.3.1 causal-conv1d==1.7.0; then
      echo "fused kernels installed"
    else
      echo "warning: could not install the fused kernels, using the reference implementation"
    fi
  fi
fi

FLAGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  FLAGS+=(--hf-repo "$HF_REPO")
else
  echo "warning: HF_TOKEN is not set, results will NOT be uploaded to $HF_REPO"
  FLAGS+=(--no-hf-upload)
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  FLAGS+=(--wandb --wandb-project "${WANDB_PROJECT:-latent-mamba}")
  [ -n "${WANDB_ENTITY:-}" ] && FLAGS+=(--wandb-entity "$WANDB_ENTITY")
else
  echo "warning: WANDB_API_KEY is not set, Weights & Biases logging is off"
fi

launch() {
  if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node "$NGPU" latent_mamba/train.py "$@"
  else
    python latent_mamba/train.py "$@"
  fi
}
run_dir() { echo "$RUNS/$DATASET-$MODEL-$1"; }

echo "=== $(date -u) | model $MODEL | dataset $DATASET | methods: $METHODS | $NGPU GPU(s) | workdir $WORKDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# 1. Data
if [ ! -f "$DATA/${DATASET}_test.json" ]; then
  python latent_mamba/train.py data --dataset "$DATASET" --data-dir "$DATA"
  [ -f "$DATA/${DATASET}_test.json" ] || { echo "no $DATASET data (ProntoQA must be generated first, see above)"; false; }
fi

# 2. Training
for method in $METHODS; do
  init=()
  if [ "$method" = latent ] && [ "$DATASET" = gsm8k ]; then
    # Coconut starts GSM8k's latent curriculum from the language-CoT checkpoint.
    cot_ckpt="$(run_dir cot)/cot.pt"
    [ -f "$cot_ckpt" ] || { echo "latent on gsm8k needs the cot run first (METHODS=\"cot latent\")"; false; }
    init=(--init-from "$cot_ckpt" --start-stage 1)
  fi
  echo "=== train $method"
  # shellcheck disable=SC2086  # TRAIN_ARGS is split on purpose
  launch train --model "$MODEL" --dataset "$DATASET" --method "$method" --data-dir "$DATA" \
    --out "$(run_dir "$method")" ${init[@]+"${init[@]}"} "${FLAGS[@]}" $TRAIN_ARGS
done

# 3. Evaluation on the test split
for method in $METHODS; do
  dir=$(run_dir "$method")
  case $method in
    cot) ckpt=$dir/cot.pt ;;
    nocot) ckpt=$dir/nocot.pt ;;
    latent) ckpt=$dir/ponder.pt; [ -f "$ckpt" ] || ckpt=$dir/final.pt ;;
  esac
  result="$dir/eval_test_$(basename "$ckpt" .pt).json"
  if [ -f "$result" ]; then
    echo "=== eval $method: $result exists, skipping"
    continue
  fi
  echo "=== eval $method ($ckpt)"
  # shellcheck disable=SC2086
  launch eval --ckpt "$ckpt" --split test --data-dir "$DATA" "${FLAGS[@]}" $EVAL_ARGS
done

# 4. Memory and latency against thinking budget, against the size-matched model
if [ "$BENCH" = 1 ]; then
  case $MODEL in
    mamba2-130m) pair="mamba2-130m gpt2" ;;
    mamba2-370m) pair="mamba2-370m gpt2-medium" ;;
    gpt2) pair="gpt2 mamba2-130m" ;;
    gpt2-medium) pair="gpt2-medium mamba2-370m" ;;
    *) pair="$MODEL" ;;
  esac
  bench="$RUNS/bench/bench-$MODEL.json"
  if [ ! -f "$bench" ]; then
    echo "=== bench $pair"
    bench_flags=()
    [ -n "${WANDB_API_KEY:-}" ] && bench_flags+=(--wandb --wandb-project "${WANDB_PROJECT:-latent-mamba}")
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python latent_mamba/train.py bench --models $pair --dtype bf16 \
      --out "$bench" ${bench_flags[@]+"${bench_flags[@]}"}
  fi
fi

# 5. Report over every run on this volume, then upload the report and benchmark
python latent_mamba/train.py report --runs "$RUNS" --out "$RUNS/report.md"
if [ -n "${HF_TOKEN:-}" ]; then
  python latent_mamba/train.py upload --path "$RUNS/report.md" --path-in-repo report.md --hf-repo "$HF_REPO"
  [ -f "$RUNS/bench/bench-$MODEL.json" ] && \
    python latent_mamba/train.py upload --path "$RUNS/bench/bench-$MODEL.json" --path-in-repo "bench/bench-$MODEL.json" --hf-repo "$HF_REPO"
fi

trap - ERR
finish "=== $(date -u) pipeline finished: https://huggingface.co/$HF_REPO"
