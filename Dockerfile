# Latent Mamba training image.
#
#   docker build --platform linux/amd64 -t <dockerhub-user>/latent-mamba:0.1 .
#   docker push <dockerhub-user>/latent-mamba:0.1
#
# The container runs scripts/run_pipeline.sh, configured through environment
# variables (MODEL, DATASET, HF_TOKEN, WANDB_API_KEY, ...). See docs/RUNPOD.md.
#
# The image includes mamba_ssm's fused scan kernels, which training uses by
# default on the GPU (compiling them can add ~20 min to the build). Build with
# --build-arg INSTALL_MAMBA_SSM=0 to skip them; the reference scan is then used.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WORKDIR=/workspace \
    HF_HOME=/workspace/hf-cache \
    WANDB_DIR=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
# Install into the image's Python, reusing its CUDA build of PyTorch.
RUN uv pip install --system -r pyproject.toml --extra train

ARG INSTALL_MAMBA_SSM=1
RUN if [ "$INSTALL_MAMBA_SSM" = "1" ]; then \
        uv pip install --system packaging ninja && uv pip install --system --no-build-isolation mamba-ssm; \
    fi

COPY latent_mamba/ latent_mamba/
COPY scripts/ scripts/

CMD ["bash", "scripts/run_pipeline.sh"]
