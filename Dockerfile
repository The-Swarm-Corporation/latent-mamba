# Latent Mamba training image.
#
#   docker build --platform linux/amd64 -t <dockerhub-user>/latent-mamba:0.1 .
#   docker push <dockerhub-user>/latent-mamba:0.1
#
# The container runs scripts/run_pipeline.sh, configured through environment
# variables (MODEL, DATASET, HF_TOKEN, WANDB_API_KEY, ...). See docs/RUNPOD.md.
#
# Build with --build-arg INSTALL_MAMBA_SSM=1 to compile mamba_ssm's fused scan
# kernels (adds ~20 min to the build). Then add --fused-scan to TRAIN_ARGS.
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

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG INSTALL_MAMBA_SSM=0
RUN if [ "$INSTALL_MAMBA_SSM" = "1" ]; then \
        pip install packaging ninja && pip install --no-build-isolation mamba-ssm; \
    fi

COPY latent_mamba/ latent_mamba/
COPY scripts/ scripts/

CMD ["bash", "scripts/run_pipeline.sh"]
