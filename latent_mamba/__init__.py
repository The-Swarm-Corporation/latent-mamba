"""Latent Mamba: adaptive continuous reasoning in constant memory.

The model needs only PyTorch. The training pipeline (``train.py``) and the chat
CLI (``chat.py``) are scripts with extra dependencies and are not imported here:

    uv run latent_mamba/train.py --help
    uv run latent_mamba/chat.py
"""

from latent_mamba.main import (
    IGNORE_INDEX,
    Ablation,
    GenerationOutput,
    HaltingHead,
    HaltRule,
    LatentLossOutput,
    LatentMamba,
    LatentMambaConfig,
    LayerState,
    Mamba2Mixer,
    MambaBackbone,
    MambaCache,
    PonderLossOutput,
    ResidualBlock,
    RMSNorm,
    ThoughtProjector,
    ssd_chunked_scan,
    conv_step,
    ssd_step,
)
from latent_mamba.tests import PointerChasingTask, run_demo, self_test

__version__ = "0.2.0"

__all__ = [
    "IGNORE_INDEX",
    "Ablation",
    "GenerationOutput",
    "HaltRule",
    "HaltingHead",
    "LatentLossOutput",
    "LatentMamba",
    "LatentMambaConfig",
    "LayerState",
    "Mamba2Mixer",
    "MambaBackbone",
    "MambaCache",
    "PointerChasingTask",
    "PonderLossOutput",
    "RMSNorm",
    "ResidualBlock",
    "ThoughtProjector",
    "run_demo",
    "self_test",
    "conv_step",
    "ssd_chunked_scan",
    "ssd_step",
]
