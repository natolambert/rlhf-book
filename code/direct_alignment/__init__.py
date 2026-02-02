# Direct Alignment Algorithms
#
# Educational implementations of DPO and related direct alignment methods
# for RLHF Book (https://rlhfbook.com). See Chapter 8 for theory.
#
# Algorithms implemented:
# - DPO: Direct Preference Optimization (Rafailov et al., 2023)
# - cDPO: Conservative DPO with label smoothing
# - IPO: Identity Preference Optimization (Azar et al., 2023)
# - SimPO: Simple Preference Optimization (Meng et al., 2024)
# - ORPO: Odds Ratio Preference Optimization (Hong et al., 2024)
# - KTO: Kahneman-Tversky Optimization (Ethayarajh et al., 2024)
#
# Usage:
#   uv run python -m direct_alignment.train --config configs/dpo.yaml
#   uv run python -m direct_alignment.train --loss dpo --max_samples 1000

from .config import Config, load_config
from .data import PreferenceBatch, create_dataloader, load_preference_dataset
from .loss import (
    DPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
    compute_logprobs,
    get_loss_function,
)
from .train import main, main_cli

__all__ = [
    # Config
    "Config",
    "load_config",
    # Data
    "PreferenceBatch",
    "create_dataloader",
    "load_preference_dataset",
    # Loss functions
    "DPOLoss",
    "IPOLoss",
    "KTOLoss",
    "ORPOLoss",
    "SimPOLoss",
    "compute_logprobs",
    "get_loss_function",
    # Training
    "main",
    "main_cli",
]
