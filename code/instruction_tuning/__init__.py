# Instruction Tuning (Supervised Fine-Tuning) for RLHF Book.
#
# Educational single-GPU SFT example. See Chapter 4: Instruction Fine-Tuning
# (https://rlhfbook.com) for the math and chat-template background.
#
# Usage:
#   uv run python -m instruction_tuning.train \
#       --config instruction_tuning/configs/sft_olmo2_1b.yaml

from .config import Config, load_config
from .train import main, main_cli


__all__ = [
    "Config",
    "load_config",
    "main",
    "main_cli",
]
