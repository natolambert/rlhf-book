# Configuration for Rejection Sampling
#
# Mirrors the structure of policy_gradients/config.py: a single Pydantic
# `Config` loaded from YAML via `load_config()`.

from typing import Literal

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """GSM8k dataset slice used for both rollouts and evaluation."""

    name: str = "openai/gsm8k"
    subset: str = "main"
    train_split: str = "train"
    test_split: str = "test"
    # None means "use the full split".
    max_train_samples: int | None = 1000
    max_test_samples: int | None = 200


class SelectionConfig(BaseModel):
    """Which rejection-sampling selection rule to apply in Stage 3."""

    strategy: Literal[
        "top_per_prompt",
        "top_k_overall",
        "random_per_prompt",
        "random_k_overall",
    ]
    # Only consulted when strategy ∈ {"top_k_overall", "random_k_overall"}.
    top_k: int = 500


class Config(BaseModel):
    """Full rejection-sampling configuration.

    The three selection-strategy configs are expected to share identical
    generation/scoring parameters so that they all hit the same rollout cache.
    """

    # Data
    data: DatasetConfig

    # Models
    model_name: str = "Qwen/Qwen3-1.7B"
    reward_model_name: str = "nvidia/AceMath-7B-RM"

    # Generation (Stage 1)
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 512
    num_completions_per_prompt: int = 8
    rollout_batch_size: int = 8

    # Scoring (Stage 2)
    score_batch_size: int = 2

    # Selection (Stage 3a)
    selection: SelectionConfig

    # SFT (Stage 3b)
    lr: float = 5e-6
    num_epochs: int = 2
    train_batch_size: int = 1
    batch_acc: int = 8
    max_norm: float = 1.0
    max_seq_length: int = 1024

    # Eval (Stage 3c)
    eval_temperature: float = 0.0
    eval_max_new_tokens: int = 512
    eval_batch_size: int = 8

    # Misc
    seed: int = 42
    model_device_id: int = 0
    reward_model_device_id: int = 0
    output_dir: str = "./rejection_sampling/output"
    save_checkpoint: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def load_config(config_path: str) -> Config:
    """Load a rejection-sampling configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
