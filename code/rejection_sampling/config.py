# Configuration for rejection sampling. Loaded from YAML via `load_config()`.

from typing import Literal

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str = "openai/gsm8k"
    subset: str = "main"
    train_split: str = "train"
    test_split: str = "test"
    max_train_samples: int | None = 1000  # None = full split
    max_test_samples: int | None = 200


class SelectionConfig(BaseModel):
    strategy: Literal[
        "top_per_prompt",
        "top_k_overall",
        "random_per_prompt",
        "random_k_overall",
    ]
    top_k: int = 500  # only used by the *_k_overall strategies


class Config(BaseModel):
    data: DatasetConfig

    model_name: str = "Qwen/Qwen3-1.7B"
    reward_model_name: str = "nvidia/AceMath-7B-RM"

    # Stage 1 — generation
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 512
    num_completions_per_prompt: int = 8
    rollout_batch_size: int = 8

    # Stage 2 — scoring
    score_batch_size: int = 2

    # Stage 3a — selection
    selection: SelectionConfig

    # Stage 3b — SFT
    lr: float = 5e-6
    num_epochs: int = 2
    train_batch_size: int = 1
    batch_acc: int = 8
    max_norm: float = 1.0
    max_seq_length: int = 1024

    # Stage 3c — eval
    eval_temperature: float = 0.0
    eval_max_new_tokens: int = 512
    eval_batch_size: int = 8

    seed: int = 42
    model_device_id: int = 0
    reward_model_device_id: int = 0
    output_dir: str = "./rejection_sampling/output"
    save_checkpoint: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        return Config(**yaml.safe_load(f))
