"""Configuration helpers for reward model training."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    """Configuration for preference reward model training."""

    # Model
    model_id: str = "Qwen/Qwen3-0.6B-Base"

    # Dataset
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    dataset_split: str = "train"
    samples: int = 5000
    max_length: int = 512

    # Training
    batch_size: int = 2
    grad_accum_steps: int = 8
    epochs: int = 2
    lr: float = 5e-5
    warmup_ratio: float = 0.1
    val_ratio: float = 0.1
    eval_interval: int = 25
    lr_scheduler: str = "linear_decay"
    seed: int = 123

    # Logging / demo
    use_wandb: bool = True
    skip_demo: bool = False


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def save_config(config: Config, config_path: str | Path) -> None:
    """Save configuration to a YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False)
