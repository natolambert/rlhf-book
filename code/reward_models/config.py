"""Configuration helpers for reward model training."""

from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class Config(BaseModel):
    """Configuration for preference reward model training."""

    model_config = ConfigDict(validate_assignment=True)

    # Model
    model_id: str = "Qwen/Qwen3-0.6B-Base"
    freeze_backbone: bool = False

    # Dataset
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    dataset_split: str = "train"
    samples: int = 5000
    max_length: int = 512

    # PRM-specific dataset limits
    max_steps: int = 20
    max_tokens: int = 5500

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

    # Hardware
    device: str = "cuda"
    device_id: int = 0

    # Logging / demo
    use_wandb: bool = True
    skip_demo: bool = False

    @model_validator(mode="after")
    def validate_config(self):
        if self.lr_scheduler not in ("linear_decay", "warmup_only"):
            raise ValueError(
                f'Unsupported lr_scheduler={self.lr_scheduler!r}. Expected "linear_decay" or "warmup_only".'
            )
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but device is set to 'cuda'")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available, but device is set to 'mps'")

        return self

    def get_device(self) -> str:
        """Return the device string for PyTorch."""
        if self.device == "cuda" and torch.cuda.is_available():
            return f"cuda:{self.device_id}"
        return self.device


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def save_config(config: Config, config_path: str | Path) -> None:
    """Save configuration to a YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
