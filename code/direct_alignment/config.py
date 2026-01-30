# Direct Alignment Configuration
#
# Configuration dataclasses for DPO and related algorithms.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Config:
    """Configuration for direct alignment training."""

    # Model settings
    model_name: str = "allenai/OLMo-2-0425-1B-SFT"
    ref_model_name: str | None = None  # Defaults to model_name if None

    # Training settings
    loss: Literal["dpo", "cdpo", "ipo", "simpo", "orpo", "kto"] = "dpo"
    beta: float = 0.1  # KL penalty / temperature
    gamma: float = 0.5  # SimPO target margin
    label_smoothing: float = 0.0  # For cDPO (overridden if loss=cdpo)

    # Dataset settings
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned"
    dataset_split: str = "train"
    max_samples: int | None = 1000  # Limit samples for quick experiments
    max_length: int = 512  # Max sequence length

    # Training hyperparameters
    learning_rate: float = 5e-7  # DPO typically uses very low LR
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Hardware settings
    device: str = "cuda"
    gradient_checkpointing: bool = True
    bf16: bool = True

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    log_every: int = 10
    eval_every: int = 100

    # Output
    output_dir: str = "./outputs"
    save_model: bool = False

    # Misc
    seed: int = 42

    def __post_init__(self):
        """Set defaults based on loss type."""
        if self.ref_model_name is None:
            self.ref_model_name = self.model_name

        # Loss-specific defaults
        if self.loss == "cdpo":
            self.label_smoothing = 0.1
        elif self.loss == "simpo":
            # SimPO typically uses higher beta
            if self.beta == 0.1:  # Only override if default
                self.beta = 2.0


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def save_config(config: Config, config_path: str | Path) -> None:
    """Save configuration to YAML file."""
    import dataclasses
    with open(config_path, "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False)
