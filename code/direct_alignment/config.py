# Direct Alignment Configuration
#
# Configuration dataclasses for DPO and related algorithms.

from dataclasses import dataclass
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

    # Sequence length settings (TRL-style controls)
    max_length: int = 512  # Max total sequence length (prompt + completion)
    max_prompt_length: int | None = None  # Max prompt length (truncated from left if exceeded)
    max_completion_length: int | None = None  # Max completion length (truncated from right if exceeded)
    truncation_mode: Literal["keep_end", "keep_start"] = "keep_end"  # How to truncate: keep_end preserves response

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
    sample_every: int = 25  # Generate sample outputs every N steps (0 to disable)
    sample_max_tokens: int = 128  # Max tokens for sample generation
    sample_max_input_tokens: int = 512  # Prompt token limit for sample generation
    sample_num_prompts: int = 3  # Number of prompts per sample logging event
    sample_prompt_strategy: Literal["fixed", "round_robin", "random"] = "round_robin"
    sample_prompts_file: str | None = None  # Optional .txt/.json prompt list
    sample_do_sample: bool = True
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9

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

        if self.sample_num_prompts < 1:
            raise ValueError("sample_num_prompts must be >= 1")
        if not 0.0 < self.sample_top_p <= 1.0:
            raise ValueError("sample_top_p must be in (0, 1]")
        if self.sample_temperature <= 0.0:
            raise ValueError("sample_temperature must be > 0")


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
