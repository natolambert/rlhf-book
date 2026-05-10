# Configuration for Instruction Tuning (SFT).

from typing import Literal

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Full SFT training configuration.

    Attributes:
        model_name: Base HuggingFace model identifier (we want a base model
            without an instruction-tuned chat template, e.g. OLMo-2-0425-1B).
        chat_template_source: Tokenizer to lift ``chat_template`` from when the
            base tokenizer has none. Set to ``null`` to disable.
        dataset_name, dataset_split: HuggingFace dataset identifier and split.
        max_samples: Optional cap on training rows for quick experiments.
        max_length: Maximum total sequence length (prompt + response).
        masking: ``final_assistant`` trains on the last assistant turn only;
            ``user_only`` trains on every assistant turn (chapter 4).

        lr, num_epochs, batch_size, gradient_accumulation_steps, warmup_ratio,
        weight_decay, max_grad_norm: standard AdamW SFT knobs.

        bf16, gradient_checkpointing, model_device_id: hardware/memory.

        sample_*: in-loop generation logging settings. ``sample_every`` fires
            at step 0 (base model) and every ``sample_every`` optimizer steps
            after, so the W&B run shows the base model alongside the SFT model.

        wandb_project, wandb_run_name: W&B logging.
    """

    # Model
    model_name: str = "allenai/OLMo-2-0425-1B"
    chat_template_source: str | None = "allenai/OLMo-2-0425-1B-SFT"

    # Data
    dataset_name: str = "HuggingFaceH4/no_robots"
    dataset_split: str = "train"
    max_samples: int | None = None
    max_length: int = 2048
    masking: Literal["final_assistant", "user_only"] = "final_assistant"

    # Training
    lr: float = 5.0e-6
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    model_device_id: int = 0

    # In-loop generation
    sample_every: int = 50
    sample_max_tokens: int = 128
    sample_max_input_tokens: int = 512
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9
    sample_do_sample: bool = True

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
