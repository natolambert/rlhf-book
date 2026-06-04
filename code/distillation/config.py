import yaml
from pydantic import BaseModel


class Config(BaseModel):
    # Data
    split: str = "train"
    max_tests: int | None = None

    # Model
    model_name: str = "Qwen/Qwen3-1.7B"

    # Loss
    loss: str = "sdpo"

    # SDPO
    kl_top_k: int = 100  # logits kept per position for the distillation KL
    success_reward_threshold: float = 1.0  # reward at/above which a rollout is a demo

    # Generation
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024
    max_prompt_len: int = 2048  # left-truncation cap for student and teacher prompts
    enable_thinking: bool = True

    # Training
    lr: float = 1e-6
    warmup_ratio: float = 0.05  # fraction of num_steps for linear LR warmup (then held constant)
    prompts_per_step: int = 4  # prompts generated and accumulated per optimizer step
    num_rollouts: int = 8  # rollouts per prompt
    num_steps: int = 200
    max_norm: float = 1.0
    seed: int = 42
    model_device_id: int = 0

    # Logging
    wandb_entity: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
