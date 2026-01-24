# Configuration for Policy Gradient Training
#
# Original implementation by Zarif Stojano (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0

from typing import Any

import yaml
from pydantic import BaseModel, model_validator


class DatasetSpec(BaseModel):
    """Specification for a single dataset in the training mixture."""

    name: str
    weight: int = 1
    config: dict[str, Any] = {}


class DataConfig(BaseModel):
    """Configuration for the training data."""

    specs: list[DatasetSpec]
    size: int = 3000


class Config(BaseModel):
    """Full training configuration.

    Attributes:
        data: Dataset configuration
        loss: Loss function to use (reinforce, rloo, ppo, grpo, drgrpo, gspo, cispo)
        model_name: HuggingFace model identifier
        clip_eps_lo/hi: Clipping bounds for policy ratio
        clip_eps_val: Clipping bound for value function (PPO)
        beta: KL penalty coefficient (0 = no KL penalty)
        lr: Learning rate
        gamma: Discount factor (PPO)
        lam: GAE lambda (PPO)
        vf_coef: Value function loss coefficient (PPO)
        temperature: Sampling temperature
        top_p/top_k/min_p: Sampling parameters
        max_new_tokens: Maximum tokens to generate
        prompts_per_step: Number of unique prompts per training step
        num_rollouts: Number of rollouts per prompt
        rollout_batch_size: Batch size during generation
        train_batch_size: Batch size during training
        batch_acc: Gradient accumulation steps
        max_norm: Gradient clipping norm
        seed: Random seed
        model_device_id: GPU for policy model
        ref_model_device_id: GPU for reference model
        val_model_device_id: GPU for value model (PPO)
        wandb_project/wandb_run_name: Weights & Biases logging
    """

    data: DataConfig
    loss: str
    model_name: str = "Qwen/Qwen3-1.7B"
    clip_eps_lo: float = 0.2
    clip_eps_hi: float = 0.2
    clip_eps_val: float = 0.2
    beta: float = 0.0
    lr: float = 5e-6
    gamma: float = 0.99
    lam: float = 0.95
    vf_coef: float = 0.1
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 512
    prompts_per_step: int = 4
    num_rollouts: int = 8
    rollout_batch_size: int = 8
    train_batch_size: int = 2
    batch_acc: int = 4
    max_norm: float = 1.0
    seed: int = 42
    model_device_id: int = 0
    ref_model_device_id: int = 1
    val_model_device_id: int = 2
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    @model_validator(mode="after")
    def validate_rollout_batch_size(self) -> "Config":
        if self.num_rollouts > 1 and self.rollout_batch_size != self.num_rollouts:
            raise ValueError("When num_rollouts > 1, rollout_batch_size must equal num_rollouts.")
        if (self.prompts_per_step * self.num_rollouts) % self.rollout_batch_size != 0:
            raise ValueError("prompts_per_step * num_rollouts must be divisible by rollout_batch_size.")
        return self


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
