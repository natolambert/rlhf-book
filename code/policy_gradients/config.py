# Configuration for Policy Gradient Training
#
# Original implementation by Zafir Stojanovski (@zafstojano)
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
        loss: Loss function (reinforce, rloo, ppo, grpo, drgrpo, gspo, cispo)
        model_name: HuggingFace model identifier

        # Clipping (GRPO, DrGRPO, GSPO, CISPO, PPO)
        clip_eps_lo: Lower clipping bound for policy ratio
        clip_eps_hi: Upper clipping bound for policy ratio

        # PPO-specific
        clip_eps_val: Clipping bound for value function
        gamma: Discount factor for GAE
        lam: Lambda for GAE
        vf_coef: Value function loss coefficient
        val_model_device_id: GPU for value model

        # Optional KL penalty (REINFORCE, RLOO, GRPO, etc.)
        beta: KL penalty coefficient (0 = disabled)
        ref_model_device_id: GPU for reference model (when beta > 0)

        # Generation
        temperature, top_p, top_k, min_p: Sampling parameters
        max_new_tokens: Maximum tokens to generate

        # Training
        lr: Learning rate
        prompts_per_step: Prompts per training step
        num_rollouts: Rollouts per prompt (1 for REINFORCE/PPO, >1 for GRPO/RLOO)
        rollout_batch_size: Batch size during generation
        train_batch_size: Batch size during training
        batch_acc: Gradient accumulation steps
        max_norm: Gradient clipping norm
        seed: Random seed
        model_device_id: GPU for policy model

        # Logging
        wandb_project, wandb_run_name: Weights & Biases logging
    """

    data: DataConfig
    loss: str
    model_name: str = "Qwen/Qwen3-1.7B"

    # Clipping params (used by GRPO, DrGRPO, GSPO, CISPO, PPO)
    clip_eps_lo: float = 0.2
    clip_eps_hi: float = 0.2

    # PPO-specific params
    clip_eps_val: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    vf_coef: float = 0.1
    val_model_device_id: int = 0

    # KL penalty (optional, for REINFORCE/RLOO/GRPO when beta > 0)
    beta: float = 0.0
    ref_model_device_id: int = 0

    # Generation params
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 512

    # Training params
    lr: float = 5e-6
    prompts_per_step: int = 4
    num_rollouts: int = 8
    rollout_batch_size: int = 8
    train_batch_size: int = 2
    batch_acc: int = 4
    max_norm: float = 1.0
    seed: int = 42
    model_device_id: int = 0

    # Logging
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
