"""Base classes and utilities for reward model training.

This module provides common functionality shared across ORM, PRM, and Preference RM:
- BaseRewardModel: Full fine-tuning setup (frozen backbone, trainable head)
- Training utilities: wandb init, optimizer, training loop helpers
- Data utilities: collate functions

Note: We use full fine-tuning for simplicity with small models (0.6B-1.7B).
For larger models, consider using LoRA/QLoRA (see commented code below).
"""

import os
from typing import Callable, Iterator

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Commented out LoRA imports - kept for reference if needed for larger models
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import BitsAndBytesConfig


# =============================================================================
# Model Base Class
# =============================================================================


class BaseRewardModel(nn.Module):
    """Base class for reward models with full fine-tuning.

    Uses a frozen or partially-frozen backbone with a trainable reward head.
    For small models (0.6B-1.7B), we fine-tune all parameters.

    Subclasses should implement:
    - forward(): Define the forward pass and loss computation
    - Optionally override _build_head() for custom reward heads
    """

    def __init__(
        self,
        model_id: str,
        head_dim: int = 1,
        freeze_backbone: bool = False,
        # LoRA params kept for potential future use
        # lora_r: int = 16,
        # lora_alpha: int = 32,
        # lora_dropout: float = 0.05,
    ):
        super().__init__()

        # BF16 loading - simple for small models
        device_map = {"": 0} if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="bfloat16",  # Use string to avoid deprecation warning
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False

        # Optionally freeze the backbone (for head-only training)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Build head with same dtype as model
        self.head = self._build_head(self.model.config.hidden_size, head_dim)
        self.head = self.head.to(torch.bfloat16)

        # --- Commented out LoRA setup for reference ---
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        # lora_config = LoraConfig(
        #     r=lora_r,
        #     lora_alpha=lora_alpha,
        #     lora_dropout=lora_dropout,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
        #                     "gate_proj", "up_proj", "down_proj"],
        # )
        # base = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     quantization_config=bnb_config,
        #     device_map=device_map,
        #     trust_remote_code=True,
        # )
        # base = prepare_model_for_kbit_training(base)
        # self.model = get_peft_model(base, lora_config)

    def _build_head(self, hidden_size: int, output_dim: int) -> nn.Module:
        """Build the reward head. Override for custom architectures."""
        return nn.Linear(hidden_size, output_dim, bias=output_dim > 1)

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get the last hidden states from the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Tokenizer Utilities
# =============================================================================


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with proper padding setup."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =============================================================================
# Wandb Utilities
# =============================================================================


def init_wandb(
    default_run_name: str,
    config: dict,
    use_wandb: bool = True,
) -> bool:
    """Initialize wandb with environment variable support.

    Args:
        default_run_name: Default name if WANDB_RUN_NAME not set
        config: Training config to log
        use_wandb: Whether to enable wandb

    Returns:
        True if wandb is enabled, False otherwise
    """
    wandb_project = os.environ.get("WANDB_PROJECT")

    if use_wandb and wandb_project:
        wandb.init(
            project=wandb_project,
            name=os.environ.get("WANDB_RUN_NAME", default_run_name),
            config=config,
        )
        return True
    else:
        wandb.init(mode="disabled")
        return False


def log_metrics(metrics: dict, step: int | None = None):
    """Log metrics to wandb."""
    wandb.log(metrics, step=step)


def finish_wandb():
    """Finish wandb run."""
    wandb.finish()


# =============================================================================
# Training Utilities
# =============================================================================


def create_optimizer(
    model: nn.Module,
    lr: float,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer for trainable parameters."""
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )


def training_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    compute_loss_and_metrics: Callable,
    epochs: int = 1,
    grad_accum_steps: int = 1,
    log_interval: int = 10,
    use_amp: bool = True,
) -> None:
    """Generic training loop for reward models.

    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        compute_loss_and_metrics: Function(model, batch) -> (loss, metrics_dict)
        epochs: Number of epochs
        grad_accum_steps: Gradient accumulation steps
        log_interval: How often to log (in steps)
        use_amp: Whether to use automatic mixed precision
    """
    device = next(model.parameters()).device
    autocast_enabled = use_amp and torch.cuda.is_available()
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        optimizer.zero_grad()

        for step_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, metrics = compute_loss_and_metrics(model, batch)

            (loss / grad_accum_steps).backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v

            if step_idx % log_interval == 0:
                print(f"Epoch {epoch} step {step_idx} | loss {loss.item():.4f}")
                log_metrics({"loss": loss.item(), **metrics})

        # Log epoch summary
        avg_loss = epoch_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in epoch_metrics.items()}
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | {avg_metrics}")
        log_metrics({"epoch_loss": avg_loss, "epoch": epoch, **{f"epoch_{k}": v for k, v in avg_metrics.items()}})


# =============================================================================
# Data Utilities
# =============================================================================


def pad_sequences(
    sequences: list[list[int]],
    pad_value: int,
    return_tensors: bool = True,
) -> torch.Tensor | list[list[int]]:
    """Pad sequences to the same length."""
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded.append(seq + [pad_value] * (max_len - len(seq)))

    if return_tensors:
        return torch.tensor(padded, dtype=torch.long)
    return padded


def create_collate_fn(tokenizer: AutoTokenizer, fields: list[str]):
    """Create a collate function for the given fields.

    Args:
        tokenizer: Tokenizer (for pad_token_id)
        fields: List of field names to collate. Fields ending in '_ids' use
                pad_token_id, fields ending in '_mask' use 0, others use -100.
    """
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        result = {}
        for field in fields:
            sequences = [item[field] for item in batch]

            # Determine pad value based on field name
            if field.endswith("_ids"):
                pad_value = tokenizer.pad_token_id
            elif field.endswith("_mask"):
                pad_value = 0
            else:  # labels
                pad_value = -100

            result[field] = pad_sequences(sequences, pad_value)

        return result

    return collate_fn
