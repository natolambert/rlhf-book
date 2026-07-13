#!/usr/bin/env python3
"""Preference-based Reward Model Training (Bradley-Terry)

Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

This script trains a standard preference-based reward model using the
Bradley-Terry model. Given pairs of (chosen, rejected) responses, the model
learns to assign higher rewards to chosen responses.

The loss function is: -log(sigmoid(r_chosen - r_rejected))

This is the standard approach used in InstructGPT, Llama 2, and most RLHF
pipelines. See Chapter 5 (Reward Models) of RLHF Book for theoretical background.

Usage:
    uv run python -m reward_models.train_preference_rm --config reward_models/configs/preference_rm.yaml
    uv run python -m reward_models.train_preference_rm --config reward_models/configs/preference_rm.yaml --samples 2000
"""

import argparse
import os
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from reward_models.base import (
    BaseRewardModel,
    create_optimizer,
    finish_wandb,
    init_wandb,
    load_tokenizer,
    log_metrics,
    pad_sequences,
)
from reward_models.config import Config, load_config


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_DATASET = "argilla/ultrafeedback-binarized-preferences-cleaned"
DEFAULT_SAMPLES = 5000
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 8
DEFAULT_MAX_LENGTH = 512
DEFAULT_EPOCHS = 2
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_VAL_RATIO = 0.1
DEFAULT_EVAL_INTERVAL = 25
DEFAULT_LR_SCHEDULER = "linear_decay"
DEFAULT_SEED = 123


# =============================================================================
# Data Preparation
# =============================================================================


def format_conversation(messages: List[Dict]) -> str:
    """Format a conversation as a simple string."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        result.append(f"{role}: {content}")
    return "\n".join(result)


def tokenize_messages(
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    max_length: int,
    return_tensors: str | None = None,
) -> Dict:
    """Tokenize messages with the tokenizer chat template when available."""
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_dict=True,
            return_tensors=return_tensors,
        )

    return tokenizer(
        format_conversation(messages),
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors=return_tensors,
    )


def build_preference_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str = DEFAULT_DATASET,
    dataset_split: str = "train",
    limit: int = DEFAULT_SAMPLES,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Build preference dataset from UltraFeedback.

    Each example contains:
    - chosen_ids: Token IDs for the chosen response
    - rejected_ids: Token IDs for the rejected response
    """
    random.seed(seed)

    if os.path.exists(dataset_name):
        raw = load_from_disk(dataset_name)
        if hasattr(raw, "keys"):
            raw = raw[dataset_split]
    else:
        raw = load_dataset(dataset_name, split=dataset_split)

    # Shuffle and limit
    raw = raw.shuffle(seed=seed).select(range(min(limit, len(raw))))

    records = []
    for ex in raw:
        # Extract prompt and responses
        prompt = ex.get("prompt", "")
        chosen = ex.get("chosen", [])
        rejected = ex.get("rejected", [])

        # Handle different dataset formats
        if isinstance(chosen, list) and len(chosen) > 0:
            # Conversation format
            chosen_messages = chosen
            rejected_messages = rejected
        elif isinstance(chosen, str):
            chosen_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]
            rejected_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected},
            ]
        else:
            continue

        chosen_tokens = tokenize_messages(tokenizer, chosen_messages, max_length)
        rejected_tokens = tokenize_messages(tokenizer, rejected_messages, max_length)

        records.append(
            {
                "chosen_ids": chosen_tokens["input_ids"],
                "chosen_mask": chosen_tokens["attention_mask"],
                "rejected_ids": rejected_tokens["input_ids"],
                "rejected_mask": rejected_tokens["attention_mask"],
            }
        )

    return Dataset.from_list(records)


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Collate function that pads chosen and rejected sequences."""
    return {
        "chosen_ids": pad_sequences([x["chosen_ids"] for x in batch], tokenizer.pad_token_id),
        "chosen_mask": pad_sequences([x["chosen_mask"] for x in batch], 0),
        "rejected_ids": pad_sequences([x["rejected_ids"] for x in batch], tokenizer.pad_token_id),
        "rejected_mask": pad_sequences([x["rejected_mask"] for x in batch], 0),
    }


# =============================================================================
# Model Definition
# =============================================================================


class PreferenceRewardModel(BaseRewardModel):
    """Preference-based Reward Model with full fine-tuning.

    Architecture:
    - Base LLM (e.g., Qwen3) loaded in bfloat16
    - Linear head mapping last hidden state to scalar reward

    The model outputs a single scalar reward for each sequence.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs):
        super().__init__(model_id, head_dim=1, **kwargs)

    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute scalar reward for a sequence.

        Returns the reward from the last non-padding token position.
        """
        hidden = self.get_hidden_states(input_ids, attention_mask)

        # Get last non-padding position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, seq_lengths]

        reward = self.head(last_hidden).squeeze(-1)
        return reward

    def forward(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Bradley-Terry preference loss.

        Returns:
            loss: -log(sigmoid(r_chosen - r_rejected))
            r_chosen: Rewards for chosen responses
            r_rejected: Rewards for rejected responses
        """
        r_chosen = self.get_reward(chosen_ids, chosen_mask)
        r_rejected = self.get_reward(rejected_ids, rejected_mask)

        # Bradley-Terry loss
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        return loss, r_chosen, r_rejected


# =============================================================================
# Evaluation helpers
# =============================================================================


@torch.no_grad()
def evaluate_preference_rm(
    model: PreferenceRewardModel,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
) -> dict[str, float]:
    """Evaluate Bradley-Terry loss and pairwise ranking metrics."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    total_r_chosen = 0.0
    total_r_rejected = 0.0
    total_margin = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss, r_chosen, r_rejected = model(**batch)

        batch_size = r_chosen.size(0)
        margin = r_chosen - r_rejected

        total_loss += loss.item() * batch_size
        total_correct += (margin > 0).sum().item()
        total_pairs += batch_size
        total_r_chosen += r_chosen.sum().item()
        total_r_rejected += r_rejected.sum().item()
        total_margin += margin.sum().item()

    n = max(1, total_pairs)
    return {
        "val/loss": total_loss / n,
        "val/accuracy": total_correct / n,
        "val/r_chosen_mean": total_r_chosen / n,
        "val/r_rejected_mean": total_r_rejected / n,
        "val/reward_margin": total_margin / n,
    }


# =============================================================================
# Training
# =============================================================================


def train_preference_rm(
    model_id: str = DEFAULT_MODEL_ID,
    dataset_name: str = DEFAULT_DATASET,
    dataset_split: str = "train",
    samples: int = DEFAULT_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = True,
) -> PreferenceRewardModel:
    """Train a preference-based reward model on UltraFeedback.

    Args:
        model_id: HuggingFace model ID for base model
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        samples: Number of preference pairs to use
        batch_size: Training batch size
        grad_accum_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        epochs: Number of training epochs
        lr: Learning rate
        warmup_ratio: Fraction of total steps for linear LR warmup
        val_ratio: Fraction of examples held out for validation
        eval_interval: Run validation every N optimizer steps. Set <= 0 to disable mid-epoch eval.
        lr_scheduler: LR scheduler type. Use "linear_decay" for warmup + linear decay,
            or "warmup_only" to keep the previous behavior.
        seed: Random seed
        use_wandb: Whether to log to wandb

    Returns:
        Trained PreferenceRewardModel
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    init_wandb(
        default_run_name="preference_rm",
        config={
            "model_id": model_id,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "samples": samples,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "max_length": max_length,
            "epochs": epochs,
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "val_ratio": val_ratio,
            "eval_interval": eval_interval,
            "lr_scheduler": lr_scheduler,
        },
        use_wandb=use_wandb,
    )

    # Load tokenizer
    tokenizer = load_tokenizer(model_id)

    # Build dataset
    print(f"Building preference dataset with {samples} pairs...")
    data = build_preference_dataset(
        tokenizer,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        limit=samples,
        max_length=max_length,
        seed=seed,
    )
    print(f"Dataset size: {len(data)} pairs")

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")

    if val_ratio > 0.0:
        splits = data.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
        train_data = splits["train"]
        val_data = splits["test"]
    else:
        train_data = data
        val_data = None

    print(f"Train size: {len(train_data)} pairs")
    if val_data is not None:
        print(f"Validation size: {len(val_data)} pairs")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_data) > batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    val_loader = (
        DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
        )
        if val_data is not None
        else None
    )

    # Initialize model
    print(f"Loading model: {model_id}")
    model = PreferenceRewardModel(model_id=model_id).to(device)
    print(f"Trainable parameters: {model.count_trainable_params() / 1e6:.2f}M")

    # Optimizer and LR scheduler
    optimizer = create_optimizer(model, lr)
    total_optimizer_steps = -(-len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_optimizer_steps * warmup_ratio)

    if lr_scheduler == "linear_decay":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
    elif lr_scheduler == "warmup_only":
        scheduler = (
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            if warmup_steps > 0
            else None
        )
    else:
        raise ValueError(
            f'Unsupported lr_scheduler={lr_scheduler!r}. Expected "linear_decay" or "warmup_only".'
        )

    print(
        f"LR scheduler: {lr_scheduler} | "
        f"total optimizer steps: {total_optimizer_steps} | "
        f"warmup steps: {warmup_steps}"
    )

    # Mixed precision
    autocast_enabled = torch.cuda.is_available()

    # Training loop
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_pairs = 0
        optimizer.zero_grad()

        # Accumulators for logging per optimizer step
        accum_loss = 0.0
        accum_correct = 0
        accum_pairs = 0
        accum_r_chosen = 0.0
        accum_r_rejected = 0.0
        accum_microbatches = 0

        for step_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, r_chosen, r_rejected = model(**batch)

            # Use a fixed grad_accum_steps divisor even for the final partial accumulation
            # window. This can slightly under-weight the final update when len(train_loader)
            # is not divisible by grad_accum_steps, but keeps the teaching example simple
            # and follows the common constant-divisor convention.

            (loss / grad_accum_steps).backward()

            # Accumulate metrics over the grad_accum window
            accum_loss += loss.item()
            correct = (r_chosen > r_rejected).sum().item()
            accum_correct += correct
            accum_pairs += r_chosen.size(0)
            accum_r_chosen += r_chosen.mean().item()
            accum_r_rejected += r_rejected.mean().item()
            accum_microbatches += 1

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_pairs += r_chosen.size(0)

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log averaged metrics over the full effective batch
                n = max(1, accum_pairs)
                mb = accum_microbatches
                avg_loss = accum_loss / mb
                acc = accum_correct / n
                print(f"Epoch {epoch} step {global_step} | loss {avg_loss:.4f} | acc {acc:.3f}")
                log_metrics(
                    {
                        "train/loss": avg_loss,
                        "train/accuracy": acc,
                        "train/r_chosen_mean": accum_r_chosen / mb,
                        "train/r_rejected_mean": accum_r_rejected / mb,
                        "train/reward_margin": (accum_r_chosen - accum_r_rejected) / mb,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

                # Run validation every N optimizer steps.
                # evaluate_preference_rm() switches model to eval mode, so switch back to train after.
                if (
                    val_loader is not None
                    and eval_interval > 0
                    and global_step % eval_interval == 0
                ):
                    val_metrics = evaluate_preference_rm(
                        model,
                        val_loader,
                        device,
                        autocast_enabled,
                    )
                    print(
                        f"Eval step {global_step} | "
                        f"Val Loss: {val_metrics['val/loss']:.4f} | "
                        f"Val Accuracy: {val_metrics['val/accuracy']:.3f} | "
                        f"Val Margin: {val_metrics['val/reward_margin']:.4f}"
                    )
                    log_metrics(val_metrics, step=global_step)
                    model.train()

                # Reset accumulators
                accum_loss = 0.0
                accum_correct = 0
                accum_pairs = 0
                accum_r_chosen = 0.0
                accum_r_rejected = 0.0
                accum_microbatches = 0

        avg_loss = epoch_loss / len(train_loader)
        accuracy = epoch_correct / max(1, epoch_pairs)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}")

        # Also run validation at epoch end, unless we already evaluated on this exact step.
        should_run_epoch_eval = val_loader is not None and (
            eval_interval <= 0 or global_step % eval_interval != 0
        )

        if should_run_epoch_eval:
            val_metrics = evaluate_preference_rm(model, val_loader, device, autocast_enabled)
            print(
                f"Epoch {epoch} | Val Loss: {val_metrics['val/loss']:.4f} | "
                f"Val Accuracy: {val_metrics['val/accuracy']:.3f} | "
                f"Val Margin: {val_metrics['val/reward_margin']:.4f}"
            )
            log_metrics({**val_metrics, "epoch": epoch}, step=global_step)
            model.train()

    finish_wandb()
    return model


# =============================================================================
# Evaluation
# =============================================================================


def demo_scoring(model: PreferenceRewardModel, tokenizer: AutoTokenizer):
    """Demo: Score some example responses."""
    device = next(model.parameters()).device
    model.eval()

    # Example prompt and responses
    prompt = "Explain quantum computing in simple terms."
    good_response = """Quantum computing uses quantum bits (qubits) instead of regular bits.
While regular bits are either 0 or 1, qubits can be both at once (superposition).
This lets quantum computers try many solutions simultaneously, making them faster
for certain problems like breaking codes or simulating molecules."""

    bad_response = """Quantum computing is complicated. It uses physics.
Computers are electronic devices. I don't really know much about it."""

    good_tokens = tokenize_messages(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": good_response},
        ],
        max_length=512,
        return_tensors="pt",
    )
    bad_tokens = tokenize_messages(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": bad_response},
        ],
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        good_reward = model.get_reward(
            good_tokens["input_ids"].to(device),
            good_tokens["attention_mask"].to(device),
        )
        bad_reward = model.get_reward(
            bad_tokens["input_ids"].to(device),
            bad_tokens["attention_mask"].to(device),
        )

    print("=" * 60)
    print("Prompt:", prompt)
    print("=" * 60)
    print(f"\nGood response reward: {good_reward.item():.4f}")
    print(f"Bad response reward: {bad_reward.item():.4f}")
    print(f"\nModel correctly prefers good response: {good_reward.item() > bad_reward.item()}")


# =============================================================================
# Main
# =============================================================================


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply non-None CLI arguments to a loaded config."""
    override_map = {
        "model_id": "model_id",
        "dataset_name": "dataset_name",
        "dataset_split": "dataset_split",
        "samples": "samples",
        "batch_size": "batch_size",
        "grad_accum": "grad_accum_steps",
        "max_length": "max_length",
        "epochs": "epochs",
        "lr": "lr",
        "warmup_ratio": "warmup_ratio",
        "val_ratio": "val_ratio",
        "eval_interval": "eval_interval",
        "lr_scheduler": "lr_scheduler",
        "seed": "seed",
    }
    for arg_name, cfg_name in override_map.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)

    if args.skip_demo:
        cfg.skip_demo = True
    if args.no_wandb:
        cfg.use_wandb = False

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Train preference-based reward model on UltraFeedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-id", type=str, default=None, help="Base model ID")
    parser.add_argument("--dataset-name", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--dataset-split", type=str, default=None, help="Dataset split")
    parser.add_argument("--samples", type=int, default=None, help="Number of preference pairs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Fraction of steps for LR warmup",
    )
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Run validation every N optimizer steps. Set <= 0 to disable mid-epoch eval.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["linear_decay", "warmup_only"],
        default=None,
        help="LR scheduler type: linear_decay uses warmup + linear decay; warmup_only preserves the previous behavior.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--skip-demo", action="store_true", help="Skip scoring demo after training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    cfg = apply_overrides(cfg, args)

    model = train_preference_rm(
        model_id=cfg.model_id,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        samples=cfg.samples,
        batch_size=cfg.batch_size,
        grad_accum_steps=cfg.grad_accum_steps,
        max_length=cfg.max_length,
        epochs=cfg.epochs,
        lr=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        val_ratio=cfg.val_ratio,
        eval_interval=cfg.eval_interval,
        lr_scheduler=cfg.lr_scheduler,
        seed=cfg.seed,
        use_wandb=cfg.use_wandb,
    )

    if not cfg.skip_demo:
        tokenizer = load_tokenizer(cfg.model_id)
        demo_scoring(model, tokenizer)


if __name__ == "__main__":
    main()
