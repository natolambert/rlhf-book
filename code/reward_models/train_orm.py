#!/usr/bin/env python3
"""Outcome Reward Model (ORM) Training

Original implementation by @myhott163com
Source: https://github.com/myhott163com/RLHF_ORM_PRM
License: MIT

Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

This script trains a minimal outcome reward model by fine-tuning a LLM
on verifier-labeled Qwen3-0.6B rollouts from GSM8K. The model learns to classify
solution correctness via per-token BCE loss on completion tokens.

See Chapter 5 (Reward Models) of RLHF Book for theoretical background.

Usage:
    uv run python -m reward_models.train_orm --config reward_models/configs/orm.yaml
"""

import argparse
import random

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from reward_models.base import (
    BaseRewardModel,
    create_collate_fn,
    create_optimizer,
    finish_wandb,
    init_wandb,
    load_tokenizer,
    log_metrics,
)
from reward_models.config import Config, load_config


ROLLOUT_SYSTEM_PROMPT = (
    "Solve the math problem step by step. End with exactly one line in the form "
    "#### <number>. Do not write anything after that line."
)


# =============================================================================
# Data Preparation
# =============================================================================


def tokenize_prompt(question: str, tokenizer: AutoTokenizer) -> list[int]:
    """Tokenize the chat context used to generate the stored rollouts."""
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ROLLOUT_SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
    )
    return encoded["input_ids"]


def pack_example(
    question: str, completion: str, label: int, tokenizer: AutoTokenizer
) -> dict[str, list[int]]:
    """Pack a (question, completion, label) into tokenized format.

    The label is applied to all completion tokens, with prompt tokens masked (-100).
    """
    prompt_ids = tokenize_prompt(question, tokenizer)
    completion_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = prompt_ids + completion_ids
    attention = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + [label] * len(completion_ids)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def preprocess_rollout_batch(
    batch: dict[str, list],
    indices: list[int],
    tokenizer: AutoTokenizer,
    max_length: int,
    rollouts_per_prompt: int,
) -> dict[str, list]:
    """Flatten, tokenize, and length-filter grouped rollout rows."""
    prefixes = []
    completions = []
    rewards = []
    prompt_ids = []
    for prompt_id, prompt, prompt_completions, prompt_rewards in zip(
        indices,
        batch["prompt"],
        batch["completions"],
        batch["rewards"],
        strict=True,
    ):
        sample_size = min(rollouts_per_prompt, len(prompt_completions))
        if sample_size < rollouts_per_prompt:
            print(
                f"Warning: prompt {prompt_id} has {sample_size} rollouts; "
                f"using all instead of {rollouts_per_prompt}"
            )
        prefix = tokenize_prompt(prompt, tokenizer)
        for completion, reward in zip(
            prompt_completions[:sample_size],
            prompt_rewards[:sample_size],
            strict=True,
        ):
            prefixes.append(prefix)
            completions.append(completion + tokenizer.eos_token)
            rewards.append(reward)
            prompt_ids.append(prompt_id)

    suffix_ids = tokenizer(completions, add_special_tokens=False)["input_ids"]

    output = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "prompt_id": [],
        "outcome": [],
    }
    for prompt_id, reward, prefix, suffix in zip(
        prompt_ids,
        rewards,
        prefixes,
        suffix_ids,
        strict=True,
    ):
        input_ids = prefix + suffix
        if len(input_ids) > max_length:
            continue
        label = int(reward)
        output["input_ids"].append(input_ids)
        output["attention_mask"].append([1] * len(input_ids))
        output["labels"].append([-100] * len(prefix) + [label] * len(suffix))
        output["prompt_id"].append(prompt_id)
        output["outcome"].append(bool(label))
    return output


def preprocess_rollouts(
    candidates: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    rollouts_per_prompt: int,
    split: str,
) -> Dataset:
    """Tokenize rollout candidates and report sequences above ``max_length``."""
    data = candidates.map(
        preprocess_rollout_batch,
        batched=True,
        batch_size=max(1, 512 // rollouts_per_prompt),
        with_indices=True,
        remove_columns=candidates.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "rollouts_per_prompt": rollouts_per_prompt,
        },
        load_from_cache_file=True,
        desc=f"Tokenizing ORM {split.lower()} rollouts",
    )

    scanned = sum(
        min(rollouts_per_prompt, len(completions)) for completions in candidates["completions"]
    )
    dropped = scanned - len(data)
    percentage = 100 * dropped / scanned if scanned else 0.0
    print(
        f"{split} length filter: dropped {dropped}/{scanned} scanned completions "
        f"({percentage:.1f}%) over max_length={max_length}"
    )

    if not data:
        raise ValueError(f"No ORM {split.lower()} completions fit max_length={max_length}")
    return data


def build_orm_dataset(
    tokenizer: AutoTokenizer,
    config: Config,
) -> tuple[Dataset, Dataset | None]:
    """Build an ORM dataset from stored verifier-labeled rollout rows."""
    raw = load_dataset(config.dataset_name, split=config.dataset_split)
    raw = raw.shuffle(seed=config.seed).select(range(min(config.samples, len(raw))))
    if config.val_ratio > 0:
        splits = raw.train_test_split(test_size=config.val_ratio, seed=config.seed)
        train_rows, val_rows = splits["train"], splits["test"]
    else:
        train_rows, val_rows = raw, None

    train_data = preprocess_rollouts(
        train_rows,
        tokenizer,
        config.max_length,
        config.rollouts_per_prompt,
        "Training",
    )
    train_data = train_data.remove_columns(["prompt_id", "outcome"])
    if val_rows is None:
        return train_data, None

    val_data = preprocess_rollouts(
        val_rows,
        tokenizer,
        config.max_length,
        config.rollouts_per_prompt,
        "Validation",
    )
    return train_data, val_data


# =============================================================================
# Model Definition
# =============================================================================


def last_token_values(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select each sequence's final non-padding value."""
    last_indices = attention_mask.sum(dim=1) - 1
    return values.gather(1, last_indices.unsqueeze(1)).squeeze(1)


class OutcomeRewardModel(BaseRewardModel):
    """Outcome Reward Model with full fine-tuning.

    Architecture:
    - LLM (e.g., Qwen3) with FP32 parameters and BF16 CUDA autocast
    - Linear head mapping hidden states to scalar reward

    The model outputs per-token logits which are trained with BCE loss
    on completion tokens only (prompt tokens are masked).
    """

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, head_dim=1, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Forward pass computing reward logits and optional loss.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Per-token labels (0/1 for completion, -100 for masked) [batch, seq_len]

        Returns:
            loss: BCE loss on completion tokens (None if labels not provided)
            logits: Per-token reward logits [batch, seq_len]
        """
        hidden = self.get_hidden_states(input_ids, attention_mask)
        logits = self.head(hidden).squeeze(-1)

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())
            else:
                loss = logits.sum() * 0

        return loss, logits


# =============================================================================
# Evaluation
# =============================================================================


def compute_eval_metrics(
    rewards: list[float], outcomes: list[bool], prompt_ids: list[int]
) -> dict[str, float]:
    """Compare ORM rewards with verifier outcomes and prompt-level ranking."""
    if not rewards or len(rewards) != len(outcomes) or len(rewards) != len(prompt_ids):
        raise ValueError("Rewards, outcomes, and prompt IDs must be nonempty and aligned")

    metrics = {}
    correct_rewards = [reward for reward, outcome in zip(rewards, outcomes, strict=True) if outcome]
    incorrect_rewards = [
        reward for reward, outcome in zip(rewards, outcomes, strict=True) if not outcome
    ]
    if correct_rewards:
        metrics["val/reward_correct_mean"] = sum(correct_rewards) / len(correct_rewards)
    if incorrect_rewards:
        metrics["val/reward_incorrect_mean"] = sum(incorrect_rewards) / len(incorrect_rewards)

    groups = {}
    for index, prompt_id in enumerate(prompt_ids):
        groups.setdefault(prompt_id, []).append(index)

    top1_scores = []
    pairwise_score = 0.0
    pairwise_count = 0
    for indices in groups.values():
        maximum = max(rewards[index] for index in indices)
        tied_top = [index for index in indices if rewards[index] == maximum]
        top1_scores.append(sum(outcomes[index] for index in tied_top) / len(tied_top))

        correct_indices = [index for index in indices if outcomes[index]]
        incorrect_indices = [index for index in indices if not outcomes[index]]
        if not correct_indices or not incorrect_indices:
            continue

        for correct_index in correct_indices:
            for incorrect_index in incorrect_indices:
                pairwise_score += (
                    1.0
                    if rewards[correct_index] > rewards[incorrect_index]
                    else 0.5
                    if rewards[correct_index] == rewards[incorrect_index]
                    else 0.0
                )
                pairwise_count += 1

    metrics["val/top1_accuracy"] = sum(top1_scores) / len(top1_scores)
    if pairwise_count:
        metrics["val/pairwise_accuracy"] = pairwise_score / pairwise_count
    return metrics


def val_collate_fn(
    batch: list[dict],
    tokenizer: AutoTokenizer,
) -> dict[str, torch.Tensor]:
    """Pad model inputs and preserve scalar validation metadata."""
    collated = create_collate_fn(tokenizer, ["input_ids", "attention_mask"])(batch)
    collated["prompt_id"] = torch.tensor([example["prompt_id"] for example in batch])
    collated["outcome"] = torch.tensor([example["outcome"] for example in batch])
    return collated


@torch.no_grad()
def evaluate_orm(
    model: OutcomeRewardModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Score validation rollouts and compare final-token rewards with outcomes."""
    model.eval()
    rewards = []
    outcomes = []
    prompt_ids = []

    for batch in loader:
        outcomes.extend(batch.pop("outcome").tolist())
        prompt_ids.extend(batch.pop("prompt_id").tolist())
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, logits = model(**batch)
        scores = torch.sigmoid(last_token_values(logits, batch["attention_mask"]))
        rewards.extend(scores.float().cpu().tolist())

    return compute_eval_metrics(rewards, outcomes, prompt_ids)


# =============================================================================
# Training
# =============================================================================


def train_orm(
    config: Config,
) -> OutcomeRewardModel:
    """Train an Outcome Reward Model on GSM8K rollouts.

    Args:
        config: Configuration object containing training parameters.

    Returns:
        Trained OutcomeRewardModel
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.get_device())

    # Initialize wandb
    init_wandb(
        default_run_name="orm_gsm8k",
        config=config.model_dump(),
        use_wandb=config.use_wandb,
    )

    # Load tokenizer
    tokenizer = load_tokenizer(config.model_id)
    print(
        f"Building ORM dataset from up to {config.samples} prompt rows "
        f"and up to {config.rollouts_per_prompt} rollouts per prompt..."
    )
    train_data, val_data = build_orm_dataset(tokenizer, config)

    print(f"Train size: {len(train_data)} examples")
    if val_data is not None:
        print(f"Validation size: {len(val_data)} examples")

    collate = create_collate_fn(tokenizer, ["input_ids", "attention_mask", "labels"])
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=len(train_data) > config.batch_size,
        collate_fn=collate,
    )
    val_loader = (
        DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda batch: val_collate_fn(batch, tokenizer),
        )
        if val_data is not None
        else None
    )

    # Initialize model
    print(f"Loading model: {config.model_id}")
    model = OutcomeRewardModel(
        model_id=config.model_id,
        freeze_backbone=config.freeze_backbone,
        device=device,
    ).to(device)
    print(f"Trainable parameters: {model.count_trainable_params() / 1e6:.2f}M")

    # Optimizer and LR scheduler with linear warmup
    optimizer = create_optimizer(model, config.lr)
    total_optimizer_steps = -(-len(loader) // config.grad_accum_steps) * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    if config.lr_scheduler == "linear_decay":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
    elif config.lr_scheduler == "warmup_only":
        scheduler = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_steps,
            )
            if warmup_steps > 0
            else None
        )

    autocast_enabled = device.type == "cuda"

    # Training loop
    global_step = 0
    grad_accum_steps = config.grad_accum_steps
    eval_interval = config.eval_interval
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0
        optimizer.zero_grad(set_to_none=True)

        # Accumulators for logging per optimizer step
        accum_loss = 0.0
        accum_correct = 0
        accum_examples = 0
        accum_microbatches = 0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, logits = model(**batch)

            (loss / grad_accum_steps).backward()

            # Accumulate metrics over the grad_accum window
            loss_value = loss.item()
            accum_loss += loss_value
            sequence_logits = last_token_values(logits.detach(), batch["attention_mask"])
            sequence_labels = last_token_values(batch["labels"], batch["attention_mask"])

            preds = (torch.sigmoid(sequence_logits) > 0.5).long()
            correct = (preds == sequence_labels).sum().item()

            examples = sequence_labels.numel()
            accum_correct += correct
            accum_examples += examples
            accum_microbatches += 1

            epoch_loss += loss_value
            epoch_correct += correct
            epoch_examples += examples

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Log averaged metrics over the full effective batch
                avg_loss = accum_loss / accum_microbatches
                acc = accum_correct / max(1, accum_examples)
                print(f"Epoch {epoch} step {global_step} | loss {avg_loss:.4f} | acc {acc:.3f}")
                log_metrics(
                    {
                        "train/loss": avg_loss,
                        "train/accuracy": acc,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

                # Run validation every N optimizer steps.
                # evaluate_orm() switches model to eval mode, so switch back to train after.
                if (
                    val_loader is not None
                    and eval_interval > 0
                    and global_step % eval_interval == 0
                ):
                    metrics = evaluate_orm(model, val_loader, device)
                    summary = " | ".join(
                        f"{name.removeprefix('val/')}: {value:.3f}"
                        for name, value in metrics.items()
                    )
                    print(f"Eval step {global_step} | {summary}")
                    log_metrics(metrics, step=global_step)
                    model.train()

                # Reset accumulators
                accum_loss = 0.0
                accum_correct = 0
                accum_examples = 0
                accum_microbatches = 0

        avg_loss = epoch_loss / len(loader)
        accuracy = epoch_correct / max(1, epoch_examples)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}")
        log_metrics(
            {"epoch_loss": avg_loss, "epoch_accuracy": accuracy, "epoch": epoch},
            step=global_step,
        )

        # Also run validation at epoch end, unless we already evaluated on this exact step.
        should_run_epoch_eval = val_loader is not None and (
            eval_interval <= 0 or global_step % eval_interval != 0
        )

        if should_run_epoch_eval:
            metrics = evaluate_orm(model, val_loader, device)
            summary = " | ".join(
                f"{name.removeprefix('val/')}: {value:.3f}" for name, value in metrics.items()
            )
            print(f"Eval step {global_step} | {summary}")
            log_metrics(metrics, step=global_step)
            model.train()

    finish_wandb()
    return model


# =============================================================================
# Evaluation
# =============================================================================


def score_completion(
    model: OutcomeRewardModel,
    tokenizer: AutoTokenizer,
    question: str,
    completion: str,
    device: torch.device,
) -> float:
    """Score a single completion using the trained ORM.

    Returns the final completion token's correctness probability.
    """
    example = pack_example(question, completion, 1, tokenizer)  # Label doesn't matter here
    batch = create_collate_fn(tokenizer, ["input_ids", "attention_mask"])([example])
    batch = {k: v.to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        _, logits = model(**batch)
        score = torch.sigmoid(last_token_values(logits, batch["attention_mask"]))

    return score[0].item()


def demo_scoring(model: OutcomeRewardModel, tokenizer: AutoTokenizer, config: Config):
    """Demo: Score an unseen GSM8K test question."""
    device = next(model.parameters()).device
    sample = load_dataset(config.dataset_name, split="test").shuffle(seed=config.seed)[0]
    question = sample["prompt"].strip()
    retained = [
        (completion, int(reward))
        for completion, reward in zip(sample["completions"], sample["rewards"], strict=True)
        if len(pack_example(question, completion, int(reward), tokenizer)["input_ids"])
        <= config.max_length
    ]
    correct_completion = next((completion for completion, label in retained if label == 1), None)
    incorrect_completion = next((completion for completion, label in retained if label == 0), None)
    if correct_completion is None or incorrect_completion is None:
        print("Selected test prompt does not have both correct and incorrect retained rollouts")
        return

    print("=" * 60)
    print("Question:", question)
    print("=" * 60)

    correct_score = score_completion(model, tokenizer, question, correct_completion, device)
    print("\nCorrect completion:")
    print(correct_completion[:200] + "..." if len(correct_completion) > 200 else correct_completion)
    print(f"Score: {correct_score:.3f}")

    incorrect_score = score_completion(model, tokenizer, question, incorrect_completion, device)
    print("\nIncorrect completion:")
    print(
        incorrect_completion[:200] + "..."
        if len(incorrect_completion) > 200
        else incorrect_completion
    )
    print(f"Score: {incorrect_score:.3f}")

    print(f"\nModel correctly prefers correct answer: {correct_score > incorrect_score}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Outcome Reward Model on GSM8K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = train_orm(config=cfg)

    if not cfg.skip_demo:
        tokenizer = load_tokenizer(cfg.model_id)
        demo_scoring(model, tokenizer, cfg)


if __name__ == "__main__":
    main()
