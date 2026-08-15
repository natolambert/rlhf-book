#!/usr/bin/env python3
"""Outcome Reward Model (ORM) Training

Original implementation by @myhott163com
Source: https://github.com/myhott163com/RLHF_ORM_PRM
License: MIT

Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

This script trains a minimal outcome reward model by fine-tuning a base LLM
on GSM8K-derived correct/incorrect math answers. For each question,
we parse the gold numeric answer and synthesize wrong completions by adding
random offsets. The model learns to classify solution correctness via per-token
BCE loss on completion tokens.

See Chapter 5 (Reward Models) of RLHF Book for theoretical background.

Usage:
    uv run python -m reward_models.train_orm --config reward_models/configs/orm.yaml
"""

import argparse
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from reward_models.base import (
    BaseRewardModel,
    create_optimizer,
    finish_wandb,
    init_wandb,
    load_tokenizer,
    log_metrics,
)
from reward_models.config import Config, load_config


# =============================================================================
# Data Preparation
# =============================================================================


def parse_answer(text: str) -> int | None:
    """Extract numeric answer from GSM8K solution text.

    GSM8K answers are formatted as "#### <number>" at the end.
    This function extracts that number, handling commas and edge cases.
    """
    if "####" in text:
        tail = text.split("####")[-1]
    else:
        sentences = [seg.strip() for seg in text.strip().split("\n") if seg.strip()]
        tail = sentences[-1] if sentences else text

    tokens = tail.replace(",", "").split()
    for token in reversed(tokens):
        digits = "".join(ch for ch in token if ch.isdigit() or ch == "-")
        if digits:
            try:
                return int(digits)
            except ValueError:
                continue
    return None


def pack_example(
    prompt: str, completion: str, label: int, tokenizer: AutoTokenizer
) -> Dict[str, List[int]]:
    """Pack a (prompt, completion, label) into tokenized format.

    The label is applied to all completion tokens, with prompt tokens masked (-100).
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = prompt_ids + completion_ids
    attention = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + [label] * len(completion_ids)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def _pack_raw_examples(raw_rows, tokenizer: AutoTokenizer) -> list[dict]:
    """Pack raw GSM8K rows into paired positive/negative examples."""
    packed = []
    for ex in raw_rows:
        question = ex["question"].strip()
        prompt = f"Question: {question}\nAnswer:"
        answer = ex["answer"].strip()
        value = parse_answer(answer)

        if value is None:
            continue

        # Correct example
        packed.append(pack_example(prompt, answer, 1, tokenizer))

        # Incorrect example (add random offset to answer)
        wrong = value + random.randint(1, 9)
        wrong_solution = answer + f"\nTherefore, the answer is {wrong}."
        packed.append(pack_example(prompt, wrong_solution, 0, tokenizer))
    return packed


def build_orm_dataset(
    tokenizer: AutoTokenizer,
    config: Config,
) -> Dataset | tuple[Dataset, Dataset]:
    """Build ORM training dataset from GSM8K.

    Splits raw GSM8K rows first so both completions for a given question stay
    in the same split, then packs paired positive/negative examples inside
    each split.

    For each question:
    - Creates a positive example with the correct solution (label=1)
    - Creates a negative example with a corrupted answer (label=0)
    """
    random.seed(config.seed)
    raw = load_dataset(
        config.dataset_name,
        "main",
        split=config.dataset_split,
    )

    samples = min(config.samples, len(raw))
    raw = raw.shuffle(seed=config.seed).select(range(samples))

    if config.val_ratio > 0.0:
        raw_splits = raw.train_test_split(
            test_size=config.val_ratio, seed=config.seed, shuffle=True
        )
        train_rows = _pack_raw_examples(raw_splits["train"], tokenizer)
        val_rows = _pack_raw_examples(raw_splits["test"], tokenizer)
        return Dataset.from_list(train_rows), Dataset.from_list(val_rows)

    rows = _pack_raw_examples(raw, tokenizer)
    return Dataset.from_list(rows)


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader - pads sequences to same length."""
    max_len = max(len(x["input_ids"]) for x in batch)
    inputs = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(inputs)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        length = len(item["input_ids"])
        inputs[i, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        attn[i, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
        labels[i, :length] = torch.tensor(item["labels"], dtype=torch.long)

    return {"input_ids": inputs, "attention_mask": attn, "labels": labels}


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
    - Base LLM (e.g., Qwen3) loaded in bfloat16
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


@torch.no_grad()
def evaluate_orm(
    model: OutcomeRewardModel,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
) -> dict[str, float]:
    """Evaluate ORM loss and completion-correctness accuracy on a loader."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss, logits = model(**batch)

        sequence_logits = last_token_values(logits, batch["attention_mask"])
        sequence_labels = last_token_values(batch["labels"], batch["attention_mask"])

        preds = (torch.sigmoid(sequence_logits) > 0.5).long()
        correct = (preds == sequence_labels).sum().item()
        examples = sequence_labels.numel()

        total_loss += loss.item() * examples
        total_correct += correct
        total_examples += examples

    n = max(1, total_examples)
    return {
        "val/loss": total_loss / n,
        "val/accuracy": total_correct / n,
    }


# =============================================================================
# Training
# =============================================================================


def train_orm(
    config: Config,
) -> OutcomeRewardModel:
    """Train an Outcome Reward Model on GSM8K.

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

    # Build dataset
    print(f"Building ORM dataset with {config.samples} samples...")
    data = build_orm_dataset(tokenizer, config)

    if isinstance(data, tuple):
        train_data, val_data = data
    else:
        train_data, val_data = data, None

    print(f"Train size: {len(train_data)} examples")
    if val_data is not None:
        print(f"Validation size: {len(val_data)} examples")

    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=len(train_data) > config.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    val_loader = (
        DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
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

    # Mixed precision
    autocast_enabled = torch.cuda.is_available()

    # Training loop
    global_step = 0
    grad_accum_steps = config.grad_accum_steps
    eval_interval = config.eval_interval
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0
        optimizer.zero_grad()

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
            accum_loss += loss.item()
            sequence_logits = last_token_values(logits, batch["attention_mask"])
            sequence_labels = last_token_values(batch["labels"], batch["attention_mask"])

            preds = (torch.sigmoid(sequence_logits) > 0.5).long()
            correct = (preds == sequence_labels).sum().item()

            examples = sequence_labels.numel()
            accum_correct += correct
            accum_examples += examples
            accum_microbatches += 1

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_examples += examples

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
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
                    val_metrics = evaluate_orm(model, val_loader, device, autocast_enabled)
                    print(
                        f"Eval step {global_step} | "
                        f"Val Loss: {val_metrics['val/loss']:.4f} | "
                        f"Val Accuracy: {val_metrics['val/accuracy']:.3f}"
                    )
                    log_metrics(val_metrics, step=global_step)
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
            val_metrics = evaluate_orm(model, val_loader, device, autocast_enabled)
            print(
                f"Epoch {epoch} | Val Loss: {val_metrics['val/loss']:.4f} | "
                f"Val Accuracy: {val_metrics['val/accuracy']:.3f}"
            )
            log_metrics({**val_metrics, "epoch": epoch}, step=global_step)
            model.train()

    finish_wandb()
    return model


# =============================================================================
# Evaluation
# =============================================================================


def score_completion(
    model: OutcomeRewardModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
) -> float:
    """Score a single completion using the trained ORM.

    Returns the final completion token's probability.
    """
    example = pack_example(prompt, completion, 1, tokenizer)  # Label doesn't matter for inference
    batch = collate_fn([example], tokenizer)
    batch = {k: v.to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        _, logits = model(**batch)
        score = torch.sigmoid(last_token_values(logits, batch["attention_mask"]))

    return score[0].item()


def demo_scoring(model: OutcomeRewardModel, tokenizer: AutoTokenizer, config: Config):
    """Demo: Score an unseen GSM8K test question."""
    device = next(model.parameters()).device
    random.seed(config.seed)

    # Get a random test example
    test_data = load_dataset(
        config.dataset_name,
        "main",
        split="test",
    )
    sample = random.choice(test_data)

    question = sample["question"].strip()
    answer = sample["answer"].strip()
    value = parse_answer(answer)

    if value is None:
        print("Could not parse answer from sample")
        return

    prompt = f"Question: {question}\nAnswer:"

    # Create correct and incorrect completions
    wrong_value = value + random.randint(1, 9)
    wrong_answer = answer + f"\nTherefore, the answer is {wrong_value}."

    print("=" * 60)
    print("Question:", question)
    print("=" * 60)

    correct_score = score_completion(model, tokenizer, prompt, answer, device)
    print(f"\nCorrect completion (answer={value}):")
    print(answer[:200] + "..." if len(answer) > 200 else answer)
    print(f"Score: {correct_score:.3f}")

    incorrect_score = score_completion(model, tokenizer, prompt, wrong_answer, device)
    print(f"\nIncorrect completion (answer={wrong_value}):")
    print(wrong_answer[:200] + "..." if len(wrong_answer) > 200 else wrong_answer)
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
