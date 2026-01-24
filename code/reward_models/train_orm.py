#!/usr/bin/env python3
"""Outcome Reward Model (ORM) Training

Original implementation by @myhott163com
Source: https://github.com/myhott163com/RLHF_ORM_PRM
License: MIT

Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

This script trains a minimal outcome reward model by fine-tuning a base LLM
with LoRA on GSM8K-derived correct/incorrect math answers. For each question,
we parse the gold numeric answer and synthesize wrong completions by adding
random offsets. The model learns to classify solution correctness via per-token
BCE loss on completion tokens.

See Chapter 7 (Reward Models) of RLHF Book for theoretical background.

Usage:
    uv run python -m reward_models.train_orm
    uv run python -m reward_models.train_orm --samples 500 --epochs 2
"""

import argparse
import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from reward_models.base import (
    BaseRewardModel,
    create_optimizer,
    finish_wandb,
    init_wandb,
    load_tokenizer,
    log_metrics,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
DEFAULT_DATASET = "gsm8k"
DEFAULT_SAMPLES = 200
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 1
DEFAULT_LR = 5e-6  # Lower LR for full fine-tuning (vs 5e-5 for LoRA)
DEFAULT_SEED = 7


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
    completion_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids
    attention = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + [label] * len(completion_ids)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def build_orm_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str = DEFAULT_DATASET,
    limit: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Build ORM training dataset from GSM8K.

    For each question:
    - Creates a positive example with the correct solution (label=1)
    - Creates a negative example with a corrupted answer (label=0)
    """
    random.seed(seed)
    raw = load_dataset(dataset_name, "main", split=f"train[:{limit}]")
    rows = []

    for ex in raw:
        question = ex["question"].strip()
        prompt = f"Question: {question}\nAnswer:"
        answer = ex["answer"].strip()
        value = parse_answer(answer)

        if value is None:
            continue

        # Correct example
        rows.append(pack_example(prompt, answer, 1, tokenizer))

        # Incorrect example (add random offset to answer)
        wrong = value + random.randint(1, 9)
        wrong_solution = answer + f"\nTherefore, the answer is {wrong}."
        rows.append(pack_example(prompt, wrong_solution, 0, tokenizer))

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


class OutcomeRewardModel(BaseRewardModel):
    """Outcome Reward Model with LoRA fine-tuning.

    Architecture:
    - Base LLM (e.g., Qwen3) with 4-bit quantization
    - LoRA adapters on attention projections
    - Linear head mapping hidden states to scalar reward

    The model outputs per-token logits which are trained with BCE loss
    on completion tokens only (prompt tokens are masked).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs):
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
# Training
# =============================================================================


def train_orm(
    model_id: str = DEFAULT_MODEL_ID,
    samples: int = DEFAULT_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = True,
) -> OutcomeRewardModel:
    """Train an Outcome Reward Model on GSM8K.

    Args:
        model_id: HuggingFace model ID for base model
        samples: Number of GSM8K samples to use
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        use_wandb: Whether to log to wandb

    Returns:
        Trained OutcomeRewardModel
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    init_wandb(
        default_run_name="orm_gsm8k",
        config={
            "model_id": model_id,
            "samples": samples,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        },
        use_wandb=use_wandb,
    )

    # Load tokenizer
    tokenizer = load_tokenizer(model_id)

    # Build dataset
    print(f"Building ORM dataset with {samples} samples...")
    data = build_orm_dataset(tokenizer, limit=samples, seed=seed)
    print(f"Dataset size: {len(data)} examples")

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Initialize model
    print(f"Loading model: {model_id}")
    model = OutcomeRewardModel(model_id=model_id).to(device)
    print(f"Trainable parameters: {model.count_trainable_params() / 1e6:.2f}M")

    # Optimizer
    optimizer = create_optimizer(model, lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model(**batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # Compute accuracy on completion tokens
            mask = batch["labels"] != -100
            preds = (torch.sigmoid(logits[mask]) > 0.5).long()
            total_correct += (preds == batch["labels"][mask]).sum().item()
            total_tokens += mask.sum().item()

            if step % 10 == 0:
                acc = total_correct / total_tokens if total_tokens > 0 else 0
                print(f"Epoch {epoch} step {step} loss {loss.item():.4f}")
                log_metrics({"loss": loss.item(), "accuracy": acc})

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}")
        log_metrics({"epoch_loss": avg_loss, "epoch_accuracy": accuracy, "epoch": epoch})

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

    Returns average token probability for the completion.
    """
    example = pack_example(prompt, completion, 1, tokenizer)  # Label doesn't matter for inference
    batch = collate_fn([example], tokenizer)
    batch = {k: v.to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        _, logits = model(**batch)
        probs = torch.sigmoid(logits)

    mask = batch["labels"][0] != -100
    return probs[0][mask].mean().item()


def demo_scoring(model: OutcomeRewardModel, tokenizer: AutoTokenizer, seed: int = DEFAULT_SEED):
    """Demo: Score an unseen GSM8K test question."""
    device = next(model.parameters()).device
    random.seed(seed)

    # Get a random test example
    test_index = random.randint(0, 1000)
    sample = load_dataset(DEFAULT_DATASET, "main", split=f"test[{test_index}:{test_index + 1}]")[0]

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
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base model ID")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of training samples")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--skip-demo", action="store_true", help="Skip scoring demo after training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    model = train_orm(
        model_id=args.model_id,
        samples=args.samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )

    if not args.skip_demo:
        tokenizer = load_tokenizer(args.model_id)
        demo_scoring(model, tokenizer, seed=args.seed)


if __name__ == "__main__":
    main()
