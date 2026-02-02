#!/usr/bin/env python3
"""Process Reward Model (PRM) Training

Original implementation by @myhott163com
Source: https://github.com/myhott163com/RLHF_ORM_PRM
License: MIT

Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

This script trains a process reward model by fine-tuning a base LLM with LoRA
on PRM800K-style chain-of-thought traces. Each reasoning step has a label in
{-1, 0, 1} (bad, neutral, good). The model learns to classify step quality
via cross-entropy loss on step terminator tokens.

Unlike ORM which only judges final answers, PRM provides step-level feedback,
enabling more granular credit assignment during RL training.

See Chapter 5 (Reward Models) of RLHF Book for theoretical background.

Usage:
    uv run python -m reward_models.train_prm
    uv run python -m reward_models.train_prm --samples 1000 --epochs 2
"""

import argparse
import random
from typing import Any, Dict, List

import torch
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

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"  # Smaller model to fit in memory
DEFAULT_PRM_DATASET = "tasksource/PRM800K"
DEFAULT_SAMPLES = 2000
DEFAULT_BATCH_SIZE = 1  # PRM traces are long
DEFAULT_GRAD_ACCUM = 2
DEFAULT_MAX_STEPS = 20  # Max reasoning steps per sample
DEFAULT_MAX_TOKENS = 5500  # Max tokens per sample
DEFAULT_EPOCHS = 1
DEFAULT_LR = 3e-6  # Lower LR for full fine-tuning (vs 3e-5 for LoRA)
DEFAULT_SEED = 13

STEP_SEPARATOR = "\n<step>\n"
PRM_CLASS_VALUES = [-1, 0, 1]  # Bad, Neutral, Good
PRM_CLASS_TO_IDX = {value: idx for idx, value in enumerate(PRM_CLASS_VALUES)}


# =============================================================================
# Data Preparation
# =============================================================================


def to_plain_text(value: Any) -> str:
    """Convert various data types to plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "value", "content"):
            if key in value and isinstance(value[key], str):
                return value[key]
        return " ".join(str(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def get_problem_text(example: Dict) -> str:
    """Extract problem text from PRM800K example."""
    question_block = example.get("question") or {}
    raw = (
        question_block.get("problem")
        or question_block.get("question")
        or question_block.get("prompt")
        or question_block.get("problem_statement")
        or question_block.get("content")
        or example.get("problem")
        or example.get("prompt")
        or ""
    )
    return to_plain_text(raw)


def get_steps_and_labels(example: Dict) -> tuple[List[str], List[int]]:
    """Extract reasoning steps and their labels from PRM800K example.

    Returns:
        steps: List of step text strings
        labels: List of step ratings (-1, 0, or 1)
    """
    label_block = example.get("label") or {}
    steps_struct = label_block.get("steps") or []

    steps: List[str] = []
    parsed_labels: List[int] = []

    for step in steps_struct:
        completions = step.get("completions") or []
        found = False

        for comp in completions:
            text = comp.get("text")
            rating = comp.get("rating")
            if text is None or rating is None:
                continue
            text = to_plain_text(text).strip()
            if not text:
                continue
            try:
                rating_int = int(rating)
            except (TypeError, ValueError):
                continue
            steps.append(text)
            parsed_labels.append(rating_int)
            found = True

        if found:
            continue

        # Fallback to other fields
        text = step.get("human_completion") or step.get("text") or step.get("completion")
        rating = step.get("rating") or step.get("score")
        if text and rating is not None:
            text = to_plain_text(text).strip()
            if text:
                try:
                    rating_int = int(rating)
                    steps.append(text)
                    parsed_labels.append(rating_int)
                except (TypeError, ValueError):
                    continue

    return steps, parsed_labels


def build_prm_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str = DEFAULT_PRM_DATASET,
    limit: int = DEFAULT_SAMPLES,
    max_steps_per_sample: int = DEFAULT_MAX_STEPS,
    max_tokens_per_sample: int = DEFAULT_MAX_TOKENS,
) -> Dataset:
    """Build PRM training dataset from PRM800K.

    Each example contains:
    - Problem text as prompt
    - Reasoning steps separated by STEP_SEPARATOR
    - Labels only on step terminator tokens (not prompt or step content)
    """
    stream = load_dataset(dataset_name, split="train", streaming=True)
    records = []

    for example in stream:
        if len(records) >= limit:
            break

        problem = get_problem_text(example).strip()
        steps, labels = get_steps_and_labels(example)

        if not problem or not steps or not labels:
            continue

        prompt = f"Problem: {problem}\nReasoning trace:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Chunk very long traces to avoid OOM
        for start in range(0, len(steps), max_steps_per_sample):
            if len(records) >= limit:
                break

            chunk_steps = steps[start : start + max_steps_per_sample]
            chunk_labels = labels[start : start + max_steps_per_sample]

            if not chunk_steps or not chunk_labels:
                continue

            input_ids = list(prompt_ids)
            attention_mask = [1] * len(input_ids)
            label_ids = [-100] * len(input_ids)

            for step_text, lbl in zip(chunk_steps, chunk_labels):
                step_payload = step_text.strip() + STEP_SEPARATOR
                encoded = tokenizer(step_payload, add_special_tokens=False)["input_ids"]
                input_ids.extend(encoded)
                attention_mask.extend([1] * len(encoded))

                # Only label the step terminator token
                step_labels = [-100] * len(encoded)
                cls_id = PRM_CLASS_TO_IDX.get(int(lbl), PRM_CLASS_TO_IDX[0])
                step_labels[-1] = cls_id
                label_ids.extend(step_labels)

            # Skip pathologically long traces
            if len(input_ids) > max_tokens_per_sample:
                continue

            records.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids,
            })

    if not records:
        raise ValueError("No PRM examples loaded. Check dataset path/permissions.")

    return Dataset.from_list(records[:limit])


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    max_len = max(len(item["input_ids"]) for item in batch)
    inputs = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(inputs)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for idx, item in enumerate(batch):
        length = len(item["input_ids"])
        inputs[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        attn[idx, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long)

    return {"input_ids": inputs, "attention_mask": attn, "labels": labels}


# =============================================================================
# Model Definition
# =============================================================================


class ProcessRewardModel(BaseRewardModel):
    """Process Reward Model with full fine-tuning.

    Architecture:
    - Base LLM (e.g., Qwen3) in BF16
    - Linear head mapping hidden states to 3-class logits

    The model outputs per-token logits which are trained with cross-entropy
    loss on step terminator tokens only (all other tokens masked).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs):
        # 3-class head for PRM: bad (-1), neutral (0), good (1)
        super().__init__(model_id, head_dim=len(PRM_CLASS_VALUES), **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Forward pass computing step logits and optional loss.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Per-token class labels (0/1/2 for steps, -100 for masked)

        Returns:
            loss: Cross-entropy loss on step tokens (None if labels not provided)
            logits: Per-token class logits [batch, seq_len, 3]
        """
        hidden = self.get_hidden_states(input_ids, attention_mask)
        logits = self.head(hidden)

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                loss = F.cross_entropy(logits[mask], labels[mask])
            else:
                loss = logits.sum() * 0

        return loss, logits


# =============================================================================
# Training
# =============================================================================


def train_prm(
    model_id: str = DEFAULT_MODEL_ID,
    samples: int = DEFAULT_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = True,
) -> ProcessRewardModel:
    """Train a Process Reward Model on PRM800K.

    Args:
        model_id: HuggingFace model ID for base model
        samples: Number of PRM800K samples to use
        batch_size: Training batch size
        grad_accum_steps: Gradient accumulation steps
        epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        use_wandb: Whether to log to wandb

    Returns:
        Trained ProcessRewardModel
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    init_wandb(
        default_run_name="prm_prm800k",
        config={
            "model_id": model_id,
            "samples": samples,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "epochs": epochs,
            "lr": lr,
        },
        use_wandb=use_wandb,
    )

    # Load tokenizer
    tokenizer = load_tokenizer(model_id)

    # Build dataset
    print(f"Building PRM dataset with {samples} samples...")
    data = build_prm_dataset(tokenizer, limit=samples)
    print(f"Dataset size: {len(data)} examples")

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Initialize model
    print(f"Loading model: {model_id}")
    model = ProcessRewardModel(model_id=model_id).to(device)
    print(f"Trainable parameters: {model.count_trainable_params() / 1e6:.2f}M")

    # Optimizer
    optimizer = create_optimizer(model, lr)

    # Mixed precision for memory efficiency
    autocast_enabled = torch.cuda.is_available()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_steps = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, logits = model(**batch)

            (loss / grad_accum_steps).backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # Compute step-level accuracy
            mask = batch["labels"] != -100
            preds = logits[mask].argmax(dim=-1)
            total_correct += (preds == batch["labels"][mask]).sum().item()
            total_steps += mask.sum().item()

            if step_idx % 100 == 0:
                acc = total_correct / max(1, total_steps)
                print(f"Epoch {epoch} step {step_idx} loss {loss.item():.4f}")
                log_metrics({"loss": loss.item(), "step_accuracy": acc})

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / max(1, total_steps)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Step Accuracy: {accuracy:.3f}")
        log_metrics({"epoch_loss": avg_loss, "epoch_accuracy": accuracy, "epoch": epoch})

    finish_wandb()
    return model


# =============================================================================
# Evaluation
# =============================================================================


def score_trace(
    model: ProcessRewardModel,
    tokenizer: AutoTokenizer,
    problem: str,
    steps: List[str],
    device: torch.device,
) -> List[Dict[str, float]]:
    """Score each step in a reasoning trace.

    Returns list of dicts with probabilities for each class.
    """
    prompt = f"Problem: {problem}\nReasoning trace:\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    input_ids = list(prompt_ids)
    attention_mask = [1] * len(input_ids)
    boundaries = []

    for step_text in steps:
        step_payload = step_text.strip() + STEP_SEPARATOR
        encoded = tokenizer(step_payload, add_special_tokens=False)["input_ids"]
        input_ids.extend(encoded)
        attention_mask.extend([1] * len(encoded))
        boundaries.append(len(input_ids) - 1)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device),
    }

    model.eval()
    with torch.no_grad():
        _, logits = model(**batch)
        probs = torch.softmax(logits[0], dim=-1)

    results = []
    for boundary in boundaries:
        prob_vec = probs[boundary].cpu().tolist()
        results.append({cls: prob_vec[PRM_CLASS_TO_IDX[cls]] for cls in PRM_CLASS_VALUES})

    return results


def demo_scoring(model: ProcessRewardModel, tokenizer: AutoTokenizer, seed: int = DEFAULT_SEED):
    """Demo: Score an unseen PRM800K test trace."""
    device = next(model.parameters()).device
    random.seed(seed)

    # Get a random test example
    test_stream = load_dataset(DEFAULT_PRM_DATASET, split="test", streaming=True)
    target_idx = random.randint(0, 500)

    sample = None
    for idx, item in enumerate(test_stream):
        if idx == target_idx:
            sample = item
            break

    if sample is None:
        print("Could not fetch test example")
        return

    problem = get_problem_text(sample).strip()
    steps, labels = get_steps_and_labels(sample)

    if not steps:
        print("No steps found in example")
        return

    print("=" * 60)
    print("Problem:", problem[:300] + "..." if len(problem) > 300 else problem)
    print("=" * 60)

    scores = score_trace(model, tokenizer, problem, steps, device)

    for idx, (step_text, true_label, step_scores) in enumerate(zip(steps, labels, scores)):
        label_name = true_label
        pred_class = max(step_scores, key=step_scores.get)
        print(f"\nStep {idx} (true label: {label_name}, predicted: {pred_class}):")
        print(step_text[:150] + "..." if len(step_text) > 150 else step_text)
        print(f"  Probs: -1={step_scores[-1]:.3f}, 0={step_scores[0]:.3f}, 1={step_scores[1]:.3f}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Process Reward Model on PRM800K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base model ID")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of training samples")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--skip-demo", action="store_true", help="Skip scoring demo after training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    model = train_prm(
        model_id=args.model_id,
        samples=args.samples,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
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
