#!/usr/bin/env python3
"""Process Reward Model (PRM) Training

Original implementation by @myhott163com
Source: https://github.com/myhott163com/RLHF_ORM_PRM
License: MIT

Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

This script trains a process reward model by fine-tuning a base LLM
on PRM800K-style chain-of-thought traces. Each reasoning step has a label in
{-1, 0, 1} (bad, neutral, good). The model learns to classify step quality
via cross-entropy loss on step terminator tokens.

Unlike ORM which only judges final answers, PRM provides step-level feedback,
enabling more granular credit assignment during RL training.

See Chapter 5 (Reward Models) of RLHF Book for theoretical background.

Usage:
    uv run python -m reward_models.train_prm --config reward_models/configs/prm.yaml
"""

import argparse
import json
import random
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import HfFileSystem
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
from reward_models.config import Config, load_config


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"  # Smaller model to fit in memory
DEFAULT_PRM_DATASET = "tasksource/PRM800K"
DEFAULT_SAMPLES = 2000
DEFAULT_MAX_STEPS = 20  # Max reasoning steps per sample
DEFAULT_MAX_TOKENS = 5500  # Max tokens per sample
DEFAULT_SEED = 13

STEP_SEPARATOR = "\n<step>\n"
PRM_CLASS_VALUES = [-1, 0, 1]  # Bad, Neutral, Good
PRM_CLASS_TO_IDX = {value: idx for idx, value in enumerate(PRM_CLASS_VALUES)}


def _stream_prm_jsonl(dataset_name: str, split: str):
    """Stream raw JSONL records from a PRM800K-style Hugging Face dataset.

    This loader expects a repo laid out as `phase1_{split}.jsonl` and
    `phase2_{split}.jsonl` (as PRM800K is). We bypass the ``datasets`` JSON
    loader because it intermittently fails to cast the PRM800K schema under
    recent ``datasets``/``pyarrow`` versions.
    """
    fs = HfFileSystem()
    repo_prefix = f"datasets/{dataset_name}"
    filenames = [f"phase1_{split}.jsonl", f"phase2_{split}.jsonl"]

    found = False
    for fname in filenames:
        remote_path = f"{repo_prefix}/{fname}"
        try:
            with fs.open(remote_path, "r") as f:
                found = True
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            # Some splits may only exist in one phase file.
            continue

    if not found:
        raise FileNotFoundError(
            f"Could not find any of {filenames} in dataset '{dataset_name}'. "
            "This loader only supports PRM800K-style repos with "
            "`phase{{1,2}}_{split}.jsonl` files."
        )


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
        rating = step.get("rating")
        if rating is None or rating == "":
            rating = step.get("score")
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


def _pack_raw_prm_examples(
    raw_rows, tokenizer: AutoTokenizer, max_steps_per_sample: int, max_tokens_per_sample: int
) -> list[dict]:
    """Pack raw PRM800K rows into tokenized examples."""
    records = []

    for example in raw_rows:
        problem = get_problem_text(example).strip()
        steps, labels = get_steps_and_labels(example)

        if not problem or not steps or not labels:
            continue

        prompt = f"Problem: {problem}\nReasoning trace:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Chunk very long traces to avoid OOM
        for start in range(0, len(steps), max_steps_per_sample):
            chunk_steps = steps[start : start + max_steps_per_sample]
            chunk_labels = labels[start : start + max_steps_per_sample]

            if not chunk_steps or not chunk_labels:
                continue

            input_ids = list(prompt_ids)
            attention_mask = [1] * len(input_ids)
            label_ids = [-100] * len(input_ids)

            for step_text, lbl in zip(chunk_steps, chunk_labels, strict=True):
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

            records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids,
                }
            )

    return records


def build_prm_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str = DEFAULT_PRM_DATASET,
    split: str = "train",
    limit: int = DEFAULT_SAMPLES,
    max_steps_per_sample: int = DEFAULT_MAX_STEPS,
    max_tokens_per_sample: int = DEFAULT_MAX_TOKENS,
    val_ratio: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> Dataset | tuple[Dataset, Dataset]:
    """Build PRM training dataset from PRM800K.

    Optionally holds out a fraction of raw PRM800K rows as a validation split
    before tokenizing, so chunks derived from the same problem stay together.

    Args:
        tokenizer: Tokenizer for encoding the problem and reasoning steps.
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to load (e.g. "train", "validation", "test").
        limit: Maximum number of raw examples to load.
        max_steps_per_sample: Max reasoning steps packed into a single example.
        max_tokens_per_sample: Max tokens per packed example.
        val_ratio: Optional fraction of raw rows to hold out for validation.
        seed: Random seed used for shuffling/splitting.

    Returns:
        A single Dataset when val_ratio == 0, otherwise a (train, val) tuple.

    Each example contains:
    - Problem text as prompt
    - Reasoning steps separated by STEP_SEPARATOR
    - Labels only on step terminator tokens (not prompt or step content)
    """
    stream = _stream_prm_jsonl(dataset_name, split)
    raw_records = []

    for example in stream:
        if len(raw_records) >= limit:
            break

        problem = get_problem_text(example).strip()
        steps, labels = get_steps_and_labels(example)
        if problem and steps and labels:
            raw_records.append(example)

    if not raw_records:
        raise ValueError("No PRM examples loaded. Check dataset path/permissions.")

    raw_dataset = Dataset.from_list(raw_records[:limit])

    if val_ratio > 0.0:
        raw_splits = raw_dataset.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
        train_records = _pack_raw_prm_examples(
            raw_splits["train"], tokenizer, max_steps_per_sample, max_tokens_per_sample
        )
        val_records = _pack_raw_prm_examples(
            raw_splits["test"], tokenizer, max_steps_per_sample, max_tokens_per_sample
        )
        if not train_records or not val_records:
            raise ValueError(
                "Split produced an empty partition; reduce val_ratio or increase limit."
            )
        return Dataset.from_list(train_records), Dataset.from_list(val_records)

    records = _pack_raw_prm_examples(
        raw_dataset, tokenizer, max_steps_per_sample, max_tokens_per_sample
    )
    if not records:
        raise ValueError("No PRM examples loaded. Check dataset path/permissions.")

    return Dataset.from_list(records)


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
    - Base LLM (e.g., Qwen3) with FP32 parameters and BF16 CUDA autocast
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
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_prm(
    model: ProcessRewardModel,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
) -> dict[str, float]:
    """Evaluate PRM cross-entropy loss and step-classification accuracy."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss, logits = model(**batch)

        mask = batch["labels"] != -100
        preds = logits[mask].argmax(dim=-1)
        correct = (preds == batch["labels"][mask]).sum().item()
        tokens = mask.sum().item()

        total_loss += loss.item() * tokens
        total_correct += correct
        total_tokens += tokens

    n = max(1, total_tokens)
    return {
        "val/loss": total_loss / n,
        "val/step_accuracy": total_correct / n,
    }


# =============================================================================
# Training
# =============================================================================


def train_prm(config: Config) -> ProcessRewardModel:
    """Train a Process Reward Model on PRM800K.

    Args:
        config: Configuration object containing training parameters.

    Returns:
        Trained ProcessRewardModel
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.get_device())

    # Initialize wandb
    init_wandb(
        default_run_name="prm_prm800k",
        config=config.model_dump(),
        use_wandb=config.use_wandb,
    )

    # Load tokenizer
    tokenizer = load_tokenizer(config.model_id)

    # Build dataset
    print(f"Building PRM dataset with {config.samples} samples...")
    data = build_prm_dataset(
        tokenizer,
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        limit=config.samples,
        max_steps_per_sample=config.max_steps,
        max_tokens_per_sample=config.max_tokens,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

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
    model = ProcessRewardModel(
        model_id=config.model_id,
        freeze_backbone=config.freeze_backbone,
        device=device,
    ).to(device)
    print(f"Trainable parameters: {model.count_trainable_params() / 1e6:.2f}M")

    # Optimizer and LR scheduler with linear warmup
    optimizer = create_optimizer(model, config.lr)
    total_optimizer_steps = -(-len(loader) // config.grad_accum_steps) * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = (
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        if warmup_steps > 0
        else None
    )

    # Mixed precision for memory efficiency
    autocast_enabled = torch.cuda.is_available()

    # Training loop
    global_step = 0
    grad_accum_steps = config.grad_accum_steps
    eval_interval = config.eval_interval
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0
        optimizer.zero_grad()

        # Accumulators for logging per optimizer step
        accum_loss = 0.0
        accum_correct = 0
        accum_tokens = 0
        accum_microbatches = 0

        for step_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, logits = model(**batch)

            (loss / grad_accum_steps).backward()

            # Accumulate metrics over the grad_accum window
            accum_loss += loss.item()
            mask = batch["labels"] != -100
            preds = logits[mask].argmax(dim=-1)
            correct = (preds == batch["labels"][mask]).sum().item()
            tokens = mask.sum().item()
            accum_correct += correct
            accum_tokens += tokens
            accum_microbatches += 1

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_tokens += tokens

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log averaged metrics over the full effective batch
                avg_loss = accum_loss / accum_microbatches
                acc = accum_correct / max(1, accum_tokens)
                print(f"Epoch {epoch} step {global_step} | loss {avg_loss:.4f} | acc {acc:.3f}")
                log_metrics(
                    {
                        "train/loss": avg_loss,
                        "train/step_accuracy": acc,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

                # Run validation every N optimizer steps.
                # evaluate_prm() switches model to eval mode, so switch back to train after.
                if (
                    val_loader is not None
                    and eval_interval > 0
                    and global_step % eval_interval == 0
                ):
                    val_metrics = evaluate_prm(model, val_loader, device, autocast_enabled)
                    print(
                        f"Eval step {global_step} | "
                        f"Val Loss: {val_metrics['val/loss']:.4f} | "
                        f"Val Step Accuracy: {val_metrics['val/step_accuracy']:.3f}"
                    )
                    log_metrics(val_metrics, step=global_step)
                    model.train()

                # Reset accumulators
                accum_loss = 0.0
                accum_correct = 0
                accum_tokens = 0
                accum_microbatches = 0

        avg_loss = epoch_loss / len(loader)
        accuracy = epoch_correct / max(1, epoch_tokens)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Step Accuracy: {accuracy:.3f}")
        log_metrics(
            {"epoch_loss": avg_loss, "epoch_accuracy": accuracy, "epoch": epoch},
            step=global_step,
        )

        # Also run validation at epoch end, unless we already evaluated on this exact step.
        should_run_epoch_eval = val_loader is not None and (
            eval_interval <= 0 or global_step % eval_interval != 0
        )

        if should_run_epoch_eval:
            val_metrics = evaluate_prm(model, val_loader, device, autocast_enabled)
            print(
                f"Epoch {epoch} | Val Loss: {val_metrics['val/loss']:.4f} | "
                f"Val Step Accuracy: {val_metrics['val/step_accuracy']:.3f}"
            )
            log_metrics({**val_metrics, "epoch": epoch}, step=global_step)
            model.train()

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
    test_stream = _stream_prm_jsonl(DEFAULT_PRM_DATASET, "test")
    target_idx = random.randint(0, 500)

    sample = None
    try:
        for idx, item in enumerate(test_stream):
            if idx == target_idx:
                sample = item
                break
    except Exception as e:
        print(f"Skipping demo: failed reading PRM800K test sample ({e})")
        return

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

    for idx, (step_text, true_label, step_scores) in enumerate(
        zip(steps, labels, scores, strict=True)
    ):
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
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = train_prm(config=cfg)

    if not cfg.skip_demo:
        tokenizer = load_tokenizer(cfg.model_id)
        demo_scoring(model, tokenizer, seed=cfg.seed)


if __name__ == "__main__":
    main()
