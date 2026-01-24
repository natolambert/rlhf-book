#!/usr/bin/env python3
"""Preference-based Reward Model Training (Bradley-Terry)

Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

This script trains a standard preference-based reward model using the
Bradley-Terry model. Given pairs of (chosen, rejected) responses, the model
learns to assign higher rewards to chosen responses.

The loss function is: -log(sigmoid(r_chosen - r_rejected))

This is the standard approach used in InstructGPT, Llama 2, and most RLHF
pipelines. See Chapter 7 (Reward Models) of RLHF Book for theoretical background.

Usage:
    uv run python -m reward_models.train_preference_rm
    uv run python -m reward_models.train_preference_rm --samples 5000 --epochs 1
"""

import argparse
import os
import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_DATASET = "argilla/ultrafeedback-binarized-preferences-cleaned"
DEFAULT_SAMPLES = 2000
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_LENGTH = 512
DEFAULT_EPOCHS = 1
DEFAULT_LR = 1e-5
DEFAULT_SEED = 42


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


def build_preference_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str = DEFAULT_DATASET,
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

    # Load dataset
    raw = load_dataset(dataset_name, split="train")

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
            chosen_text = format_conversation(chosen)
            rejected_text = format_conversation(rejected)
        elif isinstance(chosen, str):
            chosen_text = f"user: {prompt}\nassistant: {chosen}"
            rejected_text = f"user: {prompt}\nassistant: {rejected}"
        else:
            continue

        # Tokenize
        chosen_tokens = tokenizer(
            chosen_text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )
        rejected_tokens = tokenizer(
            rejected_text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )

        records.append({
            "chosen_ids": chosen_tokens["input_ids"],
            "chosen_mask": chosen_tokens["attention_mask"],
            "rejected_ids": rejected_tokens["input_ids"],
            "rejected_mask": rejected_tokens["attention_mask"],
        })

    return Dataset.from_list(records)


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Collate function that pads chosen and rejected sequences."""

    def pad_sequence(sequences, pad_value):
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded

    chosen_ids = pad_sequence([x["chosen_ids"] for x in batch], tokenizer.pad_token_id)
    chosen_mask = pad_sequence([x["chosen_mask"] for x in batch], 0)
    rejected_ids = pad_sequence([x["rejected_ids"] for x in batch], tokenizer.pad_token_id)
    rejected_mask = pad_sequence([x["rejected_mask"] for x in batch], 0)

    return {
        "chosen_ids": chosen_ids,
        "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids,
        "rejected_mask": rejected_mask,
    }


# =============================================================================
# Model Definition
# =============================================================================


class PreferenceRewardModel(nn.Module):
    """Preference-based Reward Model with LoRA fine-tuning.

    Architecture:
    - Base LLM (e.g., Qwen3) with 4-bit quantization
    - LoRA adapters on attention projections
    - Linear head mapping last hidden state to scalar reward

    The model outputs a single scalar reward for each sequence.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        device_map = {"": 0} if torch.cuda.is_available() else None
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        base = prepare_model_for_kbit_training(base)
        base.config.use_cache = False

        self.model = get_peft_model(base, lora_config)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)

    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute scalar reward for a sequence.

        Returns the reward from the last non-padding token position.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]

        # Get last non-padding position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, seq_lengths]

        reward = self.reward_head(last_hidden).squeeze(-1)
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
# Training
# =============================================================================


def train_preference_rm(
    model_id: str = DEFAULT_MODEL_ID,
    samples: int = DEFAULT_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = True,
) -> PreferenceRewardModel:
    """Train a preference-based reward model on UltraFeedback.

    Args:
        model_id: HuggingFace model ID for base model
        samples: Number of preference pairs to use
        batch_size: Training batch size
        grad_accum_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        use_wandb: Whether to log to wandb

    Returns:
        Trained PreferenceRewardModel
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb_project = os.environ.get("WANDB_PROJECT")
    if use_wandb and wandb_project:
        wandb.init(
            project=wandb_project,
            name=os.environ.get("WANDB_RUN_NAME", "preference_rm"),
            config={
                "model_id": model_id,
                "samples": samples,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "max_length": max_length,
                "epochs": epochs,
                "lr": lr,
            },
        )
    else:
        wandb.init(mode="disabled")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    print(f"Building preference dataset with {samples} pairs...")
    data = build_preference_dataset(tokenizer, limit=samples, max_length=max_length, seed=seed)
    print(f"Dataset size: {len(data)} pairs")

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Initialize model
    print(f"Loading model: {model_id}")
    model = PreferenceRewardModel(model_id=model_id).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # Mixed precision
    autocast_enabled = torch.cuda.is_available()

    # Training loop
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_pairs = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, r_chosen, r_rejected = model(**batch)

            (loss / grad_accum_steps).backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item()

            # Accuracy: how often is r_chosen > r_rejected?
            correct = (r_chosen > r_rejected).sum().item()
            total_correct += correct
            total_pairs += r_chosen.size(0)

            if step_idx % 50 == 0:
                acc = total_correct / max(1, total_pairs)
                print(f"Epoch {epoch} step {step_idx} | loss {loss.item():.4f} | acc {acc:.3f}")
                wandb.log({
                    "loss": loss.item(),
                    "accuracy": acc,
                    "r_chosen_mean": r_chosen.mean().item(),
                    "r_rejected_mean": r_rejected.mean().item(),
                    "reward_margin": (r_chosen - r_rejected).mean().item(),
                }, step=global_step)

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / max(1, total_pairs)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}")

    wandb.finish()
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

    # Format as conversations
    good_text = f"user: {prompt}\nassistant: {good_response}"
    bad_text = f"user: {prompt}\nassistant: {bad_response}"

    # Tokenize
    good_tokens = tokenizer(good_text, return_tensors="pt", max_length=512, truncation=True)
    bad_tokens = tokenizer(bad_text, return_tensors="pt", max_length=512, truncation=True)

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


def main():
    parser = argparse.ArgumentParser(
        description="Train preference-based reward model on UltraFeedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base model ID")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of preference pairs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--skip-demo", action="store_true", help="Skip scoring demo after training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    model = train_preference_rm(
        model_id=args.model_id,
        samples=args.samples,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )

    if not args.skip_demo:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        demo_scoring(model, tokenizer)


if __name__ == "__main__":
    main()
