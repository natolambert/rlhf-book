# Direct Alignment Data Loading
#
# Utilities for loading and processing preference datasets.
# Supports common formats: UltraFeedback, Anthropic HH, custom paired data.

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


@dataclass
class PreferenceBatch:
    """A batch of preference data for DPO-style training."""

    chosen_input_ids: torch.Tensor  # (batch, seq_len)
    chosen_attention_mask: torch.Tensor  # (batch, seq_len)
    chosen_labels: torch.Tensor  # (batch, seq_len) - for computing loss
    chosen_response_mask: torch.Tensor  # (batch, seq_len) - 1 for response tokens, 0 for prompt/padding
    rejected_input_ids: torch.Tensor  # (batch, seq_len)
    rejected_attention_mask: torch.Tensor  # (batch, seq_len)
    rejected_labels: torch.Tensor  # (batch, seq_len)
    rejected_response_mask: torch.Tensor  # (batch, seq_len) - 1 for response tokens, 0 for prompt/padding

    def to(self, device: str | torch.device) -> "PreferenceBatch":
        """Move batch to device."""
        return PreferenceBatch(
            chosen_input_ids=self.chosen_input_ids.to(device),
            chosen_attention_mask=self.chosen_attention_mask.to(device),
            chosen_labels=self.chosen_labels.to(device),
            chosen_response_mask=self.chosen_response_mask.to(device),
            rejected_input_ids=self.rejected_input_ids.to(device),
            rejected_attention_mask=self.rejected_attention_mask.to(device),
            rejected_labels=self.rejected_labels.to(device),
            rejected_response_mask=self.rejected_response_mask.to(device),
        )


def format_chat_prompt(
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str | None = None,
) -> str:
    """Format a prompt-response pair using the tokenizer's chat template.

    Args:
        prompt: User message / question
        response: Assistant response
        tokenizer: Tokenizer with chat template
        system_prompt: Optional system message

    Returns:
        Formatted string ready for tokenization
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def format_prompt_only(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str | None = None,
) -> str:
    """Format just the prompt (without response) for computing prompt length.

    Args:
        prompt: User message / question
        tokenizer: Tokenizer with chat template
        system_prompt: Optional system message

    Returns:
        Formatted prompt string with generation prompt added
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Add the assistant turn start
    )


def extract_ultrafeedback_pairs(example: dict) -> dict:
    """Extract prompt and chosen/rejected from UltraFeedback format.

    UltraFeedback format:
    {
        "prompt": "...",
        "chosen": [{"role": "user", ...}, {"role": "assistant", ...}],
        "rejected": [{"role": "user", ...}, {"role": "assistant", ...}],
    }
    """
    prompt = example.get("prompt", "")

    # Handle both message list and direct response formats
    chosen = example.get("chosen", [])
    rejected = example.get("rejected", [])

    if isinstance(chosen, list):
        # Message list format
        chosen_response = ""
        for msg in chosen:
            if msg.get("role") == "assistant":
                chosen_response = msg.get("content", "")
                break
    else:
        chosen_response = str(chosen)

    if isinstance(rejected, list):
        rejected_response = ""
        for msg in rejected:
            if msg.get("role") == "assistant":
                rejected_response = msg.get("content", "")
                break
    else:
        rejected_response = str(rejected)

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def extract_anthropic_hh_pairs(example: dict) -> dict:
    """Extract from Anthropic HH format (conversation strings)."""
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    # Extract last turn as the response
    # Format: "Human: ... Assistant: ... Human: ... Assistant: ..."
    def get_last_assistant_turn(text: str) -> tuple[str, str]:
        parts = text.split("Assistant:")
        if len(parts) >= 2:
            response = parts[-1].strip()
            # Get everything before last assistant turn as prompt
            prompt = "Assistant:".join(parts[:-1]).strip()
            if prompt.startswith("Human:"):
                prompt = prompt[6:].strip()
            return prompt, response
        return text, ""

    prompt, chosen_response = get_last_assistant_turn(chosen)
    _, rejected_response = get_last_assistant_turn(rejected)

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def load_preference_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    """Load a preference dataset and normalize to common format.

    Returns dataset with columns: prompt, chosen, rejected

    Supported datasets:
    - argilla/ultrafeedback-binarized-preferences-cleaned
    - Anthropic/hh-rlhf
    - Any dataset with prompt/chosen/rejected columns
    """
    dataset = load_dataset(dataset_name, split=split)

    # Limit samples if requested
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Detect format and normalize
    columns = dataset.column_names

    if "chosen" in columns and isinstance(dataset[0]["chosen"], list):
        # UltraFeedback format (message lists)
        dataset = dataset.map(extract_ultrafeedback_pairs, remove_columns=columns)
    elif "chosen" in columns and "Human:" in str(dataset[0].get("chosen", "")):
        # Anthropic HH format
        dataset = dataset.map(extract_anthropic_hh_pairs, remove_columns=columns)
    elif all(col in columns for col in ["prompt", "chosen", "rejected"]):
        # Already in correct format
        pass
    else:
        raise ValueError(
            f"Unknown dataset format. Expected columns: prompt, chosen, rejected. "
            f"Got: {columns}"
        )

    return dataset


class PreferenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for preference data."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.dataset[idx]

        # Format with chat template
        chosen_text = format_chat_prompt(
            example["prompt"],
            example["chosen"],
            self.tokenizer,
        )
        rejected_text = format_chat_prompt(
            example["prompt"],
            example["rejected"],
            self.tokenizer,
        )

        # Get prompt-only text to compute prompt length
        prompt_only_text = format_prompt_only(example["prompt"], self.tokenizer)

        # Tokenize prompt-only to get prompt length (without padding)
        prompt_tokens = self.tokenizer(
            prompt_only_text,
            add_special_tokens=False,  # Chat template already adds them
            return_tensors="pt",
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Tokenize full sequences
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Create response masks (1 for response tokens, 0 for prompt/padding)
        # Response starts after prompt_len tokens
        seq_len = self.max_length
        chosen_response_mask = torch.zeros(seq_len, dtype=torch.long)
        rejected_response_mask = torch.zeros(seq_len, dtype=torch.long)

        # Mask is 1 where we have response tokens (after prompt, before padding)
        chosen_response_mask[prompt_len:] = chosen_tokens["attention_mask"].squeeze(0)[prompt_len:]
        rejected_response_mask[prompt_len:] = rejected_tokens["attention_mask"].squeeze(0)[prompt_len:]

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "chosen_response_mask": chosen_response_mask,
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
            "rejected_response_mask": rejected_response_mask,
        }


def collate_preference_batch(examples: list[dict]) -> PreferenceBatch:
    """Collate function for DataLoader."""
    chosen_input_ids = torch.stack([ex["chosen_input_ids"] for ex in examples])
    chosen_attention_mask = torch.stack([ex["chosen_attention_mask"] for ex in examples])
    chosen_response_mask = torch.stack([ex["chosen_response_mask"] for ex in examples])
    rejected_input_ids = torch.stack([ex["rejected_input_ids"] for ex in examples])
    rejected_attention_mask = torch.stack([ex["rejected_attention_mask"] for ex in examples])
    rejected_response_mask = torch.stack([ex["rejected_response_mask"] for ex in examples])

    # Labels are same as input_ids for autoregressive LM
    return PreferenceBatch(
        chosen_input_ids=chosen_input_ids,
        chosen_attention_mask=chosen_attention_mask,
        chosen_labels=chosen_input_ids.clone(),
        chosen_response_mask=chosen_response_mask,
        rejected_input_ids=rejected_input_ids,
        rejected_attention_mask=rejected_attention_mask,
        rejected_labels=rejected_input_ids.clone(),
        rejected_response_mask=rejected_response_mask,
    )


def create_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_samples: int | None = None,
    max_length: int = 512,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for preference data.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer for the model
        split: Dataset split
        max_samples: Limit number of samples
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader yielding PreferenceBatch objects
    """
    raw_dataset = load_preference_dataset(dataset_name, split, max_samples)
    torch_dataset = PreferenceDataset(raw_dataset, tokenizer, max_length)

    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_preference_batch,
        pin_memory=True,
        num_workers=0,  # Avoid multiprocessing issues with tokenizer
    )
