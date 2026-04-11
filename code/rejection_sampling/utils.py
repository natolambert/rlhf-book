# Shared utilities for rejection sampling
#
# Small, self-contained helpers: VRAM cleanup, cache-key hashing, and the
# GSM8k answer extraction/formatting used by both rollout scoring and eval.

import gc
import hashlib
import json
import re

import torch

from .config import Config


# AceMath-7B-RM was trained to score solutions written in this format. We must
# therefore use the same system prompt during Stage 1 generation so that the
# rollouts are in the distribution the reward model knows how to score.
ACEMATH_SYSTEM_PROMPT = (
    "Please reason step by step, and check your final answer within \\boxed{}."
)


def free_memory(*objs) -> None:
    """Delete a list of objects and release their VRAM.

    The caller should drop their own references as well (e.g. ``model = None``
    after calling this) — Python only frees the underlying object once the
    last reference goes away.
    """
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_memory_gb() -> float:
    """Return currently allocated CUDA memory in gigabytes (0 on CPU-only)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def cache_key(cfg: Config) -> str:
    """Short deterministic hash of every field that affects the rollout cache.

    Selection strategy is deliberately excluded — the three strategies share
    the same rollout+score cache and only differ in how they pick training
    pairs from it.
    """
    payload = {
        "reward_model_name": cfg.reward_model_name,
        "model_name": cfg.model_name,
        "dataset_name": cfg.data.name,
        "dataset_subset": cfg.data.subset,
        "train_split": cfg.data.train_split,
        "max_train_samples": cfg.data.max_train_samples,
        "num_completions_per_prompt": cfg.num_completions_per_prompt,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "max_new_tokens": cfg.max_new_tokens,
        "seed": cfg.seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def format_gsm8k_gold(answer_field: str) -> str:
    """Extract the gold numeric answer from a GSM8k ``answer`` field.

    GSM8k's gold answer lives after ``####`` at the end of the chain-of-thought.
    """
    if "####" in answer_field:
        gold = answer_field.split("####", 1)[1].strip()
    else:
        gold = answer_field.strip()
    return gold.replace(",", "")


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract a numeric answer from a model completion.

    Priority:
    1. Last ``\\boxed{...}`` group (matches the AceMath system prompt format).
    2. Last numeric substring in the completion (fallback for runs that drop
       the box).
    Commas are stripped so "1,000" and "1000" compare equal.
    """
    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        inside = boxed_matches[-1]
        numbers = _NUMBER_RE.findall(inside)
        if numbers:
            return numbers[-1].replace(",", "")
        return inside.strip().replace(",", "") or None

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None
