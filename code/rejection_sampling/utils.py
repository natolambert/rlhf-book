# Shared helpers: VRAM cleanup, cache-key hashing, GSM8k answer parsing.

import gc
import hashlib
import json
import re

import torch

from .config import Config


# AceMath-7B-RM is trained on solutions in this format, so we must use the
# same system prompt at generation time for the scores to be meaningful.
ACEMATH_SYSTEM_PROMPT = "Please reason step by step, and check your final answer within \\boxed{}."


def free_memory(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def cache_key(cfg: Config) -> str:
    """Hash of every field that affects the rollout cache.

    Selection strategy is excluded — all strategies share the same cache and
    only differ in how they pick training pairs from it.
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
    """GSM8k gold answers live after `####` at the end of the CoT."""
    gold = (
        answer_field.split("####", 1)[1].strip() if "####" in answer_field else answer_field.strip()
    )
    return gold.replace(",", "")


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> str | None:
    """Pull a numeric answer out of a completion — last `\\boxed{...}` first, then last number."""
    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        inside = boxed_matches[-1]
        numbers = _NUMBER_RE.findall(inside)
        if numbers:
            return numbers[-1].replace(",", "")
        return inside.strip().replace(",", "") or None

    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def answers_match(predicted: str | None, gold: str) -> bool:
    """Numeric comparison with string fallback for GSM8K answers."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return predicted.strip() == gold.strip()
