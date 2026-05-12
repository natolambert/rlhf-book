# Shared helpers: VRAM cleanup, cache-key hashing, GSM8K answer parsing.

import gc
import hashlib
import json
import os
import platform
import random
import re
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config


# AceMath-7B-RM is trained on solutions in this format, so we must use the
# same system prompt at generation time for the scores to be meaningful.
ACEMATH_SYSTEM_PROMPT = "Please reason step by step, and check your final answer within \\boxed{}."


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_attn_implementation() -> str:
    """Determine the best attention implementation for this platform.

    Returns 'flash_attention_2' on x86_64 with flash-attn installed,
    otherwise 'sdpa' (PyTorch's native SDPA, faster on DGX Spark/Blackwell).
    """
    if platform.machine() != "x86_64":
        return "sdpa"  # aarch64 / DGX Spark - use SDPA with cuDNN

    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model(model_name: str, device_map: Any, gradient_checkpointing: bool = True):
    """Load model and tokenizer with automatic attention implementation selection."""
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    # Many decoder-only models (LLaMA, GPT-2) don't define pad_token
    # Set it to eos_token to enable batch padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def print_step_header(console: Console, step: int, total: int) -> None:
    """Print a header for the current training step."""
    console.rule(f"[bold cyan]STEP {step + 1}/{total}[/bold cyan]", style="cyan")


def progress_bar(console: Console) -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("*"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_model_info(console: Console, model) -> None:
    """Print model configuration information."""
    console.print(
        Panel(
            f"[bold magenta]Model:[/bold magenta] {model}\n"
            f"[dim]Parameters:[/dim] {sum(p.numel() for p in model.parameters()):,}\n"
            f"[dim]Device:[/dim] {model.device}",
            title="[bold magenta]Configuration[/bold magenta]",
            border_style="magenta",
        )
    )


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
    """GSM8K gold answers live after `####` at the end of the CoT."""
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
