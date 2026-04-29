# Utility Functions for Policy Gradient Training
#
# Original implementation by Zafir Stojanovski (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0

import os
import platform
import random
import re
from typing import Any

import numpy as np
import reasoning_gym as rg
import torch
import torch.nn as nn
import torch.nn.functional as F
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer
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
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from .buffer import ReplayBuffer
from .config import Config
from .loss import (
    CISPOLoss,
    DAPOLoss,
    GRPOLoss,
    GSPOLoss,
    PPOLoss,
    ReinforceLoss,
    SAPOLoss,
    get_approx_kl,
    masked_mean,
)


console = Console()


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


def get_ref_model(model_name: str, device_map: Any, beta: float):
    """Load reference model for KL penalty (only if beta > 0)."""
    if not beta:
        return None
    ref_model, _ = load_model(model_name, device_map, gradient_checkpointing=False)
    ref_model.eval()
    return ref_model


def get_val_model(model_name: str, device_map: Any, loss: str, gradient_checkpointing: bool = True):
    """Load value model."""
    if loss not in ["ppo"]:
        return None
    val_model, _ = load_model(model_name, device_map, gradient_checkpointing)
    val_model.lm_head = nn.Linear(
        val_model.lm_head.in_features, 1, bias=False, device=val_model.device, dtype=torch.bfloat16
    )
    return val_model


def get_loss_objective(loss: str, **kwargs) -> nn.Module:
    """Get the loss function module for the specified algorithm."""
    if loss in ["grpo", "drgrpo"]:
        return GRPOLoss(**kwargs)
    elif loss == "gspo":
        return GSPOLoss(**kwargs)
    elif loss in ["rloo", "reinforce"]:
        return ReinforceLoss(**kwargs)
    elif loss == "cispo":
        return CISPOLoss(**kwargs)
    elif loss == "sapo":
        return SAPOLoss(**kwargs)
    elif loss == "ppo":
        return PPOLoss(**kwargs)
    elif loss == "dapo":
        return DAPOLoss(**kwargs)
    raise ValueError(f"Unsupported loss type: {loss}")


def create_dataset(cfg: Config) -> ProceduralDataset:
    """Create the training dataset from config."""
    specs = [DatasetSpec(name=s.name, weight=s.weight, config=s.config) for s in cfg.data.specs]
    return rg.create_dataset("composite", size=cfg.data.size, seed=cfg.seed, datasets=specs)


def apply_reward_kl(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: torch.Tensor,
    beta: float,
    loss: str,
    kl_estimator: str,
) -> torch.Tensor:
    """Apply KL penalty to rewards (for REINFORCE/RLOO/PPO)."""
    if not beta or loss not in ["ppo", "rloo", "reinforce"]:
        return rewards
    log_probs = log_probs.to(rewards.device)
    log_probs_ref = log_probs_ref.to(rewards.device)
    action_mask = action_mask.to(rewards.device)
    kl_div = get_approx_kl(kl_estimator, log_probs, log_probs_ref, action_mask)
    kl_div = masked_mean(kl_div, mask=action_mask, dim=-1, keepdim=True)
    rewards = rewards - beta * kl_div
    return rewards


def compute_standardized_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute standardized advantages (GRPO, GSPO, CISPO, DAPO)"""
    return (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + eps)


def compute_nonstandardized_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute non-standardized advantages (Dr. GRPO)."""
    return rewards - rewards.mean(dim=0, keepdim=True)


def compute_loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute leave-one-out advantages (RLOO)."""
    K = rewards.shape[0]
    return (K / (K - 1)) * (rewards - rewards.mean(dim=0, keepdim=True))


def compute_gae(
    rewards: torch.Tensor, action_mask: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (PPO)."""
    B, S = action_mask.size()
    device = action_mask.device
    last_action_indices = action_mask.long().cumsum(dim=-1).argmax(dim=-1, keepdim=True)
    indices = torch.arange(S, device=device).unsqueeze(0)
    done = (indices >= last_action_indices).float()

    rewards = torch.zeros_like(action_mask, device=device, dtype=torch.float32).scatter_(
        dim=-1, index=last_action_indices, src=rewards
    )

    values = values.to(device)
    advantages = torch.zeros_like(action_mask, dtype=torch.float32, device=device)
    next_values = torch.zeros(B, device=device, dtype=torch.float32)
    running = torch.zeros(B, device=device, dtype=torch.float32)

    for t in reversed(range(S)):
        not_done = 1.0 - done[:, t]
        delta = rewards[:, t] + not_done * gamma * next_values - values[:, t]
        running = delta + not_done * gamma * lam * running
        advantages[:, t] = running
        next_values = values[:, t]

    advantages = advantages * action_mask
    return advantages


def compute_advantages(
    rewards: torch.Tensor,
    loss: str,
    action_mask: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    gamma: float | None = None,
    lam: float | None = None,
) -> torch.Tensor:
    """Compute advantages using the appropriate method for the loss function."""
    if loss in ["grpo", "gspo", "cispo", "sapo", "dapo"]:
        return compute_standardized_advantages(rewards)
    elif loss in ["drgrpo"]:
        return compute_nonstandardized_advantages(rewards)
    elif loss in ["rloo"]:
        return compute_loo_advantages(rewards)
    elif loss in ["ppo"]:
        if action_mask is None or values is None or gamma is None or lam is None:
            raise ValueError("PPO requires action_mask, values, gamma, and lam to compute GAE.")
        return compute_gae(rewards, action_mask, values, gamma, lam)
    else:
        return rewards


def compute_log_probs(
    model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor | None:
    """Compute log probabilities for each token in the sequence."""
    if not model:
        return None
    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(model.device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    return target_log_probs


def compute_values(
    model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor | None:
    """Compute value estimates for each position (PPO)."""
    if not model:
        return None
    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(model.device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    values = output.logits[:, :-1, :].squeeze(-1).to(torch.float32)
    return values


def _correctness_reward(
    dataset: ProceduralDataset,
    completions: list[str],
    entries: list[dict],
) -> list[float]:
    """Compute raw reward scores from the environment"""

    def score_correctness_answer(dataset: ProceduralDataset, completion: str, entry: dict) -> float:
        answer = extract_answer(completion)
        return float(dataset.score_answer(answer, entry))

    return [
        score_correctness_answer(dataset=dataset, completion=completion, entry=entry)
        for completion, entry in zip(completions, entries, strict=True)
    ]


def _response_penalties(lengths: list[int], cfg: Config) -> list[float]:
    """Compute penalties of responses (e.g. based on length)"""

    def dapo_length_penalty(completion_len: int, l_cache: int, l_max: int) -> float:
        safe_len = l_max - l_cache
        if completion_len <= safe_len:
            return 0.0
        if completion_len <= l_max:
            return (safe_len - completion_len) / l_cache
        return -1.0

    if cfg.loss == "dapo":
        return [dapo_length_penalty(length, cfg.l_cache, cfg.l_max) for length in lengths]
    return [0 for _ in lengths]


def _format_reward(completions: list[str]) -> list[float]:
    """Compute format reward based on presence of thinking/answer tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        if re.search(r"\s*<think>\s*", text):
            count += 0.25
        if re.search(r"\s*</think>\s*", text):
            count += 0.25
        if re.search(r"\s*<answer>\s*", text):
            count += 0.25
        if re.search(r"\s*</answer>\s*", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


def compute_rewards(
    entries: list[dict],
    completions: list[str],
    lengths: list[int],
    dataset: ProceduralDataset,
    cfg: Config,
) -> tuple[list[float], list[float]]:
    """Compute training rewards and raw correctness rewards for filtering."""
    correctness_rewards = _correctness_reward(dataset, completions, entries)
    response_penalties = _response_penalties(lengths, cfg)
    format_rewards = _format_reward(completions)
    combined_rewards = [
        acc + pen + cfg.format_weight * fmt
        for acc, pen, fmt in zip(
            correctness_rewards, response_penalties, format_rewards, strict=True
        )
    ]
    return combined_rewards, correctness_rewards


def print_step_header(consumed: int, total: int) -> None:
    """Print a header showing dataset progress."""
    pct = 100.0 * consumed / total if total else 0.0
    console.rule(f"[bold cyan]{consumed}/{total} ({pct:.1f}%)[/bold cyan]", style="cyan")


def progress_bar() -> Progress:
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


def print_model_info(model) -> None:
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


def print_rollout_sample(buf: ReplayBuffer, tokenizer) -> None:
    """Print step-level avg reward / correctness plus one randomly sampled experience."""
    if len(buf) == 0:
        return
    avg_reward = torch.stack([e.rewards for e in buf.buffer]).mean().item()
    avg_correctness = torch.stack([e.correctness for e in buf.buffer]).mean().item()

    exp = random.choice(buf.buffer)
    target_ids = exp.sequence_ids[1:]
    action = exp.action_mask.bool()
    prompt = tokenizer.decode(target_ids[~action], skip_special_tokens=True)
    completion = tokenizer.decode(target_ids[action], skip_special_tokens=True)
    correctness = exp.correctness.item()

    console.print(
        Panel(
            f"[bold green]Avg Reward:[/bold green] {avg_reward:.4f}    "
            f"[bold green]Avg Correctness:[/bold green] {avg_correctness:.4f}",
            title="[bold cyan]Rollout Results[/bold cyan]",
            border_style="cyan",
        )
    )

    def preview(text: str, limit: int = 1000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "[dim]... (truncated)[/dim]"

    _, _, rest = prompt.partition("\nuser\n")
    user_part = rest.partition("\nassistant\n")[0].rstrip() if rest else prompt

    table = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    table.add_column("Label", style="dim", width=12)
    table.add_column("Content")
    table.add_row("User:", user_part)
    table.add_row("Assistant:", preview(completion))
    table.add_row("Correctness:", f"{correctness:.2f}")
    console.print(Panel(table, title="[bold cyan]Sample[/bold cyan]", border_style="dim"))
    console.print()
