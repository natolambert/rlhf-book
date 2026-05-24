import os
import platform
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from .loss import SDPOLoss


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
    """Pick the best attention implementation for this platform."""
    if platform.machine() != "x86_64":
        return "sdpa"  # aarch64 / DGX Spark
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def get_loss_objective(loss: str, **kwargs) -> nn.Module:
    """Get the loss module for the specified algorithm."""
    if loss == "sdpo":
        return SDPOLoss(**kwargs)
    raise ValueError(f"Unsupported loss type: {loss}")


def load_model(model_name: str, device_map: Any, gradient_checkpointing: bool = True):
    """Load model and tokenizer with automatic attention implementation selection."""
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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


def print_model_info(model) -> None:
    console.print(
        Panel(
            f"[dim]Parameters:[/dim] {sum(p.numel() for p in model.parameters()):,}\n"
            f"[dim]Device:[/dim] {model.device}",
            title="[bold magenta]Configuration[/bold magenta]",
            border_style="magenta",
        )
    )


def print_step_header(step: int, total: int) -> None:
    console.rule(f"[bold cyan]Step {step + 1}/{total}[/bold cyan]", style="cyan")


def print_rollout_sample(
    problem_id: str, reward: float, acc: float, feedback: str, completion: str
) -> None:
    """Print a rollout group's reward/accuracy plus one sampled completion."""
    console.print(
        Panel(
            f"[bold green]reward:[/bold green] {reward:.4f}    "
            f"[bold green]acc:[/bold green] {acc:.4f}",
            title="[bold cyan]Rollout Results[/bold cyan]",
            border_style="cyan",
        )
    )

    def preview(text: str, limit: int = 1000) -> str:
        return text if len(text) <= limit else text[:limit] + "[dim]... (truncated)[/dim]"

    table = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    table.add_column("Label", style="dim", width=12)
    table.add_column("Content")
    table.add_row("Problem:", str(problem_id))
    table.add_row("Feedback:", preview(feedback) or "[dim](none)[/dim]")
    table.add_row("Completion:", preview(completion))
    console.print(Panel(table, title="[bold cyan]Sample[/bold cyan]", border_style="dim"))
    console.print()
