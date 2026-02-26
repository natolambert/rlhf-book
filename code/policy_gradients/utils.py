# Utility Functions for Policy Gradient Training
#
# Original implementation by Zafir Stojanovski (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0

from __future__ import annotations

import random
from typing import Any

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


def speedrun_opts(
    enabled: bool = True,
    target_reward: float | None = None,
    metrics_file: str = "logs/speedrun/speedrun_metrics.json",
) -> dict[str, Any]:
    """Return speedrun options for main(). Use as: main(cfg, **speedrun_opts(target_reward=0.85))"""
    return {
        "speedrun": enabled,
        "speedrun_target_reward": target_reward,
        "speedrun_metrics_file": metrics_file,
    }


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


def print_rollout_sample(
    console: Console, reward: float, rollout_completions: list, reward_100avg: float | None = None
) -> None:
    """Print a sample from the rollouts with the average reward.

    From step 100 onwards, also displays the 100-step rolling average (reward_100avg).
    """
    sample_q, sample_a, sample_completion = random.choice(rollout_completions)
    reward_lines = [f"[bold green]Average Reward:[/bold green] {reward:.4f}"]
    if reward_100avg is not None:
        reward_lines.append(f"[bold yellow]100-step avg:[/bold yellow] {reward_100avg:.4f}")
    console.print(
        Panel(
            "\n".join(reward_lines),
            title="[bold cyan]Rollout Results[/bold cyan]",
            border_style="cyan",
        )
    )
    sample_preview = sample_completion[:1000]
    if len(sample_completion) > 1000:
        sample_preview += "[dim]... (truncated)[/dim]"
    sample_table = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    sample_table.add_column("Label", style="dim", width=12)
    sample_table.add_column("Content")
    sample_table.add_row("Question:", sample_q[:150] + ("..." if len(sample_q) > 150 else ""))
    sample_table.add_row("Oracle:", str(sample_a))
    sample_table.add_row("Completion:", sample_preview)
    console.print(Panel(sample_table, title="[bold cyan]Sample[/bold cyan]", border_style="dim"))
    console.print()
