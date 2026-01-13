#!/usr/bin/env python3
"""
Generate multi-lane token strip diagrams for ORM and Value Function.

These diagrams show:
- Where targets come from (offline labels vs on-policy rollouts)
- What the model outputs
- How outputs are used (aggregate score vs advantage)

This makes ORM and Value Function visually distinct despite similar structure.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as path_effects
import numpy as np


def draw_token_box(ax, x, y, w, h, text, highlighted=False, masked=False):
    """Draw a single token box."""
    if highlighted:
        facecolor = "#90EE90"
        edgecolor = "#228B22"
        linewidth = 2
    elif masked:
        facecolor = "#D3D3D3"
        edgecolor = "#808080"
        linewidth = 1
    else:
        facecolor = "#FFFFFF"
        edgecolor = "#000000"
        linewidth = 1

    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(rect)

    text_color = "#606060" if masked else "#000000"
    txt = ax.text(
        x + w/2, y + h/2, text,
        ha="center", va="center",
        fontsize=8, fontfamily="monospace",
        color=text_color,
        fontweight="bold" if highlighted else "normal",
    )
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])


def draw_label_circle(ax, x, y, label, color="#4CAF50"):
    """Draw a small circle with label (for ORM labels)."""
    circle = plt.Circle((x, y), 0.15, facecolor=color, edgecolor="#2E7D32", linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white", fontweight="bold")


def draw_value_bar(ax, x, y, height, max_height, width=0.6):
    """Draw a bar for value estimate."""
    bar = Rectangle(
        (x - width/2, y), width, height * max_height,
        facecolor="#64B5F6", edgecolor="#1976D2", linewidth=1,
        alpha=0.8,
    )
    ax.add_patch(bar)


def render_orm_diagram(output_path: Path, fmt: str = "png", dpi: int = 150):
    """
    Render ORM diagram with 3 lanes:
    1. Tokens (prompt masked, completion highlighted)
    2. Labels from verifier (z_t ∈ {0,1})
    3. Model outputs p_t with BCE loss and aggregate
    """
    tokens = ["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"]
    n_tokens = len(tokens)
    prompt_end = 5  # tokens 0-4 are prompt

    fig, ax = plt.subplots(figsize=(12, 5))

    box_w, box_h = 0.8, 0.5
    lane_gap = 0.8

    # Lane positions (y coordinates)
    y_tokens = 3.0
    y_labels = 2.0
    y_outputs = 1.0

    # === Title ===
    ax.text(
        n_tokens * box_w / 2, 4.2,
        "Outcome RM: Offline Labels → Per-Token BCE",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )

    # === Lane 1: Tokens ===
    ax.text(-0.8, y_tokens + box_h/2, "Tokens", ha="right", va="center", fontsize=10, fontweight="bold")
    for i, tok in enumerate(tokens):
        x = i * box_w
        masked = i < prompt_end
        highlighted = i >= prompt_end
        draw_token_box(ax, x, y_tokens, box_w - 0.1, box_h, tok, highlighted=highlighted, masked=masked)

    # === Lane 2: Labels from verifier ===
    ax.text(-0.8, y_labels + 0.15, "Labels z_t", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(-0.8, y_labels - 0.15, "(from verifier)", ha="right", va="center", fontsize=8, color="#606060")

    # Draw "Verifier / Dataset" box
    verifier_box = FancyBboxPatch(
        (-2.5, y_labels - 0.3), 1.5, 0.8,
        boxstyle="round,pad=0.05",
        facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=2
    )
    ax.add_patch(verifier_box)
    ax.text(-1.75, y_labels + 0.1, "Verifier/\nDataset", ha="center", va="center", fontsize=8, fontweight="bold")

    # Arrow from verifier to labels
    arrow = FancyArrowPatch(
        (-1.0, y_labels + 0.1), (prompt_end * box_w - 0.3, y_labels + 0.1),
        arrowstyle="->,head_width=0.15", color="#FF9800", linewidth=2,
        connectionstyle="arc3,rad=0.1"
    )
    ax.add_patch(arrow)

    # Label circles for completion tokens
    for i in range(prompt_end, n_tokens):
        x = i * box_w + box_w/2 - 0.05
        draw_label_circle(ax, x, y_labels + 0.15, "1")  # All correct in this example

    # Masked indicator for prompt
    ax.text(prompt_end * box_w / 2 - 0.05, y_labels + 0.15, "−100 (masked)",
            ha="center", va="center", fontsize=8, color="#808080", style="italic")

    # === Lane 3: Model outputs ===
    ax.text(-0.8, y_outputs + 0.15, "Output p_t", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(-0.8, y_outputs - 0.15, "p(correct|t)", ha="right", va="center", fontsize=8, color="#606060")

    probs = [".92", ".88", ".95", ".99", ".97"]
    for i, p in enumerate(probs):
        x = (prompt_end + i) * box_w + box_w/2 - 0.05
        ax.text(x, y_outputs + 0.15, f"p={p}", ha="center", va="center",
                fontsize=8, color="#0066CC", fontweight="bold")

    # === Loss and Aggregate ===
    # BCE loss box
    loss_box = FancyBboxPatch(
        (n_tokens * box_w + 0.3, y_labels - 0.2), 1.8, 0.6,
        boxstyle="round,pad=0.05",
        facecolor="#FFEBEE", edgecolor="#F44336", linewidth=2
    )
    ax.add_patch(loss_box)
    ax.text(n_tokens * box_w + 1.2, y_labels + 0.1, "BCE(p_t, z_t)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#C62828")

    # Arrow to loss
    arrow_loss = FancyArrowPatch(
        ((n_tokens - 1) * box_w + 0.4, y_labels + 0.1),
        (n_tokens * box_w + 0.3, y_labels + 0.1),
        arrowstyle="->,head_width=0.12", color="#404040", linewidth=1.5
    )
    ax.add_patch(arrow_loss)

    # Aggregate box
    agg_box = FancyBboxPatch(
        (n_tokens * box_w + 0.3, y_outputs - 0.2), 1.8, 0.6,
        boxstyle="round,pad=0.05",
        facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=2
    )
    ax.add_patch(agg_box)
    ax.text(n_tokens * box_w + 1.2, y_outputs + 0.1, "Aggregate\nmean/last/min",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#2E7D32")

    # Arrow to aggregate
    arrow_agg = FancyArrowPatch(
        ((n_tokens - 1) * box_w + 0.4, y_outputs + 0.1),
        (n_tokens * box_w + 0.3, y_outputs + 0.1),
        arrowstyle="->,head_width=0.12", color="#404040", linewidth=1.5
    )
    ax.add_patch(arrow_agg)

    # Final score
    ax.text(n_tokens * box_w + 2.5, y_outputs + 0.1, "→ 0.94",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#2E7D32")

    # === Notes at bottom ===
    notes = "Offline supervision  •  Fixed dataset labels  •  Use: verify/filter/rerank"
    ax.text(n_tokens * box_w / 2, 0.2, notes,
            ha="center", va="center", fontsize=9, color="#606060",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#E0E0E0"))

    ax.set_xlim(-3, n_tokens * box_w + 3.5)
    ax.set_ylim(-0.2, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def render_value_diagram(output_path: Path, fmt: str = "png", dpi: int = 150):
    """
    Render Value Function diagram with 4 lanes:
    1. Tokens (prompt masked, completion highlighted)
    2. Rewards r_t (sparse, from RM + KL penalty)
    3. Return targets V̂_t (computed from rollouts)
    4. Model outputs V_t with regression loss and advantage
    """
    tokens = ["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"]
    n_tokens = len(tokens)
    prompt_end = 5

    fig, ax = plt.subplots(figsize=(12, 6.5))

    box_w, box_h = 0.8, 0.5

    # Lane positions
    y_tokens = 4.5
    y_rewards = 3.5
    y_targets = 2.5
    y_outputs = 1.5

    # === Title ===
    ax.text(
        n_tokens * box_w / 2, 5.7,
        "Value Function: On-Policy Rollouts → Regression → Advantage",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )

    # === Source indicator (on-policy) ===
    policy_box = FancyBboxPatch(
        (-2.8, y_tokens - 0.5), 1.8, 1.2,
        boxstyle="round,pad=0.05",
        facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=2
    )
    ax.add_patch(policy_box)
    ax.text(-1.9, y_tokens + 0.1, "πθ generates\n(on-policy)", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#1565C0")

    arrow_policy = FancyArrowPatch(
        (-1.0, y_tokens + 0.1), (-0.2, y_tokens + 0.1),
        arrowstyle="->,head_width=0.12", color="#1976D2", linewidth=2
    )
    ax.add_patch(arrow_policy)

    # === Lane 1: Tokens ===
    ax.text(-0.8, y_tokens + box_h/2, "Tokens", ha="right", va="center", fontsize=10, fontweight="bold")
    for i, tok in enumerate(tokens):
        x = i * box_w
        masked = i < prompt_end
        highlighted = i >= prompt_end
        draw_token_box(ax, x, y_tokens, box_w - 0.1, box_h, tok, highlighted=highlighted, masked=masked)

    # Rollout start indicator
    ax.annotate("", xy=(prompt_end * box_w - 0.1, y_tokens - 0.15),
                xytext=(prompt_end * box_w - 0.1, y_tokens + box_h + 0.15),
                arrowprops=dict(arrowstyle="-", color="#1976D2", lw=2, ls="--"))
    ax.text(prompt_end * box_w - 0.1, y_tokens + box_h + 0.25, "rollout starts",
            ha="center", va="bottom", fontsize=7, color="#1976D2", style="italic")

    # === Lane 2: Rewards (sparse) ===
    ax.text(-0.8, y_rewards + 0.15, "Rewards r_t", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(-0.8, y_rewards - 0.15, "(from RM)", ha="right", va="center", fontsize=8, color="#606060")

    # Sparse rewards - mostly 0, with RM reward at end and optional KL
    for i in range(prompt_end, n_tokens - 1):
        x = i * box_w + box_w/2 - 0.05
        ax.text(x, y_rewards + 0.1, "0", ha="center", va="center", fontsize=8, color="#909090")

    # Final reward from RM
    x_final = (n_tokens - 1) * box_w + box_w/2 - 0.05
    ax.text(x_final, y_rewards + 0.1, "R=0.73", ha="center", va="center",
            fontsize=9, color="#E65100", fontweight="bold")
    ax.text(x_final, y_rewards - 0.2, "−β·KL", ha="center", va="center",
            fontsize=7, color="#7B1FA2", style="italic")

    # === Lane 3: Return targets (computed) ===
    ax.text(-0.8, y_targets + 0.15, "Target V̂_t", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(-0.8, y_targets - 0.15, "(GAE/returns)", ha="right", va="center", fontsize=8, color="#606060")

    # Compute box
    compute_box = FancyBboxPatch(
        (-2.8, y_targets - 0.3), 1.8, 0.8,
        boxstyle="round,pad=0.05",
        facecolor="#F3E5F5", edgecolor="#9C27B0", linewidth=2
    )
    ax.add_patch(compute_box)
    ax.text(-1.9, y_targets + 0.1, "Compute\nreturns/GAE", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#7B1FA2")

    arrow_compute = FancyArrowPatch(
        (-1.0, y_targets + 0.1), (prompt_end * box_w - 0.3, y_targets + 0.1),
        arrowstyle="->,head_width=0.12", color="#9C27B0", linewidth=2
    )
    ax.add_patch(arrow_compute)

    # Return values (decreasing as we approach end)
    returns = [".73", ".73", ".73", ".73", ".73"]  # With γ=1, all equal to final R
    for i, ret in enumerate(returns):
        x = (prompt_end + i) * box_w + box_w/2 - 0.05
        ax.text(x, y_targets + 0.1, f"V̂={ret}", ha="center", va="center",
                fontsize=8, color="#7B1FA2", fontweight="bold")

    # === Lane 4: Model outputs ===
    ax.text(-0.8, y_outputs + 0.15, "Output V_t", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(-0.8, y_outputs - 0.15, "(learned)", ha="right", va="center", fontsize=8, color="#606060")

    # Value predictions
    values = [".45", ".55", ".62", ".68", ".71"]
    for i, v in enumerate(values):
        x = (prompt_end + i) * box_w + box_w/2 - 0.05
        ax.text(x, y_outputs + 0.1, f"V={v}", ha="center", va="center",
                fontsize=8, color="#0066CC", fontweight="bold")

    # === Loss box ===
    loss_box = FancyBboxPatch(
        (n_tokens * box_w + 0.3, y_targets - 0.2), 2.0, 0.6,
        boxstyle="round,pad=0.05",
        facecolor="#FFEBEE", edgecolor="#F44336", linewidth=2
    )
    ax.add_patch(loss_box)
    ax.text(n_tokens * box_w + 1.3, y_targets + 0.1, "MSE(V_t, V̂_t)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#C62828")

    # === Advantage box (key distinction!) ===
    adv_box = FancyBboxPatch(
        (n_tokens * box_w + 0.3, y_outputs - 0.3), 2.0, 0.8,
        boxstyle="round,pad=0.05",
        facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=2
    )
    ax.add_patch(adv_box)
    ax.text(n_tokens * box_w + 1.3, y_outputs + 0.15, "Advantage",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#2E7D32")
    ax.text(n_tokens * box_w + 1.3, y_outputs - 0.1, "A_t = V̂_t − V_t",
            ha="center", va="center", fontsize=8, color="#2E7D32")

    # Arrow to advantage
    arrow_adv = FancyArrowPatch(
        ((n_tokens - 1) * box_w + 0.4, y_outputs + 0.1),
        (n_tokens * box_w + 0.3, y_outputs + 0.1),
        arrowstyle="->,head_width=0.12", color="#404040", linewidth=1.5
    )
    ax.add_patch(arrow_adv)

    # Policy gradient usage
    ax.text(n_tokens * box_w + 2.7, y_outputs, "→ Policy\n   gradient",
            ha="left", va="center", fontsize=9, fontweight="bold", color="#2E7D32")

    # === Notes at bottom ===
    notes = "On-policy rollouts  •  Targets change with πθ  •  Use: baseline for advantage estimation"
    ax.text(n_tokens * box_w / 2, 0.5, notes,
            ha="center", va="center", fontsize=9, color="#606060",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#E0E0E0"))

    ax.set_xlim(-3.5, n_tokens * box_w + 4)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-lane token strip diagrams")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "generated",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf"],
        default="png",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    render_orm_diagram(args.output_dir / f"orm_multilane.{args.format}", fmt=args.format)
    render_value_diagram(args.output_dir / f"value_fn_multilane.{args.format}", fmt=args.format)

    print(f"\nGenerated multi-lane diagrams in {args.output_dir}")


if __name__ == "__main__":
    main()
