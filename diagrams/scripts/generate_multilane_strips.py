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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects


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


def render_orm_diagram(output_path: Path, fmt: str = "png", dpi: int = 150):
    """
    Render ORM diagram with 3 lanes - cleaner layout.
    """
    tokens = ["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"]
    n_tokens = len(tokens)
    prompt_end = 5

    fig, ax = plt.subplots(figsize=(12, 5))

    box_w, box_h = 0.8, 0.5
    x_offset = 1.5  # Start position for tokens

    # Lane positions (y coordinates)
    y_tokens = 3.0
    y_labels = 2.0
    y_outputs = 1.0

    # === Title with source indicator ===
    ax.text(
        x_offset + n_tokens * box_w / 2, 4.2,
        "Outcome RM: Offline Labels → Per-Token BCE",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )
    # Source indicator as subtitle
    ax.text(
        x_offset + n_tokens * box_w / 2, 3.95,
        "Labels from: Verifier / Dataset (offline)",
        ha="center", va="top", fontsize=9, color="#E65100", style="italic"
    )

    # === Lane 1: Tokens ===
    ax.text(x_offset - 0.2, y_tokens + box_h/2, "Tokens", ha="right", va="center",
            fontsize=10, fontweight="bold")
    for i, tok in enumerate(tokens):
        x = x_offset + i * box_w
        masked = i < prompt_end
        highlighted = i >= prompt_end
        draw_token_box(ax, x, y_tokens, box_w - 0.1, box_h, tok,
                      highlighted=highlighted, masked=masked)

    # === Lane 2: Labels ===
    ax.text(x_offset - 0.2, y_labels + 0.1, "Labels z_t", ha="right", va="center",
            fontsize=10, fontweight="bold")

    # Masked indicator for prompt
    prompt_center = x_offset + prompt_end * box_w / 2 - 0.05
    ax.text(prompt_center, y_labels + 0.1, "−100 (masked)",
            ha="center", va="center", fontsize=8, color="#808080", style="italic")

    # Label circles for completion tokens
    for i in range(prompt_end, n_tokens):
        x = x_offset + i * box_w + box_w/2 - 0.05
        draw_label_circle(ax, x, y_labels + 0.1, "1")

    # === Lane 3: Model outputs (what the model learns) ===
    # Add background highlight to show this is the learned output
    model_bg = FancyBboxPatch(
        (x_offset + prompt_end * box_w - 0.15, y_outputs - 0.2),
        (n_tokens - prompt_end) * box_w + 0.2, 0.6,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=1.5, linestyle="--"
    )
    ax.add_patch(model_bg)

    ax.text(x_offset - 0.2, y_outputs + 0.2, "Model", ha="right", va="center",
            fontsize=9, fontweight="bold", color="#1565C0")
    ax.text(x_offset - 0.2, y_outputs - 0.05, "predicts p_t", ha="right", va="center",
            fontsize=9, fontweight="bold", color="#1565C0")

    probs = [".92", ".88", ".95", ".99", ".97"]
    for i, p in enumerate(probs):
        x = x_offset + (prompt_end + i) * box_w + box_w/2 - 0.05
        ax.text(x, y_outputs + 0.1, f"p={p}", ha="center", va="center",
                fontsize=9, color="#0066CC", fontweight="bold")

    # === Right side: Loss and Usage ===
    right_x = x_offset + n_tokens * box_w + 0.3

    # BCE loss box
    loss_box = FancyBboxPatch(
        (right_x, y_labels - 0.15), 1.5, 0.5,
        boxstyle="round,pad=0.05",
        facecolor="#FFEBEE", edgecolor="#F44336", linewidth=2
    )
    ax.add_patch(loss_box)
    ax.text(right_x + 0.75, y_labels + 0.1, "BCE(p_t, z_t)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#C62828")

    # Aggregate box
    agg_box = FancyBboxPatch(
        (right_x, y_outputs - 0.15), 1.5, 0.5,
        boxstyle="round,pad=0.05",
        facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=2
    )
    ax.add_patch(agg_box)
    ax.text(right_x + 0.75, y_outputs + 0.1, "Aggregate",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#2E7D32")

    # Final score
    ax.text(right_x + 1.85, y_outputs + 0.1, "→ 0.94",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#2E7D32")

    # === Notes at bottom ===
    notes = "Offline supervision  •  Fixed dataset labels  •  Use: verify / filter / rerank"
    ax.text(x_offset + n_tokens * box_w / 2, 0.2, notes,
            ha="center", va="center", fontsize=9, color="#606060",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#E0E0E0"))

    ax.set_xlim(-0.3, x_offset + n_tokens * box_w + 3.2)
    ax.set_ylim(-0.2, 4.7)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def render_value_diagram(output_path: Path, fmt: str = "png", dpi: int = 150):
    """
    Render Value Function diagram with 4 lanes - cleaner layout.
    """
    tokens = ["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"]
    n_tokens = len(tokens)
    prompt_end = 5

    fig, ax = plt.subplots(figsize=(12, 6))

    box_w, box_h = 0.8, 0.5
    x_offset = 1.5

    # Lane positions
    y_tokens = 4.0
    y_rewards = 3.0
    y_targets = 2.0
    y_outputs = 1.0

    # === Title with source indicator ===
    ax.text(
        x_offset + n_tokens * box_w / 2, 5.2,
        "Value Function: On-Policy Rollouts → Regression → Advantage",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )
    # Source indicator as subtitle
    ax.text(
        x_offset + n_tokens * box_w / 2, 4.95,
        "Targets from: πθ rollouts + GAE (on-policy, changes during training)",
        ha="center", va="top", fontsize=9, color="#1565C0", style="italic"
    )

    # === Lane 1: Tokens ===
    ax.text(x_offset - 0.2, y_tokens + box_h/2, "Tokens", ha="right", va="center",
            fontsize=10, fontweight="bold")
    for i, tok in enumerate(tokens):
        x = x_offset + i * box_w
        masked = i < prompt_end
        highlighted = i >= prompt_end
        draw_token_box(ax, x, y_tokens, box_w - 0.1, box_h, tok,
                      highlighted=highlighted, masked=masked)

    # Rollout start indicator - positioned below the tokens row
    rollout_x = x_offset + prompt_end * box_w - 0.05
    ax.plot([rollout_x, rollout_x], [y_rewards + 0.3, y_tokens],
            color="#1976D2", linewidth=2, linestyle="--")
    ax.text(rollout_x + 0.05, y_rewards + 0.35, "↑ rollout starts",
            ha="left", va="bottom", fontsize=7, color="#1976D2", style="italic")

    # === Lane 2: Rewards (sparse) ===
    ax.text(x_offset - 0.2, y_rewards + 0.1, "Rewards r_t", ha="right", va="center",
            fontsize=10, fontweight="bold")

    # Sparse rewards
    for i in range(prompt_end, n_tokens - 1):
        x = x_offset + i * box_w + box_w/2 - 0.05
        ax.text(x, y_rewards + 0.1, "0", ha="center", va="center",
                fontsize=8, color="#909090")

    # Final reward
    x_final = x_offset + (n_tokens - 1) * box_w + box_w/2 - 0.05
    ax.text(x_final, y_rewards + 0.15, "R=0.73", ha="center", va="center",
            fontsize=9, color="#E65100", fontweight="bold")
    ax.text(x_final, y_rewards - 0.12, "−β·KL", ha="center", va="center",
            fontsize=7, color="#7B1FA2", style="italic")

    # === Lane 3: Return targets (GAE computed) ===
    ax.text(x_offset - 0.2, y_targets + 0.1, "Target V̂_t", ha="right", va="center",
            fontsize=10, fontweight="bold")

    # Return values (with γ=1, equal to final R)
    returns = [".73", ".73", ".73", ".73", ".73"]
    for i, ret in enumerate(returns):
        x = x_offset + (prompt_end + i) * box_w + box_w/2 - 0.05
        ax.text(x, y_targets + 0.1, f"V̂={ret}", ha="center", va="center",
                fontsize=8, color="#7B1FA2", fontweight="bold")

    # === Lane 4: Model outputs (what the model learns) ===
    # Add background highlight to show this is the learned output
    model_bg = FancyBboxPatch(
        (x_offset + prompt_end * box_w - 0.15, y_outputs - 0.2),
        (n_tokens - prompt_end) * box_w + 0.2, 0.6,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=1.5, linestyle="--"
    )
    ax.add_patch(model_bg)

    ax.text(x_offset - 0.2, y_outputs + 0.2, "Model", ha="right", va="center",
            fontsize=9, fontweight="bold", color="#1565C0")
    ax.text(x_offset - 0.2, y_outputs - 0.05, "predicts V_t", ha="right", va="center",
            fontsize=9, fontweight="bold", color="#1565C0")

    values = [".45", ".55", ".62", ".68", ".71"]
    for i, v in enumerate(values):
        x = x_offset + (prompt_end + i) * box_w + box_w/2 - 0.05
        ax.text(x, y_outputs + 0.1, f"V={v}", ha="center", va="center",
                fontsize=9, color="#0066CC", fontweight="bold")

    # === Right side: Loss and Advantage ===
    right_x = x_offset + n_tokens * box_w + 0.3

    # MSE loss box
    loss_box = FancyBboxPatch(
        (right_x, y_targets - 0.15), 1.6, 0.5,
        boxstyle="round,pad=0.05",
        facecolor="#FFEBEE", edgecolor="#F44336", linewidth=2
    )
    ax.add_patch(loss_box)
    ax.text(right_x + 0.8, y_targets + 0.1, "MSE(V_t, V̂_t)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#C62828")

    # Advantage box
    adv_box = FancyBboxPatch(
        (right_x, y_outputs - 0.15), 1.6, 0.5,
        boxstyle="round,pad=0.05",
        facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=2
    )
    ax.add_patch(adv_box)
    ax.text(right_x + 0.8, y_outputs + 0.15, "Advantage",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#2E7D32")
    ax.text(right_x + 0.8, y_outputs - 0.08, "A_t = V̂_t − V_t",
            ha="center", va="center", fontsize=8, color="#2E7D32")

    # Policy gradient usage
    ax.text(right_x + 2.0, y_outputs, "→ Policy\n   gradient",
            ha="left", va="center", fontsize=9, fontweight="bold", color="#2E7D32")

    # === Notes at bottom ===
    notes = "On-policy rollouts  •  Targets change with πθ  •  Use: baseline for advantage estimation"
    ax.text(x_offset + n_tokens * box_w / 2, 0.2, notes,
            ha="center", va="center", fontsize=9, color="#606060",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#E0E0E0"))

    ax.set_xlim(-0.3, x_offset + n_tokens * box_w + 3.5)
    ax.set_ylim(-0.2, 5.7)
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
