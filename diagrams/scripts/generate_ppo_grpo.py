#!/usr/bin/env python3
"""
Generate PPO vs GRPO architecture comparison diagram.

Based on DeepSeek-Math/GRPO paper Figure 1.
Shows the key difference: PPO needs 4 models (Policy, Reference, Reward, Value)
while GRPO only needs 2 (Policy, Reward) by using group-based advantage estimation.

Usage:
    uv run python scripts/generate_ppo_grpo.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath
import numpy as np


# Colors matching the original diagram exactly
COLOR_TRAINED = "#FFF2CC"  # Light yellow for trained models
COLOR_TRAINED_BORDER = "#D6B656"  # Darker gold border
COLOR_FROZEN = "#DAE8FC"  # Light blue for frozen models
COLOR_FROZEN_BORDER = "#6C8EBF"  # Darker blue border
COLOR_WHITE = "#FFFFFF"
COLOR_BORDER = "#666666"
COLOR_TEXT = "#000000"
COLOR_LEGEND_BG = "#F5F5F5"


def draw_rounded_box(ax, x, y, w, h, fill, border, border_width=1.5, zorder=2):
    """Draw a rounded rectangle."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        facecolor=fill,
        edgecolor=border,
        linewidth=border_width,
        zorder=zorder,
    )
    ax.add_patch(rect)
    return rect


def draw_text(ax, x, y, text, fontsize=10, italic=False, bold=False, zorder=3, use_math=False):
    """Draw text at position."""
    if use_math:
        # Use mathtext for subscripts
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal", color=COLOR_TEXT, zorder=zorder)
    else:
        style = "italic" if italic else "normal"
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontstyle=style, fontweight=weight, color=COLOR_TEXT, zorder=zorder)


def draw_arrow(ax, start, end, dashed=False, zorder=1):
    """Draw an arrow between two points."""
    style = "Simple, tail_width=0.4, head_width=4, head_length=3"
    linestyle = (0, (4, 3)) if dashed else "-"

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color="#333333",
        linestyle=linestyle,
        linewidth=1.0,
        zorder=zorder,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)


def draw_curved_arrow(ax, start, end, rad=0.2, dashed=False, zorder=1):
    """Draw a curved arrow."""
    style = "Simple, tail_width=0.4, head_width=4, head_length=3"
    linestyle = (0, (4, 3)) if dashed else "-"

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color="#333333",
        linestyle=linestyle,
        linewidth=1.0,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_circle_node(ax, x, y, r, text, zorder=2):
    """Draw a circle with text."""
    circle = Circle((x, y), r, facecolor=COLOR_WHITE, edgecolor=COLOR_BORDER,
                    linewidth=1.5, zorder=zorder)
    ax.add_patch(circle)
    ax.text(x, y, text, ha="center", va="center", fontsize=12,
            color=COLOR_TEXT, zorder=zorder+1)


def draw_organic_boundary(ax, x, y, w, h, zorder=0):
    """Draw an organic curved boundary like in the original (pill/capsule shape)."""
    # Create a pill-shaped path
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.25",
        facecolor="none",
        edgecolor="#888888",
        linewidth=1.2,
        zorder=zorder,
    )
    ax.add_patch(rect)


def draw_ppo_section(ax, y_base: float):
    """Draw the PPO section."""

    # Section label
    ax.text(0.15, y_base + 1.7, "PPO", fontsize=24, fontweight="bold",
            color=COLOR_TEXT, ha="left", va="center")

    # Dimensions
    var_w, var_h = 0.38, 0.34
    model_w, model_h = 0.88, 0.52

    # q input
    draw_rounded_box(ax, 0.5, y_base + 0.82, var_w, var_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, 0.5 + var_w/2, y_base + 0.82 + var_h/2, "$q$", use_math=True)

    # Policy Model
    draw_rounded_box(ax, 1.35, y_base + 0.73, model_w, model_h,
                     COLOR_TRAINED, COLOR_TRAINED_BORDER, border_width=2)
    draw_text(ax, 1.35 + model_w/2, y_base + 0.73 + model_h/2, "Policy\nModel", fontsize=9)

    # o output
    draw_rounded_box(ax, 2.65, y_base + 0.82, var_w, var_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, 2.65 + var_w/2, y_base + 0.82 + var_h/2, "$o$", use_math=True)

    # Organic curved boundary around the 3 models
    draw_organic_boundary(ax, 3.35, y_base + 0.05, 1.1, 1.9)

    # Reference Model
    ref_y = y_base + 1.35
    draw_rounded_box(ax, 3.48, ref_y, model_w, model_h,
                     COLOR_FROZEN, COLOR_FROZEN_BORDER, border_width=2)
    draw_text(ax, 3.48 + model_w/2, ref_y + model_h/2, "Reference\nModel", fontsize=9)

    # Reward Model
    rew_y = y_base + 0.73
    draw_rounded_box(ax, 3.48, rew_y, model_w, model_h,
                     COLOR_FROZEN, COLOR_FROZEN_BORDER, border_width=2)
    draw_text(ax, 3.48 + model_w/2, rew_y + model_h/2, "Reward\nModel", fontsize=9)

    # Value Model
    val_y = y_base + 0.12
    draw_rounded_box(ax, 3.48, val_y, model_w, model_h,
                     COLOR_TRAINED, COLOR_TRAINED_BORDER, border_width=2)
    draw_text(ax, 3.48 + model_w/2, val_y + model_h/2, "Value\nModel", fontsize=9)

    # ⊕ operator - positioned to the right of the boundary
    plus_x, plus_y = 4.9, y_base + 1.25
    draw_circle_node(ax, plus_x, plus_y, 0.14, "⊕")

    # KL label - above/right of the dashed arrow
    ax.text(4.7, y_base + 1.65, "$KL$", fontsize=10, ha="center", va="bottom",
            color=COLOR_TEXT, fontstyle="italic")

    # r output
    r_x = 5.35
    draw_rounded_box(ax, r_x, y_base + 1.08, var_w, var_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, r_x + var_w/2, y_base + 1.08 + var_h/2, "$r$", use_math=True)

    # v output
    v_x = 5.35
    draw_rounded_box(ax, v_x, y_base + 0.22, var_w, var_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, v_x + var_w/2, y_base + 0.22 + var_h/2, "$v$", use_math=True)

    # GAE box
    gae_x, gae_y = 6.1, y_base + 0.65
    draw_rounded_box(ax, gae_x, gae_y, 0.58, 0.44, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, gae_x + 0.29, gae_y + 0.22, "GAE", fontsize=10)

    # A output
    draw_rounded_box(ax, 7.0, y_base + 0.73, var_w, var_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, 7.0 + var_w/2, y_base + 0.73 + var_h/2, "$A$", use_math=True)

    # Arrows
    # q -> Policy
    draw_arrow(ax, (0.5 + var_w, y_base + 0.99), (1.35, y_base + 0.99))

    # Policy -> o
    draw_arrow(ax, (1.35 + model_w, y_base + 0.99), (2.65, y_base + 0.99))

    # o -> boundary (single arrow entering the group)
    draw_arrow(ax, (2.65 + var_w, y_base + 0.99), (3.35, y_base + 0.99))

    # Inside: arrows to each model (curved)
    # These fan out from the entry point
    entry_x = 3.42
    draw_curved_arrow(ax, (entry_x, y_base + 0.99), (3.48, ref_y + model_h/2), rad=-0.2)
    draw_arrow(ax, (entry_x, y_base + 0.99), (3.48, rew_y + model_h/2))
    draw_curved_arrow(ax, (entry_x, y_base + 0.99), (3.48, val_y + model_h/2), rad=0.2)

    # Reference -> ⊕ (dashed curved arrow going up then to plus)
    draw_curved_arrow(ax, (3.48 + model_w, ref_y + model_h/2),
                      (plus_x - 0.1, plus_y + 0.08), rad=-0.25, dashed=True)

    # Reward -> ⊕
    draw_curved_arrow(ax, (3.48 + model_w, rew_y + model_h/2),
                      (plus_x - 0.1, plus_y - 0.05), rad=-0.1)

    # ⊕ -> r
    draw_arrow(ax, (plus_x + 0.14, plus_y), (r_x, y_base + 1.25))

    # Value -> v
    draw_arrow(ax, (3.48 + model_w, val_y + model_h/2), (v_x, y_base + 0.39))

    # r -> GAE (curved down)
    draw_curved_arrow(ax, (r_x + var_w, y_base + 1.25), (gae_x, gae_y + 0.36), rad=-0.2)

    # v -> GAE (curved up)
    draw_curved_arrow(ax, (v_x + var_w, y_base + 0.39), (gae_x, gae_y + 0.08), rad=0.2)

    # GAE -> A
    draw_arrow(ax, (gae_x + 0.58, gae_y + 0.22), (7.0, y_base + 0.9))


def draw_grpo_section(ax, y_base: float):
    """Draw the GRPO section."""

    # Section label
    ax.text(0.15, y_base + 1.5, "GRPO", fontsize=24, fontweight="bold",
            color=COLOR_TEXT, ha="left", va="center")

    # Dimensions
    var_w, var_h = 0.38, 0.34
    small_w, small_h = 0.38, 0.32
    model_w, model_h = 0.88, 0.52

    # q input
    draw_rounded_box(ax, 0.5, y_base + 0.68, var_w, var_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, 0.5 + var_w/2, y_base + 0.68 + var_h/2, "$q$", use_math=True)

    # Policy Model
    draw_rounded_box(ax, 1.35, y_base + 0.59, model_w, model_h,
                     COLOR_TRAINED, COLOR_TRAINED_BORDER, border_width=2)
    draw_text(ax, 1.35 + model_w/2, y_base + 0.59 + model_h/2, "Policy\nModel", fontsize=9)

    # Output group box (o₁, o₂, ..., oG) - YELLOW background like trained
    group_x = 2.55
    draw_rounded_box(ax, group_x, y_base + 0.12, 0.55, 1.28,
                     COLOR_TRAINED, COLOR_TRAINED_BORDER, border_width=1.5, zorder=1)

    # Individual outputs inside the group
    ox = group_x + 0.085
    draw_rounded_box(ax, ox, y_base + 1.0, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, ox + small_w/2, y_base + 1.0 + small_h/2, "$o_1$", fontsize=9, use_math=True)

    draw_rounded_box(ax, ox, y_base + 0.6, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, ox + small_w/2, y_base + 0.6 + small_h/2, "$o_2$", fontsize=9, use_math=True)

    ax.text(ox + small_w/2, y_base + 0.45, "⋮", fontsize=14,
            ha="center", va="center", color=COLOR_TEXT)

    draw_rounded_box(ax, ox, y_base + 0.2, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, ox + small_w/2, y_base + 0.2 + small_h/2, "$o_G$", fontsize=9, use_math=True)

    # Reference Model (just receives KL, no output)
    ref_x = 3.5
    ref_y = y_base + 0.98
    draw_rounded_box(ax, ref_x, ref_y, model_w, model_h,
                     COLOR_FROZEN, COLOR_FROZEN_BORDER, border_width=2)
    draw_text(ax, ref_x + model_w/2, ref_y + model_h/2, "Reference\nModel", fontsize=9)

    # Reward Model
    rew_y = y_base + 0.32
    draw_rounded_box(ax, ref_x, rew_y, model_w, model_h,
                     COLOR_FROZEN, COLOR_FROZEN_BORDER, border_width=2)
    draw_text(ax, ref_x + model_w/2, rew_y + model_h/2, "Reward\nModel", fontsize=9)

    # KL label - above the dashed arrow
    ax.text(3.25, y_base + 1.38, "$KL$", fontsize=10, ha="center", va="bottom",
            color=COLOR_TEXT, fontstyle="italic")

    # Reward outputs (r₁, r₂, ..., rG) - NO group box, just individual white boxes
    r_x = 4.75
    draw_rounded_box(ax, r_x, y_base + 1.0, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, r_x + small_w/2, y_base + 1.0 + small_h/2, "$r_1$", fontsize=9, use_math=True)

    draw_rounded_box(ax, r_x, y_base + 0.6, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, r_x + small_w/2, y_base + 0.6 + small_h/2, "$r_2$", fontsize=9, use_math=True)

    ax.text(r_x + small_w/2, y_base + 0.45, "⋮", fontsize=14,
            ha="center", va="center", color=COLOR_TEXT)

    draw_rounded_box(ax, r_x, y_base + 0.2, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, r_x + small_w/2, y_base + 0.2 + small_h/2, "$r_G$", fontsize=9, use_math=True)

    # Group Computation box
    gc_x, gc_y = 5.5, y_base + 0.48
    draw_rounded_box(ax, gc_x, gc_y, 0.9, 0.55, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, gc_x + 0.45, gc_y + 0.275, "Group\nComputation", fontsize=8)

    # Advantage outputs (A₁, A₂, ..., AG) - BLUE group background
    a_x = 6.75
    draw_rounded_box(ax, a_x, y_base + 0.12, 0.55, 1.28,
                     COLOR_FROZEN, COLOR_FROZEN_BORDER, border_width=1.5, zorder=1)

    ax2 = a_x + 0.085
    draw_rounded_box(ax, ax2, y_base + 1.0, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, ax2 + small_w/2, y_base + 1.0 + small_h/2, "$A_1$", fontsize=9, use_math=True)

    draw_rounded_box(ax, ax2, y_base + 0.6, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, ax2 + small_w/2, y_base + 0.6 + small_h/2, "$A_2$", fontsize=9, use_math=True)

    ax.text(ax2 + small_w/2, y_base + 0.45, "⋮", fontsize=14,
            ha="center", va="center", color=COLOR_TEXT)

    draw_rounded_box(ax, ax2, y_base + 0.2, small_w, small_h, COLOR_WHITE, COLOR_BORDER)
    draw_text(ax, ax2 + small_w/2, y_base + 0.2 + small_h/2, "$A_G$", fontsize=9, use_math=True)

    # Arrows
    # q -> Policy
    draw_arrow(ax, (0.5 + var_w, y_base + 0.85), (1.35, y_base + 0.85))

    # Policy -> outputs group
    draw_arrow(ax, (1.35 + model_w, y_base + 0.85), (group_x, y_base + 0.76))

    # outputs -> Reference (dashed KL)
    draw_curved_arrow(ax, (group_x + 0.55, y_base + 1.16),
                      (ref_x, ref_y + model_h/2), rad=-0.15, dashed=True)

    # outputs -> Reward
    draw_arrow(ax, (group_x + 0.55, y_base + 0.58), (ref_x, rew_y + model_h/2))

    # Reward -> r outputs
    draw_arrow(ax, (ref_x + model_w, rew_y + model_h/2), (r_x, y_base + 0.76))

    # r outputs -> Group Computation
    draw_arrow(ax, (r_x + small_w, y_base + 0.76), (gc_x, gc_y + 0.275))

    # Group Computation -> A outputs
    draw_arrow(ax, (gc_x + 0.9, gc_y + 0.275), (a_x, y_base + 0.76))


def draw_legend(ax, x, y):
    """Draw the legend matching original style."""
    # Background - light gray
    draw_rounded_box(ax, x, y, 0.95, 0.9, COLOR_LEGEND_BG, "#CCCCCC", border_width=1, zorder=5)

    # Trained models - yellow box
    draw_rounded_box(ax, x + 0.08, y + 0.52, 0.38, 0.28,
                     COLOR_TRAINED, COLOR_TRAINED_BORDER, border_width=2, zorder=6)
    ax.text(x + 0.55, y + 0.66, "Trained\nModels", fontsize=8, va="center", ha="left",
            color=COLOR_TEXT, zorder=6)

    # Frozen models - blue box
    draw_rounded_box(ax, x + 0.08, y + 0.12, 0.38, 0.28,
                     COLOR_FROZEN, COLOR_FROZEN_BORDER, border_width=2, zorder=6)
    ax.text(x + 0.55, y + 0.26, "Frozen\nModels", fontsize=8, va="center", ha="left",
            color=COLOR_TEXT, zorder=6)


def render_diagram(output_path: Path, fmt: str = "png", dpi: int = 150):
    """Render the complete PPO vs GRPO comparison diagram."""

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Draw PPO section (top)
    draw_ppo_section(ax, y_base=2.5)

    # Draw separator line (dashed)
    ax.axhline(y=2.35, xmin=0.01, xmax=0.99, color="#AAAAAA",
               linestyle="--", linewidth=1.5, dashes=(10, 5))

    # Draw GRPO section (bottom)
    draw_grpo_section(ax, y_base=0.35)

    # Draw legend
    draw_legend(ax, 7.65, 3.55)

    # Set axis properties
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Save
    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PPO vs GRPO architecture diagram"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "generated",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output format",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"ppo_vs_grpo.{args.format}"
    render_diagram(output_path, fmt=args.format)


if __name__ == "__main__":
    main()
