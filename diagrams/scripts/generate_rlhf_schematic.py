#!/usr/bin/env python3
"""
Generate a simplified version of the Christiano et al. (2017) RLHF schematic.

Shows the core RL-from-human-preferences loop:
- RL algorithm <-> environment (observation / action)
- reward predictor -> RL algorithm (predicted reward)
- human feedback -> reward predictor (dotted, asynchronous)
- environment -> reward predictor (trajectories)

Based on Figure 1 from "Deep Reinforcement Learning from Human Preferences"
(Christiano et al., 2017).

Usage:
    uv run python scripts/generate_rlhf_schematic.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -- Colors --
BOX_FACE = "#FFFFFF"
BOX_EDGE = "#1A1A1A"
ARROW_COLOR = "#666666"
LABEL_COLOR = "#333333"
HUMAN_COLOR = "#E65100"  # orange accent for human feedback

# Consistent arrow style
ARROW_STYLE = dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.0, mutation_scale=16)


def draw_box(ax, cx, cy, w, h, label, fontsize=13, edge_color=BOX_EDGE):
    """Draw a rounded box centered at (cx, cy)."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor=BOX_FACE,
        edgecolor=edge_color,
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#1A1A1A",
        zorder=4,
    )
    return rect


def render_schematic(output_path: Path, fmt: str = "png", dpi: int = 200):
    """Render the Christiano et al. RLHF schematic."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Box positions (center x, center y)
    rl_x, rl_y = 2.0, 1.5
    env_x, env_y = 8.0, 1.5
    rp_x, rp_y = 5.0, 5.0
    human_x, human_y = 9.5, 5.0

    box_w, box_h = 2.8, 1.2

    # Draw boxes
    draw_box(ax, rl_x, rl_y, box_w, box_h, "RL Algorithm", fontsize=15)
    draw_box(ax, env_x, env_y, box_w, box_h, "Environment", fontsize=15)
    draw_box(ax, rp_x, rp_y, box_w, box_h, "Reward\nPredictor", fontsize=15)

    # -- Arrows (consistent style) --

    # Action: RL algorithm -> environment (bottom arrow, right)
    ax.annotate(
        "",
        xy=(env_x - box_w / 2 - 0.05, rl_y - 0.2),
        xytext=(rl_x + box_w / 2 + 0.05, rl_y - 0.2),
        arrowprops=ARROW_STYLE,
        zorder=2,
    )
    ax.text(
        (rl_x + env_x) / 2,
        rl_y - 0.6,
        "action",
        ha="center",
        va="top",
        fontsize=12,
        color=LABEL_COLOR,
    )

    # Observation: environment -> RL algorithm (top arrow, left)
    ax.annotate(
        "",
        xy=(rl_x + box_w / 2 + 0.05, rl_y + 0.2),
        xytext=(env_x - box_w / 2 - 0.05, rl_y + 0.2),
        arrowprops=ARROW_STYLE,
        zorder=2,
    )
    ax.text(
        (rl_x + env_x) / 2,
        rl_y + 0.6,
        "observation",
        ha="center",
        va="bottom",
        fontsize=12,
        color=LABEL_COLOR,
    )

    # Trajectories: environment -> reward predictor (curved arrow)
    ax.annotate(
        "",
        xy=(rp_x + box_w / 2 - 0.3, rp_y - box_h / 2 - 0.05),
        xytext=(env_x, env_y + box_h / 2 + 0.05),
        arrowprops=dict(**ARROW_STYLE, connectionstyle="arc3,rad=-0.25"),
        zorder=2,
    )
    ax.text(
        env_x - 0.2,
        (env_y + rp_y) / 2 + 0.15,
        "trajectories",
        ha="left",
        va="center",
        fontsize=12,
        color=LABEL_COLOR,
    )

    # Predicted reward: reward predictor -> RL algorithm (curved arrow)
    ax.annotate(
        "",
        xy=(rl_x, rl_y + box_h / 2 + 0.05),
        xytext=(rp_x - box_w / 2 + 0.3, rp_y - box_h / 2 - 0.05),
        arrowprops=dict(**ARROW_STYLE, connectionstyle="arc3,rad=0.25"),
        zorder=2,
    )
    ax.text(
        rl_x - 0.3,
        (rl_y + rp_y) / 2 + 0.1,
        "predicted\nreward",
        ha="right",
        va="center",
        fontsize=12,
        color=LABEL_COLOR,
        linespacing=1.2,
    )

    # Human feedback: human -> reward predictor (dotted arrow)
    ax.annotate(
        "",
        xy=(rp_x + box_w / 2 + 0.05, rp_y),
        xytext=(human_x - 0.5, human_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=HUMAN_COLOR,
            lw=2.0,
            mutation_scale=16,
            linestyle=(0, (3, 3)),  # dashed
        ),
        zorder=2,
    )
    ax.text(
        human_x + 0.15,
        human_y,
        "human\nfeedback",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=HUMAN_COLOR,
        linespacing=1.2,
    )

    # -- Axis cleanup --
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.2, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Christiano et al. RLHF schematic"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "generated" / "png",
    )
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = Path(__file__).parent.parent / "generated" / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    render_schematic(
        args.output_dir / f"rlhf_schematic.{args.format}",
        fmt=args.format,
        dpi=args.dpi,
    )
    if args.format == "png":
        render_schematic(svg_dir / "rlhf_schematic.svg", fmt="svg")


if __name__ == "__main__":
    main()
