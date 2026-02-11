#!/usr/bin/env python3
"""
Generate annotated trajectory equation diagram for the thermostat example.

Maps each symbol in the trajectory distribution equation to its
thermostat interpretation:

  p_π(τ) = ρ₀(s₀) · ∏ π(aₜ|sₜ) · p(s_{t+1}|sₜ,aₜ)

Usage:
    uv run python scripts/generate_thermostat_equation.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl


# Colors matching the book diagram style
COLOR_MATH = "#2C3E50"
COLOR_ANNOTATION = "#4A90D9"
COLOR_ARROW = "#7F8C8D"
COLOR_BG = "white"


def render_diagram(output_path: Path, fmt: str = "png", dpi: int = 300):
    """Render the annotated trajectory equation diagram."""

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)

    # Main equation
    eq_y = 0.75
    ax.text(
        0.5, eq_y,
        r"$p_{\pi}(\tau) \;=\; \rho_0(s_0)"
        r"\;\prod_{t=0}^{T-1}"
        r"\;\pi(a_t \mid s_t)"
        r"\;\, p(s_{t+1} \mid s_t, a_t)$",
        fontsize=20,
        color=COLOR_MATH,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    # Annotation positions (in axes coords) — tuned to sit under each term
    # 4 annotations: trajectory, initial state, policy, transition
    annotations = [
        (0.08, 0.08, [
            r"Trajectory probability:",
            r"sequence of temps,",
            r"actions, and outcomes",
        ]),
        (0.25, 0.25, [
            r"Prob. of starting at",
            r"initial temp, e.g.",
            r"P($s_0$ = 65°F)",
        ]),
        (0.50, 0.50, [
            r"Policy: prob. of action",
            r"given temp, e.g.",
            r"P(heater on | 65°F) = 0.9",
        ]),
        (0.79, 0.79, [
            r"Transition: prob. of next",
            r"temp given current temp",
            r"and action (room physics)",
        ]),
    ]

    arrow_top = eq_y - 0.13
    arrow_bot = eq_y - 0.25
    line_start = eq_y - 0.33
    line_spacing = 0.12

    for arrow_x, text_x, lines in annotations:
        # Arrow from equation down to label
        ax.annotate(
            "",
            xy=(arrow_x, arrow_bot),
            xytext=(arrow_x, arrow_top),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->,head_width=0.25,head_length=0.15",
                color=COLOR_ARROW,
                lw=1.5,
            ),
        )
        # Multi-line annotation
        for i, line in enumerate(lines):
            ax.text(
                text_x, line_start - i * line_spacing, line,
                fontsize=10,
                color=COLOR_ANNOTATION,
                ha="center",
                va="top",
                transform=ax.transAxes,
                style="italic",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.savefig(
        output_path,
        format=fmt,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor=COLOR_BG,
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate thermostat equation annotation diagram"
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
    output_path = args.output_dir / f"thermostat_equation.{args.format}"
    render_diagram(output_path, fmt=args.format)


if __name__ == "__main__":
    main()
