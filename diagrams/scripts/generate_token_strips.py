#!/usr/bin/env python3
"""
Generate token strip visualizations for reward model diagrams.

These show where supervision attaches along token sequences for different RM types:
- Preference RM: highlight EOS only
- ORM: highlight completion tokens (prompt masked)
- PRM: highlight step boundary tokens only
- Value function: highlight all tokens

Usage:
    python generate_token_strips.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as path_effects


@dataclass
class TokenStrip:
    """Specification for a token strip visualization."""

    name: str
    title: str
    tokens: list[str]
    highlight: set[int]  # indices to highlight (supervised positions)
    masked: set[int] = field(default_factory=set)  # indices to gray out
    annotation: str = ""


# Define token strips for each reward model type
STRIPS = [
    TokenStrip(
        name="pref_rm_tokens",
        title="Preference RM: EOS-only supervision",
        tokens=["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"],
        highlight={9},  # EOS only
        masked=set(),
        annotation="Scalar r(x,y) from EOS representation",
    ),
    TokenStrip(
        name="orm_tokens",
        title="Outcome RM: Completion tokens supervised",
        tokens=["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"],
        highlight={5, 6, 7, 8, 9},  # completion tokens
        masked={0, 1, 2, 3, 4},  # prompt tokens
        annotation="Prompt masked | Completion: p(correct)",
    ),
    TokenStrip(
        name="prm_tokens",
        title="Process RM: Step boundary supervision",
        tokens=[
            "<s>",
            "Q:",
            "2+2",
            "Step1:",
            "2+2=4",
            "\\n",
            "Step2:",
            "verify",
            "\\n",
            "Ans:",
            "4",
            "</s>",
        ],
        highlight={5, 8},  # newlines as step boundaries
        masked={0, 1, 2, 3, 4, 6, 7, 9, 10, 11},  # everything else
        annotation="Labels only at step boundaries (\\n)",
    ),
    TokenStrip(
        name="value_fn_tokens",
        title="Value Function: All tokens have state values",
        tokens=["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"],
        highlight={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},  # all tokens
        masked=set(),
        annotation="V(s) at each token/state",
    ),
]


def render_token_strip(
    strip: TokenStrip,
    output_path: Path,
    fmt: str = "png",
    box_w: float = 1.0,
    box_h: float = 0.6,
    dpi: int = 150,
) -> None:
    """Render a token strip visualization to file."""

    # Colors
    color_highlight = "#90EE90"  # light green for supervised
    color_masked = "#D3D3D3"  # light gray for masked
    color_normal = "#FFFFFF"  # white for normal
    color_border_highlight = "#228B22"  # dark green border for highlight
    color_border_normal = "#000000"  # black border

    n_tokens = len(strip.tokens)
    fig_w = max(8, 0.9 * n_tokens)
    fig, ax = plt.subplots(figsize=(fig_w, 2.0))

    for i, tok in enumerate(strip.tokens):
        x = i * box_w

        # Determine styling
        if i in strip.highlight:
            facecolor = color_highlight
            edgecolor = color_border_highlight
            linewidth = 2.5
            alpha = 1.0
        elif i in strip.masked:
            facecolor = color_masked
            edgecolor = "#808080"
            linewidth = 1.0
            alpha = 0.7
        else:
            facecolor = color_normal
            edgecolor = color_border_normal
            linewidth = 1.0
            alpha = 1.0

        # Draw rounded rectangle
        rect = FancyBboxPatch(
            (x + 0.05, 0.1),
            box_w - 0.1,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Add token text
        text_color = "#404040" if i in strip.masked else "#000000"
        fontweight = "bold" if i in strip.highlight else "normal"
        txt = ax.text(
            x + box_w / 2,
            0.1 + box_h / 2,
            tok,
            ha="center",
            va="center",
            fontsize=10,
            fontfamily="monospace",
            color=text_color,
            fontweight=fontweight,
        )
        # Add subtle outline for readability
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=2, foreground="white")]
        )

    # Title
    ax.text(
        n_tokens * box_w / 2,
        box_h + 0.35,
        strip.title,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    # Annotation below
    ax.text(
        n_tokens * box_w / 2,
        -0.15,
        strip.annotation,
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="#505050",
    )

    # Legend
    legend_x = n_tokens * box_w + 0.3
    legend_items = [
        (color_highlight, color_border_highlight, "Supervised"),
        (color_masked, "#808080", "Masked"),
    ]

    for idx, (fc, ec, label) in enumerate(legend_items):
        y = box_h - idx * 0.3
        legend_rect = FancyBboxPatch(
            (legend_x, y),
            0.3,
            0.2,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
        ax.add_patch(legend_rect)
        ax.text(
            legend_x + 0.4, y + 0.1, label, va="center", fontsize=8, color="#404040"
        )

    # Adjust axes
    ax.set_xlim(-0.2, n_tokens * box_w + 1.5)
    ax.set_ylim(-0.35, box_h + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Save
    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate token strip visualizations")
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

    for strip in STRIPS:
        output_path = args.output_dir / f"{strip.name}.{args.format}"
        render_token_strip(strip, output_path, fmt=args.format)

    print(f"\nGenerated {len(STRIPS)} token strip diagrams in {args.output_dir}")


if __name__ == "__main__":
    main()
