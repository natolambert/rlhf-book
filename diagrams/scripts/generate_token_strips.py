#!/usr/bin/env python3
"""
Generate token strip visualizations for reward model diagrams.

These show where supervision attaches along token sequences for different RM types:
- Preference RM: highlight EOS only (show chosen vs rejected)
- ORM: highlight completion tokens (prompt masked), show p(correct)
- PRM: highlight step boundary tokens only, show 3-class labels
- Value function: highlight all tokens with V(s)

Usage:
    uv run python scripts/generate_token_strips.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
    # Optional: per-token labels to show above tokens
    token_labels: Optional[dict[int, str]] = None
    # For multi-strip diagrams (e.g., chosen vs rejected)
    secondary_strip: Optional["TokenStrip"] = None
    secondary_label: str = ""
    primary_label: str = ""


# Colors
COLOR_HIGHLIGHT = "#90EE90"  # light green for supervised
COLOR_MASKED = "#D3D3D3"  # light gray for masked
COLOR_NORMAL = "#FFFFFF"  # white for normal
COLOR_BORDER_HIGHLIGHT = "#228B22"  # dark green border
COLOR_BORDER_NORMAL = "#000000"  # black border
COLOR_REJECTED = "#FFB6C1"  # light red for rejected
COLOR_BORDER_REJECTED = "#DC143C"  # crimson border


def render_single_strip(
    ax,
    strip: TokenStrip,
    y_offset: float = 0,
    box_w: float = 1.0,
    box_h: float = 0.6,
    show_legend: bool = True,
    use_rejected_color: bool = False,
    label_prefix: str = "",
):
    """Render a single token strip on the given axes."""
    n_tokens = len(strip.tokens)

    # Draw label prefix (e.g., "Chosen:" or "Rejected:")
    if label_prefix:
        ax.text(
            -0.3,
            y_offset + box_h / 2,
            label_prefix,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#404040",
        )

    for i, tok in enumerate(strip.tokens):
        x = i * box_w

        # Determine styling
        if i in strip.highlight:
            if use_rejected_color:
                facecolor = COLOR_REJECTED
                edgecolor = COLOR_BORDER_REJECTED
            else:
                facecolor = COLOR_HIGHLIGHT
                edgecolor = COLOR_BORDER_HIGHLIGHT
            linewidth = 2.5
            alpha = 1.0
        elif i in strip.masked:
            facecolor = COLOR_MASKED
            edgecolor = "#808080"
            linewidth = 1.0
            alpha = 0.7
        else:
            facecolor = COLOR_NORMAL
            edgecolor = COLOR_BORDER_NORMAL
            linewidth = 1.0
            alpha = 1.0

        # Draw rounded rectangle
        rect = FancyBboxPatch(
            (x + 0.05, y_offset + 0.05),
            box_w - 0.1,
            box_h - 0.1,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Add token text
        text_color = "#606060" if i in strip.masked else "#000000"
        fontweight = "bold" if i in strip.highlight else "normal"
        txt = ax.text(
            x + box_w / 2,
            y_offset + box_h / 2,
            tok,
            ha="center",
            va="center",
            fontsize=9,
            fontfamily="monospace",
            color=text_color,
            fontweight=fontweight,
        )
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=2, foreground="white")]
        )

        # Add per-token labels if specified
        if strip.token_labels and i in strip.token_labels:
            ax.text(
                x + box_w / 2,
                y_offset + box_h + 0.15,
                strip.token_labels[i],
                ha="center",
                va="bottom",
                fontsize=7,
                color="#0066CC",
                fontweight="bold",
            )

    return n_tokens


def render_token_strip(
    strip: TokenStrip,
    output_path: Path,
    fmt: str = "png",
    box_w: float = 1.0,
    box_h: float = 0.6,
    dpi: int = 150,
) -> None:
    """Render a token strip visualization to file."""

    n_tokens = len(strip.tokens)
    has_secondary = strip.secondary_strip is not None
    has_labels = strip.token_labels is not None

    # Calculate figure dimensions
    fig_w = max(9, 0.9 * n_tokens + 2)
    fig_h = 2.0 if not has_secondary else 3.2
    if has_labels:
        fig_h += 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Determine y positions
    if has_secondary:
        y_primary = box_h + 0.5
        y_secondary = 0
    else:
        y_primary = 0.1 if not has_labels else 0.5
        y_secondary = None

    # Render primary strip
    render_single_strip(
        ax,
        strip,
        y_offset=y_primary,
        box_w=box_w,
        box_h=box_h,
        label_prefix=strip.primary_label,
    )

    # Render secondary strip if present
    if has_secondary and strip.secondary_strip:
        render_single_strip(
            ax,
            strip.secondary_strip,
            y_offset=y_secondary,
            box_w=box_w,
            box_h=box_h,
            use_rejected_color=True,
            label_prefix=strip.secondary_label,
        )

    # Title
    title_y = y_primary + box_h + 0.35
    if has_labels:
        title_y += 0.3
    ax.text(
        n_tokens * box_w / 2,
        title_y,
        strip.title,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    # Annotation below
    if has_secondary:
        annot_y = -0.35
    else:
        annot_y = y_primary - 0.25
    ax.text(
        n_tokens * box_w / 2,
        annot_y,
        strip.annotation,
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="#505050",
    )

    # Legend - only show items that are used
    legend_x = n_tokens * box_w + 0.4
    legend_y = y_primary + box_h / 2
    legend_items = []

    if strip.highlight:
        legend_items.append((COLOR_HIGHLIGHT, COLOR_BORDER_HIGHLIGHT, "Supervised"))
    if strip.masked:
        legend_items.append((COLOR_MASKED, "#808080", "Masked"))
    if has_secondary:
        legend_items.append((COLOR_REJECTED, COLOR_BORDER_REJECTED, "Rejected"))

    for idx, (fc, ec, label) in enumerate(legend_items):
        y = legend_y - idx * 0.35
        legend_rect = FancyBboxPatch(
            (legend_x, y - 0.1),
            0.3,
            0.2,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
        ax.add_patch(legend_rect)
        ax.text(
            legend_x + 0.4, y, label, va="center", fontsize=8, color="#404040"
        )

    # Adjust axes
    y_min = -0.5 if has_secondary else annot_y - 0.2
    y_max = title_y + 0.4
    ax.set_xlim(-0.8 if (strip.primary_label or strip.secondary_label) else -0.2,
                n_tokens * box_w + 1.8)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    # Save
    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


# Define improved token strips for each reward model type
STRIPS = [
    # Preference RM: Show chosen vs rejected with EOS only highlighted
    TokenStrip(
        name="pref_rm_tokens",
        title="Preference RM: Pairwise Comparison at EOS",
        tokens=["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"],
        highlight={9},  # EOS only
        masked=set(),
        annotation="Loss: L = -log σ(r_c - r_r)  |  Only score difference matters",
        primary_label="Chosen:",
        secondary_label="Rejected:",
        secondary_strip=TokenStrip(
            name="",
            title="",
            tokens=["<s>", "What", "is", "2+2", "?", "It", "equals", "5", "!", "</s>"],
            highlight={9},  # EOS only
            masked=set(),
        ),
    ),
    # ORM: Show per-token probabilities on completion
    TokenStrip(
        name="orm_tokens",
        title="Outcome RM: Per-Token Correctness",
        tokens=["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"],
        highlight={5, 6, 7, 8, 9},  # completion tokens
        masked={0, 1, 2, 3, 4},  # prompt tokens
        annotation="Loss: BCE per token  |  Prompt masked (label=-100), completion supervised",
        token_labels={
            5: "p=.92",
            6: "p=.88",
            7: "p=.95",
            8: "p=.99",
            9: "p=.97",
        },
    ),
    # PRM: Show step boundaries with 3-class labels
    TokenStrip(
        name="prm_tokens",
        title="Process RM: Step Boundary Supervision",
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
        annotation="3-class labels at boundaries: {+1: correct, 0: neutral, -1: incorrect}",
        token_labels={
            5: "+1",
            8: "+1",
        },
    ),
    # Value function: Show V(s) at each token
    TokenStrip(
        name="value_fn_tokens",
        title="Value Function: Per-Token State Values",
        tokens=["<s>", "What", "is", "2+2", "?", "The", "answer", "is", "4", "</s>"],
        highlight={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},  # all tokens
        masked=set(),
        annotation="V(s_t) = expected future return from state t  |  Regression loss",
        token_labels={
            0: "V=.12",
            1: "V=.15",
            2: "V=.18",
            3: "V=.22",
            4: "V=.31",
            5: "V=.45",
            6: "V=.62",
            7: "V=.78",
            8: "V=.91",
            9: "V=1.0",
        },
    ),
]


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
