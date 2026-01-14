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
    # For multi-strip diagrams (e.g., chosen vs rejected, or training vs inference)
    secondary_strip: Optional["TokenStrip"] = None
    secondary_label: str = ""
    primary_label: str = ""
    secondary_color_mode: str = "rejected"  # "rejected" or "inference"


# Colors
COLOR_HIGHLIGHT = "#90EE90"  # light green for supervised/training
COLOR_MASKED = "#D3D3D3"  # light gray for masked
COLOR_NORMAL = "#FFFFFF"  # white for normal
COLOR_BORDER_HIGHLIGHT = "#228B22"  # dark green border
COLOR_BORDER_NORMAL = "#000000"  # black border
COLOR_REJECTED = "#FFB6C1"  # light red for rejected
COLOR_BORDER_REJECTED = "#DC143C"  # crimson border
COLOR_INFERENCE = "#ADD8E6"  # light blue for inference
COLOR_BORDER_INFERENCE = "#1E90FF"  # dodger blue border


def render_single_strip(
    ax,
    strip: TokenStrip,
    y_offset: float = 0,
    box_w: float = 1.0,
    box_h: float = 0.6,
    show_legend: bool = True,
    color_mode: str = "normal",  # "normal", "rejected", or "inference"
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
            if color_mode == "rejected":
                facecolor = COLOR_REJECTED
                edgecolor = COLOR_BORDER_REJECTED
            elif color_mode == "inference":
                facecolor = COLOR_INFERENCE
                edgecolor = COLOR_BORDER_INFERENCE
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
            color_mode=strip.secondary_color_mode,
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
    if has_secondary:
        legend_y += 0.3  # Move legend up when there are two rows
    legend_items = []

    if strip.highlight:
        legend_items.append((COLOR_HIGHLIGHT, COLOR_BORDER_HIGHLIGHT, "Supervised"))
    if strip.masked:
        legend_items.append((COLOR_MASKED, "#808080", "Masked"))
    if has_secondary:
        if strip.secondary_color_mode == "inference":
            legend_items.append((COLOR_INFERENCE, COLOR_BORDER_INFERENCE, "Inference"))
        else:
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


# Define token strips using realistic tokenization (dolma2-tokenizer style)
# Based on GSM8K: "Joy can read 8 pages... How many hours to read 120 pages?"
# Correct answer: 5, Wrong answer: 3
# Using ... to indicate truncated tokens (not a real token, visual indicator)
STRIPS = [
    # Preference RM: Show chosen vs rejected with EOS only highlighted
    # Tokens from: <|eos|> Joy can ... ? The answer is 5 . <|eos|>
    TokenStrip(
        name="pref_rm_training",
        title="Training a Preference RM: Pairwise Comparison at EOS",
        tokens=["<|eos|>", "Joy", "can", "...", "?", "The", "answer", "is", "5", ".", "<|eos|>"],
        highlight={10},  # EOS only
        masked=set(),
        annotation=r"Loss: $\mathcal{L} = -\log \sigma(r_c - r_r)$  |  Only score difference matters",
        primary_label="Chosen:",
        secondary_label="Rejected:",
        secondary_strip=TokenStrip(
            name="",
            title="",
            tokens=["<|eos|>", "Joy", "can", "...", "?", "The", "answer", "is", "3", ".", "<|eos|>"],
            highlight={10},  # EOS only
            masked=set(),
        ),
    ),
    # PRM: Show step boundaries - training labels vs inference scores
    # Training: Joy example (8/20 pages per min, 120 pages -> 5 hours)
    # Inference: James example (3*2*2*52 = 624 pages) - different problem!
    TokenStrip(
        name="prm_training_inference",
        title="Process RM: Training Labels vs Inference Scores",
        tokens=[
            "<|eos|>", "Joy", "...", "?",
            "8", "/", "20", "=", "...", "\\n",
            "120", "/", "...", "\\n",
            "=", "5", ".", "<|eos|>",
        ],
        highlight={9, 13},  # newlines as step boundaries
        masked={0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17},  # everything else
        annotation="Training: 3-class labels at step boundaries  |  Inference: predicted step scores",
        token_labels={
            9: "0",
            13: "+1",
        },
        primary_label="Training:",
        secondary_label="Inference:",
        secondary_color_mode="inference",
        secondary_strip=TokenStrip(
            name="",
            title="",
            tokens=[
                "<|eos|>", "James", "...", "?",
                "3", "*", "2", "=", "\\n", "6",  # \n moved left one
                "...", "=", "12", "...", "\\n",  # \n moved right one
                "=", "624", ".", "<|eos|>",
            ],
            highlight={8, 14},  # newlines at different positions than training
            masked={0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18},
            token_labels={
                8: "p=.95",
                14: "p=.92",
            },
        ),
    ),
    # ORM: Show per-token probabilities on completion (inference time)
    # Prompt masked, completion supervised
    TokenStrip(
        name="orm_inference",
        title="Using an Outcome RM: Per-Token Correctness",
        tokens=["<|eos|>", "Joy", "can", "...", "?", "The", "answer", "is", "5", ".", "<|eos|>"],
        highlight={5, 6, 7, 8, 9, 10},  # completion tokens
        masked={0, 1, 2, 3, 4},  # prompt tokens
        annotation="Loss: BCE per token  |  Prompt masked (e.g. label=-100), completion supervised",
        token_labels={
            5: "p=.92",
            6: "p=.88",
            7: "p=.95",
            8: "p=.99",
            9: "p=.97",
            10: "p=.94",
        },
    ),
    # Value function: Show V(s) on completion tokens (prompt masked like ORM)
    TokenStrip(
        name="value_fn_inference",
        title="Value Function: Per-Token State Values",
        tokens=["<|eos|>", "Joy", "can", "...", "?", "The", "answer", "is", "5", ".", "<|eos|>"],
        highlight={5, 6, 7, 8, 9, 10},  # completion tokens only
        masked={0, 1, 2, 3, 4},  # prompt masked
        annotation=r"$V(s_t)$ = expected future return from state $t$  |  Regression loss on completion",
        token_labels={
            5: "V=.45",
            6: "V=.55",
            7: "V=.70",
            8: "V=.85",
            9: "V=.95",
            10: "V=1.0",
        },
    ),
    # NOTE: orm_multilane and value_fn_multilane show richer detail
    # (where targets come from, how outputs are used)
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
