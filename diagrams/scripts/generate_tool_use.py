#!/usr/bin/env python3
"""
Generate tool use generation diagram showing how tool calls are interleaved in generation.

Two-row layout:
- Top row: User prompt (masked)
- Bottom row: Model response with tool call, execution, and continuation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects


def draw_token_box(ax, x, y, w, h, text, style="normal", fontsize=9):
    """Draw a single token box with different styles."""
    styles = {
        "normal": {"facecolor": "#E8F5E9", "edgecolor": "#4CAF50", "textcolor": "#2E7D32"},
        "tool_call": {"facecolor": "#FFF3E0", "edgecolor": "#E65100", "textcolor": "#E65100"},
        "tool_output": {"facecolor": "#EDE7F6", "edgecolor": "#7B1FA2", "textcolor": "#7B1FA2"},
        "masked": {"facecolor": "#EEEEEE", "edgecolor": "#9E9E9E", "textcolor": "#757575"},
    }

    s = styles.get(style, styles["normal"])

    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=s["facecolor"],
        edgecolor=s["edgecolor"],
        linewidth=1.5 if style in ["tool_call", "tool_output"] else 1,
    )
    ax.add_patch(rect)

    txt = ax.text(
        x + w/2, y + h/2, text,
        ha="center", va="center",
        fontsize=fontsize, fontfamily="monospace",
        color=s["textcolor"],
        fontweight="bold" if style in ["tool_call", "tool_output"] else "normal",
    )
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])


def draw_brace(ax, x1, x2, y, text="", color="#666666"):
    """Draw a horizontal brace with label below."""
    height = 0.15
    mid = (x1 + x2) / 2

    # Simple bracket using lines
    ax.plot([x1, x1], [y, y - height], color=color, linewidth=1.5)
    ax.plot([x1, mid], [y - height, y - height * 1.5], color=color, linewidth=1.5)
    ax.plot([mid, x2], [y - height * 1.5, y - height], color=color, linewidth=1.5)
    ax.plot([x2, x2], [y - height, y], color=color, linewidth=1.5)

    ax.text(mid, y - 0.4, text, ha="center", va="top",
            fontsize=9, color=color, style="italic", zorder=10,
            bbox=dict(facecolor='white', edgecolor='none', pad=1))


def render_tool_use_diagram(output_path: Path, fmt: str = "png", dpi: int = 200):
    """
    Render tool use diagram with two-row layout.
    Top: prompt, Bottom: response with tool use.
    """
    fig, ax = plt.subplots(figsize=(14, 4.2))

    box_w, box_h = 1.0, 0.6
    x_offset = 1.5

    # Two rows - tighter vertical spacing
    y_prompt = 3.2
    y_response = 2.0

    # Prompt tokens (top row) - simple question
    prompt_tokens = [
        ("What's", "masked"),
        ("123", "masked"),
        ("*", "masked"),
        ("456?", "masked"),
    ]

    # Response tokens (bottom row)
    response_tokens = [
        ("Let", "normal"),
        ("me", "normal"),
        ("calc:", "normal"),
        ("<code>", "tool_call"),
        ("123*456", "tool_call"),
        ("</code>", "tool_call"),
        ("<out>", "tool_output"),
        ("56088", "tool_output"),
        ("</out>", "tool_output"),
        ("Answer:", "normal"),
        ("56,088", "normal"),
    ]

    # === Title ===
    ax.text(
        x_offset + len(response_tokens) * box_w / 2, 4.1,
        "Tool Use: Interleaved Generation with External Execution",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )

    # === Draw prompt tokens (top row) ===
    ax.text(x_offset - 0.2, y_prompt + box_h/2, "Prompt",
            ha="right", va="center", fontsize=10, fontweight="bold", color="#757575")
    for i, (tok, style) in enumerate(prompt_tokens):
        x = x_offset + i * box_w
        draw_token_box(ax, x, y_prompt, box_w - 0.1, box_h, tok, style=style)

    # Masked label
    ax.text(x_offset + len(prompt_tokens) * box_w + 0.2, y_prompt + box_h/2,
            "(masked)", ha="left", va="center", fontsize=9, color="#9E9E9E", style="italic")

    # === Draw response tokens (bottom row) ===
    ax.text(x_offset - 0.2, y_response + box_h/2, "Response",
            ha="right", va="center", fontsize=10, fontweight="bold", color="#2E7D32")
    for i, (tok, style) in enumerate(response_tokens):
        x = x_offset + i * box_w
        draw_token_box(ax, x, y_response, box_w - 0.1, box_h, tok, style=style)

    # === Executor box (gray) - very close to response ===
    exec_y = 0.6
    tool_mid = x_offset + 4.5 * box_w
    output_mid = x_offset + 7.5 * box_w

    exec_box = FancyBboxPatch(
        (tool_mid - 0.3, exec_y - 0.25), 3.5, 0.5,
        boxstyle="round,pad=0.05",
        facecolor="#F5F5F5", edgecolor="#9E9E9E", linewidth=1.5,
        zorder=1
    )
    ax.add_patch(exec_box)
    ax.text(tool_mid + 1.45, exec_y, "External Executor",
            ha="center", va="center", fontsize=9, color="#616161", fontweight="bold", zorder=3)

    # Arrows (drawn first, lower z-order)
    ax.annotate('', xy=(tool_mid + 0.3, exec_y + 0.25), xytext=(tool_mid + 0.3, y_response - 0.1),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=2), zorder=1)
    ax.annotate('', xy=(output_mid, y_response - 0.1), xytext=(output_mid, exec_y + 0.25),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=2), zorder=1)

    # === Phase labels below response (drawn after arrows, higher z-order) ===
    draw_brace(ax, x_offset, x_offset + 3 * box_w - 0.1, y_response - 0.1,
               text="generates", color="#4CAF50")
    draw_brace(ax, x_offset + 3 * box_w, x_offset + 6 * box_w - 0.1, y_response - 0.1,
               text="tool call", color="#E65100")
    draw_brace(ax, x_offset + 6 * box_w, x_offset + 9 * box_w - 0.1, y_response - 0.1,
               text="system output", color="#7B1FA2")
    draw_brace(ax, x_offset + 9 * box_w, x_offset + 11 * box_w - 0.1, y_response - 0.1,
               text="continues", color="#4CAF50")

    ax.set_xlim(-0.3, x_offset + len(response_tokens) * box_w + 1)
    ax.set_ylim(0.0, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tool use diagram")
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

    render_tool_use_diagram(args.output_dir / f"tool_use_generation.{args.format}",
                           fmt=args.format, dpi=args.dpi)
    if args.format == "png":
        render_tool_use_diagram(svg_dir / "tool_use_generation.svg", fmt="svg")


if __name__ == "__main__":
    main()
