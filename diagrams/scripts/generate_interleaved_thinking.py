#!/usr/bin/env python3
"""
Generate interleaved thinking diagram showing the action/thinking loop
in reasoning models and how intermediate thinking is discarded on multi-turn.

Two panels:
- Top: Single turn — think → act → think → act → think → respond
- Bottom: Multi-turn — previous intermediate thinking is dropped

Block-level layout — designed for slide decks and talks.

Usage:
    uv run python scripts/generate_interleaved_thinking.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# -- Color palette --
COLORS = {
    "thinking": {"face": "#E8EAF6", "edge": "#3949AB", "text": "#1A237E"},
    "action":   {"face": "#FFF3E0", "edge": "#E65100", "text": "#BF360C"},
    "response": {"face": "#E8F5E9", "edge": "#2E7D32", "text": "#1B5E20"},
    "user":     {"face": "#E3F2FD", "edge": "#1565C0", "text": "#0D47A1"},
    "dropped":  {"face": "#FAFAFA", "edge": "#D0D0D0", "text": "#C0C0C0"},
}

ROUNDING = 0.12
BLOCK_H = 0.7


def draw_block(ax, x, y, w, h, text, style, fontsize=11, dashed=False):
    """Draw a rounded block with label."""
    c = COLORS[style]
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.03,rounding_size={ROUNDING}",
        facecolor=c["face"],
        edgecolor=c["edge"],
        linewidth=2.0,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontfamily="sans-serif",
        color=c["text"], fontweight="bold",
    )


def draw_arrow(ax, x1, x2, y, color="#9E9E9E"):
    """Draw a right-pointing arrow between blocks."""
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                        mutation_scale=14),
    )


def render_interleaved_thinking(output_path: Path, fmt: str = "png", dpi: int = 200):
    """Render the two-panel interleaved thinking diagram."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6),
                                    gridspec_kw={"height_ratios": [1, 1],
                                                 "hspace": 0.35})

    gap = 0.35
    y0 = 0.4  # baseline y for blocks

    # =========================================================
    # Panel 1: Single turn — the think/act loop
    # =========================================================
    ax1.set_title("Single Turn: The Think / Act Loop",
                  fontsize=14, fontweight="bold", pad=12, loc="left")

    blocks_t1 = [
        ("User\nprompt",   "user",     1.6),
        ("Think",          "thinking", 1.2),
        ("Act",            "action",   1.0),
        ("Think",          "thinking", 1.2),
        ("Act",            "action",   1.0),
        ("Think",          "thinking", 1.2),
        ("Response",       "response", 1.8),
    ]

    x = 0.3
    positions = []
    for i, (label, style, w) in enumerate(blocks_t1):
        draw_block(ax1, x, y0, w, BLOCK_H, label, style)
        positions.append((x, w, style))
        if i < len(blocks_t1) - 1:
            x_next = x + w + gap
            draw_arrow(ax1, x + w + 0.04, x_next - 0.04, y0 + BLOCK_H / 2)
            x = x_next
        else:
            x = x + w

    # Clean loop bracket above Think/Act region
    t1_x, t1_w, _ = positions[1]   # first Think
    a2_x, a2_w, _ = positions[4]   # second Act

    bracket_l = t1_x
    bracket_r = a2_x + a2_w
    bracket_top = y0 + BLOCK_H + 0.15
    tick_h = 0.12
    loop_color = "#78909C"

    # Bracket shape: ┌──────────────────┐  with arrow on left tick
    ax1.plot([bracket_l, bracket_l], [bracket_top, bracket_top + tick_h],
             color=loop_color, lw=2, solid_capstyle="round")
    ax1.plot([bracket_l, bracket_r], [bracket_top + tick_h, bracket_top + tick_h],
             color=loop_color, lw=2, solid_capstyle="round")
    ax1.plot([bracket_r, bracket_r], [bracket_top + tick_h, bracket_top],
             color=loop_color, lw=2, solid_capstyle="round")
    # Small arrowhead on left descending tick
    ax1.annotate(
        "", xy=(bracket_l, bracket_top),
        xytext=(bracket_l, bracket_top + tick_h),
        arrowprops=dict(arrowstyle="-|>", color=loop_color, lw=2,
                        mutation_scale=12),
    )

    mid_loop = (bracket_l + bracket_r) / 2
    ax1.text(mid_loop, bracket_top + tick_h + 0.08, "repeats until ready to respond",
             ha="center", va="bottom", fontsize=10,
             color=loop_color, fontstyle="italic")

    ax1.set_xlim(-0.1, x + 0.5)
    ax1.set_ylim(0.0, bracket_top + tick_h + 0.5)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # =========================================================
    # Panel 2: Multi-turn — intermediate thinking dropped
    # =========================================================
    ax2.set_title("(Optional) Multi-Turn Context Management",
                  fontsize=14, fontweight="bold", pad=12, loc="left")

    x = 0.3

    # User prompt 1
    draw_block(ax2, x, y0, 1.6, BLOCK_H, "User\nprompt", "user")
    x += 1.6 + gap

    # Dropped thinking/action blocks (faded + dashed)
    dropped_blocks = [
        ("Think", 1.0), ("Act", 0.8), ("Think", 1.0), ("Act", 0.8), ("Think", 1.0),
    ]
    drop_start = x
    for label, w in dropped_blocks:
        draw_block(ax2, x, y0, w, BLOCK_H, label, "dropped",
                   fontsize=10, dashed=True)
        x += w + 0.15
    drop_end = x - 0.15

    # Red "dropped" bracket above
    mid_drop = (drop_start + drop_end) / 2
    drop_bracket_y = y0 + BLOCK_H + 0.08
    drop_tick = 0.1
    drop_color = "#EF5350"
    ax2.plot([drop_start, drop_start], [drop_bracket_y, drop_bracket_y + drop_tick],
             color=drop_color, lw=2, solid_capstyle="round")
    ax2.plot([drop_start, drop_end], [drop_bracket_y + drop_tick, drop_bracket_y + drop_tick],
             color=drop_color, lw=2, solid_capstyle="round")
    ax2.plot([drop_end, drop_end], [drop_bracket_y + drop_tick, drop_bracket_y],
             color=drop_color, lw=2, solid_capstyle="round")
    ax2.text(mid_drop, drop_bracket_y + drop_tick + 0.06,
             "dropped from context on next turn",
             ha="center", va="bottom", fontsize=10,
             color=drop_color, fontweight="bold")

    x += gap - 0.15

    # Response 1 (kept)
    draw_block(ax2, x, y0, 1.6, BLOCK_H, "Response", "response")
    x += 1.6 + gap

    # Arrow to turn 2
    draw_arrow(ax2, x - gap + 0.04, x - 0.04, y0 + BLOCK_H / 2, color="#1565C0")

    # User prompt 2
    draw_block(ax2, x, y0, 1.6, BLOCK_H, "User\nprompt 2", "user")
    x += 1.6 + gap

    # New think/act cycle begins
    draw_block(ax2, x, y0, 1.0, BLOCK_H, "Think", "thinking")
    x += 1.0 + 0.25
    ax2.text(x, y0 + BLOCK_H / 2, "...", fontsize=18,
             ha="center", va="center", color="#757575", fontweight="bold")

    ax2.set_xlim(-0.1, x + 0.6)
    ax2.set_ylim(0.0, drop_bracket_y + drop_tick + 0.55)
    ax2.set_aspect("equal")
    ax2.axis("off")

    # =========================================================
    # Compact legend (top-right of panel 1, using dead space)
    # =========================================================
    legend_items = [
        ("user", "User"),
        ("thinking", "Think"),
        ("action", "Act"),
        ("response", "Response"),
    ]
    # Place in axes coords of ax1, top-right area
    legend_ax_y = bracket_top + tick_h + 0.35
    swatch_size = (0.25, 0.18)
    cursor_x = x - 0.2  # right-align to near end of panel

    for style, label in reversed(legend_items):
        c = COLORS[style]
        text_w = len(label) * 0.12 + 0.15
        lx = cursor_x - text_w
        ax1.add_patch(FancyBboxPatch(
            (lx, legend_ax_y - swatch_size[1] / 2), *swatch_size,
            boxstyle=f"round,pad=0.01,rounding_size=0.04",
            facecolor=c["face"], edgecolor=c["edge"], linewidth=1.0,
        ))
        ax1.text(lx + swatch_size[0] + 0.08, legend_ax_y, label,
                 va="center", fontsize=9, color="#505050")
        cursor_x = lx - 0.25

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interleaved thinking diagram")
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

    render_interleaved_thinking(
        args.output_dir / f"interleaved_thinking.{args.format}",
        fmt=args.format, dpi=args.dpi,
    )
    if args.format == "png":
        render_interleaved_thinking(svg_dir / "interleaved_thinking.svg", fmt="svg")


if __name__ == "__main__":
    main()
