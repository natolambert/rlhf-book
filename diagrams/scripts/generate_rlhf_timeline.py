#!/usr/bin/env python3
"""
Generate RLHF history timeline for Chapter 2 (Key Related Works).

Shows the chronological development of RLHF across three eras:
- Origins to 2018: RL on Preferences
- 2019 to 2022: RLHF on Language Models
- 2023 to Present: ChatGPT Era

Usage:
    uv run python scripts/generate_rlhf_timeline.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -- Era definitions --
ERAS = [
    {
        "label": "RL on Preferences",
        "start": 2008,
        "end": 2018.9,
        "face": "#E3F2FD",
        "edge": "#1565C0",
        "text": "#0D47A1",
    },
    {
        "label": "RLHF on\nLanguage Models",
        "start": 2019,
        "end": 2022.8,
        "face": "#FFF3E0",
        "edge": "#E65100",
        "text": "#BF360C",
    },
    {
        "label": "ChatGPT\nEra",
        "start": 2022.9,
        "end": 2025.5,
        "face": "#E8F5E9",
        "edge": "#2E7D32",
        "text": "#1B5E20",
    },
]

# -- Milestones: (year, label, level) --
# level: 1, 2, 3 = above; -1, -2, -3 = below
MILESTONES = [
    # Era 1: RL on Preferences
    (2008, "TAMER\n(Knox & Stone)", 2),
    (2015, "DQN\n(Mnih et al.)", -2),
    (2017, "RLHF on Atari\n(Christiano et al.)", 3),
    (2018, "Deep TAMER\n(Warnell et al.)", -1),
    (2018.6, "Scalable Alignment\n(Leike et al.)", 1),
    # Era 2: RLHF on Language Models
    (2019.2, "Fine-tuning LMs from\nPreferences (Ziegler et al.)", -2),
    (2020, "Learning to Summarize\n(Stiennon et al.)", 2),
    (2021, "WebGPT\n(Nakano et al.)", -1),
    (2022, "InstructGPT\n(Ouyang et al.)", 3),
    (2022.35, "Sparrow,\nGopherCite", -2),
    (2022.6, "Reward Overoptimization\n(Gao et al.)", 1),
    # Era 3: ChatGPT Era
    (2023, "ChatGPT\nLaunch", -1),
    (2023.5, "Llama 2\n(Touvron et al.)", 2),
    (2024, "DPO\n(Rafailov et al.)", -2),
    (2024.6, "Llama 3, Nemotron,\nTülu 3", 1),
    (2025.2, "o1\n(OpenAI)", -1),
]


def render_timeline(output_path: Path, fmt: str = "png", dpi: int = 200):
    """Render the RLHF history timeline."""
    fig, ax = plt.subplots(figsize=(18, 8))

    year_min, year_max = 2007, 2026
    x_left, x_right = 0.5, 18.5

    def y2x(year):
        frac = (year - year_min) / (year_max - year_min)
        return x_left + frac * (x_right - x_left)

    TL_Y = 5.5
    ERA_HALF = 0.55  # taller bands to fit labels inside
    LEVEL_STEP = 0.95

    # -- Draw era bands (taller, with labels inside) --
    for era in ERAS:
        x0 = y2x(era["start"])
        x1 = y2x(era["end"])
        rect = FancyBboxPatch(
            (x0, TL_Y - ERA_HALF),
            x1 - x0,
            2 * ERA_HALF,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=era["face"],
            edgecolor=era["edge"],
            linewidth=2.0,
            alpha=0.45,
            zorder=1,
        )
        ax.add_patch(rect)
        # Label inside band
        mid_x = (x0 + x1) / 2
        ax.text(
            mid_x,
            TL_Y,
            era["label"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=era["text"],
            fontstyle="italic",
            alpha=0.65,
            zorder=2,
        )

    # -- Timeline axis (drawn on top of bands) --
    ax.plot(
        [y2x(year_min + 0.5), y2x(year_max - 0.5)],
        [TL_Y, TL_Y],
        color="#9E9E9E",
        linewidth=2.0,
        zorder=3,
        solid_capstyle="round",
    )

    # -- Year ticks --
    for year in range(2008, 2026):
        tx = y2x(year)
        ax.plot(
            [tx, tx],
            [TL_Y - 0.1, TL_Y + 0.1],
            color="#9E9E9E",
            linewidth=1.2,
            zorder=3,
        )
        if year % 2 == 0:
            ax.text(
                tx,
                TL_Y - ERA_HALF - 0.15,
                str(year),
                ha="center",
                va="top",
                fontsize=9,
                color="#757575",
            )

    # -- Milestones --
    for year, label, level in MILESTONES:
        mx = y2x(year)

        marker_color = "#616161"
        for era in ERAS:
            if era["start"] <= year <= era["end"]:
                marker_color = era["edge"]
                break

        # Dot on timeline
        ax.plot(mx, TL_Y, "o", color=marker_color, markersize=6, zorder=6)

        # Stem
        stem_end = TL_Y + level * LEVEL_STEP
        if level > 0:
            label_y = stem_end + 0.1
            va = "bottom"
            stem_start = TL_Y + ERA_HALF
        else:
            label_y = stem_end - 0.1
            va = "top"
            stem_start = TL_Y - ERA_HALF

        ax.plot(
            [mx, mx],
            [stem_start, stem_end],
            color=marker_color,
            linewidth=1.0,
            zorder=4,
            alpha=0.5,
        )
        ax.plot(mx, stem_end, "o", color=marker_color, markersize=3, zorder=5, alpha=0.6)

        ax.text(
            mx,
            label_y,
            label,
            ha="center",
            va=va,
            fontsize=8,
            color="#424242",
            fontweight="medium",
            linespacing=1.15,
        )

    # -- Title --
    ax.set_title(
        "Timeline of Key RLHF Developments",
        fontsize=16,
        fontweight="bold",
        color="#212121",
        pad=15,
    )

    ax.set_xlim(-0.2, 19.2)
    ax.set_ylim(TL_Y - 3.8, TL_Y + 4.5)
    ax.axis("off")

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate RLHF timeline for Chapter 2")
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

    render_timeline(
        args.output_dir / f"rlhf_timeline.{args.format}",
        fmt=args.format,
        dpi=args.dpi,
    )
    if args.format == "png":
        render_timeline(svg_dir / "rlhf_timeline.svg", fmt="svg")


if __name__ == "__main__":
    main()
