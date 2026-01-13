#!/usr/bin/env python3
"""
Generate input-output diagrams for reward models.

These show the inference-time flow: Input → Model → Output for each RM type.
Complements the token strip diagrams which show where supervision attaches.

Usage:
    uv run python scripts/generate_io_diagrams.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects


@dataclass
class IODiagram:
    """Specification for an input-output diagram."""

    name: str
    title: str
    subtitle: str
    # Input section
    inputs: list[str]
    input_label: str
    # Model section
    model_name: str
    model_components: list[str]
    # Output section
    output_label: str
    output_example: str
    output_type: str
    # Processing section (optional intermediate step)
    processing: Optional[str] = None
    processing_detail: Optional[str] = None
    # Additional notes
    notes: list[str] = field(default_factory=list)


# Define I/O diagrams for each reward model type
IO_DIAGRAMS = [
    IODiagram(
        name="pref_rm_io",
        title="Preference RM (Bradley-Terry)",
        subtitle="Inference: Score a single completion (trained with pairwise comparisons)",
        inputs=["prompt x", "completion y"],
        input_label="(x, y)",
        model_name="Reward Model",
        model_components=["LM Trunk", "Linear Head → scalar @ EOS"],
        output_label="Scalar Score",
        output_example="r(x,y) = 0.73",
        output_type="Single scalar (higher = better)",
        notes=[
            "Trained: L = -log σ(r_c - r_r)",
            "Inference: score single (x,y)",
            "Use: rank completions, RL reward",
        ],
    ),
    IODiagram(
        name="orm_io",
        title="Outcome Reward Model (ORM)",
        subtitle="Inference: Per-token correctness → aggregate",
        inputs=["prompt x", "completion y"],
        input_label="(x, y)",
        model_name="ORM",
        model_components=["LM Trunk", "Per-token BCE head"],
        processing="Aggregate",
        processing_detail="mean / last / max",
        output_label="Correctness",
        output_example="p(correct) = 0.91",
        output_type="Scalar (aggregated)",
        notes=[
            "Prompt masked (label=-100)",
            "Per-token p(correct|t)",
            "Use: verify solutions",
        ],
    ),
    IODiagram(
        name="prm_io",
        title="Process Reward Model (PRM)",
        subtitle="Inference: Score at each step boundary → aggregate",
        inputs=["prompt x", "steps s₁...sₖ"],
        input_label="(x, steps)",
        model_name="PRM",
        model_components=["LM Trunk", "3-class @ boundaries"],
        processing="Aggregate",
        processing_detail="min / prod / weighted",
        output_label="Process Score",
        output_example="score = 0.85 (scalar)",
        output_type="Aggregated step scores",
        notes=[
            "Classes: +1/0/-1 per step",
            "Only boundary tokens scored",
            "Use: guide search, prune paths",
        ],
    ),
    IODiagram(
        name="gen_rm_io",
        title="Generative RM (LLM-as-Judge)",
        subtitle="Inference: Generate verdict → parse to score",
        inputs=["[x + y_A + y_B + rubric]", "formatted as judge prompt"],
        input_label="Judge prompt",
        model_name="Full LLM",
        model_components=["GPT-4 / Claude / Llama", "→ \"A is better because...\""],
        processing="Parse",
        processing_detail="[[A]] / [[B]] / score",
        output_label="Preference",
        output_example="A ≻ B  (or 4/5)",
        output_type="Extracted from verdict",
        notes=[
            "No reward head",
            "Zero/few-shot/fine-tuned",
            "Explains reasoning",
        ],
    ),
]


def draw_box(ax, x, y, w, h, text, color="#FFFFFF", border_color="#000000",
             fontsize=10, fontweight="normal", text_color="#000000", alpha=1.0):
    """Draw a rounded box with text."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5,
        alpha=alpha,
    )
    ax.add_patch(rect)

    # Handle multi-line text
    lines = text.split("\n")
    line_height = 0.18
    start_y = y + h/2 + (len(lines) - 1) * line_height / 2

    for i, line in enumerate(lines):
        txt = ax.text(
            x + w/2, start_y - i * line_height, line,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])


def draw_arrow(ax, start, end, color="#404040"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->,head_width=0.15,head_length=0.1",
        color=color,
        linewidth=2,
        mutation_scale=15,
    )
    ax.add_patch(arrow)


def render_io_diagram(diagram: IODiagram, output_path: Path, fmt: str = "png", dpi: int = 150):
    """Render an input-output diagram."""

    fig_w, fig_h = 10, 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Layout constants
    box_w = 2.2
    box_h = 0.8
    arrow_gap = 0.3

    # Y positions
    y_title = 5.2
    y_main = 3.0  # main flow
    y_notes = 0.5

    # X positions for flow: Input → Model → (Processing) → Output
    has_processing = diagram.processing is not None
    if has_processing:
        x_positions = [0.5, 3.0, 5.5, 8.0]  # input, model, processing, output
    else:
        x_positions = [1.0, 4.0, 7.0, 7.0]  # input, model, output (no processing)

    # Title
    ax.text(
        fig_w / 2, y_title, diagram.title,
        ha="center", va="bottom", fontsize=16, fontweight="bold"
    )
    ax.text(
        fig_w / 2, y_title - 0.35, diagram.subtitle,
        ha="center", va="top", fontsize=11, style="italic", color="#505050"
    )

    # === INPUT BOX ===
    x_input = x_positions[0]
    input_text = "INPUT\n" + diagram.input_label
    draw_box(ax, x_input, y_main, box_w, box_h * 1.2, input_text,
             color="#E3F2FD", border_color="#1976D2", fontsize=10, fontweight="bold")

    # Input details below
    input_details = "\n".join(f"• {inp}" for inp in diagram.inputs)
    ax.text(
        x_input + box_w/2, y_main - 0.15, input_details,
        ha="center", va="top", fontsize=8, color="#404040"
    )

    # === MODEL BOX ===
    x_model = x_positions[1]
    model_text = diagram.model_name + "\n" + "\n".join(diagram.model_components)
    draw_box(ax, x_model, y_main, box_w, box_h * 1.4, model_text,
             color="#FFF3E0", border_color="#FF9800", fontsize=9, fontweight="bold")

    # Arrow: Input → Model
    draw_arrow(ax, (x_input + box_w + arrow_gap/2, y_main + box_h*0.6),
               (x_model - arrow_gap/2, y_main + box_h*0.6))

    # === PROCESSING BOX (optional) ===
    if has_processing:
        x_proc = x_positions[2]
        proc_text = diagram.processing + "\n" + diagram.processing_detail
        draw_box(ax, x_proc, y_main + 0.1, box_w * 0.8, box_h, proc_text,
                 color="#F3E5F5", border_color="#9C27B0", fontsize=9)

        # Arrow: Model → Processing
        draw_arrow(ax, (x_model + box_w + arrow_gap/2, y_main + box_h*0.6),
                   (x_proc - arrow_gap/2, y_main + box_h*0.6))

        x_output = x_positions[3]
        # Arrow: Processing → Output
        draw_arrow(ax, (x_proc + box_w*0.8 + arrow_gap/2, y_main + box_h*0.6),
                   (x_output - arrow_gap/2, y_main + box_h*0.6))
    else:
        x_output = x_positions[2]
        # Arrow: Model → Output
        draw_arrow(ax, (x_model + box_w + arrow_gap/2, y_main + box_h*0.6),
                   (x_output - arrow_gap/2, y_main + box_h*0.6))

    # === OUTPUT BOX ===
    output_text = "OUTPUT\n" + diagram.output_label
    draw_box(ax, x_output, y_main, box_w, box_h * 1.2, output_text,
             color="#E8F5E9", border_color="#4CAF50", fontsize=10, fontweight="bold")

    # Output example below
    ax.text(
        x_output + box_w/2, y_main - 0.15, diagram.output_example,
        ha="center", va="top", fontsize=11, fontweight="bold", color="#2E7D32",
        fontfamily="monospace"
    )
    ax.text(
        x_output + box_w/2, y_main - 0.45, diagram.output_type,
        ha="center", va="top", fontsize=8, color="#606060", style="italic"
    )

    # === NOTES ===
    if diagram.notes:
        notes_text = "  •  ".join(diagram.notes)
        ax.text(
            fig_w / 2, y_notes, notes_text,
            ha="center", va="center", fontsize=9, color="#606060",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#E0E0E0")
        )

    # Adjust axes
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # Save
    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate I/O diagrams for reward models")
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

    for diagram in IO_DIAGRAMS:
        output_path = args.output_dir / f"{diagram.name}.{args.format}"
        render_io_diagram(diagram, output_path, fmt=args.format)

    print(f"\nGenerated {len(IO_DIAGRAMS)} I/O diagrams in {args.output_dir}")


if __name__ == "__main__":
    main()
