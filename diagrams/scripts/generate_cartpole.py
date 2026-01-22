#!/usr/bin/env python3
"""
Generate CartPole environment diagram for RL training overview.

Shows the classic CartPole control problem with labeled components:
- Cart (movable block)
- Pole (balanced on cart)
- State variables: position, velocity, angle, angular velocity
- Actions: push left/right

Usage:
    uv run python scripts/generate_cartpole.py [--output-dir DIR] [--format FORMAT]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.patches import Arc
import numpy as np


# Colors
COLOR_CART = "#4A90D9"  # Blue for cart
COLOR_CART_BORDER = "#2E5A8A"
COLOR_POLE = "#E67E22"  # Orange for pole
COLOR_POLE_BORDER = "#A85C18"
COLOR_GROUND = "#7F8C8D"  # Gray ground
COLOR_WHEEL = "#2C3E50"  # Dark wheels
COLOR_PIVOT = "#E74C3C"  # Red pivot point
COLOR_ARROW = "#27AE60"  # Green for action arrows
COLOR_STATE = "#8E44AD"  # Purple for state annotations
COLOR_TEXT = "#2C3E50"
COLOR_BG = "#FAFAFA"


def draw_ground(ax, y_ground, x_min, x_max):
    """Draw the ground/track."""
    # Ground line
    ax.plot([x_min, x_max], [y_ground, y_ground], color=COLOR_GROUND,
            linewidth=3, zorder=1)
    # Hatching below ground
    for x in np.arange(x_min, x_max, 0.3):
        ax.plot([x, x - 0.15], [y_ground, y_ground - 0.15],
                color=COLOR_GROUND, linewidth=1, zorder=1)


def draw_cart(ax, x_cart, y_ground, cart_width=1.0, cart_height=0.5):
    """Draw the cart body and wheels."""
    wheel_radius = 0.12
    y_cart = y_ground + wheel_radius * 2

    # Cart body
    cart = FancyBboxPatch(
        (x_cart - cart_width/2, y_cart),
        cart_width, cart_height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=COLOR_CART,
        edgecolor=COLOR_CART_BORDER,
        linewidth=2,
        zorder=3
    )
    ax.add_patch(cart)

    # Wheels
    wheel_y = y_ground + wheel_radius
    for wx in [x_cart - cart_width/3, x_cart + cart_width/3]:
        wheel = Circle((wx, wheel_y), wheel_radius,
                       facecolor=COLOR_WHEEL, edgecolor="#1A252F",
                       linewidth=1.5, zorder=2)
        ax.add_patch(wheel)
        # Wheel highlight
        highlight = Circle((wx - 0.03, wheel_y + 0.03), wheel_radius * 0.3,
                          facecolor="#4A5568", edgecolor="none", zorder=2)
        ax.add_patch(highlight)

    return y_cart + cart_height  # Return top of cart


def draw_pole(ax, x_cart, y_cart_top, pole_length=1.8, angle_deg=15):
    """Draw the pole attached to cart."""
    angle_rad = np.radians(angle_deg)

    # Pole dimensions
    pole_width = 0.12

    # Calculate pole end position
    x_end = x_cart + pole_length * np.sin(angle_rad)
    y_end = y_cart_top + pole_length * np.cos(angle_rad)

    # Draw pole as a rotated rectangle
    # Create polygon for pole
    dx = pole_width/2 * np.cos(angle_rad)
    dy = pole_width/2 * np.sin(angle_rad)

    pole_corners = [
        (x_cart - dx, y_cart_top + dy),
        (x_cart + dx, y_cart_top - dy),
        (x_end + dx, y_end - dy),
        (x_end - dx, y_end + dy),
    ]

    pole = Polygon(pole_corners, facecolor=COLOR_POLE,
                   edgecolor=COLOR_POLE_BORDER, linewidth=2, zorder=4)
    ax.add_patch(pole)

    # Pivot point
    pivot = Circle((x_cart, y_cart_top), 0.1,
                   facecolor=COLOR_PIVOT, edgecolor="#C0392B",
                   linewidth=2, zorder=5)
    ax.add_patch(pivot)

    # Ball at top of pole
    ball = Circle((x_end, y_end), 0.08,
                  facecolor=COLOR_POLE, edgecolor=COLOR_POLE_BORDER,
                  linewidth=1.5, zorder=5)
    ax.add_patch(ball)

    return x_end, y_end, angle_rad


def draw_angle_arc(ax, x_cart, y_cart_top, angle_deg):
    """Draw the angle indicator arc."""
    arc_radius = 0.5

    # Draw vertical reference line (dashed) - taller
    ax.plot([x_cart, x_cart], [y_cart_top, y_cart_top + 1.4],
            color=COLOR_STATE, linestyle='--', linewidth=1.5, zorder=3)

    # Draw arc
    arc = Arc((x_cart, y_cart_top), arc_radius * 2, arc_radius * 2,
              angle=0, theta1=90 - angle_deg, theta2=90,
              color=COLOR_STATE, linewidth=2, zorder=3)
    ax.add_patch(arc)

    # Theta label - positioned to the left of the dashed line
    ax.text(x_cart - 0.15, y_cart_top + 0.85, r"$\theta$", fontsize=14, color=COLOR_STATE,
            ha="center", va="center", fontweight="bold")


def draw_state_labels(ax, x_cart, y_ground, y_cart_top, pole_end_x, pole_end_y):
    """Draw state variable annotations."""

    # Position x - horizontal arrow below cart
    y_pos = y_ground - 0.5
    ax.annotate("", xy=(x_cart + 0.8, y_pos), xytext=(x_cart - 0.8, y_pos),
                arrowprops=dict(arrowstyle="<->", color=COLOR_STATE, lw=2))
    ax.text(x_cart, y_pos - 0.05, r"$x$ (position)", fontsize=11,
            color=COLOR_STATE, ha="center", va="top")

    # Velocity - arrow to the right of cart (clear of cart corner)
    y_vel = y_ground + 0.35
    ax.annotate("", xy=(x_cart + 1.6, y_vel), xytext=(x_cart + 0.9, y_vel),
                arrowprops=dict(arrowstyle="->", color=COLOR_STATE, lw=2.5))
    ax.text(x_cart + 1.7, y_vel, r"$\dot{x}$", fontsize=11,
            color=COLOR_STATE, ha="left", va="center")

    # Angular velocity - curved arrow near pole (closer to pole per feedback)
    ax.annotate("",
                xy=(pole_end_x + 0.425, pole_end_y - 0.25),
                xytext=(pole_end_x + 0.225, pole_end_y + 0.25),
                arrowprops=dict(arrowstyle="->", color=COLOR_STATE, lw=2,
                               connectionstyle="arc3,rad=0.5"))
    ax.text(pole_end_x + 0.387, pole_end_y + 0.12, r"$\dot{\theta}$", fontsize=14,
            color=COLOR_STATE, ha="left", va="center", fontweight="bold")


def draw_action_arrows(ax, x_cart, y_cart):
    """Draw the action force arrows with ±F labels underneath."""
    arrow_y = y_cart + 0.25

    # Left action arrow
    ax.annotate("", xy=(x_cart - 1.3, arrow_y), xytext=(x_cart - 0.6, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_ARROW, lw=3,
                               mutation_scale=20))
    ax.text(x_cart - 0.95, arrow_y - 0.055, r"$-F$", fontsize=12, color=COLOR_ARROW,
            ha="center", va="top", fontweight="bold")

    # Right action arrow
    ax.annotate("", xy=(x_cart + 1.3, arrow_y), xytext=(x_cart + 0.6, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_ARROW, lw=3,
                               mutation_scale=20))
    ax.text(x_cart + 0.95, arrow_y - 0.055, r"$+F$", fontsize=12, color=COLOR_ARROW,
            ha="center", va="top", fontweight="bold")


def draw_labels(ax, x_cart, y_ground, y_cart_top):
    """Draw component labels."""
    # Cart label
    ax.text(x_cart, y_ground + 0.467, "Cart", fontsize=12, color="white",
            ha="center", va="center", fontweight="bold", zorder=10)

    # Pole label - moved to avoid overlap with theta
    ax.text(x_cart + 0.35, y_cart_top + 0.65, "Pole", fontsize=12,
            color=COLOR_POLE_BORDER, ha="left", va="center", fontweight="bold")


def draw_info_box(ax, x, y):
    """Draw information box with state and action summary."""
    box_width = 2.2
    box_height = 1.8

    box = FancyBboxPatch(
        (x, y), box_width, box_height,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor="#F8F9FA",
        edgecolor="#DEE2E6",
        linewidth=1.5,
        zorder=1
    )
    ax.add_patch(box)

    # Title
    ax.text(x + box_width/2, y + box_height - 0.2, "State Space",
            fontsize=11, color=COLOR_TEXT, ha="center", va="top", fontweight="bold")

    # State variables
    states = [
        (r"$x$", "cart position"),
        (r"$\dot{x}$", "cart velocity"),
        (r"$\theta$", "pole angle"),
        (r"$\dot{\theta}$", "angular velocity"),
    ]

    for i, (symbol, desc) in enumerate(states):
        y_line = y + box_height - 0.5 - i * 0.32
        ax.text(x + 0.15, y_line, symbol, fontsize=10, color=COLOR_STATE,
                ha="left", va="center")
        ax.text(x + 0.5, y_line, f"– {desc}", fontsize=9, color=COLOR_TEXT,
                ha="left", va="center")


def draw_action_box(ax, x, y):
    """Draw action space box."""
    box_width = 1.6
    box_height = 0.9

    box = FancyBboxPatch(
        (x, y), box_width, box_height,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor="#F8F9FA",
        edgecolor="#DEE2E6",
        linewidth=1.5,
        zorder=1
    )
    ax.add_patch(box)

    # Title
    ax.text(x + box_width/2, y + box_height - 0.15, "Action Space",
            fontsize=11, color=COLOR_TEXT, ha="center", va="top", fontweight="bold")

    # Actions - use ±F notation to match chapter text
    ax.text(x + 0.15, y + 0.35, r"$-F$", fontsize=10, color=COLOR_ARROW,
            ha="left", va="center", fontweight="bold")
    ax.text(x + 0.55, y + 0.35, "push left", fontsize=9, color=COLOR_TEXT,
            ha="left", va="center")

    ax.text(x + 0.15, y + 0.12, r"$+F$", fontsize=10, color=COLOR_ARROW,
            ha="left", va="center", fontweight="bold")
    ax.text(x + 0.55, y + 0.12, "push right", fontsize=9, color=COLOR_TEXT,
            ha="left", va="center")


def render_diagram(output_path: Path, fmt: str = "png", dpi: int = 300):
    """Render the complete CartPole diagram."""

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor(COLOR_BG)

    # Configuration
    x_cart = 0
    y_ground = 0
    pole_angle = 12  # degrees from vertical

    # Draw components
    draw_ground(ax, y_ground, -2.2, 2.5)
    y_cart_top = draw_cart(ax, x_cart, y_ground)
    pole_end_x, pole_end_y, angle_rad = draw_pole(ax, x_cart, y_cart_top,
                                                   pole_length=1.6,
                                                   angle_deg=pole_angle)

    # Draw annotations
    draw_angle_arc(ax, x_cart, y_cart_top, pole_angle)
    draw_state_labels(ax, x_cart, y_ground, y_cart_top, pole_end_x, pole_end_y)
    draw_action_arrows(ax, x_cart, y_ground + 0.4)
    draw_labels(ax, x_cart, y_ground, y_cart_top)


    # Set axis properties - tighter bounds
    ax.set_xlim(-2.2, 2.5)
    ax.set_ylim(-0.9, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Save - no padding
    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight",
                pad_inches=0, facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CartPole environment diagram"
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
    output_path = args.output_dir / f"cartpole.{args.format}"
    render_diagram(output_path, fmt=args.format)


if __name__ == "__main__":
    main()
