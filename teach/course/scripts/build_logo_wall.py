#!/usr/bin/env python
"""Build the "reasoning model cambrian explosion" logo wall for Lecture 5.

Composes a grid of org logos + model name + release date, mirroring the
per-model slides in ``lec5-chap7.md`` (in release order). Source logos live in
``teach/course/assets/logos/`` and the output is written to
``teach/course/assets/reasoning-model-wall.png``.

Run from anywhere:

    uv run python teach/course/scripts/build_logo_wall.py

Edit ``MODELS`` below to add/remove/reorder tiles, then re-run. Repeated orgs
(e.g. DeepSeek R1 and V3.2, the two MiMo entries) intentionally reuse a logo.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
LOGO_DIR = ASSETS / "logos"
OUT = ASSETS / "reasoning-model-wall.png"

# --- content (release order; mirrors the per-model slides) ------------------
# (logo file stem, model name, release date)
MODELS: list[tuple[str, str, str]] = [
    ("openai", "OpenAI o1", "Sep 12, 2024"),
    ("deepseek", "DeepSeek R1", "Jan 20, 2025"),
    ("kimi", "Kimi 1.5", "Jan 20, 2025"),
    ("stepfun", "Open-Reasoner-Zero", "Mar 31, 2025"),
    ("qwen", "Qwen 3", "Apr 29, 2025"),
    ("microsoft", "Phi-4 Reasoning", "Apr 30, 2025"),
    ("xiaomi", "MiMo", "Apr 30, 2025"),
    ("skywork", "Skywork OR-1", "May 2025"),
    ("primeintellect", "INTELLECT-2", "May 12, 2025"),
    ("openthoughts", "OpenThoughts 3", "Jun 5, 2025"),
    ("mistral", "Magistral", "Jun 10, 2025"),
    ("minimax", "MiniMax-M1", "Jun 16, 2025"),
    ("kimi", "Kimi K2", "Jul 11, 2025"),
    ("zhipu", "GLM-4.5", "Jul 28, 2025"),
    ("ai2", "Olmo 3 Think", "Nov 20, 2025"),
    ("deepseek", "DeepSeek V3.2", "Dec 1, 2025"),
    ("nvidia", "Nemotron 3 Nano", "Dec 15, 2025"),
    ("xiaomi", "MiMo-V2-Flash", "Dec 16, 2025"),
]
# Trailing "and more" tile to nod at the 25+ models that did not get a slide.
MORE_TILE = ("+ many more", "25+ models in 2025")

# --- layout ----------------------------------------------------------------
COLS = 5
SCALE = 4                       # bump for higher resolution output
TILE_W, TILE_H = 180 * SCALE, 102 * SCALE
LOGO_BOX = (150 * SCALE, 52 * SCALE)   # max logo footprint inside a tile
GUT_X, GUT_Y = 10 * SCALE, 12 * SCALE
MARGIN = 16 * SCALE

NAME_SIZE = 13 * SCALE
DATE_SIZE = 10 * SCALE
MORE_SIZE = 19 * SCALE

NAVY = (22, 36, 63)
GRAY = (138, 138, 138)
ACCENT = (242, 132, 130)        # deck section-break pink (#F28482)
WHITE = (255, 255, 255, 255)

BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
REG_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit(f"No usable font found in: {candidates}")


def fit(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    """Scale ``img`` to fit within ``box`` preserving aspect ratio."""
    bw, bh = box
    scale = min(bw / img.width, bh / img.height)
    return img.resize((max(1, round(img.width * scale)),
                       max(1, round(img.height * scale))), Image.LANCZOS)


def text_centered(draw, cx, y, text, font, fill):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (r - l) / 2, y), text, font=font, fill=fill)
    return b - t


def make_tile(name_bold, font_bold, font_reg) -> Image.Image:
    """A blank transparent tile; caller pastes logo + draws text."""
    return Image.new("RGBA", (TILE_W, TILE_H), (0, 0, 0, 0))


def build() -> Image.Image:
    font_name = load_font(BOLD_CANDIDATES, NAME_SIZE)
    font_date = load_font(REG_CANDIDATES, DATE_SIZE)
    font_more = load_font(BOLD_CANDIDATES, MORE_SIZE)

    tiles: list[Image.Image] = []
    for stem, name, date in MODELS:
        tile = Image.new("RGBA", (TILE_W, TILE_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        logo = Image.open(LOGO_DIR / f"{stem}.png").convert("RGBA")
        logo = fit(logo, LOGO_BOX)
        logo_y = 6 * SCALE
        tile.alpha_composite(logo, ((TILE_W - logo.width) // 2,
                                    logo_y + (LOGO_BOX[1] - logo.height) // 2))
        text_top = logo_y + LOGO_BOX[1] + 6 * SCALE
        h = text_centered(draw, TILE_W / 2, text_top, name, font_name, NAVY)
        text_centered(draw, TILE_W / 2, text_top + h + 5 * SCALE, date, font_date, GRAY)
        tiles.append(tile)

    # trailing "+ many more" tile
    more = Image.new("RGBA", (TILE_W, TILE_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(more)
    text_centered(d, TILE_W / 2, TILE_H / 2 - 22 * SCALE, MORE_TILE[0], font_more, ACCENT)
    text_centered(d, TILE_W / 2, TILE_H / 2 + 6 * SCALE, MORE_TILE[1], font_date, GRAY)
    tiles.append(more)

    rows = [tiles[i:i + COLS] for i in range(0, len(tiles), COLS)]
    inner_w = COLS * TILE_W + (COLS - 1) * GUT_X
    canvas_w = inner_w + 2 * MARGIN
    canvas_h = len(rows) * TILE_H + (len(rows) - 1) * GUT_Y + 2 * MARGIN
    canvas = Image.new("RGBA", (canvas_w, canvas_h), WHITE)

    for r, row in enumerate(rows):
        row_w = len(row) * TILE_W + (len(row) - 1) * GUT_X   # center short rows
        x0 = MARGIN + (inner_w - row_w) // 2
        y = MARGIN + r * (TILE_H + GUT_Y)
        for c, tile in enumerate(row):
            canvas.alpha_composite(tile, (x0 + c * (TILE_W + GUT_X), y))

    return canvas.convert("RGB")


if __name__ == "__main__":
    img = build()
    img.save(OUT)
    print(f"wrote {OUT}  ({img.width}x{img.height})")
