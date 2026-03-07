# Teaching Slides

Lecture slides for the RLHF Book, built with [colloquium](https://github.com/Interconnects-AI/colloquium).

Colloquium is under active development -- expect improvements to the slide tooling alongside the content.

## Lecture sets

- **`course/`** — Full introductory course covering the entire book (5-10 lectures, under construction)
- **`SALA-2026/`** — Upcoming talk

## Setup

Install from the repo root (this pulls colloquium from source via the `teach` extra):

```bash
uv sync --extra teach
```

Or from a local clone for development:

```bash
uv pip install -e /path/to/colloquium
```

## Building

From the repo root:

```bash
make teach
```

Or build a single lecture:

```bash
uv run colloquium build teach/course/slides.md -o build/html/teach/course/
```

Output goes to `build/html/teach/`.

## Live Preview

For development with live reload:

```bash
cd teach/SALA-2026
uv run colloquium serve talk.md
```

## Talk Assets

Each talk keeps its images in a local `assets/` directory (e.g. `teach/SALA-2026/assets/`). Reference them with relative paths in slides:

```markdown
![description](assets/image.png)
```
