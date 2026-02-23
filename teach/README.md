# Teaching Slides

Lecture slides for the RLHF Book, built with [colloquium](https://github.com/Interconnects-AI/colloquium).

Colloquium is under active development -- expect improvements to the slide tooling alongside the content.

## Lecture sets

- **`course/`** — Full introductory course covering the entire book (5-10 lectures, under construction)
- **`SALA-2026/`** — Upcoming talk

## Setup

Install colloquium from source (recommended while the tool is evolving):

```bash
uv pip install "colloquium @ git+https://github.com/Interconnects-AI/colloquium.git"
```

Or from a local clone:

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
