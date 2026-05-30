# Teaching Slides

Lecture slides for the RLHF Book, built with [colloquium](https://github.com/natolambert/colloquium).

Colloquium is under active development -- expect improvements to the slide tooling alongside the content.

## Lecture sets

- **`course/`** — Full introductory course covering the entire book (5-10 lectures, under construction)
- **`SALA-2026/`** — Upcoming talk

## Setup

Install from the repo root (this pulls `colloquium` from GitHub via the `teach` extra):

```bash
uv sync --extra teach
```

The `teach` extra intentionally tracks the GitHub source while `colloquium` is moving quickly.
When running slide commands directly, use `uv run --extra teach ...` so `colloquium` is available even in a fresh environment.

Or install from a local clone for development:

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
uv run --extra teach colloquium build teach/course/lec1-chap1-3.md -o build/html/teach/course/lec1-chap1-3/
```

Output goes to `build/html/teach/`.

## Live Preview

For course lectures, keep the generated HTML in `teach/course/` so `assets/...` image links resolve:

```bash
uv run --extra teach python -c "from colloquium.serve import serve; serve('teach/course/lec5-chap7.md', port=8081, output_dir='teach/course')"
```

Open `http://localhost:8081/lec5-chap7.html`.

For standalone talks with local assets, run from the talk directory:

```bash
cd teach/SALA-2026
uv run --extra teach colloquium serve talk.md
```

Before sharing a preview URL, check at least one expected image URL returns `200 image/...`.

## Talk Assets

Each talk keeps its images in a local `assets/` directory (e.g. `teach/SALA-2026/assets/`). Reference them with relative paths in slides:

```markdown
![description](assets/image.png)
```
