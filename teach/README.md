# Teaching Slides

Lecture slides for the RLHF Book, built with [colloquium](https://github.com/natolambert/colloquium).

More courses will come. The current one (`course/`) is an under-construction lecture set covering the entire book.
Colloquium is also under active development — expect improvements to the slide tooling alongside the content.

## Setup

Install colloquium from source (recommended while the tool is evolving):

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
