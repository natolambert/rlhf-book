# RLHF Book - Claude Code Context

## Project Overview

This is the source repository for "RLHF Book" by Nathan Lambert - a comprehensive guide to Reinforcement Learning from Human Feedback.

**Live site:** https://rlhfbook.com

## Build System

- **Pandoc + Make** for multi-format output (HTML, PDF, EPUB, DOCX)
- Run `make` to build all formats
- Run `make html` for just the HTML site
- Dependencies: pandoc, pandoc-crossref, basictex (for PDF)

## Python Commands

**Always use `uv run python` instead of bare `python`** to ensure the correct virtual environment and dependencies:

```bash
# Correct
uv run python book/scripts/some_script.py
uv run python -c "import matplotlib"

# Incorrect
python book/scripts/some_script.py
```

## Agent Skills

`AGENTS.md` is intentionally a symlink to `CLAUDE.md` for Codex compatibility. If you are Codex and the symlink is present, edit `CLAUDE.md` directly; do not replace, unlink, or edit through `AGENTS.md`.

The repo includes Claude Code skills under `.claude/skills/`.
For code experiments, use `.claude/skills/run-rlhf-code-experiment/SKILL.md`; it maps reader goals to the maintained examples under `code/` and specifies what metrics to report.
For live preview of teaching slides, use `.claude/skills/serve-course-lecture/SKILL.md`; course lecture HTML must be served from `teach/course/` or images referenced as `assets/...` will break.
For diagram review, use `.claude/skills/gemini-feedback/SKILL.md`; it sends a generated diagram and relevant mathematical context to Gemini for textbook-quality feedback.

## Directory Structure

```
book/         # Book source and build files
  chapters/   # Markdown source files (01-introduction.md, etc.)
  images/     # Image assets referenced in chapters
  assets/     # Brand assets (covers, logos)
  templates/  # Pandoc templates for each output format
  scripts/    # Build utilities
  data/       # Library data
  preorder/   # Order redirect page
  metadata.yml # Book metadata for Pandoc
code/         # Code examples and tutorials
diagrams/     # Diagram sources (Python scripts, specs)
build/        # Generated output (not tracked in git)
```

## Image Conventions

- Place images in `book/images/` directory
- Reference: `![Description](images/filename.png){#fig:label}`
- Optional sizing: `{width=450px}`
- Cross-reference with `@fig:label`

## Diagram Workflow

The `diagrams/` directory contains source files for generating figures:

1. **specs/** - YAML specifications defining diagram content
2. **scripts/** - Python/matplotlib scripts for generating diagrams
3. **tikz/** - TikZ/LaTeX diagram sources
4. **generated/** - Built outputs (PNG, PDF, and SVG)

Generate diagrams with:
```bash
cd diagrams && make all
```

Always output built diagrams to `diagrams/generated/png/`, `diagrams/generated/pdf/`, and `diagrams/generated/svg/` — not alongside source files (e.g. not in `diagrams/tikz/`). Then copy final versions to `book/images/` for use in chapters.

**Image conversion**: When converting TikZ PDFs to PNG with `magick`, **always use `-trim`** to remove whitespace, and use `-density 300` for previews (use 400 for `book/images/`). Example: `magick -density 300 input.pdf -trim -quality 100 output.png`

## Key Chapters for Diagrams

- **Chapter 5 (Reward Models)**: Bradley-Terry, ORM, PRM, Generative RM
- **Chapter 6 (Policy Gradients)**: PPO visualizations, async vs sync training
- **Chapter 8 (Direct Alignment)**: DPO visualizations

## Footer Convention

The site footer (logos + copyright line) lives in `book/templates/footer.html` and is included by every page template on rlhfbook.com:

- `book/templates/html.html` (index) — included via the Pandoc partial `$footer.html()$`
- `book/templates/chapter.html` (chapter pages) — included via `$footer.html()$`
- `book/templates/library.html` (standalone page, copied to build/) — included via the HTML sentinel `<!-- include: footer.html -->`
- `book/templates/course.html` (copied to build/) — sentinel
- `book/templates/404.html` (copied to build/ by the `files` target) — sentinel
- `book/rl-cheatsheet/index.html` (copied to build/) — sentinel

The sentinels are expanded at build time by the `$(INLINE_FOOTER)` awk command defined in the Makefile. Pandoc-templated pages use its native partial syntax.

To update the footer, edit `book/templates/footer.html`. That's it.

The Citation block (which has a different heading level across pages — h3 on index, h4 on chapters/library) is **not** part of the footer partial and remains inline in each template. Footer asset paths are absolute (`/assets/...`) so they resolve correctly even when 404.html is served as a fallback on an arbitrary URL.

## Slide Conventions (teach/course)

- **Never add `<!-- notes: ... -->` speaker notes to slides** — they get in the
  way when presenting. Put load-bearing context in the slide body or the PR
  description instead.
- **Do not add "Where to go deeper" / further-reading slides to lecture
  drafts.** Key references belong inline on the slides where the work is
  discussed (links + cite-right), not in a closing link-dump slide.
- **Never shut down the colloquium preview server to resolve file conflicts.**
  The user is often editing the same deck (and watching the preview) while the
  agent works — killing the server to get edits past a staleness check breaks
  their loop. Leave the server running.
- **On file-modified conflicts, edit via script instead of retrying.** The
  default Read→Edit pathway aborts whenever the file's mtime changes (user
  saves, server touches) and is not fault tolerant for concurrent editing.
  After one stale-file error, switch to a targeted `uv run python`/`sed`
  replacement of the exact string — verify the match count first, then
  substitute.

## Style Notes

- Both "finetuning" and "fine-tuning" are acceptable. Do not normalize existing
  `finetuning`, `finetuned`, or `finetune` to hyphenated forms during cleanup
  edits.
- Keep diagrams simple and artist-friendly
- Use consistent visual grammar across related figures
- **Arrows: flush at the source, slight gap at the target** (`shorten >=5pt` in TikZ) — reads as a more dynamic connection than edge-to-edge arrows
- **Don't waste vertical space** — keep boxes/rows close so a figure reads as one compact unit, not floating in whitespace (`-trim` only crops the outer canvas, not layout gaps)
- Prefer SVG for scalability, PNG for final book assets; TikZ PNGs rasterize at `TIKZ_DENSITY` dpi (default 800)
- Mockups are iterative - not pixel-perfect
- See `diagrams/README.md` for the full diagram style conventions and `tikz/` topic-folder layout
- Trailing whitespace at the end of Markdown lines is acceptable; do not remove it during cleanup or typo-only edits.
