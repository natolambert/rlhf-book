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
uv run python scripts/some_script.py
uv run python -c "import matplotlib"

# Incorrect
python scripts/some_script.py
```

## Directory Structure

```
chapters/     # Markdown source files (01-introduction.md, etc.)
images/       # Image assets referenced in chapters
assets/       # Brand assets (covers, logos)
templates/    # Pandoc templates for each output format
scripts/      # Build utilities
diagrams/     # Diagram sources (D2, Python scripts, specs)
build/        # Generated output (not tracked in git)
```

## Image Conventions

- Place images in `images/` directory
- Reference: `![Description](images/filename.png){#fig:label}`
- Optional sizing: `{width=450px}`
- Cross-reference with `@fig:label`

## Diagram Workflow

The `diagrams/` directory contains source files for generating figures:

1. **specs/** - YAML specifications defining diagram content
2. **d2/** - D2 language sources for pipeline diagrams
3. **scripts/** - Python scripts for token strip visualizations
4. **generated/** - Intermediate outputs

Generate diagrams with:
```bash
cd diagrams && make all
```

Then copy final versions to `images/` for use in chapters.

## Future: Multimodal Feedback Loop

Plan to integrate Gemini API for diagram feedback:
- Pass math content + generated diagrams to Gemini 2.5 Pro
- Get feedback on visual clarity, correctness, consistency
- Iterate on mockups before artist handoff

Example workflow:
```python
# Pseudocode for diagram feedback
import google.generativeai as genai

model = genai.GenerativeModel('gemini-2.5-pro')
response = model.generate_content([
    "Review this reward model diagram for accuracy:",
    diagram_image,
    "The math should show: " + latex_formula,
    "Is this correct and clear?"
])
```

## Key Chapters for Diagrams

- **Chapter 7 (Reward Models)**: Bradley-Terry, ORM, PRM, Generative RM
- **Chapter 11 (Policy Gradients)**: PPO visualizations, async vs sync training
- **Chapter 12 (DPO)**: Direct alignment visualizations

## Style Notes

- Keep diagrams simple and artist-friendly
- Use consistent visual grammar across related figures
- Prefer SVG for scalability, PNG for final book assets
- Mockups are iterative - not pixel-perfect
