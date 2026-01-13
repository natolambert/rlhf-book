# Diagram Sources for RLHF Book

This directory contains source files for generating diagrams. These are **mockups** intended for iteration with coding tools, to be refined by a professional artist.

## Directory Structure

```
diagrams/
├── specs/          # YAML specifications for each diagram type
├── d2/             # D2 diagram source files (box-and-arrow flows)
├── scripts/        # Python scripts for generating token strips and other visuals
├── generated/      # Intermediate outputs (SVG, PNG before final placement)
└── README.md       # This file
```

## Workflow

1. **Edit specs** in `specs/` to define the conceptual content
2. **Generate diagrams** using the scripts or D2 CLI
3. **Review outputs** in `generated/`
4. **Copy final versions** to `images/` for use in the book
5. **Commit both sources and outputs** for reproducibility

## Tooling Requirements

### D2 (for pipeline diagrams)

Install D2: https://d2lang.com/tour/install

```bash
# macOS
brew install d2

# or via script
curl -fsSL https://d2lang.com/install.sh | sh -s --
```

Generate SVG/PNG:
```bash
d2 d2/pref_rm_pipeline.d2 generated/pref_rm_pipeline.svg
d2 d2/pref_rm_pipeline.d2 generated/pref_rm_pipeline.png
```

### Python (for token strip visuals)

Dependencies (matplotlib) are managed via uv:
```bash
uv add matplotlib  # if not already installed
```

Generate token strips:
```bash
uv run python scripts/generate_token_strips.py
```

## Generating All Diagrams

```bash
# From repo root
cd diagrams && make all

# Or just token strips (doesn't require D2)
cd diagrams && make tokens

# Copy generated diagrams to images/
cd diagrams && make install
```

## Diagram Types

### 1. Pipeline Diagrams (D2)
Box-and-arrow flows showing: Data → Model → Output → Loss

- `pref_rm_pipeline.d2` - Bradley-Terry Preference RM
- `orm_pipeline.d2` - Outcome RM
- `prm_pipeline.d2` - Process RM
- `gen_rm_pipeline.d2` - Generative RM / LLM-as-Judge

### 2. Token Strip Visualizations (Python)
Horizontal token sequences showing where supervision attaches:

- Preference RM: highlight EOS/last token only
- ORM: highlight all completion tokens (prompt masked)
- PRM: highlight step boundary tokens only
- Value function: highlight all tokens (state values)

### 3. Inference Usage Diagrams (D2)
Simple flows showing how each RM type is used at inference time.

## Handoff to Artist

When ready for professional refinement:

1. Export all diagrams as SVG
2. Provide the YAML specs as semantic documentation
3. Include a style guide (fonts, colors, stroke widths)
4. Use consistent naming: `fig_rm_{type}_{variant}.svg`
