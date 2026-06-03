# Diagram Sources for RLHF Book

This directory contains source files for generating diagrams. These are **mockups** intended for iteration with coding tools, to be refined by a professional artist.

## Quick Start

```bash
cd diagrams

# Generate everything
make all

# Or specific targets
make tokens    # Token strip diagrams (reward models)
make tikz      # TikZ diagrams (policy gradients, distillation)
make figures   # Standalone figures (cartpole, tool_use, etc.)
make clean     # Remove generated files
make help      # List all targets
```

## Directory Structure

```
diagrams/
├── specs/          # YAML specifications for diagram content
├── scripts/        # Python/matplotlib generator scripts
├── tikz/           # TikZ/LaTeX diagram sources, organized by topic (see below)
├── generated/      # Output directory (png/, svg/, pdf/)
├── feedback/       # Gemini API diagram review feedback
├── Makefile        # Build automation
└── README.md       # This file (diagram catalog)
```

### `tikz/` topic folders

TikZ sources are grouped by topic (named after the book chapter they support).
`make tikz` discovers `*_tikz.tex` recursively, so adding a diagram is just
dropping a `<name>_tikz.tex` into the right folder.

```
tikz/
├── _shared/            # styles_rlhf.tex — shared styles, found via TEXINPUTS
├── related-works/      # Ch 2: rlhf_schematic, rlhf_timeline
├── training-overview/  # Ch 3: rl_loop, thermostat_equation
├── policy-gradients/   # Ch 6: reinforce, ppo, grpo, rloo
├── reasoning/          # Ch 7: rlvr_loop
├── tools/              # Ch 13: tooluse_rl
├── synthetic-data/     # Ch 12: knowledge_distillation, synthetic_data_distillation
├── product/            # Ch 17: persona_vectors_pipeline
└── pretraining/        # talks/intro: pretraining_next_token
```

A source in any subfolder can `\input{styles_rlhf.tex}` directly — the Makefile
adds `tikz/_shared` to `TEXINPUTS`. Output basenames stay flat, so a diagram's
generated PNG/SVG/PDF name is unchanged by which folder it lives in.

## Tooling Requirements

- **Python + matplotlib**: `uv add matplotlib`
- **TikZ/LaTeX**: `brew install --cask basictex` (for `make tikz`)
- **SVG export**: `brew install pdf2svg` (optional, for TikZ SVGs)

---

## Complete Diagram Catalog

### Token Strip Visualizations

Horizontal token sequences showing where supervision attaches for different reward model types. Based on GSM8K math examples.

| Output file | Script | Description | Book chapter |
|---|---|---|---|
| `pref_rm_training` | `generate_token_strips.py` | Preference RM: pairwise comparison at EOS (chosen vs rejected) | Ch 5 (Reward Models) |
| `prm_training_inference` | `generate_token_strips.py` | Process RM: 3-class labels at step boundaries (training vs inference) | Ch 5 (Reward Models) |
| `orm_inference` | `generate_token_strips.py` | Outcome RM: per-token correctness probability | Ch 5 (Reward Models) |
| `value_fn_inference` | `generate_token_strips.py` | Value function: per-token state values V(s) | Ch 5 (Reward Models) |
| `orm_training` | `generate_multilane_strips.py` | ORM multi-lane: tokens, labels, targets, model outputs | Ch 5 (Reward Models) |
| `value_fn_training` | `generate_multilane_strips.py` | Value function multi-lane: training data flow | Ch 5 (Reward Models) |

**Make target:** `make tokens`

### TikZ/LaTeX Architecture Diagrams

Box-and-arrow flows for RLHF architectures and related training concepts. Many use shared styles from `tikz/_shared/styles_rlhf.tex`.

| Output file | Source | Description | Book chapter |
|---|---|---|---|
| `rl_loop_tikz` | `tikz/training-overview/rl_loop_tikz.tex` | Standard RL feedback loop: agent &harr; environment with s/a/r | Ch 3 (Training Overview) |
| `rlhf_schematic_tikz` | `tikz/related-works/rlhf_schematic_tikz.tex` | RLHF loop: RL algorithm, environment, reward predictor, human feedback (Christiano et al. 2017) | Ch 2 (Related Works) |
| `rlhf_timeline_tikz` | `tikz/related-works/rlhf_timeline_tikz.tex` | Timeline of key RLHF developments across three eras | Ch 2 (Related Works) |
| `thermostat_equation_tikz` | `tikz/training-overview/thermostat_equation_tikz.tex` | Thermostat analogy for the RL objective | Ch 3 (Training Overview) |
| `reinforce_tikz` | `tikz/policy-gradients/reinforce_tikz.tex` | REINFORCE: basic policy gradient algorithm | Ch 6 (Policy Gradients) |
| `ppo_tikz` | `tikz/policy-gradients/ppo_tikz.tex` | PPO: single output, value network, GAE, KL in reward | Ch 6 (Policy Gradients) |
| `grpo_tikz` | `tikz/policy-gradients/grpo_tikz.tex` | GRPO: group of G outputs, group normalization, KL as loss | Ch 6 (Policy Gradients) |
| `rloo_tikz` | `tikz/policy-gradients/rloo_tikz.tex` | RLOO: K outputs, leave-one-out baseline, KL in reward | Ch 6 (Policy Gradients) |
| `knowledge_distillation_tikz` | `tikz/synthetic-data/knowledge_distillation_tikz.tex` | Knowledge distillation pipeline | Ch 12 (Synthetic Data) |
| `synthetic_data_distillation_tikz` | `tikz/synthetic-data/synthetic_data_distillation_tikz.tex` | Synthetic data distillation process | Ch 12 (Synthetic Data) |
| `persona_vectors_pipeline_tikz` | `tikz/product/persona_vectors_pipeline_tikz.tex` | Persona vector extraction and steering pipeline | Ch 17 (Product) |
| `pretraining_next_token_tikz` | `tikz/pretraining/pretraining_next_token_tikz.tex` | Introductory next-token prediction example with target token and loss intuition | Talks/presentations |

**Make target:** `make tikz`

### Standalone Figures

Individual diagrams for specific concepts.

| Output file | Script | Description | Book chapter |
|---|---|---|---|
| `cartpole` | `generate_cartpole.py` | CartPole environment: cart, pole, state variables, actions | Ch 6 (Policy Gradients) |
| `tool_use_generation` | `generate_tool_use.py` | Tool use: interleaved generation with external execution | Ch 13 (Tools) |
| `interleaved_thinking` | `generate_interleaved_thinking.py` | Reasoning model thinking/response block alternation | Talks/presentations |

**Make target:** `make figures`

### Data Specifications

| File | Description |
|---|---|
| `specs/reward_models.yaml` | Token examples, highlight positions, annotations for all RM types |

---

## Workflow

1. **Edit specs** in `specs/` or modify scripts directly
2. **Generate diagrams** with `make all` (or specific target)
3. **Review outputs** in `generated/{png,svg,pdf}/`
4. **Get AI feedback** with `/gemini-feedback diagrams/generated/png/<name>.png`
5. **Copy final versions** to `book/images/` for use in chapters
6. **Commit both sources and outputs** for reproducibility

## Handoff to Artist

When ready for professional refinement:

1. Export all diagrams as SVG
2. Provide the YAML specs as semantic documentation
3. Include a style guide (fonts, colors, stroke widths)
4. Use consistent naming: `fig_rm_{type}_{variant}.svg`
