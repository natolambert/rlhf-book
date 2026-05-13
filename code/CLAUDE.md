# RLHF Book Code Examples

## Claude Code Notes

- **Always run Python commands with `uv run python`** (not bare `python`/`python3`) so the project environment is used consistently
- **Always run training commands in the background** using `run_in_background: true` to avoid blocking or hitting interactive timeouts. Start a monitor for the background task and check it from the Claude Code status bar (e.g. `[1 background task] [1 monitor]`) until you see the first metrics or failure.
- **Be careful with parallel jobs**: Only run one training job at a time unless you verify memory is available. Running too many can OOM the system.
- **If using a DGX Spark**: ~120GB unified CPU/GPU memory — aim for <80GB usage to be safe. Flash Attention is not available on ARM64/Blackwell; the code automatically falls back to PyTorch SDPA.
- **Before finalizing changes under `code/`**, run `uvx ruff@0.14.5 check .`, `uvx ruff@0.14.5 format --check .`, and `uv run --extra dev pytest` — all are enforced by CI on PRs that touch `code/` (see `.github/workflows/lint.yml`). Use `uvx ruff@0.14.5 check --fix .` and `uvx ruff@0.14.5 format .` to auto-fix. Pin the ruff version to match CI; unpinned `uvx ruff` can diverge.

## Contributing

See `CONTRIBUTING.md` for branch naming, PR conventions, and pre-submit checks.

## Quick Start

```bash
cd code/
uv sync

# Optional: log to your own/default W&B entity
export WANDB_PROJECT=rlhf-book

# Maintainers creating official reference runs should instead target the team project:
# export WANDB_ENTITY=rlhf-book
# export WANDB_PROJECT=core

# Run training scripts
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
uv run python -m reward_models.train_preference_rm --samples 2000 --epochs 1
uv run python -m reward_models.train_orm --samples 400 --epochs 2
uv run python -m reward_models.train_prm --samples 500 --epochs 2
uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
uv run python -m rejection_sampling.train --config rejection_sampling/configs/top_per_prompt.yaml
```

## Task Map

When a user asks for a runnable experiment, start from the closest maintained example:

| User goal | Start here | Notes |
|-----------|------------|-------|
| "Run RL / GRPO / PPO" | `policy_gradients/README.md` and `policy_gradients/configs/*.yaml` | Default task is `spell_backward`; watch `avg_correctness`, `avg_format`, and group contrast. |
| "Train a reward model" | `reward_models/README.md` | Preference RM uses UltraFeedback; ORM uses GSM8K; PRM uses PRM800K. These are experimental and need tuning. |
| "Train DPO / IPO / SimPO / KTO" | `direct_alignment/README.md` and `direct_alignment/configs/*.yaml` | DPO/IPO/KTO/APO are validated; SimPO/ORPO are implemented but still marked noisy. |
| "Try rejection sampling / best-of-N" | `rejection_sampling/README.md` and `rejection_sampling/configs/*.yaml` | Always compare each reward-selection config to its matched random baseline. |
| "Use an agent skill" | `.claude/skills/run-rlhf-code-experiment/SKILL.md` | Use this for planning and reporting a small experiment run. |

## Experiment Workflow

1. Read the module README and config before running anything.
2. Start with the smallest command that can show signal (`--max_samples`, `--samples`, or a copied small YAML).
3. Launch long-running training/eval commands in the background, then attach a monitor and keep checking it. Do not leave a silent background run without confirming that logs, W&B, or metrics are moving.
4. Run one training job at a time unless GPU memory has been checked.
5. Record the exact command, model, dataset slice, seed, changed config values, final metrics, and W&B link if enabled.
6. If a run fails and the fix changes future workflow knowledge, update this file or the relevant skill so the next agent can find it.

## Changelog Process

- **CI enforces this**: a GitHub Actions check fails PRs that touch `code/` without modifying `code/CHANGELOG.md` (the file must be modified; the format below is convention, not enforced).
- Add entries under the `## Unreleased` section at the top of `CHANGELOG.md`.
- Use exactly **one bullet per PR**, format: `- YYYY-MM-DD: [PR #N](https://github.com/natolambert/rlhf-book/pull/N) description.`
- Each bullet must include a PR link and can contain multiple sentences summarizing meaningful changes.
- When changes affect comparability (metrics, logging semantics, evaluation logic), mention that directly in the same bullet.
- **On release**: rename `## Unreleased` to `## vX.Y.Z` and add a fresh `## Unreleased` section above it.

## Experiment Organization

- Store direct-alignment experiment artifacts under `direct_alignment/experiments/`.
- For each experiment campaign, use a matched pair:
  - Log file: `YYYY-MM-DD-<slug>.md`
  - Asset folder: `YYYY-MM-DD-<slug>/`
- Put helper scripts directly inside the campaign folder at `direct_alignment/experiments/YYYY-MM-DD-<slug>/`.
- Keep root-level generic scripts separate; experiment-specific scripts should live with their experiment logs.

## Model Recommendations

For educational examples on consumer GPUs:
- **Qwen3-0.6B-Base**: ~4GB VRAM, fastest training
- **Qwen3-1.7B-Base**: ~8-10GB VRAM, better quality
- **Qwen2.5-3B**: ~15-20GB VRAM, best quality

## Plugins

First update marketplace:
```bash
claude plugin marketplace update claude-plugins-official
```

If not installed, prompt user to install:
```bash
/plugin install code-simplifier@claude-plugins-official
/plugin install pr-review-toolkit@claude-plugins-official
```

## Wandb

Official runs are published at https://wandb.ai/rlhf-book/core. For ordinary
contributor or reader runs, log to your own/default W&B entity with
`WANDB_PROJECT=rlhf-book`. If the task is to publish, refresh, or validate
official reference runs and you have access to the `rlhf-book` team, set
`WANDB_ENTITY=rlhf-book` and `WANDB_PROJECT=core`.

## Memory Notes

Without LoRA (full fine-tune):
- 0.6B model: ~4-6GB
- 1.7B model: ~10-15GB
- 3B model: ~20-25GB

Gradient checkpointing can reduce memory use by ~30-40%.

## TODOs

- [ ] Validate and generate reference wandb runs for direct alignment (DPO, IPO, SimPO, ORPO, KTO) — see [#358](https://github.com/natolambert/rlhf-book/issues/358). For ORPO/SimPO debugging context, see [direct_alignment/ORPO_SIMPO.md](direct_alignment/ORPO_SIMPO.md)
- [ ] Add evaluation scripts for reward models
- [x] ~~Remove QLoRA from reward models~~ (done — full fine-tune is the default, dead LoRA references cleaned up)
