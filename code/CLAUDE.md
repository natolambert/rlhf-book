# RLHF Book Code Examples

## Claude Code Notes

- **Always run Python commands with `uv run python`** (not bare `python`/`python3`) so the project environment is used consistently
- **Always run training commands in background** using `run_in_background: true` to avoid blocking
- **Be careful with parallel jobs**: Only run one training job at a time unless you verify memory is available. Running too many can OOM the system.
- **If using a DGX Spark**: ~120GB unified CPU/GPU memory — aim for <80GB usage to be safe. Flash Attention is not available on ARM64/Blackwell; the code automatically falls back to PyTorch SDPA.
- **Before finalizing changes under `code/`**, run `uvx ruff@0.14.5 check .` and `uvx ruff@0.14.5 format --check .` — both are enforced by CI on PRs that touch `code/` (see `.github/workflows/lint.yml`). Use `uvx ruff@0.14.5 check --fix .` and `uvx ruff@0.14.5 format .` to auto-fix. Pin the version to match CI; unpinned `uvx ruff` can diverge.

## Contributing

See `CONTRIBUTING.md` for branch naming, PR conventions, and pre-submit checks.

## Quick Start

```bash
cd code/
uv sync

# Optional: log to wandb (public project for book examples)
export WANDB_PROJECT=rlhf-book

# Run training scripts
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
uv run python -m reward_models.train_preference_rm --samples 2000 --epochs 1
uv run python -m reward_models.train_orm --samples 400 --epochs 2
uv run python -m reward_models.train_prm --samples 500 --epochs 2
uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
uv run python -m rejection_sampling.train --config rejection_sampling/configs/top_per_prompt.yaml
```

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

Reference runs are published at https://wandb.ai/natolambert/rlhf-book (public, no login needed). When contributing new algorithms or configs, log runs to your own wandb project and include the link in your PR.

## Memory Notes

Without LoRA (full fine-tune):
- 0.6B model: ~4-6GB
- 1.7B model: ~10-15GB
- 3B model: ~20-25GB

With gradient checkpointing can reduce by ~30-40%.

## TODOs

- [ ] Validate and generate reference wandb runs for direct alignment (DPO, IPO, SimPO, ORPO, KTO) — see [#358](https://github.com/natolambert/rlhf-book/issues/358). For ORPO/SimPO debugging context, see [direct_alignment/ORPO_SIMPO.md](direct_alignment/ORPO_SIMPO.md)
- [ ] Add evaluation scripts for reward models
- [x] ~~Remove QLoRA from reward models~~ (done — full fine-tune is the default, dead LoRA references cleaned up)
