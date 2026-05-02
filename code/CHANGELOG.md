# Changelog

PRs that modify `code/` must add a one-line entry here under the current development section.
On release, entries get moved under a version heading.

## Unreleased

- 2026-05-01: [PR #391](https://github.com/natolambert/rlhf-book/pull/391) added the MaxRL loss to policy gradients. Moved the new `binary_reward` (correctness ∧ format) onto `Experience` so it is computed once in `rollout.py` and averaged in `train.py` via the simple string-only `avg` lambda. Aligned `configs/maxrl.yaml` with the GRPO/DAPO defaults (`num_rollouts: 8`, `temperature: 0.6`, `data.size: 3000`) to fix OOM on consumer hardware.
- 2026-04-29: [PR #385](https://github.com/natolambert/rlhf-book/pull/385) added the DAPO loss and refactored the policy-gradient rollout into an iterable `RolloutEngine`.
- 2026-04-17: [PR #380](https://github.com/natolambert/rlhf-book/pull/380) added Discord community link to `code/README.md` (part of the wider site-wide Discord rollout).
- 2026-04-17: [PR #377](https://github.com/natolambert/rlhf-book/pull/377) cleaned "Tune PPO hyperparameters" TODO from
  `policy_gradients/README.md` because it was addressed in PR #364 and PR #365.
- 2026-04-17: [PR #375](https://github.com/natolambert/rlhf-book/pull/375) added CONTRIBUTING.md with branch/PR conventions, pre-submit-pr skill with ruff lint/format and changelog checks.
- 2026-04-17: [PR #368](https://github.com/natolambert/rlhf-book/pull/368) added APO-Zero and APO-Down losses (Anchored Preference Optimization, D'Oosterlinck et al., 2024) to the direct alignment module, with validated reference runs on OLMo-2-1B-SFT.
- 2026-04-16: [PR #374](https://github.com/natolambert/rlhf-book/pull/374) added CI ruff lint/format check for PRs touching `code/`, applied ruff format to all existing files, fixed lint errors (unused imports, unsorted imports, `zip()` without `strict=`), and documented linting in README.
- 2026-04-16: [PR #373](https://github.com/natolambert/rlhf-book/pull/373) added reward-vs-correctness diagnostic script (`rejection_sampling/diagnostics.py`) that measures RM within-row signal on scored rollout caches, and added a `decidable_fraction` log line to `preprocess.py` so users can see how many prompts have mixed correct/incorrect completions where selection strategy matters.
- 2026-04-15: [PR #372](https://github.com/natolambert/rlhf-book/pull/372) documented build-essential requirement for Ubuntu/Debian and uv version guidance in README install section.
- 2026-04-15: [PR #370](https://github.com/natolambert/rlhf-book/pull/370) cleaned up CLAUDE.md for generic use, removed dead LoRA/QLoRA references from RM docstrings and base.py, moved ORPO/SimPO debug notes to direct_alignment/ORPO_SIMPO.md.
- 2026-04-12: [PR #365](https://github.com/natolambert/rlhf-book/pull/365) updated the canonical PPO reference run to the post-retune validation run from PR #364.
- 2026-04-12: [PR #364](https://github.com/natolambert/rlhf-book/pull/364) retuned PPO
  hyperparameters for smoother training.

## v0.2.0

- 2026-04-12: [PR #362](https://github.com/natolambert/rlhf-book/pull/362) added overview figures from the book to the policy_gradients, reward_models, and rejection_sampling module READMEs.
- 2026-04-12: [PR #354](https://github.com/natolambert/rlhf-book/pull/354) fixed warmup step count in RM training to use ceiling division, so trailing partial accumulation windows are counted correctly.
- 2026-04-12: [PR #349](https://github.com/natolambert/rlhf-book/pull/349) added SAPO (Soft Adaptive Policy Optimization, Gao et al., 2025) loss to the policy gradient module.
- 2026-04-12: [PR #350](https://github.com/natolambert/rlhf-book/pull/350) added the Chapter 9 rejection-sampling module, including matched random baselines and canonical DGX Spark reference runs.
- 2026-04-11: [PR #353](https://github.com/natolambert/rlhf-book/pull/353) added paper links to the policy gradient algorithms table in the README.
- 2026-04-11: [PR #352](https://github.com/natolambert/rlhf-book/pull/352) fixed RM training logging to report per-optimizer-step metrics instead of per-micro-batch, updated preference RM defaults (lr 1e-6→5e-5, samples 2K→5K, effective batch 8→32, added 10% LR warmup), and added `drop_last=True` to all RM DataLoaders. New reference runs reflect these changes — prior wandb links are no longer comparable.
- 2026-04-11: [PR #351](https://github.com/natolambert/rlhf-book/pull/351) made flash-attn optional so the library installs on broader hardware (ARM64, systems without CUDA).
- 2026-02-07: [PR #243](https://github.com/natolambert/rlhf-book/pull/243) stabilized ORPO/SimPO by switching to average-logprob behavior and improved direct-alignment logging/sampling instrumentation.

## v0.1.0

Initial release: policy gradient methods (REINFORCE, PPO, GRPO, RLOO, Dr. GRPO, GSPO, CISPO), reward models (preference RM, ORM, PRM), and direct alignment (DPO, IPO, SimPO, ORPO, KTO).
