# Changelog

PRs that modify `code/` must add a one-line entry here under the current development section.
On release, entries get moved under a version heading.

## Unreleased

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
