---
name: run-rlhf-code-experiment
description: Plan, run, and report a small RLHF Book code experiment.
allowed-tools: Bash(uv:*), Bash(git:*), Read, Edit
---

# Run RLHF Code Experiment

Use this skill when the user wants to run, adapt, compare, or document an experiment from `code/`.

## Pick The Starting Point

- Policy gradients / RL / GRPO / PPO: read `code/policy_gradients/README.md`.
- Reward models / ORM / PRM / Bradley-Terry RM: read `code/reward_models/README.md`.
- DPO / IPO / SimPO / ORPO / KTO / APO: read `code/direct_alignment/README.md`.
- Rejection sampling / best-of-N / GSM8K filtering: read `code/rejection_sampling/README.md`.

## Run Protocol

1. Work from the repository root unless a command explicitly says `cd code/`.
2. Install or refresh dependencies with `cd code/ && uv sync` only when needed.
3. Use `uv run python`, never bare `python`.
4. Start with a short run:
   - Reward models: lower `--samples` and `--epochs`.
   - Direct alignment: use `--max_samples` or copy a YAML with a smaller sample count.
   - Policy gradients: copy a YAML and reduce `data.size` before changing algorithm logic.
   - Rejection sampling: reduce `max_train_samples`, `max_test_samples`, or `num_completions_per_prompt` in a copied YAML.
5. Run one training job at a time unless GPU memory has been checked.
6. If W&B is not desired, set `WANDB_MODE=disabled` or use the module's no-W&B flag when available.

## What To Report

Report enough detail for another reader to reproduce the result:

- Exact command.
- Model, dataset, seed, and config file.
- Config values changed from the checked-in defaults.
- Final metrics and any observed failure mode.
- W&B run URL if logging was enabled.
- Follow-up sweep worth trying next.

## Comparison Rules

- For policy gradients, compare `avg_correctness`, `avg_format`, `avg_binary`, loss, and whether sampled groups contain reward contrast.
- For reward models, compare reward margins or correctness scores on held-out examples, not just training loss.
- For direct alignment, compare `accuracy`, `margins`, `chosen_rewards`, `rejected_rewards`, and sample generations. IPO loss scale is not directly comparable to DPO loss scale.
- For rejection sampling, always compare each reward-selected run to its matched random baseline.

## Documentation Rule

If the run exposes a new setup requirement, failure mode, or useful workflow shortcut, update the relevant README, `code/CLAUDE.md`, or this skill before finishing.
