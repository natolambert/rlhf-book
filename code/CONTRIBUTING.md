# Contributing to `code/`

The `code/` directory contains runnable training examples organized by technique: `policy_gradients/`, `direct_alignment/`, `reward_models/`, `rejection_sampling/`, and shared `scripts/`.

## Branching & PRs

- Create branches prefixed with `code/` (e.g. `code/fix-grpo-logging`)
- Title PRs with `[CODE]` (e.g. `[CODE] Fix gradient accumulation in PPO`)

## Changelog

CI requires PRs that touch `code/` to also modify `code/CHANGELOG.md` (the file must be modified; the format below is convention, not enforced). Add one bullet under `## Unreleased`:

```
- YYYY-MM-DD: [PR #N](https://github.com/natolambert/rlhf-book/pull/N) description.
```

## Before Submitting

If you use Claude Code, run `/pre-submit-pr` and paste its output in your PR description.

## Smoke Tests

When adding a new top-level code module, add its import path to `tests/test_import_smoke.py` so CI catches broken package wiring. If the module exposes a runnable CLI entrypoint, add that module to the CLI help smoke tests too.

Keep these tests lightweight: they should verify imports, entrypoint wiring, and tiny helper signatures only. They should not download datasets, load models, start training, or require GPUs.

## Adding a New Module

When adding a new top-level module under `code/`, update the surrounding repository metadata so readers, agents, and CI can find it:

- `code/pyproject.toml`: add the package to `[tool.setuptools.packages.find].include`, add any new dependencies, and add a `[project.scripts]` entry if the module has a CLI.
- `code/README.md`: add the module to the overview, quick-start commands, reader experiment path, and chapter mapping if it is user-facing.
- `code/CLAUDE.md`: add the module to the task map and quick-start list so coding agents know where to start. `code/AGENTS.md` is a symlink to this file, so do not create a separate copy.
- `code/<module>/README.md`: document the purpose, status, quick start, configs, expected metrics, memory notes, and reference runs if validated.
- `code/<module>/configs/`: include at least one small, runnable config when the module uses YAML configs.
- `code/tests/test_import_smoke.py`: add import coverage and CLI `--help` coverage for new entrypoints.
- `code/CHANGELOG.md`: add one `## Unreleased` bullet for the PR.
- Book chapter docs: if the module supports a chapter exercise, link it from the relevant chapter's "Suggested Experiments" section.

For long-running examples, document how to launch the run in the background and monitor it. Agent-facing instructions should explicitly say to use a background task plus monitor rather than a foreground command that may time out.
