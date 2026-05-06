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
