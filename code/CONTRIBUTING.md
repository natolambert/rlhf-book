# Contributing to `code/`

The `code/` directory contains runnable training examples organized by technique: `policy_gradients/`, `direct_alignment/`, `reward_models/`, `rejection_sampling/`, and shared `scripts/`.

## Branching & PRs

- Create branches prefixed with `code/` (e.g. `code/fix-grpo-logging`)
- Title PRs with `[CODE]` (e.g. `[CODE] Fix gradient accumulation in PPO`)

## Changelog

CI requires PRs that touch `code/` to also update `code/CHANGELOG.md`. Add one bullet under `## Unreleased`:

```
- YYYY-MM-DD: [PR #N](https://github.com/natolambert/rlhf-book/pull/N) description.
```

## Before Submitting

If you use Claude Code, run `/pre-submit-pr` and paste its output in your PR description.
