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

## Keeping Examples and Book Docs in Sync

When changing a runnable example, config, metric, or recommended command under `policy_gradients/`, `direct_alignment/`, `reward_models/`, or `rejection_sampling/`, check the connected documentation before submitting:

- Update the module README if commands, configs, expected metrics, memory notes, status, or reference runs changed.
- Update `code/README.md` if the reader experiment path, quick-start commands, or chapter mapping changed.
- Update `code/CLAUDE.md` if agent-facing task guidance or run workflow changed. `code/AGENTS.md` is a symlink to this file, so do not create a separate copy.
- Update the connected book chapter's "Suggested Experiments" section if the reader-facing exercise changed.
- Update `code/tests/test_import_smoke.py` if imports or CLI entrypoints changed.
- Add one `## Unreleased` bullet in `code/CHANGELOG.md`.

New top-level modules are rare, but when adding one also update `code/pyproject.toml` package/script metadata and add a module README plus a small runnable config if the module uses YAML configs.

For long-running examples, document how to launch the run in the background and monitor it. Agent-facing instructions should explicitly say to use a background task plus monitor rather than a foreground command that may time out.
