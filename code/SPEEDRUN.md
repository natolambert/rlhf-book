# Speedrun Mode

> **Note**: This feature is experimental and may be modified or removed in future versions.

Speedrun mode and the leaderboard help you record training runs and compare how quickly different setups reach the same goal.

## What it’s for

- **Comparable runs**: Metrics use the same goal definition and fields, so different setups can be lined up in the same format.

- **Goal definition**: The finish line is when the 100-step rolling average reward first reaches your `--speedrun-target-reward` value. Before step 100, no goal is evaluated.

## Workflow

**(1) Train with speedrun** — Use the policy-gradient commands and add the `--speedrun` and `--speedrun-target-reward <value>` flags. When training finishes, metrics are written under `logs/speedrun/` as JSON.

**(2) Add a record to the leaderboard** — When publishing the metrics JSON from (1) to the shared leaderboard table, use `uv run python scripts/speedrun/append_leaderboard.py`. By default, it uses the latest `logs/speedrun/*.json`, or you can pass `logs/speedrun/<id>.json` explicitly.

**(3) Remove a record if needed** — To remove a row, such as a duplicate or incorrect entry, use `remove_leaderboard.py`.

## Where the leaderboard lives

- **Detailed workflow and records table**: [`scripts/speedrun/LEADERBOARD.md`](scripts/speedrun/LEADERBOARD.md)
