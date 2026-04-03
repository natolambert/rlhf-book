# Speedrun Leaderboard

> **Note**: This feature is experimental and may be modified or removed in future versions.

This page explains a two-step speedrun workflow: (1) run training with speedrun mode to save metrics JSON files, then (2) append selected runs to the leaderboard table for comparison of walltime to reach a target reward.

Target achievement is judged by the 100-step rolling average reward, and time-to-target is reported.

[Jump to Records table](#records) (includes sort order).

With wandb enabled, `run_id` makes it easy to cross-reference wandb metrics and the leaderboard. For wandb setup, see [Weights & Biases configuration](../../README.md#configuration).

## (1) How to run training

First, choose a target reward value (e.g. 1.35) before running — this determines when the goal is considered reached. From `code/`, run the following. The base commands are in [Policy Gradient Training](../../README.md#policy-gradient-training) in `code/README.md` — add `--speedrun` and `--speedrun-target-reward <value>` to any of them for speedrun mode.

On exit, metrics are saved to `logs/speedrun/`. When wandb is enabled, each run gets a unique file (`{wandb_run_id}.json`); otherwise a timestamped file (`speedrun_YYYYMMDD_HHMMSS.json`) is created.

```bash
cd code

# Speedrun with target reward 1.35 (goal = 100-step avg)
uv run python -m policy_gradients.train \
  --config policy_gradients/configs/grpo.yaml \
  --speedrun --speedrun-target-reward 1.35
```

> **Note**: Speedrun options (`--speedrun`, `--speedrun-target-reward`) are **CLI-only**. Adding them to your YAML config file will not work — make sure to specify them on the command line as shown above.

- Other algorithms: change the YAML, e.g. `--config policy_gradients/configs/rloo.yaml`
- Custom output path: `--speedrun-metrics-file path/to/file.json`
- When wandb is enabled, output is automatically saved as `logs/speedrun/{wandb_run_id}.json` and wandb metadata (run ID, entity, project) is included in the JSON

## (2) How to add a record

After a run, append a row to the table from the speedrun JSON:

```bash
# From code/ (default: logs/speedrun/speedrun_metrics.json)
uv run python scripts/speedrun/append_leaderboard.py

# Specific run JSON file
uv run python scripts/speedrun/append_leaderboard.py logs/speedrun/{run_id}.json

# With wandb link (opt-in; only when you agree to share; set project to Public for others to view)
uv run python scripts/speedrun/append_leaderboard.py logs/speedrun/{run_id}.json --include-wandb

# With recorder name or notes
uv run python scripts/speedrun/append_leaderboard.py --recorder "your_name" --notes "1x A100"

# Re-sort existing rows only (no new row added)
uv run python scripts/speedrun/append_leaderboard.py --sort-only
```

The script writes Date, model, dataset, goal@step, time-to-target, run_id, walltime, final_reward, and algorithm to the table. For multiple runs, you can list your best or a representative run.

- **wandb**: You can use `--include-wandb` to add a wandb run link to the table (opt-in for sharing). Wandb metadata is automatically saved in the JSON when wandb is enabled during training. For others to view the link, make sure to set the project to Public in wandb project settings.

- **Date**: Run date (YYYY-MM-DD)
- **Runner**: Handle or name (optional)
- **model**: HuggingFace model name (e.g. `Qwen/Qwen3-1.7B`)
- **dataset**: Training dataset (e.g. `spell_backward`)
- **goal@step**: Target reward and first reached step (e.g. `goal(1.35)@step181`, based on 100-step rolling average)
- **time_to_target**: Walltime at first target reach (based on 100-step rolling average)
- **run_id**: Wandb run ID or JSON filename stem (corresponds to `logs/speedrun/{run_id}.json`)
- **walltime**: Total training time (e.g. `4 h 44 min 49 sec`)
- **final_reward**: Average reward at the last step
- **algorithm**: Algorithm (rloo / ppo / drgrpo / gspo / cispo / reinforce, etc.)
- **Notes**: Optional free-form notes (e.g. GPU, data size, batch setup)

## (3) How to remove a record

If a row was added by mistake (wrong metrics, duplicate entry, etc.), you can remove it by `run_id`:

```bash
# Dry-run: show which row would be removed (no changes made)
uv run python scripts/speedrun/remove_leaderboard.py <run_id> --dry-run

# Actually remove the row
uv run python scripts/speedrun/remove_leaderboard.py <run_id>
```

> **Note**: The header and separator rows are never affected — only data rows matching the given `run_id` are removed.

---

## Records

**Sort order** (runs that did not reach the target appear last within their group):

- dataset (asc): group by dataset
- model (asc): group by model within dataset
- target (desc): higher target reward first
- time_to_target (asc): shorter time-to-target first
- step (asc): fewer steps to reach the target first
- date (desc): newer date first

| Date | Runner | model | dataset | goal@step | time_to_target | run_id | walltime | final_reward | algorithm | wandb | Notes |
|------|--------|-------|---------|-----------|----------------|--------|----------|--------------|-----------|-------|-------|
| 2026-03-02 | shota | Qwen/Qwen3-1.7B | spell_backward | goal(1.35)@step196 | 9 h 21 min 34 sec | x6kixlrb | 11 h 33 min 51 sec | 1.4531 | cispo | [run](https://wandb.ai/shotakaji-independent-researcher/rlhf-book/runs/x6kixlrb) | 1x RTX 4090 Laptop, symmetric clip |
| 2026-03-02 | shota | Qwen/Qwen3-1.7B | spell_backward | goal(1.35)@step181 | 10 h 27 min 46 sec | rx89evw3 | 13 h 57 min 6 sec | 1.4812 | grpo | [run](https://wandb.ai/shotakaji-independent-researcher/rlhf-book/runs/rx89evw3) | 1x RTX 4090 Laptop, power mode changed mid-run |
| 2026-03-02 | shota | Qwen/Qwen3-1.7B | spell_backward |  |  | 5hrcbad2 | 12 h 56 min 40 sec | 1.4396 | grpo | [run](https://wandb.ai/shotakaji-independent-researcher/rlhf-book/runs/5hrcbad2) | 1x RTX 4090 Laptop, constant power mode |






