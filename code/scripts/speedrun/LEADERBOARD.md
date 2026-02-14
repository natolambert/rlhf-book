# Speedrun Leaderboard

This table records and compares **walltime** (elapsed time) and **final_reward** from policy gradient training runs.

## How to run training

From `code/`, run the following. On exit, metrics are written to `logs/speedrun/{run_id}.json` (when wandb is enabled) or `logs/speedrun/speedrun_metrics.json`.

```bash
cd code
# Single algorithm (e.g. GRPO), target reward 0.85, goal = 100-step avg
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml --speedrun --speedrun-target-reward 0.85
```

- Other algorithms: change the YAML, e.g. `--config policy_gradients/configs/rloo.yaml`
- No target reward: omit `--speedrun-target-reward`
- Custom output path: `--speedrun-metrics-file path/to/file.json`

## How to add a record

After a run, append a row to the table from `speedrun_metrics.json`:

```bash
# From code/ (default: logs/speedrun/speedrun_metrics.json)
uv run python scripts/speedrun/append_leaderboard.py

# For a wandb run: pass logs/speedrun/{run_id}.json
# Add --include-wandb to include the run link (opt-in; only when you agree to share).
uv run python scripts/speedrun/append_leaderboard.py logs/speedrun/{run_id}.json --include-wandb

# With recorder name or notes
uv run python scripts/speedrun/append_leaderboard.py --recorder "your_name" --notes "1x A100"
```

The script writes Date, walltime, final_reward, config, seed, and goal info to the table. For multiple runs, list your best or a representative run.

- **wandb**: Use `--include-wandb` only when you agree to share your run publicly. Without it, the cell is left empty.

- **Date**: Run date (YYYY-MM-DD)
- **Runner**: Handle or name (optional)
- **walltime**: Seconds from start to end (or e.g. "X min Y sec")
- **final_reward**: Average reward at the last step (accuracy + format, after KL penalty)
- **algorithm**: Algorithm (rloo / ppo / drgrpo / gspo / cispo / reinforce, etc.)
- **seed**: Random seed
- **Notes**: GPU, data size, or other details

---

## Records

| Date | Runner | walltime | final_reward | algorithm | seed | wandb | Notes |
|------|--------|----------|--------------|-----------|------|-------|-------|
| (add entries here) | | | | | | | |

---

## Notes

- Metrics are written when you run training with `uv run python -m policy_gradients.train --config ... --speedrun`. `train.py` creates the output directory if needed.
- When wandb is enabled, the JSON is saved as `logs/speedrun/{run_id}.json` (wandb run ID). Otherwise `logs/speedrun/speedrun_metrics.json` or the path given by `--speedrun-metrics-file`.
