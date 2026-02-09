# Speedrun Leaderboard

This table records and compares **walltime** (elapsed time) and **final_reward** from policy gradient training runs.

## How to run training

From `code/`, run the following. On exit, metrics are written to `logs/speedrun/speedrun_metrics.json`.

```bash
cd code
# Single algorithm (e.g. GRPO), target reward 0.85, goal = 100-step avg
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml --speedrun --speedrun-target-reward 0.85
```

- Other algorithms: change the YAML, e.g. `--config policy_gradients/configs/rloo.yaml`
- No target reward: omit `--speedrun-target-reward`
- Custom output path: `--speedrun-metrics-file path/to/file.json`

## How to add a record

After a run, generate a one-line summary from `logs/speedrun/speedrun_metrics.json` and add it to the table below.

```bash
# From code/ (default: reads logs/speedrun/speedrun_metrics.json)
uv run python scripts/speedrun/summary_from_json.py

# With a custom path
uv run python scripts/speedrun/summary_from_json.py path/to/speedrun_metrics.json
```

Example output:
```
SPEEDRUN_SUMMARY: walltime_sec=29195 final_reward=1.332 config=grpo seed=42 target_reward=0.85 goal_reached_at_step=99 goal_walltime_sec=15948
```

Copy that line or fill the table columns manually. For multiple runs of the same algorithm, list your best or a representative run.

- **Date**: Run date (YYYY-MM-DD)
- **Runner**: Handle or name (optional)
- **walltime**: Seconds from start to end (or e.g. "X min Y sec")
- **final_reward**: Average reward at the last step (accuracy + format, after KL penalty)
- **config**: Algorithm (rloo / ppo / drgrpo / gspo / cispo / reinforce, etc.)
- **seed**: Random seed
- **Notes**: GPU, data size, or other details

---

## Records

| Date | Runner | walltime | final_reward | config | seed | Notes |
|------|--------|----------|--------------|--------|------|-------|
| (add entries here) | | | | | | |

---

## Notes

- Metrics are written to `logs/speedrun/speedrun_metrics.json` (or the path given by `--speedrun-metrics-file`) when you run training with `uv run python -m policy_gradients.train --config ... --speedrun`. `train.py` creates the output directory if needed.
