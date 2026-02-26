# Speedrun Leaderboard

This page explains a two-step speedrun workflow: (1) run training with speedrun mode to save metrics JSON files, then (2) append selected runs to the leaderboard table for comparison of walltime to reach a target reward.

Target achievement is judged by the 100-step rolling average reward, and time-to-target is reported.

[Jump to Records table](#records)

## (1) How to run training

From `code/`, run the following. On exit, metrics are saved to `logs/speedrun/`. When wandb is enabled, each run gets a unique file (`{wandb_run_id}.json`); otherwise `speedrun_metrics.json` is overwritten each time.

```bash
cd code

# Speedrun with target reward 1.28 (goal = 100-step avg)
uv run python -m policy_gradients.train \
  --config policy_gradients/configs/grpo.yaml \
  --speedrun --speedrun-target-reward 1.28
```

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

# With wandb link (opt-in; only when you agree to share)
uv run python scripts/speedrun/append_leaderboard.py logs/speedrun/{run_id}.json --include-wandb

# With recorder name or notes
uv run python scripts/speedrun/append_leaderboard.py --recorder "your_name" --notes "1x A100"
```

The script writes Date, goal@step, time-to-target, run_id, walltime, final_reward, and algorithm to the table. For multiple runs, list your best or a representative run.

- **wandb**: Use `--include-wandb` to add a wandb run link to the table (opt-in for sharing). Wandb metadata is automatically saved in the JSON when wandb is enabled during training. For others to view the link, make sure your wandb project is set to Public (Project Settings → Visibility).

- **Date**: Run date (YYYY-MM-DD)
- **Runner**: Handle or name (optional)
- **goal@step**: Target reward and first reached step (e.g. `goal(1.28)@step108`, based on 100-step rolling average)
- **time_to_target**: Walltime at first target reach (based on 100-step rolling average)
- **run_id**: Wandb run ID or JSON filename stem (corresponds to `logs/speedrun/{run_id}.json`)
- **walltime**: Total training time (e.g. `4 h 44 min 49 sec`)
- **final_reward**: Average reward at the last step (accuracy + format, after KL penalty)
- **algorithm**: Algorithm (rloo / ppo / drgrpo / gspo / cispo / reinforce, etc.)
- **Notes**: Optional free-form notes (e.g. GPU, data size, batch setup)

---

## Records

| Date | Runner | goal@step | time_to_target | run_id | walltime | final_reward | algorithm | wandb | Notes |
|------|--------|-----------|----------------|--------|----------|--------------|-----------|-------|-------|

## Notes

- Metrics are written when you run training with `uv run python -m policy_gradients.train --config ... --speedrun`. `train.py` creates the output directory if needed.
