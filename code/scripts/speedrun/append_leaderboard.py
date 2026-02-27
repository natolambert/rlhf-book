#!/usr/bin/env python3
"""Append a row to LEADERBOARD.md from speedrun_metrics.json.

Usage (from code/):
  uv run python scripts/speedrun/append_leaderboard.py
  uv run python scripts/speedrun/append_leaderboard.py --recorder "shota"
  uv run python scripts/speedrun/append_leaderboard.py path/to/speedrun_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def load_metrics(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def format_walltime(sec: int) -> str:
    """Format seconds as 'X min Y sec' or 'X h Y min' for readability."""
    if sec < 60:
        return f"{sec} sec"
    if sec < 3600:
        m, s = divmod(sec, 60)
        return f"{m} min {s} sec"
    h, remainder = divmod(sec, 3600)
    m, s = divmod(remainder, 60)
    if s > 0:
        return f"{h} h {m} min {s} sec"
    return f"{h} h {m} min"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append a record to LEADERBOARD.md from speedrun_metrics.json"
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="logs/speedrun/speedrun_metrics.json",
        help="Path to speedrun_metrics.json",
    )
    parser.add_argument(
        "--recorder",
        type=str,
        default="",
        help="Runner name for the record",
    )
    parser.add_argument(
        "--leaderboard",
        type=str,
        default="scripts/speedrun/LEADERBOARD.md",
        help="Path to LEADERBOARD.md (relative to cwd)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes (GPU, data size, etc.)",
    )
    parser.add_argument(
        "--include-wandb",
        action="store_true",
        help="Include wandb run link when wandb_run_id exists in JSON (opt-in for sharing)",
    )
    args = parser.parse_args()

    try:
        d = load_metrics(args.json_path)
    except FileNotFoundError:
        print(f"Error: {args.json_path} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    date_str = datetime.now().strftime("%Y-%m-%d")
    walltime_sec = d.get("walltime_sec") or 0
    final_reward = d.get("final_reward")
    config = d.get("algorithm", d.get("config", ""))

    # Extract run_id: prefer wandb_run_id from JSON, fallback to filename stem
    run_id = d.get("wandb_run_id", "")
    if not run_id:
        json_stem = Path(args.json_path).stem
        if json_stem != "speedrun_metrics":
            run_id = json_stem
    notes = args.notes

    if final_reward is not None:
        final_reward_str = f"{float(final_reward):.4f}"
    else:
        final_reward_str = ""

    goal_reached_at_step = d.get("goal_reached_at_step")
    target = d.get("target_reward", "")
    goal_step_display = ""
    if goal_reached_at_step is not None:
        target_str = f"({target})" if target != "" else ""
        goal_step_display = f"goal{target_str}@step{goal_reached_at_step}"

    goal_walltime_sec = d.get("goal_walltime_sec")
    time_to_target_display = ""
    if goal_walltime_sec is not None:
        time_to_target_display = format_walltime(goal_walltime_sec)

    walltime_display = format_walltime(walltime_sec)

    wandb_cell = ""
    if args.include_wandb:
        wandb_run_id = d.get("wandb_run_id")
        entity = d.get("wandb_entity", "")
        project = d.get("wandb_project", "")
        if wandb_run_id and entity and project:
            url = f"https://wandb.ai/{entity}/{project}/runs/{wandb_run_id}"
            wandb_cell = f"[run]({url})"
        elif wandb_run_id:
            print(
                f"Warning: wandb_entity or wandb_project missing in JSON; cannot generate wandb link.",
                file=sys.stderr,
            )

    new_row = f"| {date_str} | {args.recorder} | {goal_step_display} | {time_to_target_display} | {run_id} | {walltime_display} | {final_reward_str} | {config} | {wandb_cell} | {notes} |"

    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.exists():
        print(f"Error: {args.leaderboard} not found", file=sys.stderr)
        sys.exit(1)

    content = leaderboard_path.read_text(encoding="utf-8")

    # Replace placeholder row if present, otherwise append after the last table row
    placeholder_pattern = r"\| \(add entries here\)[\s|]*"
    if re.search(placeholder_pattern, content):
        content = re.sub(placeholder_pattern, new_row, content, count=1)
    else:
        # Find the last markdown table row (any line starting and ending with |)
        lines = content.split("\n")
        last_table_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                last_table_idx = idx
        if last_table_idx is None:
            print("Error: could not find table in LEADERBOARD.md", file=sys.stderr)
            sys.exit(1)
        lines.insert(last_table_idx + 1, new_row)
        content = "\n".join(lines)

    leaderboard_path.write_text(content, encoding="utf-8")
    print(f"Appended record to {args.leaderboard}:")
    print(new_row)


if __name__ == "__main__":
    main()
