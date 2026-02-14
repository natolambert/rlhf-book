#!/usr/bin/env python3
"""Append a row to LEADERBOARD.md from speedrun_metrics.json.

Usage (from code/):
  uv run python scripts/speedrun/append_leaderboard.py
  uv run python scripts/speedrun/append_leaderboard.py --recorder "shota"
  uv run python scripts/speedrun/append_leaderboard.py path/to/speedrun_metrics.json
"""

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
    config = d.get("config", "")
    seed = d.get("seed", "")
    notes = args.notes

    if final_reward is not None:
        final_reward_str = f"{float(final_reward):.4f}"
    else:
        final_reward_str = ""

    if d.get("goal_reached_at_step") is not None and d.get("goal_walltime_sec") is not None:
        goal_note = f"goal@step{d.get('goal_reached_at_step')}({format_walltime(d.get('goal_walltime_sec'))})"
        notes = f"{goal_note} {notes}".strip() if notes else goal_note

    walltime_display = format_walltime(walltime_sec)

    wandb_cell = ""
    if args.include_wandb:
        run_id = d.get("wandb_run_id")
        entity = d.get("wandb_entity") or "natolambert"
        project = d.get("wandb_project") or "rlhf-book"
        if run_id:
            url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            wandb_cell = f"[run]({url})"

    new_row = f"| {date_str} | {args.recorder} | {walltime_display} | {final_reward_str} | {config} | {seed} | {wandb_cell} | {notes} |"

    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.exists():
        print(f"Error: {args.leaderboard} not found", file=sys.stderr)
        sys.exit(1)

    content = leaderboard_path.read_text(encoding="utf-8")

    placeholder_pattern = r"\| \(add entries here\) \| \| \| \| \| \| \| \|"
    if re.search(placeholder_pattern, content):
        content = re.sub(placeholder_pattern, new_row, content, count=1)
    else:
        sep_8 = r"\|------\|--------\|----------\|--------------\|--------\|------\|-------\|-------\|"
        sep_7 = r"\|------\|--------\|----------\|--------------\|--------\|------\|-------\|"
        match = re.search((sep_8 + r"|" + sep_7) + r"\n(.+?)(?=\n\n|\Z)", content, re.DOTALL)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + "\n" + new_row + content[insert_pos:]
        else:
            table_end = content.rfind("|")
            if table_end == -1:
                print("Error: could not find table in LEADERBOARD.md", file=sys.stderr)
                sys.exit(1)
            last_newline = content.rfind("\n", 0, table_end)
            content = content[: last_newline + 1] + new_row + "\n" + content[last_newline + 1 :]

    leaderboard_path.write_text(content, encoding="utf-8")
    print(f"Appended record to {args.leaderboard}:")
    print(new_row)


if __name__ == "__main__":
    main()
