#!/usr/bin/env python3
"""Read speedrun_metrics.json and print a one-line SPEEDRUN_SUMMARY for the leaderboard.

Usage (from code/):
  uv run python scripts/speedrun/summary_from_json.py
  uv run python scripts/speedrun/summary_from_json.py path/to/speedrun_metrics.json
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Print SPEEDRUN_SUMMARY line from speedrun_metrics.json")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="logs/speedrun/speedrun_metrics.json",
        help="Path to speedrun_metrics.json (default: logs/speedrun/speedrun_metrics.json)",
    )
    args = parser.parse_args()

    try:
        with open(args.json_path) as f:
            d = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.json_path} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    parts = [
        f"walltime_sec={d.get('walltime_sec')}",
        f"final_reward={d.get('final_reward')}",
        f"config={d.get('config')}",
        f"seed={d.get('seed')}",
    ]
    if d.get("target_reward") is not None:
        parts.append(f"target_reward={d.get('target_reward')}")
    if d.get("goal_reached_at_step") is not None:
        parts.append(f"goal_reached_at_step={d.get('goal_reached_at_step')}")
    if d.get("goal_walltime_sec") is not None:
        parts.append(f"goal_walltime_sec={d.get('goal_walltime_sec')}")

    print("SPEEDRUN_SUMMARY: " + " ".join(parts))


if __name__ == "__main__":
    main()
