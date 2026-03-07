#!/usr/bin/env python3
"""Remove a row from LEADERBOARD.md by run_id.

Usage (from code/):
  uv run python scripts/speedrun/remove_leaderboard.py <run_id>
  uv run python scripts/speedrun/remove_leaderboard.py <run_id> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_ID_COL = 4  # 0-indexed column position of run_id in the table


def find_rows_by_run_id(lines: list[str], run_id: str) -> list[int]:
    """Return line indices whose run_id column matches exactly."""
    matches: list[int] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cols) <= RUN_ID_COL:
            continue
        if cols[RUN_ID_COL] == run_id:
            matches.append(idx)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove a record from LEADERBOARD.md by run_id"
    )
    parser.add_argument("run_id", help="run_id of the record to remove")
    parser.add_argument(
        "--leaderboard",
        type=str,
        default="scripts/speedrun/LEADERBOARD.md",
        help="Path to LEADERBOARD.md (relative to cwd)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the row that would be removed without actually deleting it",
    )
    args = parser.parse_args()

    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.exists():
        print(f"Error: {args.leaderboard} not found", file=sys.stderr)
        sys.exit(1)

    lines = leaderboard_path.read_text(encoding="utf-8").split("\n")
    to_remove = find_rows_by_run_id(lines, args.run_id)

    if not to_remove:
        print(f"No record found with run_id '{args.run_id}'", file=sys.stderr)
        sys.exit(1)

    for idx in to_remove:
        prefix = "[dry-run] Would remove" if args.dry_run else "Removing"
        print(f"{prefix}: {lines[idx]}")

    if args.dry_run:
        return

    new_lines = [line for i, line in enumerate(lines) if i not in to_remove]
    leaderboard_path.write_text("\n".join(new_lines), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
