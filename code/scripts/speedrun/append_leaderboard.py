#!/usr/bin/env python3
"""Append a row to LEADERBOARD.md from speedrun_metrics.json.

Sort order: target_reward desc, time_to_target asc, goal@step asc, date desc.
(Within same target, fastest achievers first; non-achievers last.)

Usage (from code/):
  uv run python scripts/speedrun/append_leaderboard.py
  uv run python scripts/speedrun/append_leaderboard.py --recorder "shota"
  uv run python scripts/speedrun/append_leaderboard.py path/to/speedrun_metrics.json
  uv run python scripts/speedrun/append_leaderboard.py --sort-only  # re-sort existing rows only
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


def parse_walltime(s: str) -> int | None:
    """Parse walltime string (e.g. '9 h 21 min 34 sec') back to seconds."""
    s = s.strip()
    if not s:
        return None
    total = 0
    tokens = s.split()
    i = 0
    while i < len(tokens):
        try:
            num = int(float(tokens[i]))
        except (ValueError, IndexError):
            i += 1
            continue
        if i + 1 < len(tokens):
            unit = tokens[i + 1].lower()
            if unit in ("h", "hr", "hours"):
                total += num * 3600
                i += 2
            elif unit in ("min", "mins", "minutes"):
                total += num * 60
                i += 2
            elif unit in ("sec", "s", "secs", "seconds"):
                total += num
                i += 2
            else:
                i += 1
        else:
            i += 1
    return total if total > 0 else None


def parse_goal_step(goal_step_str: str) -> tuple[float | None, int | None]:
    """Parse 'goal(1.35)@step196' -> (1.35, 196). Return (None, None) if empty."""
    goal_step_str = goal_step_str.strip()
    if not goal_step_str:
        return (None, None)
    m = re.match(r"goal\(([\d.]+)\)@step(\d+)", goal_step_str)
    if m:
        return (float(m.group(1)), int(m.group(2)))
    return (None, None)


def row_sort_key(
    target_reward: float | None,
    time_to_target_sec: int | None,
    goal_step: int | None,
    date_str: str,
) -> tuple:
    """Sort key: target desc, time_to_target asc, step asc, date desc.
    Rows without goal (未達) go after achievers within same target.
    """
    target = -(target_reward if target_reward is not None else -999999.0)
    time_sec = time_to_target_sec if time_to_target_sec is not None else 999999999
    step = goal_step if goal_step is not None else 999999999
    try:
        date_ord = datetime.strptime(date_str.strip(), "%Y-%m-%d").toordinal()
    except ValueError:
        date_ord = 0
    return (target, time_sec, step, -date_ord)


def _run_sort_only(args: argparse.Namespace) -> None:
    """Re-sort existing table rows without adding a new one."""
    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.exists():
        print(f"Error: {args.leaderboard} not found", file=sys.stderr)
        sys.exit(1)

    content = leaderboard_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    table_rows: list[str] = []
    header_idx = None
    separator_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if len(cells) >= 10 and cells[0] == "Date":
            header_idx = idx
        elif len(cells) >= 10 and re.match(r"^-+$", cells[0]):
            separator_idx = idx
        elif header_idx is not None and separator_idx is not None and idx > separator_idx and len(cells) >= 10:
            table_rows.append(stripped)

    if not table_rows:
        print("No table rows to sort.", file=sys.stderr)
        return

    sorted_rows = sorted(table_rows, key=lambda row: get_sort_key_for_row(row, None, None))

    before = "\n".join(lines[: separator_idx + 1])
    i = separator_idx + 1
    while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
        i += 1
    after = "\n".join(lines[i:]) if i < len(lines) else ""
    new_content = before + "\n" + "\n".join(sorted_rows) + ("\n\n" + after if after else "")

    leaderboard_path.write_text(new_content, encoding="utf-8")
    print(f"Re-sorted {len(table_rows)} rows in {args.leaderboard}")


def get_sort_key_for_row(
    row: str,
    target_reward_val: float | None,
    goal_walltime_sec: int | None,
) -> tuple:
    """Extract sort key from a table row. Use target_reward_val/goal_walltime_sec for new row from JSON."""
    cells = [c.strip() for c in row.split("|")[1:-1]]
    if len(cells) < 10:
        return (0.0, 999999999, 999999999, 0)
    date_str_c = cells[0]
    goal_step_str = cells[2]
    time_to_target_str = cells[3]
    target_r, step = parse_goal_step(goal_step_str)
    time_sec = parse_walltime(time_to_target_str)
    if target_reward_val is not None and target_r is None:
        target_r = target_reward_val
    if time_sec is None and goal_walltime_sec is not None:
        time_sec = goal_walltime_sec
    return row_sort_key(target_r, time_sec, step, date_str_c)


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
    parser.add_argument(
        "--sort-only",
        action="store_true",
        help="Only re-sort existing table rows (no new row added); requires --leaderboard",
    )
    args = parser.parse_args()

    if args.sort_only:
        _run_sort_only(args)
        return

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

    # target_reward for new row (for sorting)
    try:
        target_reward_val = float(target) if target not in ("", None) else None
    except (ValueError, TypeError):
        target_reward_val = None

    new_row = f"| {date_str} | {args.recorder} | {goal_step_display} | {time_to_target_display} | {run_id} | {walltime_display} | {final_reward_str} | {config} | {wandb_cell} | {notes} |"

    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.exists():
        print(f"Error: {args.leaderboard} not found", file=sys.stderr)
        sys.exit(1)

    content = leaderboard_path.read_text(encoding="utf-8")

    # Replace placeholder row if present, otherwise merge new row and sort all rows
    placeholder_pattern = r"\| \(add entries here\)[\s|]*"
    if re.search(placeholder_pattern, content):
        content = re.sub(placeholder_pattern, new_row, content, count=1)
    else:
        # Parse all table rows (skip header and separator)
        lines = content.split("\n")
        table_rows: list[str] = []
        header_idx = None
        separator_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("|") or not stripped.endswith("|"):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(cells) >= 10 and cells[0] == "Date":
                header_idx = idx
            elif len(cells) >= 10 and re.match(r"^-+$", cells[0]):
                separator_idx = idx
            elif header_idx is not None and separator_idx is not None and idx > separator_idx and len(cells) >= 10:
                table_rows.append(stripped)

        table_rows.append(new_row)

        # Sort: target desc, time_to_target asc, step asc, date desc
        # For new row (last), pass JSON values; for existing rows, pass None
        def sort_key(idx_and_row: tuple[int, str]) -> tuple:
            idx, row = idx_and_row
            is_new = idx == len(table_rows) - 1
            return get_sort_key_for_row(
                row,
                target_reward_val if is_new else None,
                goal_walltime_sec if is_new else None,
            )

        sorted_row_strs = [row for _, row in sorted(enumerate(table_rows), key=sort_key)]

        # Rebuild content: everything before table + header + separator + sorted rows + everything after
        if header_idx is None or separator_idx is None:
            print("Error: could not find table header in LEADERBOARD.md", file=sys.stderr)
            sys.exit(1)
        before = "\n".join(lines[: separator_idx + 1])
        after_lines = []
        i = separator_idx + 1
        while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
            i += 1
        after = "\n".join(lines[i:]) if i < len(lines) else ""
        content = before + "\n" + "\n".join(sorted_row_strs) + ("\n\n" + after if after else "")

    leaderboard_path.write_text(content, encoding="utf-8")
    print(f"Appended record to {args.leaderboard}:")
    print(new_row)


if __name__ == "__main__":
    main()
