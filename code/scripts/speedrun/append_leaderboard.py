#!/usr/bin/env python3
"""Append a row to LEADERBOARD.md from speedrun_metrics.json.

See LEADERBOARD.md for usage details and sort order.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

MIN_COLUMNS = 12
_UNIT_TO_SEC = {"h": 3600, "hr": 3600, "hours": 3600, "min": 60, "mins": 60, "minutes": 60,
                "sec": 1, "s": 1, "secs": 1, "seconds": 1}


def load_metrics(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def format_walltime(sec: int) -> str:
    if sec < 60:
        return f"{sec} sec"
    if sec < 3600:
        m, s = divmod(sec, 60)
        return f"{m} min {s} sec"
    h, remainder = divmod(sec, 3600)
    m, s = divmod(remainder, 60)
    return f"{h} h {m} min {s} sec" if s > 0 else f"{h} h {m} min"


def parse_walltime(s: str) -> int | None:
    total = 0
    for num, unit in re.findall(r"(\d+)\s*([a-zA-Z]+)", s):
        total += int(num) * _UNIT_TO_SEC.get(unit.lower(), 0)
    return total or None


def parse_goal_step(s: str) -> tuple[float | None, int | None]:
    m = re.match(r"goal\(([\d.]+)\)@step(\d+)", s.strip())
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


def row_sort_key(dataset: str, model: str, target: float | None,
                 time_sec: int | None, step: int | None, date_str: str) -> tuple:
    try:
        date_ord = datetime.strptime(date_str.strip(), "%Y-%m-%d").toordinal()
    except ValueError:
        date_ord = 0
    return (dataset, model,
            -(target if target is not None else -999999.0),
            time_sec if time_sec is not None else 999999999,
            step if step is not None else 999999999,
            -date_ord)


def _parse_table(lines: list[str]) -> tuple[int | None, int | None, list[str]]:
    header_idx = separator_idx = None
    rows: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if len(cells) < MIN_COLUMNS:
            continue
        if cells[0] == "Date":
            header_idx = idx
        elif re.match(r"^-+$", cells[0]):
            separator_idx = idx
        elif header_idx is not None and separator_idx is not None and idx > separator_idx:
            rows.append(stripped)
    return header_idx, separator_idx, rows


def _rebuild_content(lines: list[str], separator_idx: int, sorted_rows: list[str]) -> str:
    before = "\n".join(lines[: separator_idx + 1])
    i = separator_idx + 1
    while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
        i += 1
    after = "\n".join(lines[i:]) if i < len(lines) else ""
    return before + "\n" + "\n".join(sorted_rows) + ("\n\n" + after if after else "")


def _sort_key_for_row(row: str, target_val: float | None = None,
                      goal_wt: int | None = None) -> tuple:
    cells = [c.strip() for c in row.split("|")[1:-1]]
    if len(cells) < MIN_COLUMNS:
        return ("", "", 0.0, 999999999, 999999999, 0)
    target_r, step = parse_goal_step(cells[4])
    time_sec = parse_walltime(cells[5])
    if target_val is not None and target_r is None:
        target_r = target_val
    if time_sec is None and goal_wt is not None:
        time_sec = goal_wt
    return row_sort_key(cells[3], cells[2], target_r, time_sec, step, cells[0])


def _sort_and_write(leaderboard_path: Path, rows: list[str],
                    target_val: float | None = None, goal_wt: int | None = None,
                    new_row_idx: int | None = None) -> str:
    """Sort rows and write to file. Returns rebuilt content."""
    lines = leaderboard_path.read_text(encoding="utf-8").split("\n")
    header_idx, separator_idx, _ = _parse_table(lines)
    if header_idx is None or separator_idx is None:
        print("Error: could not find table header in LEADERBOARD.md", file=sys.stderr)
        sys.exit(1)

    def key(idx_row: tuple[int, str]) -> tuple:
        idx, row = idx_row
        tv = target_val if idx == new_row_idx else None
        gw = goal_wt if idx == new_row_idx else None
        return _sort_key_for_row(row, tv, gw)

    sorted_rows = [r for _, r in sorted(enumerate(rows), key=key)]
    content = _rebuild_content(lines, separator_idx, sorted_rows)
    leaderboard_path.write_text(content, encoding="utf-8")
    return content


def _build_wandb_cell(d: dict) -> str:
    run_id = d.get("wandb_run_id")
    entity = d.get("wandb_entity", "")
    project = d.get("wandb_project", "")
    if run_id and entity and project:
        return f"[run](https://wandb.ai/{entity}/{project}/runs/{run_id})"
    if run_id:
        print("Warning: wandb_entity or wandb_project missing in JSON; cannot generate wandb link.",
              file=sys.stderr)
    return ""


def _find_latest_json(directory: str = "logs/speedrun") -> str | None:
    """Find the most recently modified JSON file in the directory."""
    jsons = sorted(Path(directory).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(jsons[0]) if jsons else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a record to LEADERBOARD.md")
    parser.add_argument("json_path", nargs="?", default=None, help="Path to speedrun JSON (default: latest in logs/speedrun/)")
    parser.add_argument("--recorder", default="", help="Runner name")
    parser.add_argument("--leaderboard", default="scripts/speedrun/LEADERBOARD.md")
    parser.add_argument("--notes", default="", help="Optional notes")
    parser.add_argument("--include-wandb", action="store_true")
    parser.add_argument("--sort-only", action="store_true")
    args = parser.parse_args()

    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.exists():
        print(f"Error: {args.leaderboard} not found", file=sys.stderr)
        sys.exit(1)

    if args.sort_only:
        _, _, table_rows = _parse_table(leaderboard_path.read_text(encoding="utf-8").split("\n"))
        if not table_rows:
            print("No table rows to sort.", file=sys.stderr)
            return
        _sort_and_write(leaderboard_path, table_rows)
        print(f"Re-sorted {len(table_rows)} rows in {args.leaderboard}")
        return

    json_path = args.json_path or _find_latest_json()
    if json_path is None:
        print("Error: no JSON files found in logs/speedrun/", file=sys.stderr)
        sys.exit(1)
    if not args.json_path:
        print(f"Using latest: {json_path}")

    try:
        d = load_metrics(json_path)
    except FileNotFoundError:
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract fields from JSON
    run_id = d.get("wandb_run_id", "")
    if not run_id:
        stem = Path(json_path).stem
        if stem != "speedrun_metrics":
            run_id = stem

    target = d.get("target_reward", "")
    goal_step = d.get("goal_reached_at_step")
    goal_wt = d.get("goal_walltime_sec")
    goal_display = ""
    if goal_step is not None:
        goal_display = f"goal({target})@step{goal_step}" if target != "" else f"goal@step{goal_step}"
    ttt_display = format_walltime(goal_wt) if goal_wt is not None else ""

    final_r = d.get("final_reward")
    wandb_cell = _build_wandb_cell(d) if args.include_wandb else ""

    try:
        target_val = float(target) if target not in ("", None) else None
    except (ValueError, TypeError):
        target_val = None

    new_row = (f"| {datetime.now():%Y-%m-%d} | {args.recorder} "
               f"| {d.get('model_name', '')} | {d.get('dataset', '')} "
               f"| {goal_display} | {ttt_display} | {run_id} "
               f"| {format_walltime(d.get('walltime_sec') or 0)} "
               f"| {f'{float(final_r):.4f}' if final_r is not None else ''} "
               f"| {d.get('algorithm', '')} | {wandb_cell} | {args.notes} |")

    content = leaderboard_path.read_text(encoding="utf-8")
    placeholder = r"\| \(add entries here\)[\s|]*"
    if re.search(placeholder, content):
        content = re.sub(placeholder, new_row, content, count=1)
        leaderboard_path.write_text(content, encoding="utf-8")
    else:
        _, _, table_rows = _parse_table(content.split("\n"))
        table_rows.append(new_row)
        _sort_and_write(leaderboard_path, table_rows, target_val, goal_wt,
                        new_row_idx=len(table_rows) - 1)

    print(f"Appended record to {args.leaderboard}:")
    print(new_row)


if __name__ == "__main__":
    main()
