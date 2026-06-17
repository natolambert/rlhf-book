"""Unit tests for scripts/speedrun/append_leaderboard.py (beta column schema)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
APPEND_LB_PATH = CODE_ROOT / "scripts" / "speedrun" / "append_leaderboard.py"

TABLE_PREFIX = """# Leaderboard test fixture

| Date | Runner | model | dataset | goal@step | time_to_target | run_id | walltime | final_reward | algorithm | beta | wandb | Notes |
|------|--------|-------|---------|-----------|----------------|--------|----------|--------------|-----------|------|-------|-------|
"""

PLACEHOLDER_SUFFIX = "| (add entries here) |\n"

LEGACY_ROW = (
    "| 2026-03-02 | shota | Qwen/Qwen3-1.7B | spell_backward "
    "| goal(1.35)@step196 | 9 h 21 min 34 sec | legacy_run "
    "| 11 h 33 min 51 sec | 1.4531 | grpo | | | legacy note |"
)


def _import_append_leaderboard():
    spec = importlib.util.spec_from_file_location("append_leaderboard", APPEND_LB_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _row_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.split("|")[1:-1]]


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["append_leaderboard.py", *argv])
    _import_append_leaderboard().main()


@pytest.fixture
def append_lb():
    return _import_append_leaderboard()


def test_format_beta(append_lb) -> None:
    assert append_lb.format_beta(None) == ""
    assert append_lb.format_beta(0.0) == "0"
    assert append_lb.format_beta(0.01) == "0.01"


def test_append_reflects_beta_from_json(tmp_path, monkeypatch) -> None:
    leaderboard = tmp_path / "LEADERBOARD.md"
    leaderboard.write_text(TABLE_PREFIX + PLACEHOLDER_SUFFIX, encoding="utf-8")

    metrics = tmp_path / "beta_run.json"
    metrics.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "dataset": "spell_backward",
                "target_reward": 1.35,
                "goal_reached_at_step": 100,
                "goal_walltime_sec": 3600,
                "walltime_sec": 7200,
                "final_reward": 1.4,
                "algorithm": "ppo",
                "beta": 0.01,
                "wandb_run_id": "beta_run",
            }
        ),
        encoding="utf-8",
    )

    _run_main(
        monkeypatch,
        str(metrics),
        "--leaderboard",
        str(leaderboard),
        "--recorder",
        "tester",
    )

    append_lb = _import_append_leaderboard()
    _, _, rows = append_lb._parse_table(leaderboard.read_text(encoding="utf-8").split("\n"))
    assert len(rows) == 1
    cells = _row_cells(rows[0])
    assert len(cells) == append_lb.MIN_COLUMNS
    assert cells[6] == "beta_run"
    assert cells[10] == "0.01"


def test_legacy_empty_beta_and_new_beta_row_share_schema(tmp_path, monkeypatch, append_lb) -> None:
    leaderboard = tmp_path / "LEADERBOARD.md"
    leaderboard.write_text(TABLE_PREFIX + LEGACY_ROW + "\n", encoding="utf-8")

    metrics = tmp_path / "new_beta.json"
    metrics.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "dataset": "spell_backward",
                "target_reward": 1.4,
                "goal_reached_at_step": 50,
                "goal_walltime_sec": 1800,
                "walltime_sec": 3600,
                "final_reward": 1.45,
                "algorithm": "rloo",
                "beta": 0.1,
                "wandb_run_id": "new_beta_run",
            }
        ),
        encoding="utf-8",
    )

    _run_main(
        monkeypatch,
        str(metrics),
        "--leaderboard",
        str(leaderboard),
        "--recorder",
        "tester",
    )

    _, _, rows = append_lb._parse_table(leaderboard.read_text(encoding="utf-8").split("\n"))
    assert len(rows) == 2
    for row in rows:
        assert len(_row_cells(row)) == append_lb.MIN_COLUMNS

    by_run_id = { _row_cells(row)[6]: row for row in rows }
    assert _row_cells(by_run_id["legacy_run"])[10] == ""
    assert _row_cells(by_run_id["new_beta_run"])[10] == "0.1"


def test_sort_only_is_idempotent(tmp_path, monkeypatch, append_lb) -> None:
    unsorted_rows = [
        (
            "| 2026-01-01 | a | M | ds | goal(1.2)@step10 | 1 min 0 sec | run_low "
            "| 5 min 0 sec | 1.0 | grpo | | | |"
        ),
        (
            "| 2026-01-02 | a | M | ds | goal(1.4)@step20 | 2 min 0 sec | run_high "
            "| 6 min 0 sec | 1.0 | grpo | 0.01 | | |"
        ),
    ]
    leaderboard = tmp_path / "LEADERBOARD.md"
    leaderboard.write_text(
        TABLE_PREFIX + "\n".join(reversed(unsorted_rows)) + "\n",
        encoding="utf-8",
    )

    _run_main(monkeypatch, "--sort-only", "--leaderboard", str(leaderboard))
    _, _, rows_after_first = append_lb._parse_table(
        leaderboard.read_text(encoding="utf-8").split("\n")
    )

    _run_main(monkeypatch, "--sort-only", "--leaderboard", str(leaderboard))
    _, _, rows_after_second = append_lb._parse_table(
        leaderboard.read_text(encoding="utf-8").split("\n")
    )

    assert rows_after_first == rows_after_second
    assert _row_cells(rows_after_first[0])[6] == "run_high"
