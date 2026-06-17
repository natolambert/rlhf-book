"""Unit tests for SpeedrunTracker (GPU not required)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from rich.console import Console

from policy_gradients.config import Config, DataConfig, DatasetSpec
from policy_gradients.speedrun import SpeedrunTracker

console = Console(record=True)


def _minimal_cfg() -> Config:
    return Config(
        data=DataConfig(specs=[DatasetSpec(name="spell_backward")]),
        loss="grpo",
        seed=42,
        model_name="Qwen/Qwen3-1.7B",
    )


def _started_tracker(**kwargs) -> SpeedrunTracker:
    tracker = SpeedrunTracker(**kwargs)
    tracker.start(1000.0)
    return tracker


def _record_steps(
    tracker: SpeedrunTracker,
    n: int,
    *,
    reward: float = 1.0,
) -> None:
    """record_step then check_goal, matching train.py call order."""
    for _ in range(n):
        tracker.record_step(reward)
        tracker.check_goal(console)


def test_record_step_requires_start() -> None:
    tracker = SpeedrunTracker()
    with pytest.raises(RuntimeError, match="start\\(\\) must be called"):
        tracker.record_step(1.0)


def test_reward_100avg_none_before_100_steps() -> None:
    tracker = _started_tracker(target_reward=0.5)
    _record_steps(tracker, 99)

    assert tracker.reward_100avg is None
    assert all(v is None for v in tracker.reward_100step_history)
    assert len(tracker.reward_history) == 99
    assert len(tracker.reward_100step_history) == 99


def test_goal_not_reached_at_99_records() -> None:
    tracker = _started_tracker(target_reward=0.5)
    _record_steps(tracker, 99)

    assert tracker.goal_reached_at_step is None
    assert tracker.goal_walltime_sec is None


def test_reward_100avg_at_step_100() -> None:
    tracker = _started_tracker(target_reward=0.5)
    _record_steps(tracker, 100, reward=1.0)

    assert tracker.reward_100avg == pytest.approx(1.0)
    assert tracker.reward_100step_history[-1] == pytest.approx(1.0)
    assert len(tracker.reward_history) == 100
    assert len(tracker.reward_100step_history) == 100


def test_goal_reached_at_step_100_on_crossing() -> None:
    tracker = _started_tracker(target_reward=0.5)
    _record_steps(tracker, 100, reward=1.0)

    assert tracker.goal_reached_at_step == 100
    assert tracker.goal_walltime_sec is not None


def test_goal_crossing_reported_once() -> None:
    tracker = _started_tracker(target_reward=0.5)
    _record_steps(tracker, 100, reward=1.0)
    assert tracker.goal_reached_at_step == 100

    _record_steps(tracker, 50, reward=1.0)

    assert tracker.goal_reached_at_step == 100


def test_check_goal_without_prior_record_step_off_by_one() -> None:
    """check_goal must run immediately after record_step (train.py contract)."""
    tracker = _started_tracker(target_reward=0.5)
    _record_steps(tracker, 99)

    # Bug: check_goal without a preceding record_step on the crossing iteration.
    tracker.check_goal(console)

    assert tracker.goal_reached_at_step is None

    tracker.record_step(1.0)
    tracker.check_goal(console)

    assert tracker.goal_reached_at_step == 100


def test_elapsed_sec_uses_start_time() -> None:
    tracker = SpeedrunTracker()
    tracker.start(1000.0)

    with patch("policy_gradients.speedrun.time.time", return_value=1005.7):
        tracker.record_step(1.0)

    assert tracker.walltime_at_step == [5]


def test_write_metrics_writes_json(tmp_path) -> None:
    metrics_file = tmp_path / "metrics.json"
    tracker = _started_tracker(
        target_reward=0.5,
        metrics_file=str(metrics_file),
    )
    _record_steps(tracker, 100, reward=0.9)

    with patch("policy_gradients.speedrun.time.time", return_value=1010.0):
        tracker.write_metrics(cfg=_minimal_cfg())

    assert metrics_file.exists()
    payload = json.loads(metrics_file.read_text())
    assert payload["goal_reached_at_step"] == 100
    assert payload["algorithm"] == "grpo"
    assert payload["seed"] == 42
    assert len(payload["reward_history"]) == 100
    assert len(payload["reward_100step_history"]) == 100
    assert payload["walltime_sec"] == 10
