# Speedrun Tracking for Policy Gradient Training

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from rich.console import Console

from .config import Config


class SpeedrunTracker:
    """Track training rewards over time, detect goal crossing, and write metrics JSON.

    Usage in train.py::

        tracker = SpeedrunTracker(target_reward=1.35)

        for step, batch in enumerate(dataloader):
            ...
            avg_reward = torch.cat(rollout_rewards, dim=0).mean().item()
            tracker.record_step(avg_reward)
            ...
            tracker.check_goal(step, console)

        tracker.write_metrics(
            cfg=cfg,
            wandb_run_id=wandb_run_id,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project_name,
        )
    """

    def __init__(
        self,
        *,
        target_reward: float | None = None,
        metrics_file: str = "logs/speedrun/speedrun_metrics.json",
    ):
        self.target_reward = target_reward
        self.metrics_file = metrics_file
        self.start_time = time.time()

        self.reward_history: list[float] = []
        self.reward_100step_history: list[float | None] = []
        self.walltime_at_step: list[int] = []

        self._goal_reported = False
        self.goal_reached_at_step: int | None = None
        self.goal_walltime_sec: int | None = None

    @property
    def reward_100avg(self) -> float | None:
        """100-step rolling average, available from step 100 onwards."""
        if len(self.reward_history) >= 100:
            return sum(self.reward_history[-100:]) / 100
        return None

    def record_step(self, avg_reward: float) -> None:
        """Record one training step's reward and walltime."""
        self.reward_history.append(avg_reward)
        self.walltime_at_step.append(int(time.time() - self.start_time))
        self.reward_100step_history.append(self.reward_100avg)

    def check_goal(self, step: int, console: Console) -> None:
        """Print once when 100-step rolling average first crosses the target."""
        if self._goal_reported or self.target_reward is None:
            return
        if self.reward_100avg is not None and self.reward_100avg >= self.target_reward:
            self._goal_reported = True
            self.goal_reached_at_step = step + 1
            self.goal_walltime_sec = self.walltime_at_step[-1]
            console.print(
                f"[bold green]Speedrun goal reached[/bold green] at step {self.goal_reached_at_step} "
                f"(walltime: {self.goal_walltime_sec} sec)"
            )

    def write_metrics(
        self,
        *,
        cfg: Config,
        wandb_run_id: str | None = None,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
    ) -> None:
        """Write final metrics JSON after training completes."""
        metrics_path = self._resolve_metrics_path(wandb_run_id)

        payload: dict = {
            "final_reward": self.reward_history[-1] if self.reward_history else None,
            "reward_history": self.reward_history,
            "reward_100step_history": self.reward_100step_history,
            "walltime_at_step": self.walltime_at_step,
            "walltime_sec": int(time.time() - self.start_time),
            "algorithm": cfg.loss,
            "seed": cfg.seed,
            "model_name": cfg.model_name,
            "dataset": "+".join(s.name for s in cfg.data.specs),
            "target_reward": self.target_reward,
            "goal_reached_at_step": self.goal_reached_at_step,
            "goal_walltime_sec": self.goal_walltime_sec,
        }
        if wandb_run_id:
            payload["wandb_run_id"] = wandb_run_id
        if wandb_entity:
            payload["wandb_entity"] = wandb_entity
        if wandb_project:
            payload["wandb_project"] = wandb_project

        dirpath = os.path.dirname(metrics_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _resolve_metrics_path(self, wandb_run_id: str | None) -> str:
        default = "logs/speedrun/speedrun_metrics.json"
        if wandb_run_id and self.metrics_file == default:
            return os.path.join("logs", "speedrun", f"{wandb_run_id}.json")
        if not wandb_run_id and self.metrics_file == default:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join("logs", "speedrun", f"speedrun_{ts}.json")
        return self.metrics_file
