"""Smoke tests for the educational code package boundaries.

These intentionally stop at import and CLI-help coverage. They catch broken
module wiring without downloading datasets, loading models, or requiring GPUs.
"""

from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console


CODE_ROOT = Path(__file__).resolve().parents[1]

CORE_MODULES = [
    "direct_alignment",
    "direct_alignment.config",
    "direct_alignment.data",
    "direct_alignment.loss",
    "direct_alignment.profile_memory",
    "direct_alignment.train",
    "instruction_tuning",
    "instruction_tuning.config",
    "instruction_tuning.train",
    "instruction_tuning.utils",
    "policy_gradients",
    "policy_gradients.buffer",
    "policy_gradients.config",
    "policy_gradients.loss",
    "policy_gradients.rollout",
    "policy_gradients.train",
    "policy_gradients.utils",
    "rejection_sampling",
    "rejection_sampling.config",
    "rejection_sampling.preprocess",
    "rejection_sampling.selection",
    "rejection_sampling.train",
    "rejection_sampling.utils",
    "reward_models",
    "reward_models.base",
    "reward_models.train_orm",
    "reward_models.train_preference_rm",
    "reward_models.train_prm",
]

CLI_MODULES = [
    "direct_alignment.train",
    "instruction_tuning.train",
    "policy_gradients.train",
    "rejection_sampling.preprocess",
    "rejection_sampling.train",
    "reward_models.train_orm",
    "reward_models.train_preference_rm",
    "reward_models.train_prm",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_modules_render_help(module_name: str) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": f"{CODE_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=CODE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "usage:" in result.stdout.lower()


def test_progress_bar_helper_signatures() -> None:
    pg_utils = importlib.import_module("policy_gradients.utils")
    rs_utils = importlib.import_module("rejection_sampling.utils")
    console = Console(file=io.StringIO(), force_terminal=False)

    pg_utils.print_step_header(consumed=0, total=1)
    assert pg_utils.progress_bar() is not None

    rs_utils.print_step_header(console=console, step=0, total=1)
    assert rs_utils.progress_bar(console=console) is not None
