"""Numerical-stability tests for the k3 KL estimator.

The naive k3 form ``(exp(r) - 1) - r`` catastrophically cancels near ``r = 0``:
``exp(r) - 1`` loses the leading ``r`` term, so the result is dominated by float
round-off and can even take the wrong sign. The ``expm1``-based form
``expm1(r) - r`` keeps the quadratic behavior correct.
"""

from __future__ import annotations

import pytest
import torch

from policy_gradients.loss import approx_kl3


def _naive_kl3(log_ratio: torch.Tensor) -> torch.Tensor:
    """The pre-fix form ``(exp(r) - 1) - r`` that ``approx_kl3`` used to use."""
    return (log_ratio.exp() - 1) - log_ratio


def test_kl3_matches_quadratic_near_zero() -> None:  # Pins the mathematical target
    r = torch.tensor(1e-4)
    # k3 ~ r^2/2 for small r; the naive form gives ~3.3x error here.
    assert approx_kl3(r, torch.zeros_like(r), None).item() == pytest.approx(1e-4**2 / 2, rel=0.05)


def test_expm1_vs_naive_forms_diverge_only_where_cancellation_hurts() -> None:
    """Contrast the chosen ``expm1(r) - r`` against the naive ``(exp(r) - 1) - r``.

    Mathematically the two forms are identical for every ``r``. They differ only
    in float round-off: ``exp(r) - 1`` computes ``e^r`` then subtracts exactly 1,
    destroying the leading ``r`` term for tiny ``r`` (catastrophic cancellation).
    ``expm1`` computes ``e^r - 1`` in one pass, keeping those digits. We pick
    ``expm1`` so the KL stays on the correct ``~r^2/2`` branch even at high
    confidence where ``r`` approaches 0.
    """
    tiny = torch.tensor(1e-6)
    # expm1(r) - r is tiny, positive, ~r^2/2.
    assert approx_kl3(tiny, torch.zeros_like(tiny), None).item() == pytest.approx(
        1e-6**2 / 2, abs=1e-12
    )
    # The naive form round-trips through exp(r) - 1, losing the leading r term and
    # landing in pure float noise (here: the wrong sign).
    assert _naive_kl3(tiny).item() < 0.0

    # Both forms are mathematically identical when r is not tiny (no cancellation);
    # they may differ by a few ulps of round-off but agree to ~1e-7.
    larger = torch.tensor([0.5, 1.0, 2.0, 4.0])
    ref = torch.zeros_like(larger)
    assert torch.allclose(approx_kl3(larger, ref, None), _naive_kl3(larger), atol=1e-7)


def test_kl3_respects_action_mask() -> None:
    r = torch.ones(2, 4)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    # k3(1.0) = e - 2
    expected = torch.tensor([[torch.e - 2, torch.e - 2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(approx_kl3(r, torch.zeros_like(r), mask), expected)
