"""The ``binary`` reward must stay in {0, 1} when the scorer gives partial credit.

Reasoning Gym's ``spell_backward`` scores a near miss by the fraction of matching
characters, so correctness can land strictly between 0 and 1. MaxRL treats
``binary`` as a success indicator, and ``avg_binary`` is logged as a success rate,
so a well-formatted near miss must count as a failure.
"""

from __future__ import annotations

import torch

from policy_gradients.config import Config
from policy_gradients.utils import compute_maxrl_advantages, compute_rewards, create_dataset


def _wrap(answer: str) -> str:
    return f"<think>\nreverse it\n</think>\n<answer>{answer}</answer>"


def _rewards(completions: list[str]):
    cfg = Config(
        loss="maxrl",
        data={
            "size": 4,
            "specs": [{"name": "spell_backward", "config": {"min_word_len": 6, "max_word_len": 6}}],
        },
    )
    dataset = create_dataset(cfg)
    entry = dataset[0]
    answer = entry["answer"]
    near_miss = answer[:-1] + ("x" if answer[-1].lower() != "x" else "y")
    completions = [c.format(answer=answer, near_miss=near_miss) for c in completions]
    entries = [entry] * len(completions)
    lengths = [10] * len(completions)
    return compute_rewards(entries, completions, lengths, dataset, cfg, torch.device("cpu"))


def test_binary_reward_is_zero_for_partial_credit() -> None:
    rewards = _rewards([_wrap("{answer}"), _wrap("{near_miss}"), "<answer>{answer}</answer>"])

    correctness = rewards["correctness"].squeeze(-1)
    assert 0.0 < correctness[1].item() < 1.0  # the scorer really gives partial credit
    # Exact + formatted, near miss + formatted, exact + missing think tags.
    assert rewards["binary"].squeeze(-1).tolist() == [1.0, 0.0, 0.0]


def test_binary_reward_is_zero_when_float32_would_round_up() -> None:
    # A correct reversal with trailing junk scores six additions of 1/6, i.e.
    # 0.9999999999999999, which float32 rounds to exactly 1.0.
    rewards = _rewards([_wrap("{answer}.")])

    assert rewards["binary"].squeeze(-1).tolist() == [0.0]


def test_maxrl_does_not_reinforce_near_misses() -> None:
    rewards = _rewards(
        [_wrap("{answer}"), _wrap("{near_miss}"), _wrap("{near_miss}"), _wrap("zzzzzz")]
    )

    advantages = compute_maxrl_advantages(rewards["binary"]).squeeze(-1)
    assert advantages[0].item() > 0
    # Near misses are failures: same (negative) advantage as a fully wrong answer.
    assert (advantages[1:] < 0).all()
    assert advantages[1].item() == advantages[3].item()
