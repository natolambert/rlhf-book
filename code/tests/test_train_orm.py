"""Download-free regression tests for the sequence-level ORM objective."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from reward_models.train_orm import (
    OutcomeRewardModel,
    pool_completion_logits,
    score_completion,
    sequence_accuracy_counts,
)


class _LogitModel(OutcomeRewardModel):
    """Minimal ORM whose inputs are used directly as token logits."""

    def __init__(self):
        nn.Module.__init__(self)
        self.head = nn.Identity()

    def get_hidden_states(self, input_ids, attention_mask):
        del attention_mask
        return input_ids.unsqueeze(-1)


def test_pool_completion_logits_uses_only_completion_tokens_and_extracts_targets():
    token_logits = torch.tensor(
        [
            [99.0, 0.0, 2.0, -99.0],
            [-50.0, 6.0, -70.0, 30.0],
        ]
    )
    labels = torch.tensor(
        [
            [-100, 1, 1, -100],
            [-100, 0, -100, -100],
        ]
    )

    sequence_logits, outcome_targets = pool_completion_logits(token_logits, labels)

    torch.testing.assert_close(sequence_logits, torch.tensor([1.0, 6.0]))
    torch.testing.assert_close(outcome_targets, torch.tensor([1.0, 0.0]))


def test_forward_averages_logits_before_bce_regression():
    model = _LogitModel()
    token_logits = torch.tensor([[0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([[1, 1]])

    loss, returned_logits = model(
        input_ids=token_logits,
        attention_mask=torch.ones_like(token_logits, dtype=torch.long),
        labels=labels,
    )

    torch.testing.assert_close(returned_logits, token_logits)
    assert loss.item() == pytest.approx(0.3132617)
    assert torch.sigmoid(returned_logits.mean()).item() == pytest.approx(0.7310586)


def test_forward_weights_sequences_equally_regardless_of_completion_length():
    model = _LogitModel()
    token_logits = torch.tensor(
        [
            [-2.0, 100.0, 100.0],
            [-2.0, -2.0, -2.0],
        ]
    )
    labels = torch.tensor(
        [
            [1, -100, -100],
            [0, 0, 0],
        ]
    )

    loss, _ = model(
        input_ids=token_logits,
        attention_mask=torch.ones_like(token_logits, dtype=torch.long),
        labels=labels,
    )
    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([-2.0, -2.0]), torch.tensor([1.0, 0.0])
    )
    token_weighted_loss = F.binary_cross_entropy_with_logits(
        torch.tensor([-2.0, -2.0, -2.0, -2.0]), torch.tensor([1.0, 0.0, 0.0, 0.0])
    )

    torch.testing.assert_close(loss, expected)
    assert not torch.isclose(loss, token_weighted_loss)


def test_pooling_is_invariant_to_prompt_and_padding_logits():
    labels = torch.tensor([[-100, -100, 1, 1, -100]])
    baseline = torch.tensor([[0.0, 0.0, -1.0, 3.0, 0.0]])
    changed_masked_positions = torch.tensor([[1e6, -1e6, -1.0, 3.0, math.pi]])

    baseline_logits, baseline_targets = pool_completion_logits(baseline, labels)
    changed_logits, changed_targets = pool_completion_logits(changed_masked_positions, labels)

    torch.testing.assert_close(changed_logits, baseline_logits)
    torch.testing.assert_close(changed_targets, baseline_targets)


def test_pooling_drops_empty_rows_but_keeps_valid_rows():
    token_logits = torch.tensor(
        [
            [7.0, 8.0, 9.0],
            [100.0, -1.0, 3.0],
            [-4.0, -5.0, -6.0],
        ]
    )
    labels = torch.tensor(
        [
            [-100, -100, -100],
            [-100, 0, 0],
            [-100, -100, -100],
        ]
    )

    sequence_logits, outcome_targets = pool_completion_logits(token_logits, labels)

    torch.testing.assert_close(sequence_logits, torch.tensor([1.0]))
    torch.testing.assert_close(outcome_targets, torch.tensor([0.0]))


def test_forward_returns_differentiable_zero_for_all_empty_rows():
    model = _LogitModel()
    token_logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    labels = torch.full((2, 2), -100)

    loss, _ = model(
        input_ids=token_logits,
        attention_mask=torch.ones_like(token_logits, dtype=torch.long),
        labels=labels,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    torch.testing.assert_close(token_logits.grad, torch.zeros_like(token_logits))


def test_sequence_accuracy_counts_sequences_and_ignores_empty_rows():
    token_logits = torch.tensor(
        [
            [9.0, 2.0, -1.0],
            [-9.0, -3.0, 1.0],
            [8.0, 8.0, 8.0],
            [4.0, -2.0, -2.0],
        ]
    )
    labels = torch.tensor(
        [
            [-100, 1, 1],
            [-100, 0, 0],
            [-100, -100, -100],
            [-100, 1, 1],
        ]
    )

    correct, examples = sequence_accuracy_counts(token_logits, labels)

    assert correct == 2
    assert examples == 3


class _FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "prompt":
            return {"input_ids": [99]}
        assert text == "completion<eos>"
        return {"input_ids": [0, 2]}


class _EchoModel(nn.Module):
    def forward(self, input_ids, attention_mask, labels=None):
        del attention_mask, labels
        return None, input_ids.float()


def test_score_completion_applies_sigmoid_after_mean_logit():
    score = score_completion(
        model=_EchoModel(),
        tokenizer=_FakeTokenizer(),
        prompt="prompt",
        completion="completion",
        device=torch.device("cpu"),
    )

    assert score == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())
    mean_token_probability = torch.sigmoid(torch.tensor([0.0, 2.0])).mean().item()
    assert score != pytest.approx(mean_token_probability)
