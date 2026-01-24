# Experience Buffer for Policy Gradient Training
#
# Original implementation by Zarif Stojano (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0

from dataclasses import dataclass, fields
from typing import Self

import torch
import torch.nn.functional as F


@dataclass
class Experience:
    """A single experience tuple for policy gradient training.

    Stores all information needed to compute policy gradient updates:
    - sequence_ids: Full token sequence (prompt + completion)
    - attention_mask: Mask for valid tokens
    - action_mask: Mask for generated tokens (not prompt)
    - advantages: Computed advantages for each token
    - log_probs_old: Log probabilities from the rollout policy
    - log_probs_ref: Log probabilities from the reference policy (for KL)
    - values_old: Value estimates (for PPO)
    """

    sequence_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    advantages: torch.Tensor | None = None
    log_probs_old: torch.Tensor | None = None
    log_probs_ref: torch.Tensor | None = None
    values_old: torch.Tensor | None = None

    def to(self, device: torch.device) -> Self:
        """Move all tensors to the specified device."""
        field_names = [f.name for f in fields(self)]
        moved_tensors = {
            name: getattr(self, name).to(device) for name in field_names if getattr(self, name) is not None
        }
        return Experience(**moved_tensors)


def split_experience_batch(experience: Experience) -> list[Experience]:
    """Split a batched experience into individual experiences."""
    batch_size = experience.sequence_ids.size(0)
    batch_data = [{} for _ in range(batch_size)]
    field_names = [f.name for f in fields(experience)]
    for field_name in field_names:
        val = getattr(experience, field_name)
        if val is not None:
            vals = torch.unbind(val, dim=0)
        else:
            vals = [None] * batch_size
        assert len(vals) == batch_size
        for i in range(batch_size):
            batch_data[i][field_name] = vals[i]
    return [Experience(**data) for data in batch_data]


def pad_sequences(tensor_list: list[torch.Tensor], how: str = "start") -> torch.Tensor:
    """Pad sequences to the same length."""
    assert how in ("start", "end")
    max_len = max(t.size(0) for t in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_len = max_len - tensor.size(0)
        padding = (pad_len, 0) if how == "start" else (0, pad_len)
        padded_tensors.append(F.pad(tensor, padding))
    return torch.stack(padded_tensors, dim=0)


def join_experiences_batch(experiences: list[Experience]) -> Experience:
    """Join a list of experiences into a single batched experience."""
    batch_data = {}
    field_names = [f.name for f in fields(Experience)]
    for field_name in field_names:
        vals = [getattr(exp, field_name) for exp in experiences]
        if all(v is not None for v in vals):
            data = pad_sequences(vals, how="start")
        else:
            data = None
        batch_data[field_name] = data
    return Experience(**batch_data)


class ReplayBuffer:
    """Simple replay buffer for storing experiences during rollouts."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.buffer: list[Experience] = []

    def add(self, experience: Experience) -> None:
        """Add a batched experience to the buffer (splits into individual experiences)."""
        items = split_experience_batch(experience)
        self.buffer.extend(items)
        if self.limit and len(self.buffer) > self.limit:
            excess = len(self.buffer) - self.limit
            self.buffer = self.buffer[excess:]

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Experience:
        return self.buffer[idx]
