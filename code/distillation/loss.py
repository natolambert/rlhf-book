import torch
import torch.nn as nn
import torch.nn.functional as F


def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a tail bucket so top-K log-probs form a valid (K+1) log-distribution."""
    log_sum = log_probs.logsumexp(dim=-1, keepdim=True).clamp(max=-1e-7)  # log(sum(top-k probs))
    tail = torch.log(-torch.expm1(log_sum))  # log(1 - sum(top-k probs))
    return torch.cat([log_probs, tail], dim=-1)


class SDPOLoss(nn.Module):
    def __init__(self, kl_top_k: int, rollout_chunk: int = 4) -> None:
        super().__init__()
        self.kl_top_k = kl_top_k
        self.rollout_chunk = rollout_chunk

    def _chunk_loss(
        self, model, batch: dict, sl: slice, A: int, denom: torch.Tensor
    ) -> torch.Tensor:
        """SDPO top-K KL for one rollout slice, normalized by the global token count.

        Only ``logits_to_keep = A + 1`` positions are projected through the LM head, so
        the materialized logits cover exactly the completion tokens the loss reads.

        Args:
            model: Shared teacher/student model.
            batch: Rollout batch from ``generate_batch``.
            sl: Slice selecting the rollouts in this chunk.
            A: Number of completion (action) positions.
            denom: Global action-token count to divide by, so the chunk losses sum to
                the full-group loss.

        Returns:
            Scalar loss contribution for this chunk.
        """
        s_logits = model(
            input_ids=batch["s_ids"][sl],
            attention_mask=batch["s_mask"][sl],
            use_cache=False,
            logits_to_keep=A + 1,
        ).logits[:, :-1, :]
        s_topk, idx = s_logits.topk(self.kl_top_k, dim=-1)
        s_logp = s_topk - s_logits.logsumexp(dim=-1, keepdim=True)

        with torch.no_grad():
            t_logits = model(
                input_ids=batch["t_ids"][sl],
                attention_mask=batch["t_mask"][sl],
                use_cache=False,
                logits_to_keep=A + 1,
            ).logits[:, :-1, :]
            t_logp = t_logits.gather(-1, idx) - t_logits.logsumexp(dim=-1, keepdim=True)

        s_logp, t_logp = add_tail(s_logp), add_tail(t_logp)
        # Reverse KL = KL(student || teacher): target is the student, so the gradient
        # flows through s_logp while the teacher (input) stays detached.
        kl = F.kl_div(t_logp, s_logp, reduction="none", log_target=True).sum(-1)
        return (kl * batch["action_mask"][sl]).sum() / denom

    def forward(self, model, batch: dict, scale: float = 1.0) -> float:
        """Accumulate SDPO gradients over rollout chunks, running backward per chunk.

        The student logits ``[R, A, V]`` and their gradient are the dominant memory
        cost. Splitting the ``R`` rollouts into ``rollout_chunk``-sized groups and
        backpropagating each group before the next frees those tensors between chunks,
        bounding peak memory to one chunk instead of the whole group. Because the loss
        is a sum over rollouts divided by the global token count, the chunk gradients
        accumulate to exactly the full-group gradient. Gradients add into the model
        parameters; this neither zeroes nor steps the optimizer.

        Args:
            model: Shared teacher/student model.
            batch: Rollout batch from ``generate_batch``.
            scale: Factor applied before backward, e.g. ``1 / prompts_per_step`` for
                gradient accumulation across prompts.

        Returns:
            The scaled loss value summed over chunks, as a float.
        """
        action_mask = batch["action_mask"]
        A = action_mask.shape[1]
        denom = action_mask.sum().clamp_min(1.0)
        total = 0.0
        for i in range(0, action_mask.shape[0], self.rollout_chunk):
            loss = (
                self._chunk_loss(model, batch, slice(i, i + self.rollout_chunk), A, denom) * scale
            )
            if not loss.isfinite():
                continue
            if loss.requires_grad:
                loss.backward()
            total += float(loss.detach())
        return total
