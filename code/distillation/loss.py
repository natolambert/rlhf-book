import torch
import torch.nn as nn
import torch.nn.functional as F


def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a tail bucket so top-K log-probs form a valid (K+1) log-distribution."""
    log_sum = log_probs.logsumexp(dim=-1, keepdim=True).clamp(max=-1e-7)  # log(sum(top-k probs))
    tail = torch.log(-torch.expm1(log_sum))  # log(1 - sum(top-k probs))
    return torch.cat([log_probs, tail], dim=-1)


class SDPOLoss(nn.Module):
    def __init__(self, kl_top_k: int, teacher_chunk: int = 4) -> None:
        super().__init__()
        self.kl_top_k = kl_top_k
        self.teacher_chunk = teacher_chunk

    def forward(self, model, batch: dict) -> torch.Tensor:
        action_mask = batch["action_mask"]
        A = action_mask.shape[1]

        s_logits = model(
            input_ids=batch["s_ids"],
            attention_mask=batch["s_mask"],
            use_cache=False,
            logits_to_keep=A + 1,
        ).logits[:, :-1, :]
        s_topk, idx = s_logits.topk(self.kl_top_k, dim=-1)
        s_logp = s_topk - s_logits.logsumexp(dim=-1, keepdim=True)

        with torch.no_grad():
            t_logp = self._teacher_logp(model, batch, A, idx)

        s_logp, t_logp = add_tail(s_logp), add_tail(t_logp)
        kl = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(-1)

        return (kl * action_mask).sum() / action_mask.sum().clamp_min(1.0)

    def _teacher_logp(self, model, batch: dict, A: int, idx: torch.Tensor) -> torch.Tensor:
        """Top-K teacher log-probs, computed in rollout chunks to bound peak logit memory.

        The full-vocab teacher logits are the dominant memory cost. Processing the
        rollout batch in chunks lets each chunk's logits free before the next, so the
        peak is ``teacher_chunk`` rollouts wide instead of the whole batch.

        Args:
            model: The shared teacher/student model.
            batch: Rollout batch with ``t_ids`` and ``t_mask``.
            A: Number of action (completion) positions to keep.
            idx: Student top-K vocab indices to gather teacher log-probs at.

        Returns:
            Teacher top-K log-probs aligned with ``idx``.
        """
        t_ids, t_mask = batch["t_ids"], batch["t_mask"]
        chunks = []
        for i in range(0, t_ids.shape[0], self.teacher_chunk):
            sl = slice(i, i + self.teacher_chunk)
            logits = model(
                input_ids=t_ids[sl],
                attention_mask=t_mask[sl],
                use_cache=False,
                logits_to_keep=A + 1,
            ).logits[:, :-1, :]
            chunks.append(logits.gather(-1, idx[sl]) - logits.logsumexp(dim=-1, keepdim=True))
        return torch.cat(chunks, dim=0)
