import torch
import torch.nn as nn
import torch.nn.functional as F


def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a tail bucket so top-K log-probs form a valid (K+1) log-distribution."""
    log_sum = log_probs.logsumexp(dim=-1, keepdim=True).clamp(max=-1e-7)  # log(sum(top-k probs))
    tail = torch.log(-torch.expm1(log_sum))  # log(1 - sum(top-k probs))
    return torch.cat([log_probs, tail], dim=-1)


class SDPOLoss(nn.Module):
    def __init__(self, kl_top_k: int) -> None:
        super().__init__()
        self.kl_top_k = kl_top_k

    def forward(self, model, batch: dict) -> torch.Tensor:
        action_mask = batch["action_mask"]
        A = action_mask.shape[1]

        s_logits = model(
            input_ids=batch["s_ids"], attention_mask=batch["s_mask"], use_cache=False
        ).logits[:, -A - 1 : -1, :]
        s_topk, idx = s_logits.topk(self.kl_top_k, dim=-1)
        s_logp = s_topk - s_logits.logsumexp(dim=-1, keepdim=True)

        with torch.no_grad():
            t_logits = model(
                input_ids=batch["t_ids"], attention_mask=batch["t_mask"], use_cache=False
            ).logits[:, -A - 1 : -1, :]
            t_logp = t_logits.gather(-1, idx) - t_logits.logsumexp(dim=-1, keepdim=True)

        s_logp, t_logp = add_tail(s_logp), add_tail(t_logp)
        kl = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(-1)

        return (kl * action_mask).sum() / action_mask.sum().clamp_min(1.0)
