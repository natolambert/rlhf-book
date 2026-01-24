# Policy Gradient Loss Functions
#
# Original implementation by Zarif Stojano (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0
#
# Implements:
# - REINFORCE (Williams, 1992)
# - PPO (Schulman et al., 2017)
# - GRPO (Shao et al., 2024)
# - GSPO (Zheng et al., 2025)
# - CISPO (MiniMax, 2025)

import torch
import torch.nn as nn

from .buffer import Experience


def approx_kl(log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Monte-Carlo approximation of KL divergence (k3 estimator).

    See: http://joschu.net/blog/kl-approx.html
    """
    log_ratio = log_probs - log_probs_ref
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return (log_ratio.exp() - 1) - log_ratio


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean over masked positions."""
    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    return (tensor * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)


class GRPOLoss(nn.Module):
    """Group Relative Policy Optimization loss (Shao et al., 2024).

    GRPO uses group-level advantage normalization and clipped ratio updates.
    See Chapter 11 of RLHF Book for mathematical derivation.
    """

    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, beta: float, **kwargs) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        # Policy loss with clipping
        ratio = (log_probs - experience.log_probs_old).exp()
        unclipped_term = ratio * experience.advantages
        clipped_term = ratio.clamp(1 - self.clip_eps_lo, 1 + self.clip_eps_hi) * experience.advantages
        policy_loss = -torch.min(unclipped_term, clipped_term)

        # Optional KL penalty
        if self.beta:
            kl_loss = approx_kl(log_probs, experience.log_probs_ref, experience.action_mask)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device, dtype=torch.float32)

        loss = policy_loss + self.beta * kl_loss
        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss


class GSPOLoss(nn.Module):
    """Group-Sequence Policy Optimization loss (Zheng et al., 2025).

    GSPO applies clipping to sequence-level log probability ratios.
    """

    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, beta: float, **kwargs) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        # Sequence-level ratio (average log prob difference)
        seq_logprobs = masked_mean(
            log_probs - experience.log_probs_old, mask=experience.action_mask, dim=-1, keepdim=True
        ).exp()
        unclipped_term = seq_logprobs * experience.advantages
        clipped_term = seq_logprobs.clamp(1 - self.clip_eps_lo, 1 + self.clip_eps_hi) * experience.advantages
        policy_loss = -torch.min(unclipped_term, clipped_term)

        # Optional KL penalty
        if self.beta:
            kl_loss = approx_kl(log_probs, experience.log_probs_ref, experience.action_mask)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device, dtype=torch.float32)

        loss = policy_loss + self.beta * kl_loss
        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss


class ReinforceLoss(nn.Module):
    """REINFORCE loss (Williams, 1992).

    The classic policy gradient: -log(pi) * advantage
    See Chapter 11 of RLHF Book for derivation from the policy gradient theorem.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        loss = -(log_probs * experience.advantages)
        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss


class CISPOLoss(nn.Module):
    """Clipped Importance Sampling Policy Optimization loss (MiniMax, 2025).

    CISPO uses stop-gradient on the clipped ratio, providing a different
    gradient signal than PPO/GRPO.
    """

    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, beta: float, **kwargs) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        # Stop gradient on clipped ratio
        with torch.no_grad():
            ratio = (log_probs - experience.log_probs_old).exp()
            clipped_ratio = ratio.clamp(1 - self.clip_eps_lo, 1 + self.clip_eps_hi)
        policy_loss = -clipped_ratio * experience.advantages * log_probs

        # Optional KL penalty
        if self.beta:
            kl_loss = approx_kl(log_probs, experience.log_probs_ref, experience.action_mask)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device, dtype=torch.float32)

        loss = policy_loss + self.beta * kl_loss
        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss


class PPOLoss(nn.Module):
    """Proximal Policy Optimization loss (Schulman et al., 2017).

    PPO combines clipped policy updates with a value function baseline.
    See Chapter 11 of RLHF Book for the full derivation.
    """

    def __init__(
        self,
        clip_eps_lo: float,
        clip_eps_hi: float,
        clip_eps_val: float,
        vf_coef: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.clip_eps_val = clip_eps_val
        self.vf_coef = vf_coef

    def forward(
        self, log_probs: torch.Tensor, experience: Experience, values: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # Value loss with clipping
        values = values.to(log_probs.device)
        returns = experience.advantages + experience.values_old  # A_t = G_t - V(s_t) => G_t = A_t + V(s_t)
        values_clipped = torch.clamp(
            values, experience.values_old - self.clip_eps_val, experience.values_old + self.clip_eps_val
        )
        val_unclipped_term = 0.5 * (returns - values) ** 2
        val_clipped_term = 0.5 * (returns - values_clipped) ** 2
        val_loss = torch.max(val_unclipped_term, val_clipped_term)

        # Policy loss with clipping
        policy_ratio = (log_probs - experience.log_probs_old).exp()
        policy_unclipped_term = policy_ratio * experience.advantages
        policy_clipped_term = (
            policy_ratio.clamp(1 - self.clip_eps_lo, 1 + self.clip_eps_hi) * experience.advantages
        )
        policy_loss = -torch.min(policy_unclipped_term, policy_clipped_term)

        loss = policy_loss + self.vf_coef * val_loss
        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss
