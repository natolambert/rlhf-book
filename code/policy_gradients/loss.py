# Policy Gradient Loss Functions
#
# Original implementation by Zafir Stojanovski (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0
#
# Implements:
# - REINFORCE (Williams, 1992)
# - PPO (Schulman et al., 2017)
# - GRPO (Shao et al., 2024)
# - GSPO (Zheng et al., 2025)
# - CISPO (MiniMax, 2025)
# - SAPO (Qwen Team, 2025)
# - SDPO (Hübotter et al., 2026)

import torch
import torch.nn as nn

from .buffer import Experience


def approx_kl(
    log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
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
    return (tensor * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(
        dim=dim, keepdim=keepdim
    ).clamp_min(eps)


class GRPOLoss(nn.Module):
    """Group Relative Policy Optimization loss (Shao et al., 2024).

    GRPO uses group-level advantage normalization and clipped ratio updates.
    See Chapter 6 of RLHF Book for mathematical derivation.
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
        clipped_term = (
            ratio.clamp(1 - self.clip_eps_lo, 1 + self.clip_eps_hi) * experience.advantages
        )
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
        clipped_term = (
            seq_logprobs.clamp(1 - self.clip_eps_lo, 1 + self.clip_eps_hi) * experience.advantages
        )
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
    See Chapter 6 of RLHF Book for derivation from the policy gradient theorem.
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


class SAPOLoss(nn.Module):
    """Soft Adaptive Policy Optimization loss (Qwen Team, 2025).

    Replaces hard clipping with a smooth sigmoid gate that continuously
    attenuates gradients as tokens move off-policy.
    See Chapter 6 - Further Reading of the RLHF Book
    """

    def __init__(self, sapo_temp_pos: float, sapo_temp_neg: float, beta: float, **kwargs) -> None:
        super().__init__()
        self.sapo_temp_pos = sapo_temp_pos
        self.sapo_temp_neg = sapo_temp_neg
        self.beta = beta

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        # Token-level importance ratio
        ratio = (log_probs - experience.log_probs_old).exp()

        # Asymmetric temperature: tighter curve for negative advantages
        temps = torch.where(experience.advantages > 0, self.sapo_temp_pos, self.sapo_temp_neg)

        # Soft sigmoid gate
        soft_gate = torch.sigmoid(temps * (ratio - 1)) * 4 / temps
        policy_loss = -soft_gate * experience.advantages

        # Optional KL penalty (default for SAPO = 0)
        if self.beta:
            kl_loss = approx_kl(log_probs, experience.log_probs_ref, experience.action_mask)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device, dtype=torch.float32)

        loss = policy_loss + self.beta * kl_loss
        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss


class SDPOLoss(nn.Module):
    """Self-Distillation Policy Optimization loss (Hübotter et al., 2026).

    Hybrid objective: GRPO policy loss + reverse-KL self-distillation.
    The teacher is the same policy model conditioned on a richer context that
    includes the highest-reward rollout of the current group as a demonstration.
    The distillation term uses the token-level REINFORCE surrogate for
    reverse-KL(student || teacher):

        log_ratio = (student_log_probs - teacher_log_probs).detach()
        distill_loss = log_ratio * student_log_probs

    The successful rollout itself (and samples in groups with no successful
    rollout) are excluded from the distillation term via `distill_mask`, since
    the teacher would otherwise see the target completion verbatim in its own
    context and the surrogate would collapse to SFT on that completion.
    """

    def __init__(
        self,
        clip_eps_lo: float,
        clip_eps_hi: float,
        beta: float,
        distillation_weight: float,
        sdpo_is_clip: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.grpo = GRPOLoss(clip_eps_lo=clip_eps_lo, clip_eps_hi=clip_eps_hi, beta=beta)
        self.distillation_weight = distillation_weight
        self.sdpo_is_clip = sdpo_is_clip

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
        student_comp_log_probs: torch.Tensor | None = None,
        teacher_comp_log_probs: torch.Tensor | None = None,
        comp_action_mask: torch.Tensor | None = None,
        old_student_comp_log_probs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        grpo_loss = self.grpo(log_probs=log_probs, experience=experience)

        log_ratio = (student_comp_log_probs - teacher_comp_log_probs).detach()
        distill_token_loss = log_ratio * student_comp_log_probs
        if self.sdpo_is_clip is not None and old_student_comp_log_probs is not None:
            approx_log_ratio = (student_comp_log_probs - old_student_comp_log_probs).detach()
            approx_log_ratio = approx_log_ratio.clamp(min=-20.0, max=20.0)
            distill_token_loss = distill_token_loss * approx_log_ratio.exp().clamp(max=self.sdpo_is_clip)
        # Broadcast distill_mask [B, 1] over completion tokens [B, M].
        effective_mask = comp_action_mask * experience.distill_mask
        distill_loss = masked_mean(distill_token_loss, effective_mask, dim=-1).mean(dim=0)

        return grpo_loss + self.distillation_weight * distill_loss


class PPOLoss(nn.Module):
    """Proximal Policy Optimization loss (Schulman et al., 2017).

    PPO combines clipped policy updates with a value function baseline.
    See Chapter 6 of RLHF Book for the full derivation.
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
        returns = (
            experience.advantages + experience.values_old
        )  # A_t = G_t - V(s_t) => G_t = A_t + V(s_t)
        values_clipped = torch.clamp(
            values,
            experience.values_old - self.clip_eps_val,
            experience.values_old + self.clip_eps_val,
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
