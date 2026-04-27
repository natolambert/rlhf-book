import torch
import torch.nn.functional as F

from .loss import get_approx_kl, masked_mean


def apply_reward_kl(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: torch.Tensor,
    beta: float,
    loss: str,
    kl_estimator: str,
) -> torch.Tensor:
    """Apply KL penalty to rewards (for REINFORCE/RLOO/PPO)."""
    if not beta or loss not in ["ppo", "rloo", "reinforce"]:
        return rewards
    kl_div = get_approx_kl(kl_estimator, log_probs, log_probs_ref, action_mask)
    kl_div = masked_mean(kl_div, mask=action_mask, dim=-1, keepdim=True)
    rewards = rewards - beta * kl_div
    return rewards


def compute_standardized_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute standardized advantages (GRPO, GSPO, CISPO, DAPO)"""
    return (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + eps)


def compute_nonstandardized_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute non-standardized advantages (Dr. GRPO)."""
    return rewards - rewards.mean(dim=0, keepdim=True)


def compute_loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute leave-one-out advantages (RLOO)."""
    K = rewards.shape[0]
    return (K / (K - 1)) * (rewards - rewards.mean(dim=0, keepdim=True))


def compute_gae(
    rewards: torch.Tensor, action_mask: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (PPO)."""
    B, S = action_mask.size()
    device = action_mask.device
    last_action_indices = action_mask.long().cumsum(dim=-1).argmax(dim=-1, keepdim=True)
    indices = torch.arange(S, device=device).unsqueeze(0)
    done = (indices >= last_action_indices).float()

    rewards = torch.zeros_like(action_mask, device=device, dtype=torch.float32).scatter_(
        dim=-1, index=last_action_indices, src=rewards
    )

    values = values.to(device)
    advantages = torch.zeros_like(action_mask, dtype=torch.float32, device=device)
    next_values = torch.zeros(B, device=device, dtype=torch.float32)
    running = torch.zeros(B, device=device, dtype=torch.float32)

    for t in reversed(range(S)):
        not_done = 1.0 - done[:, t]
        delta = rewards[:, t] + not_done * gamma * next_values - values[:, t]
        running = delta + not_done * gamma * lam * running
        advantages[:, t] = running
        next_values = values[:, t]

    advantages = advantages * action_mask
    return advantages


def compute_advantages(
    rewards: torch.Tensor,
    loss: str,
    action_mask: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    gamma: float | None = None,
    lam: float | None = None,
) -> torch.Tensor:
    """Compute advantages using the appropriate method for the loss function."""
    if loss in ["grpo", "gspo", "cispo", "sapo", "dapo"]:
        return compute_standardized_advantages(rewards)
    elif loss in ["drgrpo"]:
        return compute_nonstandardized_advantages(rewards)
    elif loss in ["rloo"]:
        return compute_loo_advantages(rewards)
    elif loss in ["ppo"]:
        if action_mask is None or values is None or gamma is None or lam is None:
            raise ValueError("PPO requires action_mask, values, gamma, and lam to compute GAE.")
        return compute_gae(rewards, action_mask, values, gamma, lam)
    else:
        return rewards


def compute_log_probs(
    model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor | None:
    """Compute log probabilities for each token in the sequence."""
    if not model:
        return None
    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(model.device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    return target_log_probs


def compute_values(
    model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor | None:
    """Compute value estimates for each position (PPO)."""
    if not model:
        return None
    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(model.device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    values = output.logits[:, :-1, :].squeeze(-1).to(torch.float32)
    return values
