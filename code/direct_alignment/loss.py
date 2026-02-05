# Direct Alignment Algorithm Loss Functions
#
# Educational implementations for RLHF Book (https://rlhfbook.com)
# See Chapter 8 for mathematical derivations.
#
# References:
# - DPO: Rafailov et al., 2023 (https://arxiv.org/abs/2305.18290)
# - IPO: Azar et al., 2023 (https://arxiv.org/abs/2310.12036)
# - KTO: Ethayarajh et al., 2024 (https://arxiv.org/abs/2402.01306)
# - ORPO: Hong et al., 2024 (https://arxiv.org/abs/2403.07691)
# - SimPO: Meng et al., 2024 (https://arxiv.org/abs/2405.14734)

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Compute per-sequence log probabilities from logits.

    Args:
        logits: Model logits of shape (batch, seq_len, vocab_size)
        labels: Token ids of shape (batch, seq_len)
        mask: Attention mask of shape (batch, seq_len), 1 for valid tokens
        average_log_prob: If True, return average log prob per token (SimPO style)
                         If False, return sum of log probs (DPO style)

    Returns:
        Log probabilities of shape (batch,)
    """
    # Shift for autoregressive: predict token t from tokens 0...t-1
    logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    labels = labels[:, 1:]      # (batch, seq_len-1)
    mask = mask[:, 1:]          # (batch, seq_len-1)

    # Compute per-token log probs
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Mask out padding
    per_token_logps = per_token_logps * mask

    if average_log_prob:
        # SimPO: average over valid tokens to avoid length bias
        return per_token_logps.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    else:
        # DPO: sum over all tokens
        return per_token_logps.sum(dim=-1)


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    The core DPO loss maximizes the margin between chosen and rejected responses
    in log-probability space, normalized by a reference model.

    Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    where log_ratio = log(pi(y|x) / pi_ref(y|x))

    See Chapter 8 for derivation.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        Args:
            beta: Temperature parameter controlling KL penalty strength.
                  Higher beta = stronger preference signal, risk of overfitting.
                  Lower beta = more regularization toward reference model.
                  Typical values: 0.1-0.5
            label_smoothing: For cDPO variant. Assumes this fraction of labels
                            are incorrect. 0.0 = standard DPO, 0.1 = 10% noise.
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            policy_chosen_logps: Log probs of chosen responses from policy (batch,)
            policy_rejected_logps: Log probs of rejected responses from policy (batch,)
            ref_chosen_logps: Log probs of chosen responses from reference (batch,)
            ref_rejected_logps: Log probs of rejected responses from reference (batch,)

        Returns:
            loss: Scalar loss value
            metrics: Dict with chosen_rewards, rejected_rewards, margins
        """
        # Compute log ratios (implicit rewards)
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # DPO logits: difference in log ratios
        logits = chosen_logratios - rejected_logratios

        # cDPO: label smoothing for noisy preferences
        if self.label_smoothing > 0:
            # Soft labels: [1-eps, eps] instead of [1, 0]
            # This translates to: (1-eps) * log(sigmoid(x)) + eps * log(sigmoid(-x))
            # = (1-2*eps) * log(sigmoid(x)) + eps * log(sigmoid(x)) + eps * log(sigmoid(-x))
            # = (1-2*eps) * log(sigmoid(x)) + eps * log(1)  (since sigmoid(x) + sigmoid(-x) = 1)
            # Simplified: just use soft cross-entropy
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            # Standard DPO
            losses = -F.logsigmoid(self.beta * logits)

        # Compute implicit rewards for logging
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        metrics = {
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "margins": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        }

        return losses.mean(), metrics


class IPOLoss(nn.Module):
    """Identity Preference Optimization loss (Azar et al., 2023).

    IPO uses a regression objective instead of classification, making it more
    robust to noisy preference labels. It directly regresses the preference
    probability to 0.5 when the margin is zero.

    Loss = (log_ratio_chosen - log_ratio_rejected - 1/(2*beta))^2

    This changes the optimization from "maximize margin" to "achieve target margin".
    """

    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: Controls the target margin. The optimal margin is 1/(2*beta).
                  Higher beta = smaller target margin, more regularization.
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute IPO loss."""
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        logits = chosen_logratios - rejected_logratios

        # IPO: squared error to target margin
        # Target margin derived from preference probability of 0.5
        target_margin = 1.0 / (2.0 * self.beta)
        losses = (logits - target_margin) ** 2

        # Metrics
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        metrics = {
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "margins": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        }

        return losses.mean(), metrics


class SimPOLoss(nn.Module):
    """Simple Preference Optimization loss (Meng et al., 2024).

    SimPO makes two key changes to DPO:
    1. Uses average log prob instead of sum (length normalization)
    2. Removes the reference model (implicit in length normalization)
    3. Adds a target margin gamma for stronger preference signal

    Loss = -log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected) - gamma))

    The length normalization reduces bias toward shorter responses.
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        """
        Args:
            beta: Temperature parameter (typically higher than DPO, e.g., 2.0-2.5)
            gamma: Target margin. Pushes for chosen to be gamma better than rejected.
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,  # Already length-normalized
        policy_rejected_logps: torch.Tensor,  # Already length-normalized
        **kwargs,  # Ignore ref model logps
    ) -> tuple[torch.Tensor, dict]:
        """Compute SimPO loss.

        Note: SimPO expects average log probs (set average_log_prob=True in compute_logprobs)
        """
        # SimPO: no reference model, uses length-normalized logps
        logits = policy_chosen_logps - policy_rejected_logps

        # Apply margin and compute loss
        losses = -F.logsigmoid(self.beta * logits - self.gamma)

        metrics = {
            "chosen_logps": policy_chosen_logps.mean().item(),
            "rejected_logps": policy_rejected_logps.mean().item(),
            "margins": logits.mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
        }

        return losses.mean(), metrics


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(x)) numerically stably.

    For x close to 0, use log1p(-exp(x)).
    For x far from 0, use log(-expm1(x)) which is more stable.

    See: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    # Use -0.693 (log(0.5)) as the threshold
    mask = x > -0.693
    result = torch.empty_like(x)
    result[mask] = torch.log(-torch.expm1(x[mask]))
    result[~mask] = torch.log1p(-torch.exp(x[~mask]))
    return result


class ORPOLoss(nn.Module):
    """Odds Ratio Preference Optimization loss (Hong et al., 2024).

    ORPO combines supervised fine-tuning with preference optimization,
    eliminating the need for a reference model.

    Loss = NLL(y_chosen|x) - beta * log(sigmoid(log_odds_ratio))

    where log_odds_ratio = log(odds_chosen / odds_rejected)
    and odds(y) = P(y|x) / (1 - P(y|x))

    Derived from Eqs. (4) and (7) of https://arxiv.org/abs/2403.07691:
        log_odds = log_p - log(1-p) = log_p - log1mexp(log_p)
        log_odds_ratio = log_odds_chosen - log_odds_rejected

    The SFT term encourages the model to generate the chosen response,
    while the odds ratio term creates preference separation.

    Important implementation note:
    ORPO is very sensitive to log-prob scale. We feed average per-token log-probs
    (not sequence sums), which mirrors TRL's ORPOTrainer behavior and is more stable
    for long responses.
    """

    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: Weight of the odds ratio loss (lambda in the paper).
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        chosen_nll_loss: torch.Tensor,  # SFT loss on chosen
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Compute ORPO loss.

        Args:
            policy_chosen_logps: Average log probs for chosen (batch,)
            policy_rejected_logps: Average log probs for rejected (batch,)
            chosen_nll_loss: Negative log likelihood loss on chosen response
        """
        # Cast to float for numerical stability (following TRL)
        policy_chosen_logps = policy_chosen_logps.float()
        policy_rejected_logps = policy_rejected_logps.float()

        # ORPO log-odds expects log-probs strictly below zero.
        policy_chosen_logps = policy_chosen_logps.clamp(max=-1e-6)
        policy_rejected_logps = policy_rejected_logps.clamp(max=-1e-6)

        # Compute log odds ratio using the proper formula from the paper
        # log_odds = log(p) - log(1-p) = log_p - log1mexp(log_p)
        # log_odds_ratio = log_odds_chosen - log_odds_rejected
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            log1mexp(policy_chosen_logps) - log1mexp(policy_rejected_logps)
        )

        # ORPO uses logsigmoid of the log odds ratio
        ratio = F.logsigmoid(log_odds)

        # Combined loss: SFT - beta * log(sigmoid(log_odds))
        # Note: ratio is already negative (logsigmoid output), so we subtract
        loss = chosen_nll_loss - self.beta * ratio

        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        metrics = {
            "sft_loss": chosen_nll_loss.mean().item(),
            "or_loss": (-self.beta * ratio).mean().item(),
            "log_odds_ratio": log_odds.mean().item(),
            "chosen_logps": policy_chosen_logps.mean().item(),
            "rejected_logps": policy_rejected_logps.mean().item(),
            "margins": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        }

        return loss.mean(), metrics


class KTOLoss(nn.Module):
    """Kahneman-Tversky Optimization loss (Ethayarajh et al., 2024).

    KTO works with unpaired preference data - you only need to know if a response
    is "good" or "bad", not compare two responses directly. This is based on
    prospect theory from behavioral economics.

    For desirable (chosen) responses:
        Loss = 1 - sigmoid(beta * (log_ratio - KL))

    For undesirable (rejected) responses:
        Loss = 1 - sigmoid(beta * (KL - log_ratio))

    where KL is a reference point representing the expected divergence from the
    reference model on unrelated outputs.

    The asymmetric treatment reflects loss aversion from prospect theory.

    See: https://github.com/ContextualAI/HALOs for the official implementation.
    """

    def __init__(self, beta: float = 0.1, desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
        """
        Args:
            beta: Temperature parameter.
            desirable_weight: Weight for positive examples (default 1.0).
            undesirable_weight: Weight for negative examples (typically higher, e.g., 1.33).
                              The paper suggests ~1.33 based on prospect theory.
        """
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        policy_kl_logps: torch.Tensor | None = None,
        ref_kl_logps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute KTO loss.

        This implementation converts paired preference data to unpaired format by
        treating chosen responses as desirable and rejected as undesirable.

        Args:
            policy_chosen_logps: Log probs of chosen (desirable) responses from policy
            policy_rejected_logps: Log probs of rejected (undesirable) responses from policy
            ref_chosen_logps: Log probs of chosen responses from reference model
            ref_rejected_logps: Log probs of rejected responses from reference model
            policy_kl_logps: (Optional) Log probs from policy on separate KL samples
            ref_kl_logps: (Optional) Log probs from reference on separate KL samples

        Returns:
            loss: Scalar loss value
            metrics: Dict with diagnostic values

        Note:
            If policy_kl_logps and ref_kl_logps are provided, KL is computed from
            these separate samples (matching the paper). Otherwise, KL is estimated
            from shuffled batch samples, which is a reasonable approximation for
            paired preference data.
        """
        # Compute log ratios (implicit rewards) for chosen/rejected
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Compute KL reference point
        if policy_kl_logps is not None and ref_kl_logps is not None:
            # Full KTO: use separate samples for KL estimation
            kl_logratios = policy_kl_logps - ref_kl_logps
            KL = kl_logratios.mean().detach().clamp(min=0)
        else:
            # Approximation: use shuffled batch samples for KL
            # Shuffle rejected to pair with different prompts, simulating "unrelated" samples
            batch_size = chosen_logratios.shape[0]
            if batch_size > 1:
                # Roll rejected by 1 to pair with different prompts
                shifted_rejected = rejected_logratios.roll(shifts=1, dims=0)
                kl_logratios = shifted_rejected
            else:
                # Single sample: fall back to using all samples
                kl_logratios = torch.cat([chosen_logratios, rejected_logratios], dim=0)
            KL = kl_logratios.mean().detach().clamp(min=0)

        # KTO loss from Eqn (7) of the paper
        # Desirable: want chosen to be better than reference point
        chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
        # Undesirable: want rejected to be worse than reference point
        rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))

        # Weighted combination (asymmetric weights reflect loss aversion)
        loss = (
            self.desirable_weight * chosen_losses.mean() +
            self.undesirable_weight * rejected_losses.mean()
        )

        metrics = {
            "KL": KL.item(),
            "chosen_logratios": chosen_logratios.mean().item(),
            "rejected_logratios": rejected_logratios.mean().item(),
            "chosen_loss": chosen_losses.mean().item(),
            "rejected_loss": rejected_losses.mean().item(),
            "accuracy": (chosen_logratios > rejected_logratios).float().mean().item(),
        }

        return loss, metrics


# Loss function registry
LOSS_FUNCTIONS = {
    "dpo": DPOLoss,
    "cdpo": lambda beta: DPOLoss(beta=beta, label_smoothing=0.1),
    "ipo": IPOLoss,
    "simpo": SimPOLoss,
    "orpo": ORPOLoss,
    "kto": KTOLoss,
}


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """Get loss function by name.

    Args:
        loss_type: One of 'dpo', 'cdpo', 'ipo', 'simpo', 'orpo', 'kto'
        **kwargs: Arguments passed to the loss function constructor

    Returns:
        Loss function module
    """
    if loss_type not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(LOSS_FUNCTIONS.keys())}")

    # Filter kwargs to only pass what each loss function accepts
    beta = kwargs.get("beta", 0.1)

    if loss_type == "dpo":
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        return DPOLoss(beta=beta, label_smoothing=label_smoothing or 0.0)
    elif loss_type == "cdpo":
        return DPOLoss(beta=beta, label_smoothing=0.1)
    elif loss_type == "ipo":
        return IPOLoss(beta=beta)
    elif loss_type == "simpo":
        gamma = kwargs.get("gamma")
        # Use 0.5 as default only if gamma is None (not if it's 0.0)
        return SimPOLoss(beta=beta, gamma=gamma if gamma is not None else 0.5)
    elif loss_type == "orpo":
        return ORPOLoss(beta=beta)
    elif loss_type == "kto":
        return KTOLoss(beta=beta)
    else:
        # Fallback (shouldn't reach here due to earlier check)
        return LOSS_FUNCTIONS[loss_type](**kwargs)
