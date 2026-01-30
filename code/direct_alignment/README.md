# Direct Alignment Algorithms

Educational implementations of direct alignment methods for [RLHF Book](https://rlhfbook.com).
See **Chapter 12: Direct Alignment** for mathematical derivations and intuitions.

## Algorithms

| Algorithm | Paper | Key Idea |
|-----------|-------|----------|
| **DPO** | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) | Core direct alignment method - implicit reward via log-ratios |
| **cDPO** | Same as DPO | DPO with label smoothing for noisy preferences |
| **IPO** | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) | Regression objective instead of classification - more robust |
| **SimPO** | [Meng et al., 2024](https://arxiv.org/abs/2405.14734) | Length-normalized, no reference model needed |
| **ORPO** | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) | Combines SFT + preference, no reference model |
| **KTO** | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) | Unpaired preferences, prospect theory inspired |

## Quick Start

```bash
cd /home/natolambert/dev/rlhf-book/code
uv sync

# Train DPO on 1k samples (quick test)
uv run python -m direct_alignment.train --loss dpo --max_samples 1000

# Train with config file
uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml

# Train with wandb logging
WANDB_PROJECT=rlhf-book uv run python -m direct_alignment.train --loss dpo
```

## Loss Functions

### DPO Loss

The core DPO loss maximizes the margin between chosen and rejected:

```python
# From Chapter 12, Equation 12.1
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(beta * logits)
```

### SimPO Loss

SimPO removes the reference model and uses length normalization:

```python
# Average log probs instead of sum
avg_logp_chosen = sum(log_probs) / num_tokens
logits = avg_logp_chosen - avg_logp_rejected
loss = -F.logsigmoid(beta * logits - gamma)
```

### IPO Loss

IPO uses squared error to a target margin:

```python
target_margin = 1.0 / (2.0 * beta)
loss = (logits - target_margin) ** 2
```

## Datasets

The default dataset is `argilla/ultrafeedback-binarized-preferences-cleaned`, a cleaned
version of UltraFeedback with ~60k preference pairs.

Other compatible datasets:
- `Anthropic/hh-rlhf` - Anthropic's helpfulness/harmlessness data
- Any dataset with `prompt`, `chosen`, `rejected` columns

## Key Hyperparameters

| Parameter | DPO | IPO | SimPO |
|-----------|-----|-----|-------|
| `beta` | 0.1-0.5 | 0.1 | 2.0-2.5 |
| `learning_rate` | 5e-7 | 5e-7 | 5e-7 |
| Reference model | Yes | Yes | No |

**Important**: DPO requires very low learning rates (1e-7 to 5e-6). Higher rates cause divergence.

## Memory Requirements

With gradient checkpointing on a single GPU:
- 1B model: ~8-10GB
- 3B model: ~15-20GB

## References

- [DPO Original Implementation](https://github.com/eric-mitchell/direct-preference-optimization)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL ORPOTrainer](https://huggingface.co/docs/trl/orpo_trainer)
- [KTO HALOs Repository](https://github.com/ContextualAI/HALOs)
