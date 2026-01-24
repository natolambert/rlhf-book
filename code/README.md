# RLHF Book - Code Examples

Educational code examples accompanying [RLHF Book](https://rlhfbook.com) by Nathan Lambert.

## Attribution

This code is built on the excellent work of community contributors:

### Policy Gradients

**Original Repository**: [zafstojano/policy-gradients](https://github.com/zafstojano/policy-gradients)
**Author**: Zarif Stojano ([@zafstojano](https://github.com/zafstojano))
**License**: Apache 2.0

A clean, educational implementation of policy gradient methods for reinforcement learning.
Implements REINFORCE, RLOO, PPO, GRPO, Dr. GRPO, GSPO, and CISPO with mathematical formulations
matching the book's Chapter 11 (Policy Gradient Methods).

### Reward Models (ORM/PRM)

**Original Repository**: [myhott163com/RLHF_ORM_PRM](https://github.com/myhott163com/RLHF_ORM_PRM)
**Author**: [@myhott163com](https://github.com/myhott163com)
**License**: MIT

Minimal implementations of Outcome Reward Models (ORM) and Process Reward Models (PRM),
demonstrating the concepts from Chapter 7 (Reward Models).

---

## Installation

```bash
cd code/
uv sync
```

### Platform-specific notes

**Standard x86_64 systems** (recommended): Flash Attention is installed by default for
significant speedups during training.

**DGX Spark / aarch64**: Flash Attention is not available on ARM64/Blackwell. The code
automatically falls back to PyTorch SDPA, which is actually faster on these systems due
to native cuDNN optimizations.

```bash
# On DGX Spark, just run:
uv sync
# Flash-attn will be skipped automatically on aarch64
```

## Policy Gradient Training

Train various policy gradient algorithms on procedural reasoning tasks:

```bash
# GRPO (Chapter 11)
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml

# PPO with value function
uv run python -m policy_gradients.train --config policy_gradients/configs/ppo.yaml

# REINFORCE baseline
uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml

# RLOO (Leave-One-Out)
uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml
```

### Available algorithms

| Algorithm | Config | Description |
|-----------|--------|-------------|
| REINFORCE | `reinforce.yaml` | Williams (1992) - vanilla policy gradient |
| RLOO | `rloo.yaml` | REINFORCE Leave-One-Out (Ahmadian et al., 2024) |
| PPO | `ppo.yaml` | Proximal Policy Optimization (Schulman et al., 2017) |
| GRPO | `grpo.yaml` | Group Relative Policy Optimization (Shao et al., 2024) |
| Dr. GRPO | `drgrpo.yaml` | Dr. GRPO (Liu et al., 2025) |
| GSPO | `gspo.yaml` | Group-Sequence Policy Optimization (Zheng et al., 2025) |
| CISPO | `cispo.yaml` | Clipped Importance Sampling PO (MiniMax, 2025) |

## Reward Model Training

Train reward models on math reasoning datasets:

```bash
# Outcome Reward Model (Chapter 7) - trains on GSM8K
uv run python -m reward_models.train_orm

# Process Reward Model (Chapter 7) - trains on PRM800K
uv run python -m reward_models.train_prm
```

### ORM (Outcome Reward Model)

Binary classification on solution correctness. Fine-tunes Qwen3-1.7B with LoRA on GSM8K,
learning to distinguish correct from incorrect math solutions.

### PRM (Process Reward Model)

Step-level classification on reasoning quality. Fine-tunes Qwen3-0.6B with LoRA on PRM800K,
learning to rate individual reasoning steps as {-1, 0, 1} (bad, neutral, good).

## Configuration

### Environment variables

```bash
# Weights & Biases logging (optional)
export WANDB_API_KEY="your-key"

# HuggingFace access (for gated models)
export HF_TOKEN="your-token"
```

### Memory requirements

| Training | Model | GPU Memory |
|----------|-------|------------|
| Policy gradients | Qwen3-1.7B | ~16GB (single GPU) |
| ORM | Qwen3-1.7B (4-bit) | ~8GB |
| PRM | Qwen3-0.6B (4-bit) | ~6GB |

## Book Chapters

These examples correspond to:

- **Chapter 7**: Reward Models (ORM, PRM)
- **Chapter 11**: Policy Gradient Methods (REINFORCE, PPO, GRPO, etc.)

See [rlhfbook.com](https://rlhfbook.com) for the full text.

## License

- Policy gradients code: Apache 2.0 (from zafstojano/policy-gradients)
- Reward models code: MIT (from myhott163com/RLHF_ORM_PRM)
- Adaptations and documentation: Apache 2.0
