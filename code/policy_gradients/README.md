# Policy Gradient Methods

Educational implementations of policy gradient algorithms for [RLHF Book](https://rlhfbook.com).
See **Chapter 6: Policy Gradient Methods** for mathematical derivations and intuitions.
See the parent [`code/README.md`](../README.md) for installation, configuration, and memory requirements.

## Algorithms

| Algorithm | Config | Key Idea |
|-----------|--------|----------|
| **REINFORCE** | `reinforce.yaml` | Williams (1992) — vanilla policy gradient |
| **RLOO** | `rloo.yaml` | REINFORCE Leave-One-Out (Ahmadian et al., 2024) |
| **PPO** | `ppo.yaml` | Proximal Policy Optimization (Schulman et al., 2017) |
| **GRPO** | `grpo.yaml` | Group Relative Policy Optimization (Shao et al., 2024) |
| **Dr. GRPO** | `drgrpo.yaml` | Dr. GRPO — removes length and difficulty bias (Liu et al., 2025) |
| **GSPO** | `gspo.yaml` | Group-Sequence Policy Optimization (Zheng et al., 2025) |
| **CISPO** | `cispo.yaml` | Clipped Importance Sampling PO (MiniMax, 2025) |

## Reference Runs

| Algorithm | wandb | Status |
|-----------|-------|--------|
| **REINFORCE** | [run](https://wandb.ai/natolambert/rlhf-book/runs/0uqbq4oz) | ✅ Validated |
| **RLOO** | [run](https://wandb.ai/natolambert/rlhf-book/runs/07xeasn8) | ✅ Validated |
| **PPO** | [run](https://wandb.ai/natolambert/rlhf-book/runs/yv21y1qm) | ⚠️ Experimental |
| **GRPO** | [run](https://wandb.ai/natolambert/rlhf-book/runs/vjp7lgdi) | ✅ Validated |
| **Dr. GRPO** | [run](https://wandb.ai/natolambert/rlhf-book/runs/a1swuynq) | ✅ Validated |
| **GSPO** | [run](https://wandb.ai/natolambert/rlhf-book/runs/10sxytli) | ✅ Validated |
| **CISPO** | [run](https://wandb.ai/natolambert/rlhf-book/runs/6dg0m06n) | ✅ Validated |

## Quick Start

```bash
cd /path/to/rlhf-book/code
uv sync

# GRPO (recommended starting point)
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml

# REINFORCE baseline
uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml

# PPO with value function
uv run python -m policy_gradients.train --config policy_gradients/configs/ppo.yaml
```

## TODOs for Community Contributions

- [ ] Tune PPO hyperparameters for cleaner training curves (currently experimental)
- [ ] Add evaluation on held-out reasoning benchmarks
