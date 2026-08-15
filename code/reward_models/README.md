# Reward Model Training

![Overview of the RLHF training loop.](../../book/images/rlhf-overview.png)

Educational implementations of reward model training for [RLHF Book](https://rlhfbook.com).
See **Chapter 5: Reward Models** for mathematical derivations and intuitions.

> **⚠️ IN DEVELOPMENT**: These implementations are experimental. All three reward models now include config-driven training with validation logging, but datasets and evaluation still need refinement. Contributions welcome!

## Algorithms

| Algorithm | Script | Key Idea |
|-----------|--------|----------|
| **ORM** | `train_orm.py` | Outcome Reward Model - scores full responses |
| **Preference RM** | `train_preference_rm.py` | Bradley-Terry model for pairwise preferences |
| **PRM** | `train_prm.py` | Process Reward Model - scores intermediate steps |

## Reference Runs

| Algorithm | wandb | Status |
|-----------|-------|--------|
| **ORM** | [run](https://wandb.ai/rlhf-book/core/runs/3gkoqb7f) | Experimental |
| **Preference RM** | [run](https://wandb.ai/rlhf-book/core/runs/1g3y9bcc) | Experimental |
| **PRM** | [run](https://wandb.ai/rlhf-book/core/runs/iv4d966d) | Experimental |

## Quick Start

```bash
cd code/
uv sync

# Train ORM
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_orm \
    --config reward_models/configs/orm.yaml

# Train Preference RM (Bradley-Terry)
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_preference_rm \
    --config reward_models/configs/preference_rm.yaml

# Train PRM
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_prm \
    --config reward_models/configs/prm.yaml
```

## Reward Model Configuration

The ORM, PRM, and Preference RM scripts use `reward_models/configs/orm.yaml`,
`reward_models/configs/prm.yaml`, and `reward_models/configs/preference_rm.yaml`,
respectively. For smaller runs, copy the YAML file and edit the copy.

The default config trains Qwen3-0.6B on 5k UltraFeedback preference pairs with:

- effective batch size 16
- learning rate 5e-5
- 2 epochs
- 10% validation split
- linear warmup + linear decay
- validation logging every 25 optimizer steps

These defaults were selected from a small sweep and are intended as a cleaner
educational baseline, not universally optimal hyperparameters.

Reward models are commonly trained for around one epoch to reduce overfitting. This example uses two epochs because it produced cleaner validation curves in a small local 5k-pair sweep, but users should monitor `val/loss` and `val/accuracy` during the second epoch and reduce `epochs` if validation metrics degrade.

## Known Issues

- Training curves may be noisy - hyperparameters not yet optimized
- Dataset selection and preprocessing may need refinement
- Model architectures are simplified for educational purposes

## TODOs for Community Contributions

- [ ] Evaluate on standard benchmarks (RewardBench)
- [ ] Add data augmentation and curriculum learning
