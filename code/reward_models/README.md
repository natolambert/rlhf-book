# Reward Model Training

![Overview of the RLHF training loop.](../../book/images/rlhf-overview.png)

Educational implementations of reward model training for [RLHF Book](https://rlhfbook.com).
See **Chapter 5: Reward Models** for mathematical derivations and intuitions.

> **⚠️ IN DEVELOPMENT**: These implementations are experimental. Preference RM now includes a config-driven example with validation logging and LR scheduling, but ORM/PRM configs, datasets, and evaluation still need refinement. Contributions welcome!

## Algorithms

| Algorithm | Script | Key Idea |
|-----------|--------|----------|
| **ORM** | `train_orm.py` | Outcome Reward Model - scores full responses |
| **Preference RM** | `train_preference_rm.py` | Bradley-Terry model for pairwise preferences |
| **PRM** | `train_prm.py` | Process Reward Model - scores intermediate steps |

## Reference Runs

| Algorithm | wandb | Status |
|-----------|-------|--------|
| **ORM** | [run](https://wandb.ai/rlhf-book/core/runs/xm8mlcpl) | Experimental |
| **Preference RM** | [run](https://wandb.ai/rlhf-book/core/runs/6sninll5) | Experimental |
| **PRM** | [run](https://wandb.ai/rlhf-book/core/runs/abhkbn4q) | Experimental |

## Quick Start

```bash
cd code/
uv sync

# Train ORM
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_orm --samples 400 --epochs 2

# Train Preference RM (Bradley-Terry)
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_preference_rm \
    --config reward_models/configs/preference_rm.yaml

# Or override config values for a smaller run
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_preference_rm \
    --config reward_models/configs/preference_rm.yaml \
    --samples 2000 --epochs 1

# Train PRM
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_prm --samples 500 --epochs 2
```

## Preference RM Configuration

The Preference RM script supports config-driven training via
`reward_models/configs/preference_rm.yaml`.

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

- [ ] Add config files, validation splits, and validation logging for PRM and ORM
- [ ] Evaluate on standard benchmarks (RewardBench)
- [ ] Add data augmentation and curriculum learning
