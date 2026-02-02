# Reward Model Training

Educational implementations of reward model training for [RLHF Book](https://rlhfbook.com).
See **Chapter 5: Reward Models** for mathematical derivations and intuitions.

> **⚠️ IN DEVELOPMENT**: These implementations are experimental. Hyperparameters, datasets, and model configurations have not been fully tuned for clean training curves. Contributions welcome!

## Algorithms

| Algorithm | Script | Key Idea |
|-----------|--------|----------|
| **ORM** | `train_orm.py` | Outcome Reward Model - scores full responses |
| **Preference RM** | `train_preference_rm.py` | Bradley-Terry model for pairwise preferences |
| **PRM** | `train_prm.py` | Process Reward Model - scores intermediate steps |

## Reference Runs

| Algorithm | wandb | Status |
|-----------|-------|--------|
| **ORM** | [run](https://wandb.ai/natolambert/rlhf-book/runs/xm8mlcpl) | Experimental |
| **Preference RM** | [run](https://wandb.ai/natolambert/rlhf-book/runs/6sninll5) | Experimental |
| **PRM** | [run](https://wandb.ai/natolambert/rlhf-book/runs/abhkbn4q) | Experimental |

## Quick Start

```bash
cd /home/natolambert/dev/rlhf-book/code
uv sync

# Train ORM
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_orm --samples 400 --epochs 2

# Train Preference RM (Bradley-Terry)
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_preference_rm --samples 2000 --epochs 1

# Train PRM
WANDB_PROJECT=rlhf-book uv run python -m reward_models.train_prm --samples 500 --epochs 2
```

## Known Issues

- Training curves may be noisy - hyperparameters not yet optimized
- Dataset selection and preprocessing may need refinement
- Model architectures are simplified for educational purposes

## TODOs for Community Contributions

- [ ] Tune hyperparameters for cleaner training curves
- [ ] Add config files (like direct_alignment has)
- [ ] Evaluate on standard benchmarks (RewardBench)
- [ ] Add data augmentation and curriculum learning
