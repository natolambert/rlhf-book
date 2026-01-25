# RLHF Book Code Examples

## Claude Code Notes

- **Always run training commands in background** using `run_in_background: true` to avoid blocking
- Check task progress with `tail -f /tmp/claude/.../tasks/<id>.output`
- Multiple training jobs can run in parallel if memory allows (~16GB per 0.6B model)

## Quick Start

```bash
cd /home/natolambert/dev/rlhf-book/code
uv sync

# Set wandb project (public project for book examples)
export WANDB_PROJECT=rlhf-book

# Run training scripts
uv run python -m reward_models.train_orm --samples 400 --epochs 2
uv run python -m reward_models.train_preference_rm --samples 2000 --epochs 1
uv run python -m reward_models.train_prm --samples 500 --epochs 2
uv run python -m policy_gradients.train --config configs/reinforce.yaml
```

## TODOs

### Immediate
- [x] Add .python-version to .gitignore
- [x] Refactor reward models to use base.py shared utilities
- [ ] **Remove QLoRA from reward models** - use full fine-tuning for simplicity
  - Small models (0.6B-1.7B) don't need LoRA for memory savings
  - Simplifies code and removes bitsandbytes/PEFT dependencies
  - May need to adjust learning rate (try 1e-5 or 5e-6)
- [x] Generate wandb runs for all algorithms:
  - [x] ORM (full fine-tune): https://wandb.ai/natolambert/rlhf-book/runs/xm8mlcpl
  - [x] Preference RM (Bradley-Terry): https://wandb.ai/natolambert/rlhf-book/runs/6sninll5
  - [x] PRM (Process RM): https://wandb.ai/natolambert/rlhf-book/runs/abhkbn4q
  - [x] REINFORCE: https://wandb.ai/natolambert/rlhf-book/runs/0uqbq4oz
  - [x] GRPO: https://wandb.ai/natolambert/rlhf-book/runs/vjp7lgdi
  - [x] RLOO: https://wandb.ai/natolambert/rlhf-book/runs/07xeasn8
  - [x] PPO: https://wandb.ai/natolambert/rlhf-book/runs/yv21y1qm
  - [x] Dr. GRPO: https://wandb.ai/natolambert/rlhf-book/runs/a1swuynq
  - [x] GSPO: https://wandb.ai/natolambert/rlhf-book/runs/10sxytli
  - [x] CISPO: https://wandb.ai/natolambert/rlhf-book/runs/6dg0m06n

- [x] Add README table with algorithm → wandb run links

### Reward Model Training

**Status: Experimental** - Needs tuning of hyperparameters, datasets, and models for cleaner training curves.

### Later
- [x] Refactor train_prm.py to use base.py utilities (done)
- [ ] Consider adding configs/ for reward model hyperparams
- [ ] Add evaluation scripts for reward models
- [ ] Delete commented LoRA code once full fine-tune metrics confirmed

## Model Recommendations

For educational examples on consumer GPUs:
- **Qwen3-0.6B-Base**: ~4GB VRAM, fastest training
- **Qwen3-1.7B-Base**: ~8-10GB VRAM, better quality
- **Qwen2.5-3B**: ~15-20GB VRAM, best quality

## Wandb Project

Public project: https://wandb.ai/natolambert/rlhf-book

All example runs should log here for documentation.

## Memory Notes

Without LoRA (full fine-tune):
- 0.6B model: ~4-6GB
- 1.7B model: ~10-15GB
- 3B model: ~20-25GB

With gradient checkpointing can reduce by ~30-40%.

## Learning Rate Notes

- With LoRA: 1e-4 to 5e-5 typical
- Without LoRA (full fine-tune): 1e-5 to 5e-6 typical (10x smaller)
