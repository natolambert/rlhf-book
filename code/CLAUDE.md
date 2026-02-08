# RLHF Book Code Examples

## Claude Code Notes

- **Always run Python commands with `uv run python`** (not bare `python`/`python3`) so the project environment is used consistently
- **Always run training commands in background** using `run_in_background: true` to avoid blocking
- Check task progress with `tail -f /tmp/claude/.../tasks/<id>.output`
- **Be careful with parallel jobs**: Only run one training job at a time unless you verify memory is available (`free -h`). Running too many can OOM the system.
- DGX Spark memory: ~120GB total (unified CPU/GPU), aim for <80GB usage to be safe

## Changelog Process

- Keep `CHANGELOG.md` minimal and release-oriented.
- Use exactly **one bullet per PR**.
- Each bullet must include a PR link and can contain multiple sentences summarizing meaningful changes.
- When changes affect comparability (metrics, logging semantics, evaluation logic), mention that directly in the same bullet.

## Experiment Organization

- Store direct-alignment experiment artifacts under `direct_alignment/experiments/`.
- For each experiment campaign, use a matched pair:
  - Log file: `YYYY-MM-DD-<slug>.md`
  - Asset folder: `YYYY-MM-DD-<slug>/`
- Put helper scripts directly inside the campaign folder at `direct_alignment/experiments/YYYY-MM-DD-<slug>/`.
- Keep root-level generic scripts separate; experiment-specific scripts should live with their experiment logs.

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
uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
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

### Direct Alignment (DPO and variants)

**Status: Implemented, needs testing**

- [ ] Test DPO training end-to-end on UltraFeedback
- [ ] Test IPO, SimPO, ORPO, KTO training
- [ ] Generate wandb runs for all algorithms:
  - [ ] DPO
  - [ ] IPO
  - [ ] SimPO
  - [ ] ORPO
  - [ ] KTO
- [ ] Add wandb run links to README table
- [ ] Verify OLMo-2-0425-1B-SFT works as default model
- [ ] Consider adding evaluation (e.g., AlpacaEval, MT-Bench style)
- [ ] Test data loading with Anthropic HH dataset

## Current Debug Plan (ORPO/SimPO, Feb 2026)

Context:
- ORPO/SimPO instability from extreme ORPO scales was addressed by switching to average log-probs.
- New issue is "stable but flat" learning, especially for ORPO.
- SimPO formula now uses gamma as a gamma/beta ratio: `-logsigmoid(beta * (logit_margin - gamma))`.
- Experiment log with full run ledger and W&B links: `direct_alignment/experiments/2026-02-08-orpo-simpo.md`.

Plan:
1. Run quick low-sample sanity jobs first (1 epoch, 640 samples) before long sweeps.
2. Confirm SimPO margins/accuracy move with the gamma-ratio formulation.
3. Sweep ORPO beta/LR to rebalance SFT vs preference term.
4. Only then launch 12.8K-sample full runs.

Small-run scripts:
- `direct_alignment/experiments/2026-02-08-orpo-simpo/run_simpo_small.sh`
- `direct_alignment/experiments/2026-02-08-orpo-simpo/run_orpo_small.sh`

Examples:
```bash
cd /home/natolambert/dev/rlhf-book/code

# SimPO sanity run (background)
WANDB_PROJECT=rlhf-book ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_simpo_small.sh

# ORPO sanity run (background)
WANDB_PROJECT=rlhf-book ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_orpo_small.sh
```

Useful overrides:
```bash
# SimPO: stronger margin push
GAMMA=1.0 LEARNING_RATE=1e-6 MAX_SAMPLES=640 NUM_EPOCHS=1 ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_simpo_small.sh

# ORPO: stronger preference term
BETA=1.0 LEARNING_RATE=5e-6 MAX_SAMPLES=640 NUM_EPOCHS=1 ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_orpo_small.sh
```

Acceptance criteria for small runs:
- SimPO: `accuracy` and `margins` trend upward in first ~10-20 optimizer steps.
- ORPO: `margins` move off near-zero and `or_loss` contributes non-trivially vs `sft_loss`.
- Samples remain coherent (no repetitive collapse).

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
