# RLHF Book - Code Examples

Educational code examples accompanying [RLHF Book](https://rlhfbook.com) by Nathan Lambert.

I primarily run experiments on a [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/). For setup advice, see my [dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup) guide.

*Note: There's an open PR [here](https://github.com/natolambert/rlhf-book/pull/328) exploring the idea of adding speedrun functionality to this repository — comment if you're interested in pushing this further or seeing it merged into main.*

## Attribution

This code is built on the excellent work of community contributors:

### Policy Gradients

**Original Repository**: [zafstojano/policy-gradients](https://github.com/zafstojano/policy-gradients)
**Author**: Zafir Stojanovski ([@zafstojano](https://github.com/zafstojano))
**License**: Apache 2.0

A clean, educational implementation of policy gradient methods for reinforcement learning.
Implements REINFORCE, RLOO, PPO, GRPO, Dr. GRPO, GSPO, and CISPO with mathematical formulations
matching the book's Chapter 6 (Policy Gradient Methods).

### Reward Models (ORM/PRM)

**Original Repository**: [myhott163com/RLHF_ORM_PRM](https://github.com/myhott163com/RLHF_ORM_PRM)
**Author**: [@myhott163com](https://github.com/myhott163com)
**License**: MIT

Minimal implementations of Outcome Reward Models (ORM) and Process Reward Models (PRM),
demonstrating the concepts from Chapter 5 (Reward Models).

---

## Installation

**Requires Python 3.12+** and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd code/
uv sync
```

By default, [Flash Attention](https://github.com/Dao-AILab/flash-attention) is turned off
to support a broad range of hardware, but for speedups you should consider installing it:

```bash
uv sync --extra flash
```

> **Note:** If a pre-built wheel matches your CUDA version this installs in seconds.
> If not (e.g. CUDA 13), it falls back to a source build which needs a CUDA toolkit
> and can take several minutes. If the build fails, just use the base install — the
> code automatically falls back to PyTorch SDPA and all examples will work without it.

### Platform notes

- **Standard x86_64 systems**: Flash Attention provides a ~10-20% training speedup on
  Ampere/Ada GPUs (e.g. 3090, 4090). Pre-built wheels are available for CUDA 12.x
  ([releases](https://github.com/Dao-AILab/flash-attention/releases/latest));
  as of 11 Apr. 2026 CUDA 13 requires a source build (which tends to be a pain),
  so nothing is gated on it.
- **DGX Spark / aarch64**: Flash Attention is not available on ARM64/Blackwell. The code
  automatically falls back to PyTorch SDPA, which is actually faster on these systems due
  to native cuDNN optimizations.

## Policy Gradient Training

Train various policy gradient algorithms on procedural reasoning tasks:

```bash
# GRPO (Chapter 6)
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml

# PPO with value function
uv run python -m policy_gradients.train --config policy_gradients/configs/ppo.yaml

# REINFORCE baseline
uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml

# RLOO (Leave-One-Out)
uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml
```

### Training Results

![Policy Gradient Training Results](images/wandb_policy_gradients.png)

### Available algorithms

| Algorithm | Config | Description | Example Run |
|-----------|--------|-------------|-------------|
| REINFORCE | `reinforce.yaml` | Williams (1992) - vanilla policy gradient | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/0uqbq4oz) |
| RLOO | `rloo.yaml` | REINFORCE Leave-One-Out (Ahmadian et al., 2024) | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/07xeasn8) |
| PPO | `ppo.yaml` | Proximal Policy Optimization (Schulman et al., 2017) | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/yv21y1qm) |
| GRPO | `grpo.yaml` | Group Relative Policy Optimization (Shao et al., 2024) | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/vjp7lgdi) |
| Dr. GRPO | `drgrpo.yaml` | Dr. GRPO (Liu et al., 2025) | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/a1swuynq) |
| GSPO | `gspo.yaml` | Group-Sequence Policy Optimization (Zheng et al., 2025) | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/10sxytli) |
| CISPO | `cispo.yaml` | Clipped Importance Sampling PO (MiniMax, 2025) | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/6dg0m06n) |

## Reward Model Training

> **Note: Experimental** - Reward model training needs tuning of hyperparameters, datasets, and models for cleaner training curves. Contributions welcome!

Train reward models on various datasets:

```bash
# Standard Preference RM (Chapter 5) - Bradley-Terry on UltraFeedback
uv run python -m reward_models.train_preference_rm

# Outcome Reward Model (Chapter 5) - trains on GSM8K
uv run python -m reward_models.train_orm

# Process Reward Model (Chapter 5) - trains on PRM800K
uv run python -m reward_models.train_prm
```

### Preference RM (Bradley-Terry)

Standard preference-based reward model using the Bradley-Terry loss:
`-log(sigmoid(r_chosen - r_rejected))`. This is the approach used in InstructGPT,
Llama 2, and most production RLHF systems. Trains on UltraFeedback preference data.

### ORM (Outcome Reward Model)

Binary classification on solution correctness. Fine-tunes Qwen3-0.6B on GSM8K,
learning to distinguish correct from incorrect math solutions.

### PRM (Process Reward Model)

Step-level classification on reasoning quality. Fine-tunes Qwen3-0.6B on PRM800K,
learning to rate individual reasoning steps as {-1, 0, 1} (bad, neutral, good).

### Example Runs

| Model | Description | Example Run |
|-------|-------------|-------------|
| Preference RM | Bradley-Terry on UltraFeedback | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/1g3y9bcc) |
| ORM | Outcome RM on GSM8K | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/3gkoqb7f) |
| PRM | Process RM on PRM800K | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/iv4d966d) |

## Direct Alignment Training

Train direct alignment algorithms (DPO and variants) on preference data:

```bash
# DPO (Chapter 8)
uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml

# IPO - more robust to noisy labels
uv run python -m direct_alignment.train --config direct_alignment/configs/ipo.yaml

# SimPO - no reference model needed
uv run python -m direct_alignment.train --config direct_alignment/configs/simpo.yaml

# Quick test run (1k samples)
uv run python -m direct_alignment.train --loss dpo --max_samples 1000
```

### Available algorithms

| Algorithm | Config | Description |
|-----------|--------|-------------|
| DPO | `dpo.yaml` | Direct Preference Optimization (Rafailov et al., 2023) |
| cDPO | N/A (use `--loss cdpo`) | Conservative DPO with label smoothing |
| IPO | `ipo.yaml` | Identity Preference Optimization (Azar et al., 2023) |
| SimPO | `simpo.yaml` | Simple PO - length-normalized, no ref model (Meng et al., 2024) |
| ORPO | `orpo.yaml` | Odds Ratio PO - combines SFT + preference (Hong et al., 2024) |
| KTO | `kto.yaml` | Kahneman-Tversky Optimization (Ethayarajh et al., 2024) |

### Training Results

![Direct Alignment Training Results](images/wandb_direct_alignment.png)

See Chapter 8 of RLHF Book for mathematical derivations.

## Rejection Sampling

Train the rejection sampling pipeline from Chapter 9: generate multiple
completions per prompt, score them with a reward model, select a subset, then
SFT on the selected pairs.

```bash
# Preprocess once (generate + score rollouts)
uv run python -m rejection_sampling.preprocess \
    --config rejection_sampling/configs/top_per_prompt.yaml

# Train each selection config on the cached rollouts
uv run python -m rejection_sampling.train \
    --config rejection_sampling/configs/top_per_prompt.yaml
uv run python -m rejection_sampling.train \
    --config rejection_sampling/configs/random_per_prompt.yaml
uv run python -m rejection_sampling.train \
    --config rejection_sampling/configs/top_k_overall.yaml
uv run python -m rejection_sampling.train \
    --config rejection_sampling/configs/random_k_overall.yaml
```

### Training Results

![Rejection Sampling Results](images/wandb_rejection_sampling.png)

### Example Runs

| Strategy | Description | Example Run |
|----------|-------------|-------------|
| `top_per_prompt` | Best-of-N completion per prompt | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/ohm3xnga) |
| `random_per_prompt` | Random per-prompt control | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/y3pbcla7) |
| `top_k_overall` | Best K completions across the full pool | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/w75hklzs) |
| `random_k_overall` | Random flat-pool control | [wandb](https://wandb.ai/natolambert/rlhf-book/runs/egeyr1q3) |

On the reference 1k-train / 200-test GSM8K slice, `top_k_overall` beat its
matched random baseline, while `top_per_prompt` and `random_per_prompt` were
effectively tied.

## Configuration

### Weights & Biases Logging

Training runs are logged to Weights & Biases. Configure via environment variables:

```bash
# Required: Your wandb API key
export WANDB_API_KEY="your-key"

# Optional: Override project name (default: from config file)
export WANDB_PROJECT="rlhf-book"

# Optional: Override run name
export WANDB_RUN_NAME="grpo_experiment_1"
```

The official runs for this repo are logged to: **[wandb.ai/natolambert/rlhf-book](https://wandb.ai/natolambert/rlhf-book)**

All runs are **public** - no login required to view training curves, configs, and metrics.

To disable wandb logging entirely, set `wandb_project: null` in your config or:
```bash
export WANDB_MODE="disabled"
```

### Other environment variables

```bash
# HuggingFace access (for gated models)
export HF_TOKEN="your-token"
```

### Memory requirements

| Training | Model | GPU Memory |
|----------|-------|------------|
| Policy gradients | Qwen3-1.7B | ~16GB (single GPU) |
| Reward models | Qwen3-0.6B | ~8-16GB |
| Reward models | Qwen3-1.7B | ~16-20GB |

## Book Chapters

These examples correspond to:

- **Chapter 5**: Reward Models (ORM, PRM, Preference RM)
- **Chapter 6**: Policy Gradient Methods (REINFORCE, PPO, GRPO, etc.)
- **Chapter 8**: Direct Alignment (DPO, IPO, SimPO, KTO, etc.)
- **Chapter 9**: Rejection Sampling

See [rlhfbook.com](https://rlhfbook.com) for the full text.

## Citation

To cite this book, please use the following format:

```bibtex
@book{rlhf2025,
  author       = {Nathan Lambert},
  title        = {Reinforcement Learning from Human Feedback},
  year         = {2025},
  publisher    = {Online},
  url          = {https://rlhfbook.com},
}
```

## License

- Policy gradients code (`policy_gradients/`): Apache 2.0 (from zafstojano/policy-gradients)
- Reward models code (`reward_models/`): MIT (from myhott163com/RLHF_ORM_PRM)
- Direct alignment code and other adaptations: MIT
