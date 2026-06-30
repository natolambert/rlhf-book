# Self-Distillation (SDPO)

![SDPO: the policy is run twice over completions sampled from the question-only student — once conditioned on the question plus a correct sibling demonstration (self-teacher) and once on the question alone (student) — and trained by the per-token reverse KL between the two dense next-token distributions.](../../book/images/sdpo_tikz.png)

Educational implementation of **SDPO — Self-Distillation Policy Optimization** for
[RLHF Book](https://rlhfbook.com), an on-policy distillation method for reasoning
tasks ([Hübotter et al., 2026](https://arxiv.org/abs/2601.20802)).
See the parent [`code/README.md`](../README.md) for installation, configuration, and memory requirements.

The student rolls out a [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
task (the default is `spell_backward`, a string-reversal problem, matching the
[`policy_gradients`](../policy_gradients/README.md) GRPO setup) and each completion is
verified by the environment. The only in-context signal given to the self-teacher is a
*correct sibling demonstration* drawn from the same rollout group. If no rollout in the
group is correct, there is no demonstration to distil from and the prompt is **skipped
entirely** for that step.

## Algorithms

| Algorithm | Config | Key Idea |
|-----------|--------|----------|
| **SDPO** | `sdpo.yaml` | Self-Distillation Policy Optimization — distill a demonstration-conditioned self-teacher into the student via top-K reverse KL ([Hübotter et al., 2026](https://arxiv.org/abs/2601.20802)) |

## Reference Runs

![SDPO Training Results](../images/wandb_distillation.png)

`Qwen/Qwen3-1.7B` on the default `spell_backward` task, trained in under 20 hours on a
single 24 GB consumer GPU: `reward` rises from ~0.55 to ~0.8 while `loss` and
`grad_norm` trend down.

| Algorithm | wandb | Status |
|-----------|-------|--------|
| **SDPO** | _pending maintainer run_ | ✅ Trains; reference run ID pending publication |

## Quick Start

```bash
cd /path/to/rlhf-book/code
uv sync

# SDPO on the Reasoning Gym string-reversal task
uv run python -m distillation.train --config distillation/configs/sdpo.yaml
```

The dataset is generated procedurally by Reasoning Gym from the `data.specs` mixture
in the config (see [`data.py`](data.py)); swap or add tasks by editing those specs.

## Key configuration

See [`configs/sdpo.yaml`](configs/sdpo.yaml) for the full set. The most important knobs:

| Field | Default | Meaning |
|-------|---------|---------|
| `model_name` | `Qwen/Qwen3-1.7B` | Model used as both student and self-teacher |
| `data.specs` | `spell_backward` | Reasoning Gym task mixture (name / weight / per-task config) |
| `data.size` | `3000` | Number of procedurally generated problems |
| `kl_top_k` | `20` | Logits kept per position for the distillation KL |
| `success_reward_threshold` | `1.0` | Score at/above which a rollout becomes a demo solution |
| `num_rollouts` | `8` | Rollouts sampled per problem (sibling demos come from this group) |
| `prompts_per_step` | `4` | Problems generated and gradient-accumulated per optimizer step |
| `max_new_tokens` | `512` | Generation length cap per rollout |

## Metrics to watch

Logged to W&B each optimizer step (see [`train.py`](train.py)):

- **`avg_reward`** — mean environment score across the rollout group; the primary signal
  that the student is improving.
- **`loss`** — the masked top-K KL between student and self-teacher; should trend down
  as the student internalizes the demonstration-conditioned distribution.
- **`grad_norm`** — watch for spikes that indicate instability.

Steps whose prompts are all skipped (no correct demonstration in any group) produce no
update and are not logged.

## TODOs for Community Contributions

- [ ] Explore additional Reasoning Gym task domains beyond string reversal.

