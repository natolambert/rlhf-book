# ORPO/SimPO Debugging Notes

Status: **Needs validation** — see [#358](https://github.com/natolambert/rlhf-book/issues/358)

## Context

- ORPO/SimPO instability from extreme ORPO scales was addressed in [PR #243](https://github.com/natolambert/rlhf-book/pull/243) by switching to average log-probs.
- Remaining issue is "stable but flat" learning, especially for ORPO where the preference term (`or_loss`) contributes negligibly vs the SFT loss.
- SimPO formula now uses gamma as a gamma/beta ratio: `-logsigmoid(beta * (logit_margin - gamma))`.
- Experiment log with full run ledger and W&B links: `experiments/2026-02-08-orpo-simpo.md`.

## Debugging approach

1. Run quick low-sample sanity jobs first (1 epoch, 640 samples) before long sweeps.
2. Confirm SimPO margins/accuracy move with the gamma-ratio formulation.
3. Sweep ORPO beta/LR to rebalance SFT vs preference term.
4. Only then launch 12.8K-sample full runs.

## Small-run scripts

```bash
cd code/

# SimPO sanity run
WANDB_ENTITY=rlhf-book WANDB_PROJECT=core ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_simpo_small.sh

# ORPO sanity run
WANDB_ENTITY=rlhf-book WANDB_PROJECT=core ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_orpo_small.sh
```

### Useful overrides

```bash
# SimPO: stronger margin push
GAMMA=1.0 LEARNING_RATE=1e-6 MAX_SAMPLES=640 NUM_EPOCHS=1 ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_simpo_small.sh

# ORPO: stronger preference term
BETA=1.0 LEARNING_RATE=5e-6 MAX_SAMPLES=640 NUM_EPOCHS=1 ./direct_alignment/experiments/2026-02-08-orpo-simpo/run_orpo_small.sh
```

## Acceptance criteria

- SimPO: `accuracy` and `margins` trend upward in first ~10-20 optimizer steps.
- ORPO: `margins` move off near-zero and `or_loss` contributes non-trivially vs `sft_loss`.
- Samples remain coherent (no repetitive collapse).
