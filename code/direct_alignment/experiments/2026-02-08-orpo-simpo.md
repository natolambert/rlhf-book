# ORPO/SimPO Experiment Log (2026-02-08)

Author: @natolambert

This is an experiment log for the ORPO/SimPO debugging and sweep work done in this session.

## Goal

- Get ORPO/SimPO runs on UltraFeedback to look qualitatively similar to representative DPO/IPO/KTO runs (early margin growth, accuracy climbing above chance, stable loss curves).

## Code Changes Tested

- PR #243 changes were used as the base:
  - ORPO/SimPO switched to length-normalized average log-probs for sequence scoring.
  - Added safety guard to avoid invalid positive log-probs in sequence reductions.
- `direct_alignment/train.py` logging fix:
  - Before: W&B metrics were effectively taken from the last micro-batch of an accumulation window.
  - After: metrics are averaged across all micro-batches in the optimizer step and logged once per optimizer step.
- `direct_alignment/loss.py` SimPO gamma semantics:
  - Updated to `-logsigmoid(beta * (logit_margin - gamma))` so `gamma` behaves as the gamma/beta ratio used in reference implementations.
- Added quick sweep scripts:
  - `direct_alignment/experiments/2026-02-08-orpo-simpo/run_simpo_small.sh`
  - `direct_alignment/experiments/2026-02-08-orpo-simpo/run_orpo_small.sh`
  - Both support env overrides for `MAX_SAMPLES`, `NUM_EPOCHS`, `MAX_LENGTH`, `LEARNING_RATE`, `BETA`, `GAMMA`.

## Logging Bug Notes

- Yes, there was a real logging bug.
- Previously, reported `accuracy` was often from a single 2-example micro-batch (with `batch_size=2`, `grad_acc=4`), which made curves look unnaturally binned and noisy.
- After the fix, metrics are averaged over the full optimizer step (effective 8 examples with default small-run settings), so logging is more truthful.
- Remaining quantization is expected with tiny effective batch sizes. For 8 examples, accuracy still moves in `12.5%` increments.

## Baseline Runs (Before This Sweep)

- ORPO baseline from earlier work: https://wandb.ai/natolambert/rlhf-book/runs/oulqxp2i
- SimPO baseline from earlier work: https://wandb.ai/natolambert/rlhf-book/runs/7s100ned

## W&B Runs Tried In This Session

- SimPO, `lr=1e-6`, `gamma=0.3`, `samples=3200`, default script length (512): https://wandb.ai/natolambert/rlhf-book/runs/53pqrzt4
  - Final: `loss=0.7520`, `accuracy=75%`, `margins=0.31738`.
- SimPO, `lr=1e-6`, `gamma=0.1`, `samples=3200`, default script length (512): https://wandb.ai/natolambert/rlhf-book/runs/6yzwk11y
  - Final: `loss=0.5747`, `accuracy=75%`, `margins=0.31738`.
- SimPO, `lr=2e-6`, `gamma=0.1`, `samples=3200`, default script length (512): https://wandb.ai/natolambert/rlhf-book/runs/2gz0njwu
  - Final: `loss=0.5693`, `accuracy=75%`, `margins=0.32861`.
- SimPO, `lr=2e-6`, `gamma=0.0`, `samples=3200`, default script length (512): https://wandb.ai/natolambert/rlhf-book/runs/sass9db0
  - Final: `loss=0.4944`, `accuracy=75%`, `margins=0.32764`.
- SimPO, `lr=2e-6`, `gamma=0.1`, `max_length=1024`, `samples=1600`: https://wandb.ai/natolambert/rlhf-book/runs/bxqfssoz
  - Final: `loss=0.9219`, `accuracy=50%`, `margins=0.01562`.
- SimPO, `lr=2e-6`, `gamma=0.0`, `max_length=1024`, `samples=1600`: https://wandb.ai/natolambert/rlhf-book/runs/6h01t41k
  - Final: `loss=0.8149`, `accuracy=50%`, `margins=0.01709`.
- SimPO, `lr=2e-6`, `gamma=0.0`, `max_length=512`, `samples=1600`: https://wandb.ai/natolambert/rlhf-book/runs/a57ccg0b
  - Final: `loss=0.7498`, `accuracy=50%`, `margins=0.06494`.
- SimPO, `lr=2e-6`, `gamma=0.0`, `max_length=1024`, `samples=3200`: https://wandb.ai/natolambert/rlhf-book/runs/oek7xtxa
  - Final: `loss=0.4485`, `accuracy=75%`, `margins=0.38135`.
- ORPO, `beta=0.5`, `lr=5e-6`, `max_length=2048`, `samples=1600`: https://wandb.ai/natolambert/rlhf-book/runs/ha54crna
  - Final: `loss=1.4657`, `accuracy=50%`, `margins=0.00964`, `log_odds_ratio=0.00557`, `or_loss=0.38925`.
- ORPO debug sanity, `beta=1.0`, `lr=5e-6`, `max_length=2048`, `samples=64`: https://wandb.ai/natolambert/rlhf-book/runs/g83zbxka
  - Final: `loss=1.5125`, `accuracy=50%`, `margins=0.10352`, `log_odds_ratio=0.19609`, `or_loss=0.6121`.
- ORPO long follow-up, `beta=1.0`, `lr=5e-6`, `max_length=2048`, `samples=1600`: https://wandb.ai/natolambert/rlhf-book/runs/tmh0enbt
  - Status: in progress as of last check in this session.

### Additional Runs (Later in Session)

- ORPO quick foreground sanity, `beta=2.0`, `lr=5e-6`, `max_length=2048`, `samples=64`: https://wandb.ai/natolambert/rlhf-book/runs/qnevkkls
  - Final: `loss=1.9281`, `accuracy=75%`, `margins=0.41602`, `log_odds_ratio=0.36625`, `or_loss=1.09409`, `sft_loss=0.83301`.
- ORPO probe, `beta=2.0`, `lr=5e-6`, `max_length=2048`, `samples=160`: https://wandb.ai/natolambert/rlhf-book/runs/7pruw05c
  - Final: `loss=3.2088`, `accuracy=37.5%`, `margins=-0.48047`, `log_odds_ratio=-0.35272`, `or_loss=1.87631`, `sft_loss=1.33203`.
- ORPO probe, `beta=1.0`, `lr=8e-6`, `max_length=2048`, `samples=160`: https://wandb.ai/natolambert/rlhf-book/runs/58ywuqr4
  - Final: `loss=2.2654`, `accuracy=37.5%`, `margins=-0.23877`, `log_odds_ratio=-0.35103`, `or_loss=0.93680`, `sft_loss=1.33008`.
- SimPO probe, `beta=2.0`, `gamma=0.0`, `lr=2e-6`, `max_length=1024`, `samples=160`: https://wandb.ai/natolambert/rlhf-book/runs/myvyfkat
  - Final: `loss=1.0664`, `accuracy=37.5%`, `margins=-0.24170`.
- ORPO probe (aborted at step 12), `beta=1.0`, `lr=1e-5`, `max_length=2048`, `samples=160`: https://wandb.ai/natolambert/rlhf-book/runs/hi16cuou
  - Read: same oscillatory, batch-dominated pattern through step 12 (no clear early trend improvement vs lower-LR probes).

## Non-W&B / Aborted Probes

- Several very short sanity probes were launched with `WANDB_MODE=disabled` while validating startup and crash behavior.
- A few background launches exited early before trainer initialization due environment/sandbox behavior; this was worked around by running in foreground when needed and by explicit detached launches.
- Fixed-batch overfit sanity (non-W&B): both SimPO and ORPO can drive a single batch to `accuracy=1.0` and rapidly increasing margins at `lr=1e-5`, indicating gradient path / loss wiring is functional.

## Current Read

- SimPO responds to `gamma` and `lr` changes, but behavior remains inconsistent across lengths and sample budgets, and still does not match DPO/KTO quality.
- ORPO at `beta=0.5` is clearly too weak on this setup.
- ORPO/SimPO short probes (`samples=160`, effective batch 8) showed nearly identical oscillatory step metrics across multiple hyperparameter choices. This suggests online train-step metrics are dominated by per-batch difficulty/noise at this batch size.
- `tmh0enbt` appears to be a dead run with stale `running` state (heartbeat stopped after early steps).

## Immediate Next Sweep (Post-Restart)

- Add a stable eval signal (fixed eval subset and periodic eval logging) before trusting short-run train-step curves.
- Re-run ORPO/SimPO with larger effective batch for smoother metrics (e.g., target effective batch 16-32 if memory allows).
- After picking the best-smoothed config, scale to `max_samples=640`, then `1600+`.
- Keep W&B enabled for all runs (`WANDB_PROJECT=rlhf-book`, never `WANDB_MODE=disabled` for benchmark sweeps).
