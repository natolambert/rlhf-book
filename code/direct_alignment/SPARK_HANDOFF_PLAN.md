# Spark Handoff Plan (Temporary)

Created: 2026-02-05  
Branch: `improved-daa-logging`  
PR: https://github.com/natolambert/rlhf-book/pull/243

This is a temporary run plan for DGX Spark follow-up on ORPO/SimPO and in-loop logging.

## 1. Pull And Prepare

```bash
cd /home/natolambert/dev/rlhf-book
git fetch origin
git checkout improved-daa-logging
git pull
cd code
uv sync
```

## 2. Run SimPO (Updated Config)

```bash
cd /home/natolambert/dev/rlhf-book/code
export WANDB_PROJECT=rlhf-book
export WANDB_RUN_NAME=simpo-olmo-1b-retune-$(date +%Y%m%d-%H%M)
uv run python -m direct_alignment.train \
  --config direct_alignment/configs/simpo.yaml \
  --sample_every 50
```

Compare against baseline run: `ftv5rs3x`.

## 3. Run ORPO (Updated Config + Loss Path)

```bash
cd /home/natolambert/dev/rlhf-book/code
export WANDB_PROJECT=rlhf-book
export WANDB_RUN_NAME=orpo-olmo-1b-retune-$(date +%Y%m%d-%H%M)
uv run python -m direct_alignment.train \
  --config direct_alignment/configs/orpo.yaml \
  --sample_every 50
```

Compare against baseline run: `o38ffli5`.

## 4. What To Check In W&B

- `loss`, `accuracy`, `margins`, `grad_norm`, `learning_rate`
- ORPO-specific:
  - `log_odds_ratio`
  - `or_loss`
  - `sft_loss`
  - `chosen_logps` vs `rejected_logps`
- Sample table quality:
  - Prompt ids rotate (or match strategy)
  - Table columns include decode settings (`temperature`, `top_p`, etc.)

## 5. Quick Acceptance Criteria

- SimPO:
  - No crash
  - Better/less noisy preference metrics than `ftv5rs3x`
  - Reasonable sample quality at mid + final steps
- ORPO:
  - No crash
  - `log_odds_ratio`/`or_loss` no longer explode like prior run
  - `accuracy` and `margins` trend in the right direction

## 6. If ORPO Is Still Unstable

Try these in order:

1. Lower LR to `8e-7`:
   - `--learning_rate 8e-7`
2. Lower ORPO beta to `0.05`:
   - `--beta 0.05`
3. Shorten context:
   - `--max_length 1536`
4. Increase warmup:
   - `--warmup_ratio 0.15`

## 7. Optional Prompt Pool Sweep

To test richer logging prompts:

```bash
uv run python -m direct_alignment.train \
  --config direct_alignment/configs/simpo.yaml \
  --sample_prompt_strategy random \
  --sample_num_prompts 6 \
  --sample_prompts_file /path/to/prompts.txt
```

`prompts.txt` format: one prompt per line.  
Alternative: JSON list of strings.

## 8. After Spark Run

- Post the two new W&B run URLs to PR #243.
- Add a short comparison note against `ftv5rs3x` and `o38ffli5`.
- Remove or replace this file once final settings are locked.
