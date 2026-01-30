# Direct Alignment Algorithms Code Review Report

Review date: 2026-01-30
Scope: `direct_alignment/` (config, data, loss, train, configs)

## Findings (ordered by severity)

### Fixed

- ~~High: Prompt/response masking is missing~~ **FIXED** - Added `response_mask` to `PreferenceBatch` and updated `forward_pass`/`compute_nll_loss` to use it. Only response tokens now contribute to the loss.

- ~~High: Gradient accumulation drops the final partial micro-batch~~ **FIXED** - Added flush logic at end of epoch when `len(dataloader) % gradient_accumulation_steps != 0`.

- ~~Medium: KTO "unpaired" support is not implemented as described~~ **FIXED** - KTO now computes a unified `z_ref` across all samples (chosen + rejected concatenated) as specified in the paper. This enables proper unpaired preference learning.

- ~~Low: Progress bar advancement is incorrect~~ **FIXED** - Now advances by 1 per micro-batch consistently.

- ~~Low: `get_loss_function` prevents setting `gamma=0.0`~~ **FIXED** - Changed to `gamma if gamma is not None else 0.5`.

- ~~Low: `device_map=device` issues~~ **FIXED** - Now uses `.to(device)` after loading.

### Remaining (minor)

- Low: `extract_ultrafeedback_pairs` takes the first assistant message in the list; if the dataset contains multi-turn assistant messages, later turns are ignored. If you want the latest assistant response, use the last assistant message instead. (`direct_alignment/data.py:73`)

## Testing gaps / suggested checks

- Unit test: logprob masking ensures prompt tokens do not contribute (SimPO/ORPO/KTO).
- Unit test: gradient accumulation steps the optimizer on the last partial batch.
- Smoke test: KTO produces reasonable z_ref values and loss.
- Regression test: SimPO with `gamma=0.0` uses the configured value.

## Implementation notes

### KTO unified z_ref

The KTO implementation now follows the paper's formulation:
1. Concatenates chosen and rejected log-ratios into a single tensor
2. Computes `z_ref = mean(all_logratios)` as the reference point
3. Applies prospect theory losses relative to this unified reference

This enables KTO to work correctly with both paired data (where we treat chosen as desirable and rejected as undesirable) and truly unpaired data (if the data loading were extended to support it).
