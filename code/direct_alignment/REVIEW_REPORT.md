# Direct Alignment Algorithms Code Review Report

Review date: 2026-01-30
Scope: `direct_alignment/` (config, data, loss, train, configs)

## Findings (ordered by severity)

- High: Prompt/response masking is missing, so losses are computed over the full prompt+response sequence. This changes the intended objectives for SimPO/ORPO/KTO and dilutes the preference signal with prompt tokens. In ORPO, the SFT term currently trains on prompt tokens as targets (not just assistant response), which is not the standard formulation. DPO/IPO logits mostly cancel prompt tokens, but the ref KL/log-ratio and metrics still include them, and truncation can break exact cancellation. This is called out in a comment but not implemented. A robust fix is to track prompt lengths and build a `response_mask` (or labels with -100) so only assistant tokens contribute to logprobs/NLL. (`direct_alignment/data.py:191`, `direct_alignment/data.py:230`, `direct_alignment/loss.py:18`, `direct_alignment/train.py:138`)

- High: Gradient accumulation drops the final partial micro-batch in each epoch when `len(dataloader)` is not divisible by `gradient_accumulation_steps`. There is no final optimizer step to flush remaining gradients, so you silently train on fewer samples and the scheduler step count is off. Fix by stepping once after the epoch if `batch_idx % grad_accum != grad_accum-1` or by accumulating with a counter and stepping on the last batch. (`direct_alignment/train.py:245`, `direct_alignment/train.py:467`)

- Medium: KTO “unpaired” support is not implemented as described. `chosen_mask` is accepted but never used, and the reference point is computed separately for chosen and rejected rather than as a single `z_ref` over the batch. This deviates from the paper and makes the unpaired path incorrect. If KTO is intended to be paired-only, the API and docstring should reflect that; otherwise, implement unpaired batching and a unified `z_ref`. (`direct_alignment/loss.py:331`)

- Low: Progress bar advancement is incorrect when using gradient accumulation. You advance by 1 on micro-steps, then by `gradient_accumulation_steps` on the optimizer step, so progress overshoots total. This is cosmetic but makes logging misleading. (`direct_alignment/train.py:489`)

- Low: `get_loss_function` prevents setting `gamma=0.0` for SimPO because `gamma or 0.5` treats 0 as falsy. If you ever want a zero margin, it is silently overridden. (`direct_alignment/loss.py:411`)

- Low: `device_map=device` uses a string like `"cuda"`/`"cpu"`, which is not a supported `device_map` value in some Transformers versions. This can error or place the model on CPU unexpectedly. Consider `device_map="auto"` or `.to(device)` after load. (`direct_alignment/train.py:71`, `direct_alignment/train.py:92`)

- Low: `extract_ultrafeedback_pairs` takes the first assistant message in the list; if the dataset contains multi-turn assistant messages, later turns are ignored. If you want the latest assistant response, use the last assistant message instead. (`direct_alignment/data.py:73`)

## Open questions / assumptions

- Are these implementations intended to be strictly faithful to the papers or “good-enough” educational sketches? The masking and KTO unpaired behavior are the main divergences.
- Do you want to support unpaired KTO data formats explicitly (single response + label), or keep KTO as a paired approximation? That decision affects data loading and loss signatures.
- Should ORPO’s odds-ratio term use a more faithful per-token odds computation, or is the current approximation intentional for pedagogy?

## Testing gaps / suggested checks

- Unit test: logprob masking ensures prompt tokens do not contribute (SimPO/ORPO/KTO).
- Unit test: gradient accumulation steps the optimizer on the last partial batch.
- Smoke test: KTO on unpaired data (if supported) produces non-NaN loss and uses `chosen_mask`.
- Regression test: SimPO with `gamma=0.0` uses the configured value.

## Change summary (if you decide to fix)

- Add response-only masking/labels in `PreferenceDataset` and pass masks through to `compute_logprobs` and `compute_nll_loss`.
- Handle remainder gradient accumulation at epoch end.
- Clarify/implement KTO unpaired path and reference point definition.
