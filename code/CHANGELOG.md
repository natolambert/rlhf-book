# Changelog

- 2026-04-06: [PR #337](https://github.com/natolambert/rlhf-book/pull/337) tuned preference RM default hyperparameters (lr 1e-6 → 5e-5, effective batch 8 → 32) via Bayesian sweep on 5K samples. Full 60K run trains to epoch_loss=0.4 (random chance ~0.693). [Sweep 1](https://wandb.ai/singh-adityak1-independent/rlhf-book/sweeps/ml8d95f2), [Sweep 2](https://wandb.ai/singh-adityak1-independent/rlhf-book/sweeps/cfmnk2xg), [Full run](https://wandb.ai/singh-adityak1-independent/rlhf-book/runs/rsmqd9lr).

- 2026-02-07: [PR #243](https://github.com/natolambert/rlhf-book/pull/243) stabilized ORPO/SimPO by switching to average-logprob behavior and improved direct-alignment logging/sampling instrumentation. It also fixed grad-accum metric logging to report optimizer-step averages (instead of last micro-batch snapshots), aligned SimPO `gamma` semantics, and added small ORPO/SimPO sweep scripts.
