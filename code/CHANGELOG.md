# Changelog

- 2026-02-07: [PR #243](https://github.com/natolambert/rlhf-book/pull/243) stabilized ORPO/SimPO by switching to average-logprob behavior and improved direct-alignment logging/sampling instrumentation. It also fixed grad-accum metric logging to report optimizer-step averages (instead of last micro-batch snapshots), aligned SimPO `gamma` semantics, and added small ORPO/SimPO sweep scripts.
