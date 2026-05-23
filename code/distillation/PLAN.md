# SDPO Implementation Plan

Minimal, educational port of SDPO (Self-Distilled Policy Optimization) — RL with
rich textual feedback. Starting on the LiveCodeBench v6 code environment.

## Steps

- [x] `data.py` — LiveCodeBench v6 loading + dataloader, sandboxed code execution,
      scalar reward + textual environment feedback (`compute_score`).
- [ ] Rollout: generate solutions, build multi-turn prompts that re-feed feedback.
- [ ] SDPO loss: on-policy term + self-distillation from feedback-conditioned
      next-token predictions.
- [ ] Training loop tying rollouts, scoring, and the loss together.
- [ ] Reference run + README.
