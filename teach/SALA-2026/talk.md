---
title: "An Introduction to Reinforcement Learning from Human Feedback and Post-training"
author: "Nathan Lambert"
date: "March 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
footer:
  left: "rlhfbook.com"
  center: ""
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
---

# An Introduction to Reinforcement Learning from Human Feedback and Post-training

Nathan Lambert

[SALA '26](https://lasala.ai/), Quito, Ecuador

11 March 2026

---

<!-- section -->

# Why Post-Training?

---

## Base Models Are Not Chatbots

**Prompt**: "The president of the US in 2006 was"

**Base model**: George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was... *[keeps going]*

**Post-trained model**: George W. Bush. He served two terms, from January 20, 2001, to January 20, 2009.

---

## What Changed With ChatGPT

- ChatGPT launched with one sentence: *"We trained this model using RLHF"*
- Same underlying model family — the difference was **post-training**
- Overnight, the world saw what a well-adapted language model could do

---

## Style and Format

Post-training changes **how** the model responds, not **what** it knows

- **Style**: tone, empathy, warmth vs. bluntness
- **Format**: structured answers, markdown, step-by-step breakdowns
- **Behavior**: refusals, safety, helpfulness trade-offs

---

## The Elicitation Theory

Post-training **extracts latent potential** from the base model

**OLMoE** — same base model, updated only post-training:
- Version 1: **35** benchmark average
- Version 2: **48** benchmark average

Base models determine the *ceiling*. Post-training's job is to **reach it**.

---

## The Scale of Post-Training

- **DeepSeek R1**: RL used ~147K H800 GPU hours (~5% of total training)
- Individual ablation runs: **10–100K GPU hours**
- The trend: **more compute going to post-training every year**

---

<!-- section -->

# How We Got Here

---

## RLHF Before Language Models

- **TAMER** (Knox & Stone, 2008) — humans score agent actions to learn a reward
- **Christiano et al. 2017** — RLHF on Atari trajectory preferences
- **Ziegler et al. 2019** — first RLHF on language models

---

## The Post-ChatGPT Acceleration

- **Early 2023**: Alpaca era — limited data, impressive but narrow
- **Late 2023**: DPO — direct alignment without a reward model
- **2024**: Complex multi-stage recipes (Llama 3, Tulu 3)
- **2025**: Reasoning via RL (DeepSeek R1, o1)

---

## The Classic RLHF Pipeline

1. **Instruction fine-tuning** — teach Q&A format from examples
2. **Reward model training** — learn a scoring function from human preferences
3. **RL optimization** — optimize the model against the reward model

---

## Landmark Papers

| Year | Paper | Contribution |
|------|-------|-------------|
| 2019 | Ziegler et al. | First RLHF on language models |
| 2022 | InstructGPT | The recipe behind ChatGPT |
| 2022 | Anthropic (CAI) | Constitutional AI for Claude |
| 2023 | DPO | Direct alignment, no reward model |
| 2023 | Gao et al. | Reward over-optimization scaling laws |
| 2024 | Llama 3, Tulu 3 | Modern multi-stage recipes |
| 2025 | DeepSeek R1 | Reasoning-first RL |

---

## Training Recipes Have Evolved

| | InstructGPT (2022) | Tulu 3 (2024) | DeepSeek R1 (2025) |
|---|---|---|---|
| **IFT data** | ~10K | ~1M | 100K+ |
| **Preference data** | ~100K | ~1M | On-policy |
| **RL stage** | ~100K prompts | ~10K (RLVR) | "Until convergence" |
| **Stages** | 3 | 3 | 4 |

More data, more stages, more RL compute.

---

<!-- section -->

# RLHF vs. RLVR vs. Classical RL

---

## Classical RL

- Agent takes actions in an **environment** with state transitions
- Reward is a **known function** of the environment
- Multi-step, fine-grained rewards at each timestep
- Goal: maximize cumulative return over a trajectory

---

## RLHF

- No environment — prompts sampled from a dataset
- Reward is **learned** from human preferences (a proxy)
- **Response-level** reward (bandit-style, not per-token)
- Regularized with **KL penalty** to stay close to the base model

$$J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$$

---

## RLVR

- Same RL setup as RLHF, but reward is **verifiable**
- Math: check the final answer. Code: run the tests.
- No learned reward model — **no proxy objective**
- Enables scaling RL compute on reasoning tasks

---

## Comparison

| | Classical RL | RLHF | RLVR |
|---|---|---|---|
| **Reward** | Environment | Learned (proxy) | Verifiable (exact) |
| **State transitions** | Yes | No | No |
| **Reward granularity** | Per-step | Per-response | Per-response |
| **Failure mode** | Exploration | Over-optimization | Task coverage |
| **Example** | CartPole | Chat style tuning | Math reasoning |

---

## Where Things Are Heading

- RLHF and RLVR are **complementary** — style vs. capabilities
- Modern recipes use **both** in sequence
- The boundary between them is blurring (generative reward models, self-correction)

---

# Thank You

**rlhfbook.com**
