---
title: "An Introduction to Reinforcement Learning from Human Feedback and Post-training"
author: "Nathan Lambert"
date: "March 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
footer:
  left: "rlhfbook.com"
  center: ""
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# An Introduction to Reinforcement Learning from Human Feedback and Post-training

<div class="colloquium-title-eyebrow">SALA 2026</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
<p>Quito, Ecuador<br>11 March 2026</p>
</div>

<p class="colloquium-title-note">A cursory overview of RLHF, RLVR, and modern post-training recipes for language models.</p>

---

<!-- section -->

# Why Post-Training?

---

<!-- columns: 50/50 -->
## What is a language model?

- A language model assigns probabilities to text
- Given previous tokens, it predicts the **next token**
- Repeating this produces a completion

|||

![transformer original diagram](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 30/70 -->
## Pretraining: next-token prediction

- Train on a lot of text from the web, books, code, and documents
- Objective: predict the next token in each sequence
- This does **not** directly teach "be helpful" or "answer questions"

|||

<div style="height: 430px; border: 2px dashed #c7ced9; border-radius: 18px; display: flex; align-items: center; justify-content: center; text-align: center; color: #7b8496; font-size: 1.35em; padding: 1.5em;">
TODO image<br>next-token prediction example
</div>

---

## A base model completes text

```conversation
messages:
  - role: user
    content: "The president of the United States in 2006 was"
  - role: assistant
    content: "George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator in 2006..."
```

---

## Post-training makes it answer like a chatbot

```conversation
messages:
  - role: user
    content: "The president of the United States in 2006 was"
  - role: assistant
    content: "George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009."
```

---

<!-- columns: 40/60 -->
## ChatGPT made language models usable

- Better **format**: direct answers instead of raw continuations
- Better **style**: helpful, concise, conversational
- Better **product**: people could actually use it every day

|||

```conversation
size: 0.775
messages:
  - role: user
    content: "I'm giving a talk on RLHF tomorrow. Can you help me structure it?"
  - role: assistant
    content: |
      Absolutely — here's a simple structure:

      **1. Start with the basics**
      - What a language model is
      - Why next-token prediction is not enough

      **2. Explain post-training**
      - Instruction tuning
      - Preference tuning / RLHF
      - RLVR for verifiable tasks

      **3. End with why it matters**
      - Better products like ChatGPT
      - Better reasoning in models like o1
```

---

## The Elicitation Theory

Post-training **extracts latent potential** from the base model

**OLMoE** — same base model, updated only post-training:
- Version 1: **35** benchmark average
- Version 2: **48** benchmark average

Base models determine the *ceiling*. Post-training's job is to **reach it**.

---

## o1 scaling post training

<!-- img-align: center -->
<!-- cite-right: openai2024o1 -->

![o1 AIME accuracy](assets/o1.webp)

---

## o1: train-time scaling

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->

text goes here

|||

![o1 AIME accuracy during training](assets/o1-train-time.png)



---

## o1: test-time scaling asd asds 

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->
text goes here


|||

![o1 AIME accuracy at test time](assets/o1-test-time.png)

---

<!-- class: title-hidden -->
<!-- img-fill: true -->
<!-- img-align: center -->
<!-- valign: center -->
<!-- cite-right: ouyang2022training -->

## InstructGPT image-only

![InstructGPT](assets/instructgpt.jpg)


---

## ChatGPT's success

- Turned language models into a **mass-market product**
- Made post-training feel as important as pretraining
- Changed what users expect from AI systems

---

## The Scale of Post-Training

- **DeepSeek R1**: RL used ~147K H800 GPU hours (~5% of total training)
- Individual ablation runs: **10–100K GPU hours**
- The trend: **more compute going to post-training every year**

---

<!-- section -->

# How We Got Here

---

<!-- columns: 40/60 -->
<!-- cite-right: christiano2017 -->
## RLHF Before Language Models

- **TAMER** (Knox & Stone, 2008) — humans score agent actions to learn a reward
- **Christiano et al. 2017** — RLHF on Atari trajectory preferences
- **Ziegler et al. 2019** — first RLHF on language models

|||

![Christiano et al. 2017 RLHF overview](assets/rlhf_schematic_tikz.png)

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

<!-- rows: 60/40 -->
## Landmark Papers

![RLHF timeline](assets/rlhf_timeline_tikz.png)

===

<div class="text-sm">

- **Ziegler 2019** [@ziegler2019fine] — first RLHF on language models
- **InstructGPT** [@ouyang2022training] — the recipe behind ChatGPT
- **Constitutional AI** [@bai2022constitutional] — AI feedback and the Claude line
- **DPO** [@rafailov2024direct] — direct alignment without a reward model
- **Llama 3** [@dubey2024llama] and **Tulu 3** [@lambert2024t] — modern multi-stage recipes
- **DeepSeek R1** [@guo2025deepseek] — reasoning-first RL

</div>

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

## Beyond elicitation?

- Maybe post-training does more than just **extract** existing ability
- Long RL runs may reshape how models **reason**, not just how they respond
- Open question: when does scaling RL create **new capabilities**?

---

# Thank You

**rlhfbook.com**
