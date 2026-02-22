---
title: "Chapter 1: Introduction to RLHF"
author: "Nathan Lambert"
date: "2026"
---

# Introduction to RLHF

Reinforcement Learning from Human Feedback

Nathan Lambert

---

## What is RLHF?

- A technique to incorporate **human information** into AI systems
- Emerged to solve **hard-to-specify problems** — preferences that are difficult to quantify
- Became widely known through the release of **ChatGPT**
- Core idea: "Can we solve hard problems with only basic preference signals guiding optimization?"

<!-- notes: RLHF addresses problems where we can't write down a simple reward function — like "be helpful but not harmful" -->

---

## The Classic 3-Step RLHF Pipeline

1. **Instruction fine-tuning**: Train a language model to follow user questions (Ch. 4)
2. **Reward model training**: Collect human preferences, train a reward model (Ch. 5)
3. **RL optimization**: Optimize the language model against the reward model (Ch. 6)

<!-- notes: This is the InstructGPT-era recipe. Modern recipes are more complex but build on this foundation. -->

---

## RLHF Within Post-Training

Post-training is a many-stage process using three optimization methods:

1. **Instruction / Supervised Fine-tuning (IFT/SFT)** — teaches formatting, features in language
2. **Preference Fine-tuning (PreFT)** — aligns to human preferences, style of language. **RLHF lives here.**
3. **Reinforcement Learning with Verifiable Rewards (RLVR)** — boosts performance on verifiable domains

RLHF dominates preference fine-tuning, which is more complex than instruction tuning because it involves proxy reward models and noisier data.

---

<!-- layout: section-break -->

## What Does RLHF Do?

---

## Base Model vs. Post-Trained Model

**Prompt**: "The president of the US in 2006 was"

**Base model** (Llama 3.1 405B):
> George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator... *[continues with UIGEA, plant rights amendments, 2009 calendar facts]*

**Post-trained model** (Tulu 3 405B):
> George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009.

<!-- notes: The base model continues the sentence like internet text. The post-trained model answers the question concisely. This is what post-training does. -->

---

## RLHF Changes Style

**Style** = manner, tone, empathy. Consider: "I have to give a presentation tomorrow and I'm panicking."

**Blunt style:**
> You'll be fine. Practice your slides, get some sleep, and stop overthinking it.

**Warm style:**
> That sounds really stressful, and it's a normal reaction. Do one quick practice run, pick the one or two points you most want people to remember, and give yourself permission to rest.

RLHF tunes completions at the **response level** — telling the model what a *better* response looks like, not a specific response to memorize.

---

## RLHF Changes Format

**Simple format** (early LM):
> Try to go to bed at the same time each night, avoid caffeine late in the day, and keep your bedroom quiet and dark.

**Rich format** (post-trained LM):
> **Tonight (quick wins):** 1. Dim lights for the last hour. 2. No caffeine after lunch.
> **Daily basics:** Same wake time, morning light, cool dark room.
> **Simple rule:** `wake time fixed + caffeine cutoff + wind-down routine`

Instruction fine-tuning provides basic Q&A ability. **RLHF crafts answers into reliable, warm, engaging responses.**

---

## Why RLHF Over Instruction Tuning?

| | Instruction Tuning | RLHF |
|---|---|---|
| **Optimization** | Per-token prediction | Response-level |
| **Signal** | "Output this specific response" | "This response is *better*" |
| **Feedback** | Positive only | Positive *and* negative (contrastive) |
| **Generalization** | Narrow | Generalizes far better across domains |

<!-- notes: RLHF's contrastive loss and response-level optimization are key to its generalization advantage -->

---

## RLHF Challenges

- Requires training an intermediate **reward model** — added complexity
- Prone to **over-optimization** — reward signal is a proxy objective
- **Length bias** artifacts
- More expensive in **compute, data, and time**
- Cannot solve all problems alone — needs broader post-training context

Despite these challenges, RLHF is established as **crucial** to achieving strong fine-tuned models.

---

<!-- layout: section-break -->

## Intuition for Post-Training

---

## The Elicitation Theory of Post-Training

Post-training **extracts latent potential** by amplifying valuable behaviors in the base model.

**Analogy: Formula 1 cars** — teams start the year with a new chassis and engine, then spend all year on aerodynamics and systems changes. The best teams improve *far more* during a season than chassis-to-chassis.

**Real example: OLMoE** — same base model, updated only post-training:
- Version 1: **35** average benchmark score
- Version 2: **48** average benchmark score

Base models determine the *potential*. Post-training's job is to **cultivate all of it**.

---

## Superficial Alignment Hypothesis

From the LIMA paper:

> "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used."

**Why this is incomplete:**
- Yes, you can change models substantially with few samples
- But this conflates *narrow style changes* with *capabilities*
- RL methods can teach models chain-of-thought reasoning, boosting performance on math, logic, and more
- Post-training has **far outgrown** "just vibes"

<!-- notes: The superficial alignment hypothesis was a useful early intuition but breaks down with reasoning models and RLVR -->

---

## How We Got Here

**Alpaca era** (early 2023): Limited human data + synthetic Self-Instruct → impressive but narrow results

**Skepticism phase**: "Instruction tuning is enough for alignment"

**DPO breakthrough** (late 2023): Zephyr-Beta, Tulu 2 showed preference tuning is essential

**Modern era** (2024–2025): Complex multi-stage post-training, RLVR for reasoning, dramatically increased RL compute

The trend: more optimization steps, more training algorithms, more diverse datasets and evaluations.

---

## The Scale of RL in Post-Training

**DeepSeek R1**: Post-training RL used ~147K H800 GPU hours (~5% of total training compute)

But the science is evolving fast:
- Individual ablation runs now take **10–100K GPU hours**
- OLMo 3.1 Think 32B: 4 weeks on 200 GPUs for RL stage alone
- The trend of **increased compute on post-training will continue**

---

## Book Roadmap

| Section | Chapters |
|---------|----------|
| **Introductions** | 1. Introduction, 2. Key Related Works, 3. Training Overview |
| **Core Pipeline** | 4. Instruction Tuning, 5. Reward Models, 6. Policy Gradients, 7. Reasoning, 8. Direct Alignment, 9. Rejection Sampling |
| **Data & Preferences** | 10. What are Preferences?, 11. Preference Data, 12. Synthetic Data, 13. Tool Use |
| **Practical** | 14. Over-optimization, 15. Regularization, 16. Evaluation, 17. Product & UX |

---

<!-- layout: title -->

# Summary

- RLHF solves hard-to-specify problems through human preference signals
- Changes model **style** and **format**, not just features
- Generalizes far better than instruction tuning alone
- Post-training extracts latent potential from base models
- Modern recipes are complex, multi-stage processes with increasing RL compute
