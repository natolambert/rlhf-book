---
title: "RLHF Book: Introductions"
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


---

## The Classic 3-Step RLHF Pipeline

1. **Instruction fine-tuning**: Train a language model to follow user questions (Ch. 4)
2. **Reward model training**: Collect human preferences, train a reward model (Ch. 5)
3. **RL optimization**: Optimize the language model against the reward model (Ch. 6)


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

---

# Key Related Works

A Brief History of RLHF

Nathan Lambert

---

## Why History Matters

- RLHF and its related methods are **very new**
- Many procedures were formalized only recently
- The field is **rapidly evolving** — expect uncertainty and change
- Seminal papers were often for applications **totally distinct** from modern LLMs
- For comprehensive surveys: Wirth et al. 2017, Casper et al. 2023, Kaufmann et al. 2025


---

<!-- layout: section-break -->

## Origins to 2018: RL on Preferences

---

## Early Foundations

**TAMER** (Knox & Stone, 2008)
- First approach similar to modern RLHF
- Humans iteratively scored agent actions to learn a reward model

**COACH** (MacGlashan et al., 2017)
- Actor-critic where human feedback (positive and negative) tunes the advantage function

**Deep TAMER** (Warnell et al., 2018)
- Extended TAMER with neural networks

---

## The Primary Reference: Christiano et al. 2017

Applied RLHF to preferences between **trajectories in Atari games**

Key insight: Humans choosing between trajectories can be **more effective** than directly interacting with the environment

The core RLHF loop:
1. Agent generates trajectory segments
2. Humans compare pairs of segments
3. Reward predictor trained from comparisons (asynchronously)
4. Agent maximizes predicted reward


---

## Transition to Alignment

**Ibarz et al. 2018**: Expanded with more direct reward modeling approaches

**Leike et al. 2018**: Proposed reward models as a method for studying **alignment**, not just solving RL problems

This reframing — from tool to alignment technique — set the stage for RLHF's role in language models.

---

<!-- layout: section-break -->

## 2019–2022: RLHF Meets Language Models

---

## The First RLHF on Language Models

**Ziegler et al. 2019** — *Fine-Tuning Language Models from Human Preferences*

- Many canonical terms formalized in this paper:
  - **Reward models**
  - **KL distances**
  - **Feedback diagrams**
- Only the evaluation tasks and capabilities differed from today's methods
- This was the template that everything built on

---

## Key Applications (2020–2022)

| Paper | Application |
|-------|-------------|
| Stiennon et al. 2020 | Text summarization |
| Wu et al. 2021 | Recursive summarization of books |
| Nakano et al. 2021 (WebGPT) | Browser-assisted question answering |
| Menick et al. 2022 (GopherCite) | Supporting answers with citations |
| Glaese et al. 2022 (Sparrow) | General dialogue |
| **Ouyang et al. 2022 (InstructGPT)** | **Instruction following — precursor to ChatGPT** |

---

## Foundational Research (2021–2022)

Three seminal papers defined key areas for RLHF's future:

1. **Reward model over-optimization** (Gao, Schulman, Hilton 2023)
   - RL optimizers can over-fit to preference-trained models
   - Established scaling laws for this phenomenon

2. **LMs as alignment laboratories** (Askell et al. 2021)
   - Language models as a general area for alignment study

3. **Red teaming** (Ganguli et al. 2022)
   - Process of assessing safety of RLHF-trained models

Early open-source tools emerged: **TRL** (von Werra et al.), **TrlX** (Havrilla et al.)

---

<!-- layout: section-break -->

## 2023 to Present: The ChatGPT Era

---

## ChatGPT and the RLHF Spotlight

OpenAI's ChatGPT announcement was explicit:

> "We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup."

This single sentence made RLHF the most discussed technique in AI research overnight.

---

## RLHF in Leading Models

RLHF is well-documented in:

- **Anthropic**: Constitutional AI for Claude (Bai et al. 2022)
- **Meta**: Llama 2 (2023) and Llama 3 (2024)
- **Nvidia**: Nemotron 4 340B (2024)
- **Ai2**: Tülu 3 (2024)
- **OpenAI**: GPT series, o1 reasoning models

Companies that embraced RLHF early ended up winning out.

---

## The Expanding Frontier

RLHF is growing into **Preference Fine-tuning (PreFT)**, with new applications:

- **Process reward models** for intermediate reasoning steps (Lightman et al. 2024)
- **Direct Preference Optimization** and variants (Rafailov et al. 2023)
- **Execution feedback** from code or math (Kumar et al. 2024, Singh et al. 2023)
- **Reasoning models** inspired by OpenAI's o1 (2024)
- **Self-correction** via RL (Kumar et al. 2025)

---

## Timeline Summary

| Era | Key Development |
|-----|----------------|
| **2008** | TAMER — first human-in-the-loop reward learning |
| **2017** | Christiano et al. — RLHF on Atari |
| **2019** | Ziegler et al. — first RLHF on language models |
| **2022** | InstructGPT — RLHF for instruction following |
| **2022** | ChatGPT — RLHF goes mainstream |
| **2023** | DPO — direct alignment without reward model |
| **2024** | Tülu 3, Llama 3 — complex multi-stage recipes |
| **2024–25** | DeepSeek R1, o1 — reasoning via RL |

---

<!-- layout: title -->

# Key Takeaway

RLHF went from Atari game preferences (2017) to the core technique behind ChatGPT (2022) in just **five years**

The field is still young and rapidly evolving

---

# Training Overview

The RLHF Optimization Problem

Nathan Lambert

---

## This Chapter

A cursory overview of RLHF training before the specifics:

1. **Problem formulation** — how RLHF relates to standard RL
2. **Key differences** — what makes RLHF distinct
3. **Regularization** — why and how we constrain optimization
4. **Canonical recipes** — InstructGPT, Tülu 3, DeepSeek R1

---

<!-- layout: section-break -->

## Problem Formulation

---

## Standard RL Setup

An **agent** takes actions $a_t$ from policy $\pi(a_t \mid s_t)$ given state $s_t$ to maximize reward $r(s_t, a_t)$.

The policy and dynamics induce a **trajectory distribution**:

$$p_{\pi}(\tau) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)$$

The RL optimization objective:

$$J(\pi) = \mathbb{E}_{\tau \sim p_{\pi}} \left[ \sum_{t=0}^{T-1} \gamma^t r(s_t, a_t) \right]$$

where $\gamma \in [0, 1]$ is the discount factor balancing near-term vs. future rewards.

---

## Intuition: The Thermostat

A thermostat keeping a room at 70°F:

- **State** ($s_t$): current room temperature, e.g. 65°F
- **Action** ($a_t$): turn heater on or off
- **Reward** ($r$): +1 when within 2° of target, 0 otherwise
- **Policy** ($\pi$): the rule for on/off given temperature

$$\pi(a_t = \text{on} \mid s_t) = \begin{cases} 1 & \text{if } s_t < 70°\text{F} \\ 0 & \text{otherwise} \end{cases}$$

- **Transition**: room warms when heater is on, cools when off

The core RL loop: observe state → choose action → receive reward → update policy.

---

## Example: CartPole

A richer example with continuous dynamics — balance a pole on a moving cart:

- **State**: $s_t = (x_t, \dot{x}_t, \theta_t, \dot{\theta}_t)$ — position, velocity, angle, angular velocity
- **Action**: apply left/right force $a_t \in \{-F, +F\}$
- **Reward**: $r_t = 1$ each step the pole stays balanced
- **Dynamics**: Euler integration of Newtonian mechanics

$$\ddot{\theta}_t = \frac{g \sin\theta_t - \cos\theta_t \cdot \text{temp}}{l\left(\frac{4}{3} - \frac{m_p \cos^2\theta_t}{m_c + m_p}\right)}$$

Episode terminates when $|x_t| > 2.4$ or $|\theta_t| > 12°$.


---

<!-- layout: section-break -->

## From Standard RL to RLHF

---

## Three Key Modifications

| Aspect | Standard RL | RLHF for Language Models |
|--------|-------------|--------------------------|
| **Reward** | Environment function $r(s_t, a_t)$ | Learned preference model $r_\theta(x, y)$ |
| **State transition** | Yes: dynamics $p(s_{t+1} \mid s_t, a_t)$ | No: prompts sampled from dataset |
| **Action** | Single action $a_t$ | Full completion $y$ (token sequence) |
| **Reward granularity** | Per-step, fine-grained | Response-level (bandit-style) |
| **Horizon** | Multi-step ($T > 1$) | Single-step ($T = 1$) |

---

## Modification 1: Learned Reward Model

Standard RL: reward is a **static function** of the environment

RLHF: reward comes from a **learned model** $r_\theta(s_t, a_t)$

- Trained on human preference data (pairwise comparisons)
- Gives the designer **flexibility and control**
- But at the cost of **implementation complexity**
- The reward model is an approximation — a **proxy objective**

---

## Modification 2: No State Transitions

Standard RL: actions change the environment state via dynamics $p(s_{t+1} \mid s_t, a_t)$

RLHF: prompts sampled from a dataset, completions don't affect the next prompt

This makes the problem **much simpler** — no environment to simulate, no long-horizon planning.

---

## Modification 3: Response-Level Rewards

Standard RL: reward at **each timestep** $r(s_t, a_t)$

RLHF: reward for the **entire completion** — often called a **bandit problem**

The full sequence of tokens is scored as a unit, rather than individual token decisions.

---

## The Simplified RLHF Objective

Given the single-turn, no-transition setup, the optimization simplifies to:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ r_\theta(s_t, a_t) \right]$$

While heavily inspired by RL, the actual RLHF implementation is **very distinct** from traditional RL.

---

<!-- layout: section-break -->

## Regularization

---

## Why Regularize?

In traditional RL: agent learns from **random initialization**

In RLHF: we start from a **strong pretrained model** with existing capabilities

**Problem**: unconstrained optimization can drift too far from the initial model, leading to:
- Over-optimization against the proxy reward
- Loss of capabilities learned during pretraining
- Degenerate outputs that "hack" the reward model

---

## The KL-Regularized Objective

The most common approach — add a **distance penalty** from the reference policy:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ r_\theta(s_t, a_t) \right] - \beta \, \mathcal{D}_{\text{KL}}\!\left(\pi(\cdot | s_t) \| \pi_{\text{ref}}(\cdot | s_t)\right)$$

- $\pi_{\text{ref}}$: the reference policy (starting checkpoint)
- $\beta$: controls the trade-off between reward maximization and staying close
- **KL budget**: how far the model is allowed to drift from the reference


---

<!-- layout: section-break -->

## Optimization Tools

---

## The Post-Training Toolkit

| Tool | Description | Chapter |
|------|-------------|---------|
| **Instruction fine-tuning** | Teach Q&A format by imitating examples | Ch. 4 |
| **Reward modeling** | Train a model to score text quality from preferences | Ch. 5 |
| **Policy gradients** | RL algorithms (PPO, REINFORCE) to optimize against reward | Ch. 6 |
| **Direct alignment** | Optimize directly on preference data (DPO and variants) | Ch. 8 |
| **Rejection sampling** | Filter candidates by reward model, fine-tune on the best | Ch. 9 |

Modern models always use instruction fine-tuning followed by a **mixture** of the others.

---

<!-- layout: section-break -->

## Canonical Training Recipes

---

## Recipe 1: InstructGPT (2022)

The classic 3-step recipe around the time of ChatGPT:

1. **Instruction tuning** on ~10K examples
   - Teaches question-answer format from primarily human-written data

2. **Reward model** on ~100K pairwise prompts
   - Captures diverse values for the final model
   - Trained from the instruction-tuned checkpoint

3. **RLHF optimization** on ~100K prompts
   - Model generates responses, reward model rates them
   - Policy updated to maximize reward


---

## Recipe 2: Tülu 3 (2024)

A modern, multi-stage approach — successfully applied to Llama 3.1, OLMo 2, SmolLM:

1. **Instruction tuning** on ~**1M** examples
   - Primarily synthetic data from GPT-4o and Llama 3.1 405B
   - General instruction following + math, coding skills

2. **On-policy preference data** on ~**1M** preference pairs
   - Boosts chattiness (ChatBotArena, AlpacaEval) + skill improvements

3. **RLVR** on ~**10K** prompts
   - Small-scale RL to boost math while maintaining overall performance
   - Precursor to modern reasoning models

---

## Recipe 3: DeepSeek R1 (2025)

Reasoning-focused, 4-stage recipe:

1. **Cold-start**: 100K+ on-policy reasoning samples from R1-Zero, heavily filtered

2. **Large-scale RL**: Repeatedly covers reasoning problems with RLVR "until convergence"

3. **Rejection sampling**: 75% reasoning / 25% general queries — transition to general-purpose

4. **Mixed RL**: Verifiable rewards + preference tuning to polish the final model

---

## Evolution of Scale

| | InstructGPT (2022) | Tülu 3 (2024) | DeepSeek R1 (2025) |
|---|---|---|---|
| **IFT data** | ~10K | ~1M | 100K+ (filtered) |
| **Preference data** | ~100K | ~1M | On-policy |
| **RL stage** | ~100K prompts | ~10K prompts (RLVR) | "Until convergence" |
| **Stages** | 3 | 3 | 4 |
| **Key innovation** | The recipe itself | Scale + RLVR | Reasoning-first RL |

The trend: **more data, more stages, more RL compute**.

---

<!-- layout: title -->

# Summary

The RLHF objective: maximize learned reward while staying close to the reference policy

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ r_\theta(s_t, a_t) \right] - \beta \, \mathcal{D}_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$$

Training recipes have evolved from 3 simple steps (InstructGPT) to complex multi-stage processes with increasing RL compute
