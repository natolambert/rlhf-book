---
title: "Chapter 3: Training Overview"
author: "Nathan Lambert"
date: "2026"
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

<!-- notes: CartPole is a standard RL benchmark. The four continuous state variables and physics transitions make it substantially richer than the thermostat. -->

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

<!-- notes: Much of RLHF research is about understanding how to spend a certain KL budget effectively. See Chapter 15 on Regularization. -->

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

<!-- notes: This is the foundation of modern RLHF. Everything since builds on this recipe. -->

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
