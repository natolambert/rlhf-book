---
title: "Lecture 3: Reinforcement Learning"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com"
  center: "Lecture 3"
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
    letter-spacing: 0;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 3: Reinforcement Learning

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapters 6 & 7</p>

---

<!-- rows: 50/50 -->
## Lecture 3: Reinforcement Learning

<!-- row-columns: 32/36/32 -->

```box
title: Overview
tone: muted
compact: true
content: |
  1. Introduction
  2. Key Related Works
  3. Training Overview
```

|||

```box
title: Core Training Pipeline
tone: accent
compact: true
content: |
  4. Instruction Tuning
  5. Reward Models
  6. **Reinforcement Learning**
  7. **Reasoning**
  8. Direct Alignment
  9. Rejection Sampling
```

|||

```box
title: Data & Preferences
tone: muted
compact: true
content: |
  10. What are Preferences
  11. Preference Data
  12. Synthetic Data & CAI
```

===

<!-- row-columns: 32/36/32 -->

```box
title: Practical Considerations
tone: muted
compact: true
content: |
  13. Tool Use
  14. Over-optimization
  15. Regularization
  16. Evaluation
  17. Product & Character
```

|||

```box
title: Appendices
tone: surface
compact: true
content: |
  - A. Definitions
  - B. Style & Information
  - C. Practical Issues
```

|||

```box
title: Course Home
tone: surface
compact: true
content: |
  - [rlhfbook.com](https://rlhfbook.com)
  - [GitHub repo](https://github.com/natolambert/rlhf-book)
```

---

<!-- rows: 30/70 -->
## Where we are in the pipeline

After instruction tuning and training a reward model, the RL step updates the policy against a learned reward signal.

===

![The RLHF pipeline — the RL step optimizes the policy using the reward model signal.](assets/rlhf_schematic.png)

---

## What this lecture covers

This lecture covers the **math and theory** of RL for language models. The next lecture covers implementation.

```box
title: Lecture 3 Outline
tone: accent
content: |
  1. **RLVR motivation** — why RL matters now more than ever
  2. **RL foundations review** — MDP, return, value functions, policy gradient objective
  3. **Policy gradient derivation** — log-derivative trick, step by step
  4. **REINFORCE & RLOO** — the simplest policy gradient algorithms
  5. **PPO** — trust regions, clipping, GAE, value functions
  6. **GRPO & modern variants** — GSPO, CISPO, and the trend toward simplicity
  7. **Comparison** — when to use what
```

---

<!-- layout: section-break -->

## RLVR Motivation

---

<!-- columns: 50/50 -->
## The scaling RL era

<!-- cite-right: openai2024o1 -->

RL is now the **load-bearing step** in training the most capable models.

Reasoning models (o1, DeepSeek R1, etc.) are trained with these exact algorithms on verifiable rewards.

- Train-time compute scaling via RL
- Test-time compute scaling via extended reasoning
- RL went from "cherry on top" to a core capability driver

|||

![OpenAI o1: scaling RL training compute improves reasoning performance.](assets/o1-train-time.png)

---

<!-- columns: 50/50 -->
## RLVR: same algorithms, verifiable rewards

<!-- cite-right: lambert2024t -->

The RL methods in this lecture power **both RLHF and RLVR**:

- **RLHF**: reward model scores subjective quality
- **RLVR**: verification function checks correctness (math, code)

Same policy gradient algorithms, different reward source. We will cover reasoning training in depth in a future lecture.

|||

![RLVR uses a verification function instead of a reward model, but the RL algorithms are the same.](assets/rlvr-system.png)

---

<!-- layout: section-break -->

## RL Foundations Review

---

<!-- columns: 65/35 -->
## Recall: the MDP formulation

<!-- cite-right: sutton2018reinforcement -->

A reinforcement learning problem is often written as a **Markov Decision Process (MDP)**:

- State space $\mathcal{S}$, action space $\mathcal{A}$
- Transition dynamics $P(s_{t+1} \mid s_t, a_t)$
- Reward function $r(s_t, a_t)$, discount factor $\gamma$
- Optimize cumulative return over a trajectory

$$\text{MDP } (\mathcal{S}, \mathcal{A}, P, r, \gamma)$$

|||

![](assets/rl.png)

---

<!-- columns: 50/50 -->
## Recall: Classical RL vs. RLHF

<!-- cite-right: christiano2017, ouyang2022training -->

<div class="text-sm">

**Classical RL**
- Agent takes actions $a_t$ in an environment with states $s_t$
- Reward is a known function $r(s_t, a_t)$ from the environment per step
- Optimize cumulative return over a trajectory

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

<div class="colloquium-spacer-md"></div>

**RLHF**
- No environment — prompts sampled from a dataset
- Reward is **learned** from human preferences (a proxy)
- **Response-level** reward (bandit-style, not per-token)
- Regularized with **KL penalty** to stay close to the base model

$$J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$$

</div>

|||

![](assets/rlhf.png)

---

## The return

The objective of the agent is the sum of discounted future rewards at time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

This can also be written recursively:

$$G_t = \gamma G_{t+1} + R_{t+1}$$

In RLHF: $\gamma = 1$ (no discounting) because the unit of optimization is the collective completion, not individual tokens.

---

## Value function

The value function $V(s)$ is the **expected future return** given the current state:

$$V(s) = \mathbb{E}\big[G_t \mid S_t = s\big]$$

This is the foundation for baselines and advantage estimation — core tools for reducing variance in policy gradient algorithms.

---

## Policy gradient objective

All policy gradient algorithms optimize a policy $\pi_\theta(a \mid s)$ to maximize expected return.

Where $d^{\pi_\theta}(s)$ is the state-visitation distribution induced by the policy:

$$J(\theta) = \sum_{s} d^{\pi_\theta}(s) V^{\pi_\theta}(s)$$

---

## Empirical estimate

In practice, we never compute $J(\theta)$ exactly. We estimate from data:

$$\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} R(x_i, y_i)$$

Sample prompts $x_i$ from a dataset, generate completions $y_i \sim \pi_\theta(\cdot \mid x_i)$, score with reward model, and average.

The parameter update follows:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

---

## What $\Psi_t$ can be

The general policy gradient uses a signal $\Psi_t$:

$$g = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \Psi_t\right]$$

| $\Psi_t$ | Description | Variance | Bias |
|-----------|-------------|----------|------|
| $R(\tau)$ | Total trajectory reward | Highest | None |
| $\sum_{t'=t}^{T} r_{t'}$ | Future return from $t$ | High | None |
| $\sum_{t'=t}^{T} r_{t'} - b(s_t)$ | Baselined return | Lower | None |
| $Q^{\pi}(s_t, a_t)$ | State-action value | Med | Depends |
| $A^{\pi}(s_t, a_t)$ | Advantage | **Lowest** | None |
| $r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD residual | Low | Some |

The advantage $A = Q - V$ gives the lowest theoretical variance if computed accurately.

---

## MDP vs. Bandit framing

<!-- columns: 50/50 -->

**MDP (token-level)**
- Each token $a_t$ is an action with state $s_t$ (running prefix)
- Per-token advantages via learned value function
- Used in PPO with GAE

|||

**Bandit (sequence-level)**
- Whole completion = single action, one scalar reward
- Sequence-level advantage broadcast to all tokens
- Used in RLOO, GRPO

<div class="colloquium-spacer-md"></div>

Most RLHF: **bandit-level rewards** (one score per response) but **token-level gradients** (update every token's log-prob).

---

## Notation

This lecture uses $(s, a)$ from the RL literature (states, actions).

In the language model context, you'll also see $(x, y)$ — prompt and completion.

| RL notation | LM notation | Meaning |
|:-----------:|:-----------:|---------|
| $s$ | $x$ | State / prompt |
| $a$ | $y$ | Action / completion |
| $\pi_\theta(a \mid s)$ | $\pi_\theta(y \mid x)$ | Policy |
| $r(s, a)$ | $R(x, y)$ | Reward |

Both notations used throughout. The $(s, a)$ framing is more general; $(x, y)$ is specific to LMs.

---

<!-- layout: section-break -->

## The Policy Gradient Theorem

---

## Setup: differentiating an expectation

We want $\nabla_\theta J(\theta)$ — the gradient of expected return w.r.t. the policy parameters.

The challenge: how do we differentiate an **expectation over trajectories** when the sampling distribution itself depends on $\theta$?

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int_\tau p_\theta(\tau) R(\tau) \, d\tau$$

Taking the gradient directly:

$$\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau$$

We can't sample from $\nabla_\theta p_\theta(\tau)$. We need a trick.

---

## The log-derivative trick

From the chain rule of logarithms:

$$\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$$

Rearranging:

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$$

This converts a gradient of a probability into something we can compute as an **expectation**.

---

## Applying to trajectories

Substituting the log-derivative trick into the gradient:

$$\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau$$

$$= \int_\tau p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) \, d\tau$$

$$= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\nabla_\theta \log p_\theta(\tau) \cdot R(\tau)\right]$$

Now we can estimate this with Monte Carlo sampling!

---

## Expanding log probability of trajectory

The trajectory probability factorizes:

$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)$$

Taking the log:

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

---

## Expanding log probability of trajectory

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

Now take the gradient w.r.t. $\theta$:

- $\nabla_\theta \log p(s_0) = 0$ — initial state doesn't depend on $\theta$
- $\nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$ — environment dynamics don't depend on $\theta$
- Only $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ survives!

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

---

## The policy gradient theorem

Substituting back:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t\right]$$

Where $\Psi_t$ can be total return, future return, advantage, etc. (see the table from earlier).

This is the **policy gradient theorem** — the foundation for all algorithms in this lecture.

---

## Introducing the advantage

The advantage function measures how much better an action is compared to the average:

$$A(s, a) = Q(s, a) - V(s)$$

Using the advantage as $\Psi_t$:
- **Reduces variance** without introducing bias
- Positive advantage → action was better than expected → reinforce it
- Negative advantage → action was worse than expected → suppress it
- Zero advantage → no update needed

---

## The RLHF policy gradient

Putting it together for language models:

$$\nabla_\theta J(\theta) = \mathbb{E}_{x,\, y \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(y \mid x) \cdot A(x, y)\right]$$

In practice, this becomes a per-token sum over the completion:

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t\right]$$

Every algorithm in this lecture is an instantiation of this equation with different choices for $A_t$ and different regularization.

---

<!-- layout: section-break -->

## REINFORCE

---

## REINFORCE

<!-- cite-right: williams1992simple -->

REINFORCE is the simplest instantiation of the policy gradient. The update rule:

$$\Delta\theta = \alpha \nabla_\theta \log \pi_\theta(y \mid x) \cdot (R(x, y) - b)$$

Three components:

1. **Nonnegative factor** ($\alpha$): the learning rate
2. **Offset reinforcement** ($R - b$): reward minus a baseline for stability
3. **Characteristic eligibility** ($\nabla \log \pi$): how to attribute learning per token

---

## The baseline problem

Without a baseline $b$, if all rewards are positive, the gradient pushes the probability of **every** action up — just by different amounts. This leads to high variance.

**Key insight**: subtracting a baseline $b$ from the reward **doesn't change the expected gradient** (it's still unbiased), but it **reduces variance** dramatically.

$$\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b] = b \cdot \mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s)] = 0$$

The second equality holds because the gradient of a probability distribution sums to zero.

---

## REINFORCE with baseline

The full REINFORCE gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)(G_t - b(s_t))\right]$$

Common baselines:
- **Average reward** over the batch (simplest)
- **Moving average** of recent rewards
- **Learned value function** $V_\phi(s)$

REINFORCE uses Monte Carlo estimates — no learned value function needed.

---

<!-- layout: section-break -->

## REINFORCE Leave-One-Out (RLOO)

---

## RLOO: Leave-One-Out baseline

<!-- cite-right: ahmadian2024back -->

**Key idea**: generate $K$ completions per prompt. Use the other $K-1$ rewards as the baseline:

$$b(s, a_k) = \frac{1}{K-1}\sum_{i=1,\, i \neq k}^{K} R(s, a_i)$$

The advantage for completion $k$:

$$\hat{A}(s, a_k) = R(s, a_k) - b(s, a_k)$$

This is a **per-prompt** baseline that naturally captures prompt difficulty — hard prompts get low rewards across all completions, so the baseline is low.

---

## RLOO worked example

$K = 4$ completions for one prompt, with rewards $[0.8, 0.3, 0.6, 0.5]$:

| Completion | Reward | Baseline (avg of others) | Advantage |
|:----------:|:------:|:------------------------:|:---------:|
| 1 | 0.8 | $(0.3 + 0.6 + 0.5)/3 = 0.467$ | $+0.333$ |
| 2 | 0.3 | $(0.8 + 0.6 + 0.5)/3 = 0.633$ | $-0.333$ |
| 3 | 0.6 | $(0.8 + 0.3 + 0.5)/3 = 0.533$ | $+0.067$ |
| 4 | 0.5 | $(0.8 + 0.3 + 0.6)/3 = 0.567$ | $-0.067$ |

Completion 1 (best) gets reinforced. Completion 2 (worst) gets suppressed. Completions 3 and 4 get small updates.

---

## REINFORCE / RLOO summary

**REINFORCE**: Simplest policy gradient. Needs a baseline to reduce variance. No value function required.

**RLOO**: REINFORCE + a smart, per-prompt leave-one-out baseline. Multiple completions per prompt provide the baseline for free.

Both are the foundation for everything that follows:

- No value function needed (saves memory/compute)
- Simple to implement and debug
- Strong baselines in practice [@ahmadian2024back; @wang2024helpsteer2p]

---

<!-- layout: section-break -->

## Proximal Policy Optimization (PPO)

---

## Why constrain updates?

<!-- cite-right: schulman2017proximal -->

Large gradient steps can destroy the policy:

- Language model loses coherence after one bad batch
- Reward hacking exploits reward model weaknesses
- Training becomes unstable — reward spikes then collapses

**We need trust regions**: each update should be a small, safe step near the current policy.

---

## Trust regions: intuition

**The idea**: limit how far the policy can move in a single update.

- Too big a step → policy collapses, reward crashes
- Too small a step → training is prohibitively slow
- PPO finds a middle ground: clip the update to stay within a trust region

The idea of a "trust region" comes from numerical optimization, popularized in Deep RL by Trust Region Policy Optimization (TRPO) — PPO's predecessor.

---

## Importance sampling

We want to take multiple gradient steps on a batch, but the data came from an old policy $\pi_{\theta_\text{old}}$.

Define the **importance sampling ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$$

- If $r_t = 1$: current and old policy agree on this action
- If $r_t > 1$: current policy assigns higher probability than old
- If $r_t < 1$: current policy assigns lower probability than old

This ratio reweights old-policy samples to estimate new-policy gradients.

---

## The surrogate objective

Using importance sampling, the policy gradient becomes:

$$L^{CPI}(\theta) = \mathbb{E}_t\!\left[r_t(\theta) \hat{A}_t\right]$$

**Problem**: without constraints, maximizing this can take arbitrarily large steps — the ratio $r_t$ can diverge far from 1, making the estimate unreliable.

---

## The PPO clipped objective

PPO clips the ratio to prevent large updates:

$$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_t\right)\right]$$

Where $\varepsilon$ is typically 0.1–0.2. The $\min$ selects the **more conservative** estimate.

---

## Clipping: when advantage is positive ($\hat{A}_t > 0$)

The action was **better** than expected — we want to increase its probability.

- The unclipped objective: $r_t \hat{A}_t$ — increases as we make the action more likely
- The clipped objective: $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_t$ — caps the benefit once $r_t > 1+\varepsilon$

$$\min(r_t \hat{A}_t, (1+\varepsilon) \hat{A}_t)$$

Once the action is already $1+\varepsilon$ times more likely than before, the gradient goes to **zero**. Prevents over-committing to one good action.

---

## Clipping: when advantage is negative ($\hat{A}_t < 0$)

The action was **worse** than expected — we want to decrease its probability.

- The unclipped objective: $r_t \hat{A}_t$ — becomes more negative as $r_t$ decreases (we want this)
- The clipped objective: caps at $(1-\varepsilon) \hat{A}_t$

$$\min(r_t \hat{A}_t, (1-\varepsilon) \hat{A}_t)$$

Once the action is already $(1-\varepsilon)$ times as likely, further suppression is stopped. Prevents catastrophic over-correction.

---

## Clipping summary

In both cases, the $\min$ selects the more conservative estimate:

| Scenario | Unclipped pushes toward | Clipping stops when |
|----------|------------------------|-------------------|
| $\hat{A}_t > 0$ | Increase probability | $r_t > 1 + \varepsilon$ |
| $\hat{A}_t < 0$ | Decrease probability | $r_t < 1 - \varepsilon$ |

**Within the trust region** ($1-\varepsilon \leq r_t \leq 1+\varepsilon$), PPO operates the same as standard policy gradient. The clipping only activates **outside** this region.

---

## The value function (critic)

PPO trains a **value function** $V_\phi(s)$ alongside the policy:

- Separate parameters $\phi$ (often initialized from the reward model or SFT model)
- Predicts expected return at each token position
- Trained via MSE against actual returns: $\mathcal{L}_V = \frac{1}{2}(V_\phi(s_t) - G_t)^2$

The value function serves as a learned baseline for advantage estimation.

---

## Advantage estimation

The simplest advantage: $\hat{A}_t = G_t - V_\phi(s_t)$

**Why advantages help**:
- **Centering** reduces variance — we're asking "how much better is this action than average?" rather than "how good was the total return?"
- **Credit assignment** — with per-token values, each token gets its own advantage signal

But this simple estimate has issues: high variance (from Monte Carlo returns) or high bias (from single-step TD).

---

## GAE: the TD residual

<!-- cite-right: schulman2015high -->

The temporal difference (TD) residual measures how much the actual reward exceeded the value prediction:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This is the **1-step advantage estimate**: low variance (uses learned $V$) but potentially high bias (if $V$ is inaccurate).

---

## GAE: $K$-step advantage

We can extend to $K$ steps:

$$\hat{A}_t^{(1)} = \delta_t \quad \text{(1-step: low variance, high bias)}$$

$$\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1} \quad \text{(2-step)}$$

$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l} \quad \text{(k-step: more variance, less bias)}$$

As $k \to \infty$, we recover the full Monte Carlo return (no bias, highest variance).

---

## GAE: exponential weighting

GAE uses an exponentially-weighted average across all $K$-step estimates:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

Where $\lambda \in [0, 1]$ controls the bias-variance tradeoff.

---

## GAE: bias-variance tradeoff

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

| $\lambda$ | Behavior | Variance | Bias |
|:---------:|----------|:--------:|:----:|
| $0$ | Pure TD (1-step) | Lowest | Highest |
| $0.95$ | Typical default for LLMs | Balanced | Balanced |
| $1$ | Full Monte Carlo return | Highest | None |

The $\gamma$ here is typically $1.0$ for language models (no discounting).

---

## PPO with KL penalty

The full RLHF objective combines PPO with a KL regularizer:

$$J(\theta) = \mathbb{E}\!\left[r_t(\theta) \hat{A}_t\right] - \beta \, D_{\text{KL}}\!\left[\pi_\theta \| \pi_{\text{ref}}\right]$$

**Two layers of regularization**:
1. **Clipping** — limits how far the policy moves per batch (trust region)
2. **KL penalty** — limits total drift from the reference policy across training

These serve different purposes and are not redundant.

---

## PPO training loop (conceptual)

The PPO training loop:

1. **Generate**: sample prompts, generate completions with current policy $\pi_\theta$
2. **Score**: compute rewards with reward model + KL penalty from $\pi_\text{ref}$
3. **Estimate advantages**: compute GAE using learned value function $V_\phi$
4. **Update** (K epochs): clipped policy gradient + value function loss on the same batch
5. **Repeat**: new batch, sync $\pi_{\theta_\text{old}} \leftarrow \pi_\theta$

Typical: K = 2–4 gradient steps per batch before re-generating.

---

## PPO: four models in memory

PPO requires **four** models:

| Model | Purpose | Updates? |
|-------|---------|:--------:|
| Policy $\pi_\theta$ | Generates completions | Yes |
| Value function $V_\phi$ | Estimates per-token expected return | Yes |
| Reference policy $\pi_\text{ref}$ | KL penalty anchor | Frozen |
| Reward model $r_\psi$ | Scores completions | Frozen |

This is memory-intensive — a key motivation for simpler alternatives like GRPO.

---

<!-- layout: section-break -->

## GRPO & Modern Variants

---

## GRPO: key insight

<!-- cite-right: shao2024deepseekmath -->

Group Relative Policy Optimization eliminates the value function entirely.

**Replace learned advantages with group statistics**: generate $G$ completions per prompt, use the group's reward statistics as the baseline.

Two benefits:
1. Avoids the challenge of learning a value function from an LM backbone
2. Saves memory — no extra model copy for the critic

---

## GRPO advantage

For a group of $G$ completions with rewards $r_1, \ldots, r_G$:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

Z-score normalization: positive advantage for above-average completions, negative for below.

Each token in completion $i$ gets the **same** advantage (sequence-level, not per-token).

---

## GRPO objective

Clipped ratio (like PPO) + group-normalized advantages + KL penalty directly in loss:

$$J(\theta) = \frac{1}{G}\sum_{i=1}^{G}\!\left(\min\!\left(r_i \hat{A}_i,\, \text{clip}(r_i, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right) - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_\text{ref})\right)$$

Where $r_i = \frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_\text{old}}(a_i \mid s)}$ is the importance sampling ratio.

---

## GRPO vs PPO

| | **PPO** | **GRPO** |
|---|---------|----------|
| **Value function** | Learned $V_\phi$ | None |
| **Advantage** | Per-token via GAE | Sequence-level, group z-score |
| **KL penalty** | In reward (before advantages) | In loss (separate term) |
| **Models in memory** | 4 (policy, value, ref, RM) | 3 (policy, ref, RM) |
| **Complexity** | Higher | Lower |
| **Popular for** | General RLHF | Reasoning / RLVR (DeepSeek R1) |

GRPO is PPO minus the value function, with a statistical baseline instead.

---

## RLOO vs GRPO

Both use multiple completions per prompt. The key difference is in the details:

| | **RLOO** | **GRPO** |
|---|---------|----------|
| **Baseline** | Leave-one-out mean | Group mean (z-scored) |
| **Update style** | REINFORCE (no clipping) | PPO-style clipped ratio |
| **KL penalty** | In reward | In loss |
| **Advantage** | $R_k - \frac{1}{K-1}\sum_{j \neq k} R_j$ | $\frac{r_i - \text{mean}}{\text{std}}$ |

Same principle (compare to peers), different mechanics. Dr. GRPO (without std normalization) is equivalent to RLOO up to a scaling constant.

---

## GSPO: sequence-level ratios

<!-- cite-right: zheng2025gspo -->

**Problem**: per-token importance ratio products are numerically unstable for long sequences. A single token with a large ratio can dominate the update.

**GSPO** uses a geometric mean — a single, length-normalized importance weight per response:

$$\rho_i(\theta) = \left(\frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_\text{old}}(a_i \mid s)}\right)^{1/|a_i|} = \exp\!\left(\frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s)}{\pi_{\theta_\text{old}}(a_{i,t} \mid s)}\right)$$

---

## GSPO: why geometric mean?

Products of many small numbers $\to$ underflow. Products of numbers slightly above 1 $\to$ overflow.

The geometric mean stays in a reasonable numerical range for any sequence length.

$$\text{Product ratio: } \prod_{t=1}^{T} \frac{\pi_\theta(a_t)}{\pi_{\text{old}}(a_t)} \quad \text{vs.} \quad \text{Geometric mean: } \left(\prod_{t=1}^{T} \frac{\pi_\theta(a_t)}{\pi_{\text{old}}(a_t)}\right)^{1/T}$$

Same gradient direction, better numerics. The clipping range $\varepsilon$ now operates on a per-token average scale.

---

## CISPO: clipped importance sampling

<!-- cite-right: minimax2025minimaxm1scalingtesttimecompute -->

CISPO takes a different approach: clip the **importance weights themselves** rather than the objective, using a stop-gradient:

$$J(\theta) = \sum_{i,t} \text{sg}\!\left(\hat{\rho}_{i,t}\right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s)$$

$$\hat{\rho}_{i,t} = \text{clip}\!\left(\frac{\pi_\theta(a_{i,t})}{\pi_{\text{old}}(a_{i,t})},\; 1-\varepsilon,\; 1+\varepsilon\right)$$

**Key difference from PPO**: clipping weights (not the objective) means every token still receives a gradient signal — the weight just bounds how much it's amplified.

---

## CISPO: asymmetric clipping

CISPO allows different bounds for increasing vs. decreasing probability:

$$\text{clip}(\rho,\; 1 - \varepsilon^{-},\; 1 + \varepsilon^{+})$$

Setting $\varepsilon^{+} > \varepsilon^{-}$ allows more aggressive reward-increasing updates — encouraging exploration.

This is similar to DAPO's "clip-higher" modification for reasoning models, where exploring new token sequences is crucial.

---

## Algorithm evolution

$$\text{PPO (2017)} \to \text{REINFORCE revival (2024)} \to \text{RLOO} \to \text{GRPO (2024)} \to \text{GSPO / CISPO (2025)}$$

**The trend**: simpler algorithms, comparable performance.

- PPO: powerful but complex (4 models, GAE, value function training)
- REINFORCE/RLOO: surprisingly competitive with good baselines
- GRPO: PPO-style clipping without value function — the current favorite for reasoning
- GSPO/CISPO: numerical stability for large-scale MoE models

---

<!-- layout: section-break -->

## Putting It All Together

---

## Algorithm comparison table

| Method | Reward Model | Value Function | Reference Policy |
|:-------|:------------:|:--------------:|:----------------:|
| **REINFORCE** | Yes | No | No |
| **RLOO** | Yes | No | No |
| **PPO** | Yes | Yes | Yes |
| **GRPO** | Yes | No | Yes |
| **GSPO** | Yes | No | Yes |
| **CISPO** | Yes | No | Yes |

All are on-policy in derivation (slightly off-policy in practice).

---

## Complexity spectrum

```box
title: Simple → Complex
tone: accent
content: |
  **REINFORCE** → **RLOO** → **GRPO** → **PPO**

  - Models needed: 2 → 2 → 3 → 4
  - Memory footprint: Low → Low → Medium → High
  - Implementation complexity: Low → Low → Medium → High
  - Per-token credit assignment: No → No → No → Yes (GAE)
```

---

## When to use what

| Scenario | Recommended | Why |
|----------|:-----------:|-----|
| **Reasoning / RLVR** | GRPO | Simple, effective with verifiable rewards |
| **Highest ceiling, ample resources** | PPO | Per-token advantages, proven at scale |
| **Strong simple baseline** | RLOO | No value function, competitive results |
| **Large MoE models** | GSPO / CISPO | Numerical stability at scale |
| **Quick experiments** | REINFORCE | Minimal code, fast iteration |

---

## Key insight

**Data quality and reward signal quality matter more than algorithm choice.**

All of these methods optimize the same policy gradient objective. They differ in:

- **Variance reduction**: baselines, value functions, group statistics
- **Trust region enforcement**: clipping, KL penalties, importance sampling corrections
- **Compute requirements**: number of models in memory, implementation complexity

The algorithm determines the engineering complexity and resource requirements — not the ceiling of what's achievable.

---

## Lecture summary

All methods optimize the **same** policy gradient objective:

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t\right]$$

They differ in:

1. **What $\Psi_t$ is**: total return, advantage, group-normalized advantage
2. **How updates are bounded**: no clipping (REINFORCE), objective clipping (PPO/GRPO), weight clipping (CISPO)
3. **Memory and compute**: 2–4 models, value function training or not

---

<!-- rows: 50/50 -->
## Next up: Implementation!

<!-- row-columns: 32/36/32 -->

```box
title: Overview
tone: muted
compact: true
content: |
  1. Introduction
  2. Key Related Works
  3. Training Overview
```

|||

```box
title: Core Training Pipeline
tone: accent
compact: true
content: |
  4. Instruction Tuning
  5. Reward Models
  6. **Reinforcement Learning**
  7. **Reasoning**
  8. Direct Alignment
  9. Rejection Sampling
```

|||

```box
title: Data & Preferences
tone: muted
compact: true
content: |
  10. What are Preferences
  11. Preference Data
  12. Synthetic Data & CAI
```

===

<!-- row-columns: 32/36/32 -->

```box
title: Practical Considerations
tone: muted
compact: true
content: |
  13. Tool Use
  14. Over-optimization
  15. Regularization
  16. Evaluation
  17. Product & Character
```

|||

```box
title: Appendices
tone: surface
compact: true
content: |
  - A. Definitions
  - B. Style & Information
  - C. Practical Issues
```

|||

```box
title: Course Home
tone: surface
compact: true
content: |
  - [rlhfbook.com](https://rlhfbook.com)
  - [GitHub repo](https://github.com/natolambert/rlhf-book)
```

---

<!-- rows: 85/15 -->
## Thank you

Questions / discussion

Contact: nathan@natolambert.com

Newsletter: [interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===


```builtwith
repo: natolambert/colloquium
```
