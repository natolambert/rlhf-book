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

# Lecture 3: Reinforcement Learning Motivation & Math

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 6, Part 1</p>

---

<!-- rows: 50/50 -->
## Lecture 3: Reinforcement learning (mostly the math)

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
  7. Reasoning
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

System diagram view.

===

![The RLHF training pipeline — the RL step optimizes the policy using the reward model signal.](assets/rlhf-basic.png)

---

<!-- columns: 50/50 -->
## Where we are in the pipeline

After instruction tuning and training a reward model, the RL step updates the policy against a learned reward signal.

RLHF optimization view.

|||

![The RLHF training pipeline — the RL step optimizes the policy using the reward model signal.](assets/rlhf-overview.png)

---

## What this lecture covers

This lecture covers the **math and theory** of RL for language models. The next lecture covers implementation.

```box
title: Lecture 3 Outline
tone: accent
content: |
  1. **Motivation** — why RL training matters
  2. **RL foundations review** — MDP, return, value functions, policy gradient objective
  3. **Policy gradient derivation** — log-derivative trick, step by step
  4. **REINFORCE & RLOO** — the simplest policy gradient algorithms
  5. **PPO** — trust regions, clipping, GAE, value functions
  6. **GRPO & modern variants** — GSPO, CISPO, and the trend toward simplicity
  7. **Comparison** — strengths, weaknesses, questions
```

---

## A under-documented benefit of RL

In lecture 2, we covered rejection sampling and later we'll cover direct alignment algorithms (like DPO). They're simpler, but RL has hard to measure benefits.

Implementing RL is far more complex infrastructure, but the gradient updates they provide "generally help the model a lot". This is hard to quantify, but:
- RL stages can "fix" rough edges on the model, making them easier to chat to or more robust (this could come by training them to have numerical stability inference tools like VLLM)
- RL can be done surgically in the model, the model does a good job learning where the prompt distribution lies, and RL tends to not "squash" the general capabilities of the model. 

Overall, RL losses on language models are robust, scalable, effective, and flexible, which opened large new fields of experimentation. The original method that started us down this path was RLHF work.


---

<!-- layout: section-break -->

## Why RL matters

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
## RLVR: Same algorithms, verifiable rewards

<!-- cite-right: lambert2024t -->

The RL methods in this lecture power **both RLHF and RLVR**:

- **RLHF**: reward model scores subjective quality
- **RLVR**: verification function checks correctness (math, code)

Same policy gradient algorithms, different reward source. We will cover tricks for reasoning training and a bunch of interesting models in depth in a future lecture.

|||

![RLVR uses a verification function instead of a reward model, but the RL algorithms are the same.](assets/rlvr-system.png)

---

<!-- layout: section-break -->

## Policy gradients: Core intuitions

---

<!-- columns: 50/50 -->
## Recall: Classical RL vs. RLHF

<!-- cite-right: christiano2017, ouyang2022training -->

<div class="text-sm">

**Classical RL**: agent in an MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$
- Reward is a known function from the environment per step
- Optimize cumulative return: $J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t} \gamma^t r(s_t, a_t)\right]$

**RLHF**: prompts from a dataset, no environment
- Reward is **learned** from human preferences
- **Response-level** reward, regularized with KL penalty
- $J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$

</div>


|||

![](assets/rlhf.png)


---

<!-- columns: 50/50 -->
## Notation in this lecture

<div class="text-sm">

This lecture uses $(s, a)$ notation from the reinforcement learning literature, where $s$ denotes states and $a$ denotes actions. In the language model context, you will often see $(x, y)$ instead, where $x$ is the prompt and $y$ is the completion.

The $(s, a)$ framing is more general — these algorithms were designed for sequential decision problems where actions are taken at each timestep. However, many RLHF implementations treat the entire completion as a single action, making the $(x, y)$ notation equally valid.

</div>


|||

![](assets/rlhf.png)


---

## What are we optimizing?

Choose policy parameters $\theta$ that maximize reward **on average** under the current policy.

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

In practice, we estimate this expectation with sampled rollouts:

$$J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} R(\tau_i), \qquad \tau_i \sim \pi_\theta$$

We craft a gradient/derivative that lets us optimize this.

---


## How a policy gradient works

Make actions more likely when they lead to better outcomes.

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

<div class="text-sm">

Read it as two questions answered at once:

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — **which action?**
- $\Psi_t$ — **how good was it?** A scalar scoring the outcome.

</div>

---

## How a policy gradient works

Make actions more likely when they lead to better outcomes.

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

<div class="text-sm">

Read it as two questions answered at once:

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — **which action?** How each parameter influenced the probability of taking action $a_t$ in state $s_t$. This connects the outcome back to the knobs that caused it.
- $\Psi_t$ — **how good was it?** A scalar scoring the outcome. Positive means good, negative means bad, magnitude says how much.

> An oversimplification (e.g. intuition in case of batch size of 1): the gradient is a vector with one entry per parameter. A positive entry means "increasing this parameter made the action more likely," a negative entry means the opposite. In practice, the update averages over many such vectors — what survives is the net vote across the batch.

Multiply them: $\Psi_t > 0$ updates parameters to make $a_t$ more likely, $\Psi_t < 0$ updates them to make it less likely.

The rest of this section is about choosing a smart $\Psi_t$ — different choices (total return, advantage, TD residual) trade off variance and bias, but all plug into this same update.

</div>

---

## The policy gradient equation

A preview of where we end up:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t\right]$$

The core idea is that we *sample over trials in the environment* and **estimate** the gradient.

Where $\Psi_t$ is the **learning signal** telling the optimizer how good the action was. The choice of $\Psi_t$ determines the algorithm's variance, bias, and compute cost.


---

## What $\Psi_t$ can be

<!-- cite-right: schulman2015high -->

Popular choices for $\Psi_t$ (rewards can also be discounted by $\gamma$):

| | $\Psi_t$ | Description | Variance | Bias |
|:-:|-----------|-------------|:--------:|:----:|
| 1. | $R(\tau) = \sum_{t=0}^{T} r_t$ | Total trajectory reward | Highest | None |
| 2. | $\sum_{t'=t}^{T} r_{t'}$ | Future return from $t$ (the return, $G_t$) | High | None |
| 3. | $\sum_{t'=t}^{T} r_{t'} - b(s_t)$ | Baselined return | Lower | None |
| 4. | $Q^{\pi}(s_t, a_t)$ | State-action value function | Med | Depends |
| 5. | $A^{\pi}(s_t, a_t) = Q - V$ | Advantage function | **Lowest** | None |
| 6. | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD residual | Low | Some |

A *baseline* $b(s_t)$ is any value subtracted from the reward signal to reduce variance — we'll show why this is unbiased shortly.

---

## Unpacking the key RL quantities

The $\Psi_t$ options reference three key quantities:

**Value function** $V(s)$: expected future return from state $s$

$$V(s) = \mathbb{E}\big[G_t \mid S_t = s\big] \quad \text{where } G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Action-value** $Q(s, a)$: expected return after taking action $a$ in state $s$

$$Q(s, a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]$$

**Advantage** $A(s, a) = Q(s, a) - V(s)$: how much better is action $a$ compared to average? Positive → reinforce, negative → suppress, zero → no update.

In RLHF: often $\gamma = 1$ (no discounting) because the unit of optimization is the full completion.

---


<!-- layout: section-break -->

## Deriving the policy gradient

---

## Setup: Differentiating an expectation

Ideally, we want $\nabla_\theta J(\theta)$, but we can't do that (the state sampling distribution itself depends on $\theta$).

Written as an integral, the objective is:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int_\tau p_\theta(\tau) R(\tau) \, d\tau$$

---

## Setup: Differentiating an expectation

Ideally, we want $\nabla_\theta J(\theta)$, but we can't do that (the state sampling distribution itself depends on $\theta$).

Written as an integral, the objective is:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int_\tau p_\theta(\tau) R(\tau) \, d\tau$$

Taking the gradient directly:

$$\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau$$

We can sample trajectories from $p_\theta(\tau)$, but not from $\nabla_\theta p_\theta(\tau)$, since it is not a probability distribution.

The derivation is a few key tricks to let us approximate or compute this.

---

## Applying the idea of computing the gradient from trajectories

Substituting the log-derivative identity (more on this soon) into the gradient:

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau
&& \text{differentiate the objective} \\
&= \int_\tau p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) \, d\tau
&& \text{log-derivative identity (chain rule)} \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\nabla_\theta \log p_\theta(\tau) \cdot R(\tau)\right]
&& \text{expectation form}
\end{aligned}
$$

Now we can estimate this with Monte Carlo sampling!

---

## Applying the idea of computing the gradient from trajectories

The only non-obvious step is the middle line:

$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau)
&= \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \\
\Rightarrow \quad
\nabla_\theta p_\theta(\tau)
&= p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
\end{aligned}
$$

This is the **log-derivative identity**, derived from the chain rule for $\log$.

It rewrites the gradient of a probability in terms of a log-probability, which is much easier to work with.

---

## Applying the idea of computing the gradient from trajectories

Where did the $p_\theta(\tau)$ term go?

It became the **sampling distribution** inside the expectation:

$$\int p_\theta(\tau) f(\tau)\, d\tau = \mathbb{E}_{\tau \sim p_\theta}[f(\tau)]$$

Monte Carlo then estimates that expectation by sampling trajectories from the policy:

$$\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] \approx \frac{1}{N}\sum_{i=1}^{N} f(\tau_i), \qquad \tau_i \sim p_\theta$$

So there is no explicit $p_\theta(\tau_i)$ term in code: **more likely trajectories already appear more often in the sampled batch**.

At a high level, rollout generation handles the sampling, and the loss code handles $R(\tau_i)\,\nabla_\theta \log p_\theta(\tau_i)$.

---

## Expanding log probability of trajectory to compute $\nabla_\theta$

The trajectory probability factorizes:

$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)$$

Taking the log:

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

For language models, this is the familiar autoregressive pattern: a sequence log-probability becomes a sum of token log-probabilities.

---

## Expanding log probability of trajectory to compute $\nabla_\theta$

<div class="text-sm">

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

Now take the gradient w.r.t. $\theta$:

- $\nabla_\theta \log p(s_0) = 0$ — initial state doesn't depend on $\theta$
- $\nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$ — environment dynamics don't depend on $\theta$
- Only $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ survives!

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

</div>
---

## Expanding log probability of trajectory to compute $\nabla_\theta$

<div class="text-sm">

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log p(s_{t+1} \mid s_t, a_t)$$

Now take the gradient w.r.t. $\theta$:

- $\nabla_\theta \log p(s_0) = 0$ — initial state doesn't depend on $\theta$
- $\nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$ — environment dynamics don't depend on $\theta$
- Only $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ survives!

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

In language-model code, this is why you repeatedly see:

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

Autodiff turns that summed log-probability into the corresponding sum of per-token gradients!
</div>
---

## From full return to future rewards

An action at time $t$ can't affect past rewards. So instead of weighting by the full trajectory reward $R(\tau)$:

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)\right]$$

We can replace $R(\tau)$ with the **return-to-go** $G_t = \sum_{t'=t}^{T} R_{t'}$ — only future rewards from $t$ onward:

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

This doesn't change the expected gradient, but removes noise from past rewards that the current action couldn't have influenced. This is the step from $\Psi_t$ option 1 → option 2 in our taxonomy.

---

## Baselines keep the estimator unbiased

Raw returns are noisy. The same action can appear in trajectories with very different total rewards because of later sampled actions and, in classical RL, environment randomness.

The role of a baseline is to **center** that noisy signal: instead of asking "was the return high?", we ask "was it higher or lower than expected for this state?"


---

## Baselines keep the estimator unbiased

Raw returns are noisy. The same action can appear in trajectories with very different total rewards because of later sampled actions and, in classical RL, environment randomness.

The role of a baseline is to **center** that noisy signal: instead of asking "was the return high?", we ask "was it higher or lower than expected for this state?"

Use a centered return:

$$\Psi_t = G_t - b(s_t)$$

where the baseline $b(s_t)$ depends only on the state, not on which action was sampled.

Subtracting this baseline doesn't change the expected gradient:

---

## Baselines keep the estimator unbiased

<div class="text-sm">

Raw returns are noisy. The same action can appear in trajectories with very different total rewards because of later sampled actions and, in classical RL, environment randomness.

Use a centered return, $\Psi_t = G_t - b(s_t)$.

Subtracting this baseline doesn't change the expected gradient:

$$
\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot (G_t - b(s))]
= \mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot G_t]
- \mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)]
$$

The first term is the original estimator. The second vanishes:

$$
\begin{aligned}
\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)]
&= b(s) \cdot \sum_a \nabla_\theta \pi_\theta(a \mid s) \\
&= b(s) \cdot \nabla_\theta \underbrace{\sum_a \pi_\theta(a \mid s)}_{= 1} \\
&= 0
\end{aligned}
$$

The gradient of a normalized distribution sums to zero — so we're free to subtract **any function of state** as a baseline. This is why $\Psi_t$ options 3–5 reduce variance without introducing bias.

</div>

---

## Recall: What $\Psi_t$ can be

<!-- cite-right: schulman2015high -->

Popular choices for $\Psi_t$ (rewards can also be discounted by $\gamma$):

| | $\Psi_t$ | Description | Variance | Bias |
|:-:|-----------|-------------|:--------:|:----:|
| 1. | $R(\tau) = \sum_{t=0}^{T} r_t$ | Total trajectory reward | Highest | None |
| 2. | $\sum_{t'=t}^{T} r_{t'}$ | Future return from $t$ (the return, $G_t$) | High | None |
| 3. | $\sum_{t'=t}^{T} r_{t'} - b(s_t)$ | Baselined return | Lower | None |
| 4. | $Q^{\pi}(s_t, a_t)$ | State-action value function | Med | Depends |
| 5. | $A^{\pi}(s_t, a_t) = Q - V$ | Advantage function | **Lowest** | None |
| 6. | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD residual | Low | Some |

A *baseline* $b(s_t)$ is any value subtracted from the reward signal to reduce variance — we'll show why this is unbiased shortly.

---


## The policy gradient estimator

Combining the log-derivative trick, return-to-go, and baseline subtraction:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t\right]$$

This is the **policy gradient theorem**. Every algorithm in this lecture is an instantiation with a specific choice of $\Psi_t$ and regularization.

For language models, this means the sum runs over generated tokens in the completion, while $\Psi_t$ determines how each token is weighted.

---

## Policy gradient derivation recap

$$
\begin{aligned}
J(\theta)
&= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
&& \text{objective} \\
&= \int_\tau p_\theta(\tau) R(\tau)\, d\tau
&& \text{integral form} \\
\nabla_\theta J(\theta)
&= \int_\tau \nabla_\theta p_\theta(\tau) R(\tau)\, d\tau
&& \text{differentiate} \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t\right]
&& \text{final estimator}
\end{aligned}
$$

- Log-derivative trick turns $\nabla p_\theta(\tau)$ into a sampleable expectation
- Expanding $\log p_\theta(\tau)$ gives a sum of policy log-prob terms
- $\Psi_t$ is the learning signal: return, advantage, or TD-style estimate
- In code: sample rollouts, sum token log-probs, weight by $\Psi_t$, average

---

<!-- layout: section-break -->

## REINFORCE

---

## REINFORCE

<!-- cite-right: williams1992simple -->

REINFORCE is the simplest instantiation of the policy gradient. The update rule:

<div class="text-sm">

> The name is an acronym for "REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility."

</div>

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
- **Learned value function** $V_\phi(s)$ (optional — bridges to actor-critic methods)

Basic REINFORCE needs no critic — just Monte Carlo returns and a simple baseline. Adding a learned $V_\phi$ reduces variance further but introduces the complexity of training a second model.

---

<!-- layout: section-break -->

## REINFORCE leave-one-out (RLOO)

---

## RLOO: Leave-one-out baseline

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

## From sequence reward to token-level training

The reward model gives one scalar $R(x, y)$ at the end of a completion. How does this become a per-token training signal?

1. **KL penalty shapes intermediate tokens**: at each token $t$, subtract $\beta \cdot \text{KL}_t$ from the reward → per-token reward signal
2. **Final token gets the RM score**: $r_T = R(x, y) - \beta \, \text{KL}_T$, all other tokens get $r_t = -\beta \, \text{KL}_t$
3. **GAE propagates credit backward**: the value function + TD residuals assign per-token advantages from these shaped rewards

This is how a sequence-level reward becomes token-level PPO training.

---

## If REINFORCE is so simple, why PPO?

REINFORCE and RLOO work — but they have limitations:

- **High variance**: sequence-level advantages are noisy, especially for long completions
- **No sample reuse**: each batch is used once, then discarded
- **No trust region**: a single bad batch can destabilize the policy

PPO addresses all three: per-token credit assignment via GAE, multiple gradient steps per batch via importance sampling, and clipped updates to prevent catastrophic steps.

---

<!-- layout: section-break -->

## Proximal policy optimization (PPO)

---

## Why constrain updates?

<!-- cite-right: schulman2017proximal -->

Large gradient steps can destroy the policy:

- Language model loses coherence after one bad batch
- Reward hacking exploits reward model weaknesses
- Training becomes unstable — reward spikes then collapses

The solution: **trust regions** — limit how far the policy can move in a single update. Too big a step → policy collapses, reward crashes. Too small → prohibitively slow. PPO clips the update to stay within a trust region.

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

## Clipping: When advantage is positive ($\hat{A}_t > 0$)

The action was **better** than expected — we want to increase its probability.

- The unclipped objective: $r_t \hat{A}_t$ — increases as we make the action more likely
- The clipped objective: $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_t$ — caps the benefit once $r_t > 1+\varepsilon$

$$\min(r_t \hat{A}_t, (1+\varepsilon) \hat{A}_t)$$

Once the action is already $1+\varepsilon$ times more likely than before, the gradient goes to **zero**. Prevents over-committing to one good action.

---

## Clipping: When advantage is negative ($\hat{A}_t < 0$)

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

## GAE: The TD residual

<!-- cite-right: schulman2015high -->

The temporal difference (TD) residual measures how much the actual reward exceeded the value prediction:

$$\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$$

This is the **1-step advantage estimate**: low variance (uses learned $V$) but potentially high bias (if $V$ is inaccurate).

---

## GAE: $K$-step advantage

We can extend to $K$ steps:

$$\hat{A}_t^{(1)} = \delta_t \quad \text{(1-step: low variance, high bias)}$$

$$\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1} \quad \text{(2-step)}$$

$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l} \quad \text{(k-step: more variance, less bias)}$$

As $k \to \infty$, we recover the full Monte Carlo return (no bias, highest variance).

---

## GAE: Exponential weighting

GAE uses an exponentially-weighted average across all $K$-step estimates:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

Where $\lambda \in [0, 1]$ controls the bias-variance tradeoff.

---

## GAE: Bias-variance tradeoff

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

| $\lambda$ | Behavior | Variance | Bias |
|:---------:|----------|:--------:|:----:|
| $0$ | Pure TD (1-step) | Lowest | Highest |
| $0.95$ | Typical default for LLMs | Balanced | Balanced |
| $1$ | Full Monte Carlo return | Highest | None |

The $\gamma$ here is typically $1.0$ for language models (no discounting).

---

## PPO-RLHF: Full policy objective

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

## PPO: Four models in memory

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

## GRPO & modern variants

---

## Can we keep PPO's stability without the critic?

<!-- cite-right: shao2024deepseekmath -->

Group Relative Policy Optimization eliminates the value function entirely.

**Replace learned advantages with group statistics**: generate $G$ completions per prompt, use the group's reward statistics as the baseline.

Two benefits:
1. Avoids the challenge of learning a value function from an LM backbone
2. Saves memory — no extra model copy for the critic

---

## GRPO advantage

For a group of $G$ completions with rewards $R_1, \ldots, R_G$:

$$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \ldots, R_G)}{\text{std}(R_1, \ldots, R_G)}$$

Z-score normalization: positive advantage for above-average completions, negative for below.

Each token in completion $i$ gets the **same** advantage (sequence-level, not per-token).

---

## GRPO objective

Clipped ratio (like PPO) + group-normalized advantages + KL penalty directly in loss:

$$J(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|a_i|}\sum_{t=1}^{|a_i|}\!\left(\min\!\left(r_{i,t} \hat{A}_i,\, \text{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right) - \beta \, D_{\text{KL}, i,t}\right)$$

Where the clipping applies **per-token**: $r_{i,t} = \frac{\pi_\theta(a_{i,t} \mid s_t)}{\pi_{\theta_\text{old}}(a_{i,t} \mid s_t)}$, but $\hat{A}_i$ is shared across all tokens in the completion (sequence-level advantage, per-token ratio).

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
| **KL penalty** | Optional (in reward) | In loss |
| **Advantage** | $R_k - \frac{1}{K-1}\sum_{j \neq k} R_j$ | $\frac{R_i - \text{mean}}{\text{std}}$ |

Same principle (compare to peers), different mechanics. Without std normalization, GRPO is equivalent to RLOO up to a scaling constant.

---

## GSPO: Sequence-level ratios

<!-- cite-right: zheng2025gspo -->

**Problem**: per-token importance ratio products are numerically unstable for long sequences. A single token with a large ratio can dominate the update.

**GSPO** uses a geometric mean — a single, length-normalized importance weight per response:

$$\rho_i(\theta) = \exp\!\left(\frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s)}{\pi_{\theta_\text{old}}(a_{i,t} \mid s)}\right)$$

The geometric mean stays in a reasonable numerical range for any sequence length — similar gradient signal, better numerics. The clipping range $\varepsilon$ now operates on a per-token average scale.

---

## CISPO: Clipped importance sampling

<!-- cite-right: minimax2025minimaxm1scalingtesttimecompute -->

CISPO clips the **importance weights themselves** rather than the objective, using a stop-gradient:

$$J(\theta) = \sum_{i,t} \text{sg}\!\left(\hat{\rho}_{i,t}\right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s)$$

$$\hat{\rho}_{i,t} = \text{clip}\!\left(\frac{\pi_\theta(a_{i,t})}{\pi_{\text{old}}(a_{i,t})},\; 1-\varepsilon^{-},\; 1+\varepsilon^{+}\right)$$

**Key difference from PPO**: every token still receives a gradient signal — the weight just bounds how much it's amplified. Asymmetric bounds ($\varepsilon^{+} > \varepsilon^{-}$) allow more aggressive reward-increasing updates, encouraging exploration.

---

## Algorithm evolution

$$\text{PPO (2017)} \to \text{REINFORCE revival (2024)} \to \text{RLOO} \to \text{GRPO (2024)} \to \text{GSPO / CISPO (2025)}$$

**The trend**: simpler algorithms, comparable performance.

- PPO: powerful but complex (4 models, GAE, value function training)
- REINFORCE/RLOO: surprisingly competitive with good baselines
- GRPO: PPO-style clipping without value function — the current favorite for reasoning
- GSPO/CISPO: numerical stability for large-scale MoE models

---

## Bandit-style vs MDP-style RLHF

<!-- columns: 50/50 -->

**Bandit-style**
- One reward per completion
- Sequence-level $\Psi_t$ broadcast across tokens
- Used in REINFORCE, RLOO, GRPO

|||

**MDP-style**
- Each token is treated as an action
- Per-token values or advantages
- Used in PPO with GAE

<div class="colloquium-spacer-md"></div>

Most RLHF is mixed in practice: **sequence-level rewards**, but **token-level log-prob gradients**.

---

<!-- layout: section-break -->

## Putting it all together

---

## Algorithm comparison table

| Method | Models | Rollouts | Value Function | Reference Policy |
|:-------|:------:|:--------:|:--------------:|:----------------:|
| **REINFORCE** | 2 | 1 | No | No |
| **RLOO** | 2 | K | No | No |
| **PPO** | 4 | 1 | Yes | Yes |
| **GRPO** | 3 | G | No | Yes |
| **GSPO** | 3 | G | No | Yes |
| **CISPO** | 3 | G | No | Yes |

All are on-policy in derivation (slightly off-policy in practice). Complexity spectrum: **REINFORCE → RLOO → GRPO → PPO** (2→4 models, low→high memory).

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

## Lecture summary

**In most RLHF setups, data quality and reward signal quality dominate; algorithm choice mostly determines stability, efficiency, and engineering burden.** All methods optimize the **same** policy gradient objective:

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t\right]$$

They differ in:

1. **What $\Psi_t$ is**: total return, advantage, group-normalized advantage
2. **How updates are bounded**: no clipping (REINFORCE), objective clipping (PPO/GRPO), weight clipping (CISPO)
3. **Memory and compute**: 2–4 models, value function training or not

---

## Next lecture: RL implementation

Lecture 4 turns these algorithms into working code:

- **Token-level losses**: log-prob computation, autoregressive shift, masking
- **Loss aggregation**: per-sequence vs per-token normalization and why it matters
- **Cached rollouts**: what to store (log-probs, values, ref log-probs) and what goes wrong when you don't
- **PPO/GRPO debugging**: monitoring training, identifying reward hacking, fixing common failures

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
