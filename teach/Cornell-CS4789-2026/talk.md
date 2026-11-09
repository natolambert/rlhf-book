---
title: "Policy gradients at the frontier: RL for language models"
author: "Nathan Lambert"
date: "November 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com"
  center: "Cornell CS 4789"
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

# Policy gradients at the frontier: RL for language models

<div class="colloquium-title-eyebrow">Cornell CS 4789</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
<p>Guest lecture<br>18 November 2026</p>
</div>

<p class="colloquium-title-note">From REINFORCE on tokens to training-inference mismatch and score centering.</p>

---

## What this lecture covers

1. **Policy gradients in LLM code.** How log-probs, logits, and a surrogate loss turn REINFORCE into a few lines of PyTorch.
2. **Training-inference mismatch.** A precise definition of how the model that samples differs from the model that learns.
3. **Score centering.** A frontier fix that comes from the same zero-mean identity behind baselines [@marek2026score].

Everything today builds on one fact you have already used:

$$\mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid x)\right] = 0$$

---

## Notation in this lecture

| Symbol | Meaning | In the RLHF book |
|--------|---------|------------------|
| $x$ | Prompt | $s$ |
| $a = (a_1, \dots, a_T)$ | Completion, one token at a time | $a$ |
| $v \in \mathcal{V}$ | Any token in the vocabulary | $v$ |
| $p_\theta$ | **Learner**: the trainer's next-token distribution | $\pi_\theta^{\text{learner}}$ |
| $q$ | **Sampler**: the inference engine's next-token distribution | $\pi_\theta^{\text{sampler}}$ |
| $f_\theta$ | Logits, so $p_\theta = \text{softmax}(f_\theta)$ | |
| $s_v = \nabla_\theta \log p_v$ | **Score** of token $v$ | $\boldsymbol{\xi}_v$ |
| $A$ | Advantage for the whole completion, e.g., $r - b(x)$ | $A$ |

$p_v$ and $q_v$ are shorthand for the probability of token $v$ at one position, given $x$ and the prefix $a_{<t}$.

---

<!-- layout: section-break -->

## Policy gradients in LLM code

---

## A language model is a policy over tokens

A completion is sampled one token at a time, so its probability is a product:

$$p_\theta(a \mid x) = \prod_{t=1}^{T} p_\theta(a_t \mid x, a_{<t})$$

<!-- step -->

Taking the log turns the product into a sum:

$$\log p_\theta(a \mid x) = \sum_{t=1}^{T} \log p_\theta(a_t \mid x, a_{<t})$$

<!-- step -->

Taking the gradient gives one score per generated token:

$$\nabla_\theta \log p_\theta(a \mid x) = \sum_{t=1}^{T} \nabla_\theta \log p_\theta(a_t \mid x, a_{<t}) = \sum_{t=1}^{T} s_{a_t}$$

---

## REINFORCE for a completion

The policy gradient with one advantage for the whole completion:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim p_\theta}\left[ A \sum_{t=1}^{T} s_{a_t} \right]$$

<!-- step -->

With $n$ sampled completions, this is estimated as:

$$\hat{g} = \frac{1}{n} \sum_{i=1}^{n} A_i \sum_{t=1}^{T_i} s_{a_{i,t}}$$

<!-- step -->

This is **sequence-level reward, token-level gradients**: every token in a completion gets the same weight $A_i$, whether or not it caused the outcome.

---

## From logits to log-probs

The model returns logits at every position, and we keep the log-prob of the token that was generated:

```python
# Forward pass through the model
logits = model(input_ids).logits              # (B, L, vocab_size)

# Autoregressive shift: logit at position t predicts token at t+1
logits = logits[:, :-1, :]                    # (B, L-1, vocab_size)
labels = input_ids[:, 1:]                     # (B, L-1)
completion_mask = completion_mask[:, 1:]      # (B, L-1), 0 on prompt tokens

# Log-probs over the whole vocabulary, then pick the generated token
vocab_logprobs = logits.log_softmax(dim=-1)   # (B, L-1, vocab_size)
token_logprobs = vocab_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

# log p(a | x) is a sum over the completion tokens only
seq_logprobs = (token_logprobs * completion_mask).sum(dim=-1)  # (B,)
```

---

## We write a loss, not a gradient

We never form the score vectors. We write a scalar whose gradient is the update, then let autodiff do the rest.

$$L = -\frac{1}{n} \sum_{i=1}^{n} A_i \sum_{t=1}^{T_i} \log p_\theta(a_{i,t} \mid x_i, a_{i,<t})$$

<!-- step -->

Differentiating, with $A_i$ treated as a constant:

$$-\nabla_\theta L = \frac{1}{n} \sum_{i=1}^{n} A_i \sum_{t=1}^{T_i} s_{a_{i,t}} = \hat{g}$$

<!-- step -->

```python
loss = -(advantages.detach() * seq_logprobs).mean()
loss.backward()  # autodiff computes the sum of A * score
```

A single backward pass gives the whole sum. Computing a score for each token in the vocabulary separately would take one backward pass per token.

---

## What is inside a score?

The softmax connects one token's score to every logit. Write the log-prob in terms of logits:

$$\log p_v = f_v - \log \sum_{u \in \mathcal{V}} e^{f_u}$$

<!-- step -->

Differentiate, using the chain rule on the second term:

$$s_v = \nabla_\theta f_v - \frac{\sum_{u} e^{f_u} \, \nabla_\theta f_u}{\sum_{w} e^{f_w}}$$

<!-- step -->

Each fraction $e^{f_u} / \sum_w e^{f_w}$ is exactly $p_u$:

$$s_v = \nabla_\theta f_v - \sum_{u \in \mathcal{V}} p_u \, \nabla_\theta f_u$$

<!-- step -->

Raise the sampled token's logit, and push down the logits the learner already expects. We use this form again later.

---

## Where advantages come from

<!-- cite-right: shao2024deepseekmath -->

Sample $G$ completions per prompt, and use the group as a baseline. GRPO also divides by the group's standard deviation:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

```python
# rewards: (B*G,), from a verifier or a reward model
mean_r = rewards.view(-1, G).mean(dim=1).repeat_interleave(G)
std_r = rewards.view(-1, G).std(dim=1).repeat_interleave(G)
advantages = (rewards - mean_r) / (std_r + 1e-4)  # (B*G,)
```

No value function is needed, which saves a critic the size of the policy in memory.

---

## One training step, end to end

```python
# 1. Sample completions on the inference engine (pseudocode)
input_ids, completion_mask = sampler.generate(prompts, n=G)

# 2. Score them, then center within each group
rewards = verifier(prompts, input_ids)                  # (B*G,)
advantages = group_center(rewards, G)                   # (B*G,)

# 3. Recompute log-probs on the trainer, this time with gradients
logits = model(input_ids).logits[:, :-1, :]
token_logprobs = logits.log_softmax(-1).gather(
    -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
seq_logprobs = (token_logprobs * completion_mask[:, 1:]).sum(-1)

# 4. Update, then sync the new weights back to the sampler
(-(advantages * seq_logprobs).mean()).backward()
optimizer.step()
sampler.load_weights(model)
```

Steps 1 and 3 ask two different systems about the same tokens. Do they agree?

---

<!-- layout: section-break -->

## Training-inference mismatch

---

## Two systems, not one policy

<!-- columns: 42/58 -->

<!-- cite-right: noukhovitch2024asynchronous -->

At scale, sampling and learning are separate:

- **Sampling** runs on an inference engine (e.g., vLLM), built for throughput.
- **Learning** runs on a training framework (e.g., FSDP), with gradients.
- **Asynchrony** keeps GPUs busy, but the sampler's weights can fall behind.

|||

![Synchronous training idles GPUs. Asynchronous training overlaps generation and training for full utilization, so it is off-policy.](assets/async_v_synch_rl.png)

---

## Defining the mismatch

At one position, with prompt $x$ and prefix $a_{<t}$, each system defines a next-token distribution:

$$
\begin{aligned}
p_v &= \text{softmax}\big( f_\theta(x, a_{<t}) \big)_v && \text{learner: current weights, training numerics} \\
q_v &= \text{softmax}\big( \tilde{f}_{\tilde{\theta}}(x, a_{<t}) \big)_v && \text{sampler: synced weights, inference numerics}
\end{aligned}
$$

<!-- step -->

**Training-inference mismatch** (TIM) is when $q \neq p$. There are two independent sources:

- **Staleness:** $\tilde{\theta} \neq \theta$, because the weights have moved since the last sync.
- **Numerics:** $\tilde{f} \neq f$, because of different kernels, precision, parallelism, or a quantized KV cache [@yao2025offpolicy; @marek2026score].

<!-- step -->

The second source means that **$q \neq p$ even when $\tilde{\theta} = \theta$**. RL for language models is off-policy by accident.

---

## Same tokens, different probabilities

<!-- cite-right: minimax2025minimax_m1 -->

<!-- img-align: center -->

![Training vs. inference probability for the same tokens in MiniMax-M1, before and after a fix. Agreement improves, but the two systems still differ.](assets/minimax-m1-logprob-mismatch.png)

---

## Measuring the mismatch at one token

The ratio between the two systems for a sampled token, and its log:

$$w_t = \frac{p(a_t \mid x, a_{<t})}{q(a_t \mid x, a_{<t})}, \qquad \delta_t = \log w_t = \log p(a_t \mid x, a_{<t}) - \log q(a_t \mid x, a_{<t})$$

<!-- step -->

Tokens are sampled from $q$, so take the expectation under $q$:

$$\mathbb{E}_{a_t \sim q}[\delta_t] = \sum_{v \in \mathcal{V}} q_v \left( \log p_v - \log q_v \right)$$

<!-- step -->

Flip the sign to recognize the KL divergence:

$$\mathbb{E}_{a_t \sim q}[\delta_t] = -\sum_{v \in \mathcal{V}} q_v \log \frac{q_v}{p_v} = -\text{KL}(q_t \,\Vert\, p_t)$$

<!-- step -->

If the systems agree, every $\delta_t$ is zero. Otherwise, log-ratios are negative on average.

---

## Mismatch adds up over a completion

The ratio for a whole completion is the product of the token ratios:

$$w(a) = \frac{p_\theta(a \mid x)}{q(a \mid x)} = \prod_{t=1}^{T} w_t, \qquad \log w(a) = \sum_{t=1}^{T} \delta_t$$

<!-- step -->

Take the expectation under $q$, and apply the previous slide at each position:

$$\mathbb{E}_{a \sim q}[\log w(a)] = \sum_{t=1}^{T} \mathbb{E}_{a_{<t} \sim q}\big[ \mathbb{E}_{a_t \sim q}[\delta_t] \big] = -\sum_{t=1}^{T} \mathbb{E}_{a_{<t} \sim q}\big[ \text{KL}(q_t \,\Vert\, p_t) \big]$$

<!-- step -->

This sum is the KL between the two sequence distributions, $\text{KL}\big(q(a \mid x) \,\Vert\, p_\theta(a \mid x)\big)$, so it grows with length.

<!-- step -->

For illustration, a small per-token KL of $0.001$ over $10{,}000$ tokens gives a sequence KL of $10$ nats. A typical weight is then near $e^{-10} \approx 5 \times 10^{-5}$.

---

## What training actually estimates

The gradient we want samples from the learner:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim p_\theta}\left[ A \sum_{t=1}^{T} \nabla_\theta \log p_\theta(a_t \mid x, a_{<t}) \right]$$

<!-- step -->

The gradient we compute samples from the sampler, but differentiates through the learner:

$$\hat{g} = \mathbb{E}_{a \sim q}\left[ A \sum_{t=1}^{T} \nabla_\theta \log p_\theta(a_t \mid x, a_{<t}) \right]$$

<!-- step -->

These differ whenever $q \neq p$, and the gap has two parts: the tokens are drawn from the wrong distribution, and so are the prefixes they follow.

---

## The classical fix: Importance sampling

Reweighting by the sequence ratio recovers the gradient exactly [@marek2026score]:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim q}\left[ w(a) \, A \sum_{t=1}^{T} s_{a_t} \right], \qquad w(a) = \frac{p_\theta(a \mid x)}{q(a \mid x)}$$

<!-- step -->

But $w$ can take very large values on rare tokens, which inflates the variance of the gradient. So practical methods bound it, which introduces bias:

- **Truncate:** Cap the ratio at a constant, $\min(w_t, C)$ [@ionides2008truncated].
- **Clip or mask:** Drop or bound tokens whose ratio leaves a range, as in PPO-style objectives.
- **Apply it per token:** Use $w_t$ rather than the full product $w(a)$ [@yao2025offpolicy].

<!-- step -->

Is there a correction that needs no ratio at all? To find one, we go back to the identity behind baselines.

---

<!-- layout: section-break -->

## Score centering

---

## Recall: The score has zero mean on-policy

<!-- cite-right: williams1992simple -->

At one position, the expected score under the learner, written as a sum:

$$\mathbb{E}_{v \sim p}[s_v] = \sum_{v \in \mathcal{V}} p_v \, \nabla_\theta \log p_v$$

<!-- step -->

Apply the log-derivative identity, $p_v \nabla_\theta \log p_v = \nabla_\theta p_v$:

$$= \sum_{v \in \mathcal{V}} \nabla_\theta p_v$$

<!-- step -->

Pull the gradient out of the sum. Probabilities sum to one:

$$= \nabla_\theta \sum_{v \in \mathcal{V}} p_v = \nabla_\theta 1 = \mathbf{0}$$

<!-- step -->

This is why baselines are free: $\mathbb{E}[b(x) \, s_{a_t}] = b(x) \cdot \mathbf{0}$. It relies on the expectation and the score using **the same distribution**.

---

## Off-policy, the identity breaks

Tokens now come from $q$, but the score is still computed through $p$. Define the expected score under the sampler:

$$\bar{s} = \mathbb{E}_{v \sim q}[s_v] = \sum_{v \in \mathcal{V}} q_v \, s_v$$

<!-- step -->

Substitute the score from earlier, $s_v = \nabla_\theta f_v - \sum_u p_u \nabla_\theta f_u$:

$$\bar{s} = \sum_{v} q_v \nabla_\theta f_v - \left( \sum_{v} q_v \right) \sum_{u} p_u \nabla_\theta f_u$$

<!-- step -->

Since $\sum_v q_v = 1$, and renaming $u$ to $v$:

$$\bar{s} = \sum_{v} q_v \nabla_\theta f_v - \sum_{v} p_v \nabla_\theta f_v = \sum_{v \in \mathcal{V}} (q_v - p_v) \, \nabla_\theta f_v$$

<!-- step -->

The expected score is zero when $q = p$. Otherwise, the probability gap sets its direction.

---

## A worked example with three tokens

Take the logits themselves as parameters, so $\nabla_\theta f_v$ is a unit vector and $s_{a_t} = e_{a_t} - p$.

| | Token 1 | Token 2 | Token 3 |
|---|:---:|:---:|:---:|
| Learner $p$ | 0.50 | 0.30 | 0.20 |
| Sampler $q$ | 0.60 | 0.25 | 0.15 |
| Expected score $\bar{s} = q - p$ | **+0.10** | **−0.05** | **−0.05** |

<!-- step -->

Suppose every completion earns the same advantage, $A = 1$. There is nothing to learn, since no token is better than another.

<!-- step -->

The expected update is still $1 \cdot \bar{s} = (+0.10, -0.05, -0.05)$: the learner's logits move **toward the sampler**.

---

## Zooming in on one position

Start from the estimate, and move the sum outside the expectation:

$$\hat{g} = \mathbb{E}_{a \sim q}\left[ \sum_{t} A \, s_{a_t} \right] = \sum_{t} \mathbb{E}_{a \sim q}\left[ A \, s_{a_t} \right]$$

<!-- step -->

Sequences are sampled one token at a time, so split each term into the prefix, then the token and everything after it:

$$\hat{g} = \sum_{t} \mathbb{E}_{a_{<t} \sim q}\Big[ \mathbb{E}_{a_{\geq t} \sim q}\big[ A \, s_{a_t} \big] \Big]$$

<!-- step -->

From here on, fix a prefix and study the inner term. We write it as $\mathbb{E}_q[A \, s_{a_t}]$. With the prefix fixed, the token $a_t$ and the advantage $A$ are both still random.

---

## Drift and signal

Use the covariance identity, $\mathbb{E}[XY] = \mathbb{E}[X] \, \mathbb{E}[Y] + \text{Cov}(X, Y)$, starting on-policy:

$$
\begin{aligned}
\mathbb{E}_p[A \, s_{a_t}] &= \mathbb{E}_p[A] \, \mathbb{E}_p[s_{a_t}] + \text{Cov}_p(A, s_{a_t}) \\
&= \mathbb{E}_p[A] \cdot \mathbf{0} + \text{Cov}_p(A, s_{a_t}) \\
&= \text{Cov}_p(A, s_{a_t})
\end{aligned}
$$

<!-- step -->

Now the same identity under the sampler, where $\mathbb{E}_q[s_{a_t}] = \bar{s}$:

$$
\mathbb{E}_q[A \, s_{a_t}] = \mathbb{E}_q[A] \, \mathbb{E}_q[s_{a_t}] + \text{Cov}_q(A, s_{a_t}) = \underbrace{\mathbb{E}_q[A] \, \bar{s}}_{\text{drift}} + \underbrace{\text{Cov}_q(A, s_{a_t})}_{\text{signal}}
$$

<!-- step -->

Only the covariance sees which tokens led to which outcomes. The drift depends on the advantages only through their mean.

---

## What the drift does

<!-- cite-right: marek2026score -->

The cross-entropy loss with the sampler as the teacher is:

$$\text{CE}(q, p_\theta) = -\sum_{v \in \mathcal{V}} q_v \log p_v$$

<!-- step -->

Its gradient is the negative of the expected score:

$$\nabla_\theta \text{CE}(q, p_\theta) = -\sum_{v \in \mathcal{V}} q_v \, \nabla_\theta \log p_v = -\bar{s}$$

<!-- step -->

So the drift term, $\mathbb{E}_q[A] \, \bar{s}$, is a step of **supervised distillation toward the sampler**, scaled by the mean advantage at that prefix:

- $\mathbb{E}_q[A] > 0$: The learner is pulled toward the sampler.
- $\mathbb{E}_q[A] < 0$: The learner is pushed away from the sampler.

---

## Why the drift compounds

<!-- cite-right: marek2026score -->

Distilling toward a **fixed** teacher is harmless. It converges to the best fit of that teacher.

In online RL, the teacher moves with the student:

<!-- animate: bullets -->

- The sampler is a biased copy of the learner, e.g., quantized or stale.
- Each step pulls the learner toward the sampler's errors.
- The new weights are synced back to the sampler, carrying those errors with them.
- The next sampler is biased again, so the error compounds instead of converging.
- The bias does not average out with more samples, since $\bar{s}$ is fixed at each prefix. It needs a direct correction.

---

## Do group baselines already fix this?

<!-- cite-right: marek2026score -->

Not fully. Group centering makes advantages sum to zero **over the rollouts of a prompt**:

$$\frac{1}{G} \sum_{i=1}^{G} (r_i - \bar{r}) = 0$$

<!-- step -->

But drift arises **at each prefix**, and $\mathbb{E}_q[A \mid x, a_{<t}]$ is generally not zero there:

- A prefix likely to reach a correct answer has positive expected advantage.
- A prefix that already contains a mistake has negative expected advantage.

<!-- step -->

Group centering shrinks drift relative to all-positive rewards, and it was the most stable reward setting in the paper's online experiments. It does not eliminate drift.

---

## Score centering: A baseline for the score

Baselines center the reward. Score centering centers the **score**, subtracting its expected value under the sampler:

$$\tilde{s}_{a_t} = s_{a_t} - \bar{s}$$

<!-- step -->

$\bar{s}$ is a sum over the whole vocabulary at a fixed prefix, so it is a constant with respect to the sampled token:

$$\mathbb{E}_q[\tilde{s}_{a_t}] = \mathbb{E}_q[s_{a_t} - \bar{s}] = \mathbb{E}_q[s_{a_t}] - \bar{s} = \bar{s} - \bar{s} = \mathbf{0}$$

<!-- step -->

The centered score has zero mean under the sampler, which restores the property that made baselines free. When $q = p$, $\bar{s} = \mathbf{0}$ and the correction does nothing.

---

## The drift cancels

Apply the covariance identity to the centered score:

$$
\begin{aligned}
\mathbb{E}_q[A \, \tilde{s}_{a_t}] &= \mathbb{E}_q[A] \, \mathbb{E}_q[\tilde{s}_{a_t}] + \text{Cov}_q(A, \tilde{s}_{a_t}) \\
&= \mathbb{E}_q[A] \, \mathbb{E}_q[s_{a_t} - \bar{s}] + \text{Cov}_q(A, s_{a_t} - \bar{s}) \\
&= \mathbb{E}_q[A] \left( \mathbb{E}_q[s_{a_t}] - \bar{s} \right) + \text{Cov}_q(A, s_{a_t} - \bar{s}) \\
&= \mathbb{E}_q[A] \left( \bar{s} - \bar{s} \right) + \text{Cov}_q(A, s_{a_t} - \bar{s}) \\
&= \text{Cov}_q(A, s_{a_t})
\end{aligned}
$$

<!-- step -->

The last step uses that a constant does not change a covariance: $\text{Cov}(X, Y - c) = \text{Cov}(X, Y)$.

<!-- step -->

In the worked example, the centered score is $e_{a_t} - q$. With $A = 1$, its expected value is $q - q = \mathbf{0}$, so nothing moves.

---

## What score centering does not do

<!-- cite-right: marek2026score -->

| Method | Expected update at a prefix | Drift |
|--------|-----------------------------|:-----:|
| On-policy | $\text{Cov}_p(A, s_{a_t})$ | None |
| Vanilla, under TIM | $\mathbb{E}_q[A] \, \bar{s} + \text{Cov}_q(A, s_{a_t})$ | Yes |
| Exact importance sampling | $\text{Cov}_p(A, s_{a_t})$, at the cost of variance | None |
| Clipped or masked importance sampling | Depends on the bound | Reintroduced |
| Score centering | $\text{Cov}_q(A, s_{a_t})$ | None |

<!-- step -->

Drift-free is **not** the same as on-policy: the covariance is still measured under $q$, not $p$.

<!-- step -->

The two corrections are independent. The paper recommends composing score centering with importance sampling when the learner can move far from the sampler between syncs.

---

## From the update to a loss

<!-- cite-right: marek2026score -->

We want a scalar whose gradient is $A \, (s_{a_t} - \bar{s})$.

The first piece is already a gradient, $s_{a_t} = \nabla_\theta \log p_{a_t}$. For the second:

$$\bar{s} = \sum_{v \in \mathcal{V}} q_v \, \nabla_\theta \log p_v$$

<!-- step -->

The sampler's probabilities are logged values, so we mark them as constants with a stop-gradient and pull the gradient out:

$$\bar{s} = \nabla_\theta \sum_{v \in \mathcal{V}} \text{sg}[q_v] \log p_v$$

<!-- step -->

Combine the two pieces, and multiply by $-A$:

$$L = -A \left( \log p_{a_t} - \sum_{v \in \mathcal{V}} \text{sg}[q_v] \log p_v \right), \qquad -\nabla_\theta L = A \, (s_{a_t} - \bar{s}) = A \, \tilde{s}_{a_t}$$

---

## Score centering in code

The correction is one extra term in the REINFORCE loss from earlier:

```python
# Learner: log-probs over the whole vocabulary, with gradients
learner_vocab_logprobs = logits.log_softmax(-1)              # (B*G, L, V)
learner_logprobs = learner_vocab_logprobs.gather(
    -1, tokens.unsqueeze(-1)).squeeze(-1)                    # (B*G, L)

# Sampler: logged at generation time, so it has no gradient path
sampler_probs = sampler_vocab_logprobs.exp()                 # (B*G, L, V)

# The gradient of `centering` is the expected score, s-bar
centering = (sampler_probs * learner_vocab_logprobs).sum(-1) # (B*G, L)

# advantages: (B*G, 1), broadcast to every token
per_token_loss = -advantages * (learner_logprobs - centering)
loss = ((per_token_loss * mask).sum(1) / mask.sum(1)).mean()
```

No ratios, no masks, and no clipping. On-policy, the gradient of `centering` is zero.

---

## The full distribution is too expensive

<!-- cite-right: marek2026score -->

Computing $\bar{s}$ exactly needs the sampler's next-token distribution at every position:

- Qwen3's vocabulary has 152K tokens, so one position in fp32 takes about 608KB.
- A batch of 1024 rollouts with 32K tokens each would take about 20TB.

<!-- step -->

Instead, log only the sampler's **head** $H$, its $k$ most likely tokens at that position, with $k = 128$. Model the **tail** with the learner, rescaled to carry the sampler's missing mass:

$$
\hat{q}_v =
\begin{cases}
q_v & v \in H \\
\rho \, p_v & v \notin H
\end{cases}
\qquad
\rho = \frac{1 - \sum_{v \in H} q_v}{1 - \sum_{v \in H} p_v}
$$

---

## The tail cancels

<!-- cite-right: marek2026score -->

Substitute $\hat{q}$ into the expected score, and split the sum into head and tail:

$$\bar{s} \approx \sum_{v} \hat{q}_v \, s_v = \sum_{v \in H} q_v \, s_v + \rho \sum_{v \notin H} p_v \, s_v$$

<!-- step -->

The zero-mean identity splits the same way, which rewrites the tail in terms of the head:

$$\mathbf{0} = \sum_{v} p_v \, s_v = \sum_{v \in H} p_v \, s_v + \sum_{v \notin H} p_v \, s_v \quad \Rightarrow \quad \sum_{v \notin H} p_v \, s_v = -\sum_{v \in H} p_v \, s_v$$

<!-- step -->

Substitute, and collect the terms. Only the $k$ head tokens remain:

$$\bar{s} \approx \sum_{v \in H} q_v \, s_v - \rho \sum_{v \in H} p_v \, s_v = \sum_{v \in H} (q_v - \rho \, p_v) \, s_v$$

<!-- step -->

The paper reports that $k = 128$ and $k = 32$ both perform on par with the full vocabulary, at under 1% wall-clock overhead for $k = 128$.

---

## Top-k score centering in code

<!-- cite-right: marek2026score -->

```python
# Logged during generation: sampler_topk_logprobs, sampler_topk_ids (B*G, L, k)
learner_logprobs = learner_vocab_logprobs.gather(-1, tokens[..., None])[..., 0]
head_logprobs = learner_vocab_logprobs.gather(-1, sampler_topk_ids)  # (B*G, L, k)
learner_head_probs = head_logprobs.exp()
sampler_head_probs = sampler_topk_logprobs.exp()

# rho: the ratio of the sampler's tail mass to the learner's tail mass
sampler_tail = (1 - sampler_head_probs.sum(-1)).clamp_min(1e-6)
learner_tail = (1 - learner_head_probs.sum(-1)).clamp_min(1e-6)
rho = sampler_tail / learner_tail                                    # (B*G, L)

# detach() is required: coef contains the learner's own probabilities
coef = (sampler_head_probs - rho[..., None] * learner_head_probs).detach()
centering = (coef * head_logprobs).sum(-1)                           # (B*G, L)

per_token_loss = -advantages * (learner_logprobs - centering)
```

---

<!-- layout: section-break -->

## What are labs running?

---

## StableReinforce and GRPO

Many labs are supposedly using **StableReinforce**, the loss Tinker added, rather than GRPO [@tinker2026stablereinforce]. This is unconfirmed, but the two are related.

| | GRPO | StableReinforce |
|---|------|-----------------|
| Family | Group baseline, often with PPO-style clipping | "A REINFORCE variant" |
| Question it answers | How should we center the **reward**? | How do we stay stable when **$q \neq p$**? |
| Advantages | Group mean and std | Supplied by the caller, so group centering still fits |
| Extra sampling data | None required | `record_stability_info=True` |
| KL term | Optional, to a reference model | Optional $\text{KL}(q \,\Vert\, p_\theta)$, not to a reference model |

---

## Is StableReinforce score centering?

<!-- cite-right: tinker2026stablereinforce -->

Tinker's docs describe the behavior, but not the main loss:

> "In our internal post-training, this loss function has led to extremely stable training, even under highly off-policy async RL and low precision numerics."

<!-- step -->

They do reveal a clue: sampling must record extra information, through `record_stability_info=True`.

<!-- step -->

Frontier losses are rarely published, so public ideas are rarely confirmed. Here, Martin Marek, the score centering paper's first author, tried to recover the loss from its gradients alone.

---

## Reverse-engineering a loss from its gradients

<!-- cite-right: marek2026stablereinforcethread -->

<!-- columns: 55/45 -->

Tinker does not expose the loss or the "stability info," but it does let you take a step and download the weights.

**Step 1: Recover the raw gradient.** Tinker has no SGD, so use Adam with $\epsilon = 10{,}000$. This dominates Adam's second moment:

$$\Delta w \approx -\frac{\text{lr} \cdot g}{\epsilon}$$

<!-- step -->

**Step 2: Read it in logit space.** Train a rank-1 LoRA adapter on the unembedding of Qwen3.5-4B. The gradient with respect to $B$ equals the gradient with respect to the logits, up to scale.

|||

![A rank-1 LoRA adapter added to the unembedding matrix, so the gradient on B is proportional to the logit gradient. From Marek's thread.](assets/marek-thread-lora.png)

---

## Only 8 logits move

<!-- cite-right: marek2026stablereinforcethread -->

<!-- columns: 32/68 -->

For a single-token update:

- **Reinforce** puts a nonzero gradient on all logits: 248,077 of them move.
- **StableReinforce** moves only 8.

<!-- step -->

The bars come in multiples of $1/8$, and the third row, Reinforce with score centering, is identical.

|||

![Logit gradients for a single-token update under Reinforce, StableReinforce, and Reinforce with score centering. From Marek's thread.](assets/marek-thread-logit-gradients.jpeg)

---

## Why a sparse gradient is the clue

<!-- cite-right: marek2026stablereinforcethread -->

In logit coordinates, every token's score has the same dense part, from the softmax:

$$s_v = e_v - p$$

<!-- step -->

Estimate $\bar{s}$ by drawing 8 tokens from the sampler, $v_1, \dots, v_8 \sim q$:

$$\bar{s} \approx \frac{1}{8} \sum_{i=1}^{8} s_{v_i} = \frac{1}{8} \sum_{i=1}^{8} (e_{v_i} - p) = \frac{1}{8} \sum_{i=1}^{8} e_{v_i} - p$$

<!-- step -->

Subtract it from the score of the sampled token. The dense $-p$ cancels exactly:

$$s_{a_t} - \bar{s} \approx (e_{a_t} - p) - \left( \frac{1}{8} \sum_{i=1}^{8} e_{v_i} - p \right) = e_{a_t} - \frac{1}{8} \sum_{i=1}^{8} e_{v_i}$$

<!-- step -->

This predicts the figure, with signs flipped as for a loss gradient: $-1$ on the sampled token, and $+1/8$ per draw. A token drawn twice gets $2/8$, so 7 distinct draws plus the sampled token gives 8 logits.

---

## The match

<!-- cite-right: marek2026stablereinforcethread -->

**Hypothesis:** StableReinforce is REINFORCE plus score centering, with the expected score computed over 8 randomly sampled tokens.

<!-- step -->

**Test:** Implement that as a custom Tinker loss, take an optimizer step, and compare. The gradients are within numeric precision across all layers.

<!-- img-align: center -->

![The StableReinforce gradient minus the Reinforce with score centering gradient is zero across layers. From Marek's thread.](assets/marek-thread-gradient-match.png)

---

## What this does and does not tell us

<!-- cite-right: marek2026stablereinforcethread -->

| | Paper | Hypothesized StableReinforce |
|---|------|------|
| Estimate of $\bar{s}$ | Top-$k$ head, plus a tail model | 8 tokens drawn from the sampler |
| Properties | Deterministic, slightly biased | Unbiased, higher variance |

<!-- step -->

**A caveat:** This is a reverse-engineering by the paper's own first author, not a statement from Thinking Machines. A gradient match is strong evidence, not an admission.

<!-- step -->

**Why it matters:** Frontier RL losses are usually unpublished, so we rarely get to check a public idea against a deployed system. Here, a production API turned out to match a paper from weeks earlier.

---

## Score centering in a released recipe

<!-- cite-right: ricursive2026ripple -->

Ripple-65B Preview, a 65B open-weight model from Ricursive, names score centering in its post-training:

> "Our mixed RL recipe uses truncated importance sampling, score centering, and linear length penalty with high-quality long-horizon synthetic environments."

<!-- step -->

- This is the combination the paper recommends: score centering **in addition to** an importance-sampling correction.
- Ricursive reports a TerminalBench 2.1 score of 80.5 with 1.4M rollouts.
- The release gives no loss, hyperparameters, or ablation, so we cannot say how much score centering contributed.

---

## Takeaways

<!-- animate: bullets -->

- An LLM is a policy over tokens, and REINFORCE on it is a few lines of code: log-probs, advantages, and a surrogate loss.
- At scale, the sampler and the learner are different systems, so $q \neq p$ even with identical weights.
- The zero-mean score behind baselines fails off-policy, which leaves a drift term, $\mathbb{E}_q[A] \, \bar{s}$, next to the signal.
- Drift is distillation toward the sampler, and it compounds because the sampler keeps being refreshed from the learner.
- Score centering subtracts $\bar{s}$ from the score, which cancels the drift exactly. It is one extra term in the loss, and it does not make the update fully on-policy.
