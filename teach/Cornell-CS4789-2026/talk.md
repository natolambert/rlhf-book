---
title: "From Classical RL to Frontier LLMs"
author: "Nathan Lambert"
date: "November 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
show_animations: false  # set true for final touches
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
  /* colloquium's .slide-content has flex: 1, so valign alone can't move it */
  .slide--content.valign-center > .slide-content { flex: 0 1 auto; }
  .compute-table td { vertical-align: top; padding-top: 0.75em; padding-bottom: 0.75em; }
  .compute-note { font-size: 0.7em; color: #6b7280; }
  .col-text-sm { font-size: 0.85em; }
  .diagram-md img { max-height: 430px; width: auto; }
  .notation-table { font-size: 0.95em; }
  .notation-table td, .notation-table th { padding-top: 0.35em; padding-bottom: 0.35em; }
  .notation-table td:first-child, .notation-table th { white-space: nowrap; }
  .mdp-table { font-size: 0.85em; }
  .mdp-table td, .mdp-table th { padding-top: 0.3em; padding-bottom: 0.3em; }
  .compute-table td strong { font-size: 1.15em; }
  .compute-table td small { display: block; margin-top: 0.2em; font-size: 0.68em; line-height: 1.35; color: #6b7280; }
  .compute-table td:last-child strong { font-size: 1.4em; color: #E8676A; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# From Classical RL to Frontier LLMs

<div class="colloquium-title-eyebrow">18 November 2026</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>
---

## Post-training (RL) now takes most of a frontier lab's compute

<!-- cite-right: semianalysis2026computeshare -->

<!-- img-align: center -->

![Share of combined Anthropic and OpenAI compute by workload, 1Q24 to 4Q26 (later quarters are estimates). Pretraining falls from 67% to 7%, while post-training and RL rise from 5% to 55%.](assets/semianalysis-compute-share.png)

---

## How the times have changed!

RL compute relative to pretraining compute for the same model, in GPU-hours:

<div class="compute-table">

| Model | Pretraining | RL | RL / pretraining |
|---|---|---|---|
| **DeepSeek R1** <small>Jan 2025<br>671B total, 37B active</small> | **2.66M** <small>H800 GPU-hours [@deepseekai2025deepseekv3technicalreport]</small> | **0.14M** <small>H800 GPU-hours [@guo2025deepseeknature]</small> | **~5%** |
| **Beam** <small>Oct 2026<br>501B total, 23B active</small> | **< 4.1M** <small>6,144 GB300s for under 4 weeks</small> | **~7.1M** <small>10.5K GB300s for 4 weeks [@reflection2026beam]</small> | **> 1.7×** |
| **Inkling 2 Small** <small>Nov 2026<br>594B total, 17B active</small> | **~0.11M\*** <small>~1.5K GPUs for under 3 days</small> | **~0.44M** <small>2,048 GPUs for 9 days [@tml2026inkling2small]</small> | **~4×\*** |

</div>

<div class="compute-note">*Estimated: Thinking Machines report the pretraining duration but not the GPU count. GPU-hours ignore utilization. RL spends much of its time sampling, so its share of FLOPs is smaller than its share of GPU-hours.</div>

---

<!-- valign: center -->
## What this lecture covers

How RL is approached with LLMs and what the challenges of RL are for frontier models.

RL is mostly an engineering, stability, and throughput problem at the frontier. The learning signal is important, and mostly comes through data engineering (not the focus of this lecture).
---

## What this lecture covers

How RL is approached with LLMs and what the challenges of RL are for frontier models.

Some examples we include: 

1. **Policy gradients in LLM code.** How LLM log-probs, logits, and a surrogate loss turn REINFORCE into a few lines of PyTorch.
2. **Asynchronous RL systems.** How RL training is usually done for large LLMs.
3. **Training-inference mismatch.** A common challenge that many of the latest RL algorithms address.

---

## Notation in this lecture

<div class="notation-table">

| Symbol | Meaning | Book / lectures |
|--------|---------|-----------------|
| $x$ | Prompt | $s$ |
| $a = (a_1, \dots, a_T)$ | Completion, one token at a time | $a$ |
| $v \in \mathcal{V}$ | Any token in the vocabulary | $v$ |
| $p_\theta$ | **Learner**: the training framework's (e.g., FSDP) next-token distribution | $\pi_\theta^{\text{learner}}$ |
| $q$ | **Sampler**: the inference engine's (e.g., vLLM) next-token distribution (no gradient updates) | $\pi_\theta^{\text{sampler}}$ |
| $f_\theta$ | Logits, so $p_\theta = \text{softmax}(f_\theta)$ | |
| $\boldsymbol{\xi}_v = \nabla_\theta \log p_v$ | Gradient "**score**" of token $v$ | $\boldsymbol{\xi}_v$ |
| $A$ | Advantage for the whole completion, e.g., $r - b(x)$ | $A$ |

</div>

$p_v$ and $q_v$ are shorthand for the probability of token $v$ at one position, given $x$ and the prefix $a_{<t}$. CS 4789 uses the same $x$ and $a$.

---

<!-- layout: section-break -->

## Implementing RL basics with an LLM

---


<!-- columns: 50/50 -->
<!-- valign: center -->
## RL review: The agent–environment loop

**Classical RL**: agent in an MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$

- Reward is a known function from the environment per step
- Optimize cumulative return: $J(\pi) = \mathbb{E}_{\tau \sim p_\pi}\!\left[\sum_{t} \gamma^t r(s_t, a_t)\right]$

|||

![](assets/rl.png)

---

## Language models as episodic MDPs

| MDP concept | Language model |
|-------------|---------------|
| State $s_t$ | Prompt + tokens generated so far: $(x, a_{<t})$ |
| Action $a_t$ | Next token $a_t$ |
| Transition | Deterministic-ish: append token to sequence (tokens are sampled probabilistically) |
| Policy $\pi_\theta(a_t \mid s_t)$ | LM next-token distribution |
| Episode | One prompt → completion |
| Terminal reward | RM score, verifier output, judge-scores |
| $\gamma$ | Typically 1.0 (no discounting) |

<!-- step -->

Two implementation-level views of this MDP: **token-level** (each token is a separate action, used in PPO with GAE) vs. **sequence-level** (entire completion is one action, used in REINFORCE/GRPO).

---

## Agentic RL complicates the MDP

<div class="mdp-table">

| MDP concept | RLVR | Agentic RL |
|-------------|------|------------|
| State $s_t$ | Prompt + tokens so far | Context (prompt + tokens + tool outputs), plus hidden environment state (files, services, other users) |
| Action $a_t$ | Next token | Next token (many are tool calls) |
| Transition | Deterministic-ish: append token | More stochastic: the environment runs the call and appends its output |
| Episode | One prompt → completion | Many turns; Inkling 2 Small's p99 rollout takes 4 hours [@tml2026inkling2small] |
| Terminal reward | Verifier on the final answer | Tests on the final state of the environment |
| Loss | Every completion token | Tool-output tokens are masked: the model must not learn to predict the environment |

</div>

Math RL needed prompts and answer checkers. Agentic RL needs **environments**: containers, real file systems, services, verification tests [@ivison2026tmax]. **Building worlds for LLMs is hard.**

---


<!-- class: diagram-md -->
## RL for LLMs: RLVR

<!-- img-align: center -->

![](assets/rlvr-system.png)

---

<!-- class: diagram-md -->
## RL for LLMs: Agentic RL

<!-- img-align: center -->

![](assets/tool_use_rl_loop.png)

---

## Recall: The policy gradient

The objective and its gradient:

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

The gradient says: for each sampled token, compute the direction in parameter space that makes it more likely ($\nabla \log \pi$), then scale by how much better than expected it was ($\Psi_t$).

---

## Recall: Why losses are sums of log-probs

The policy gradient derivation gives:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

So we write a loss whose negative gradient is the policy gradient, treating each advantage $A_i$ as a constant across the trajectory:

$$
\begin{aligned}
L(\theta) &= -\frac{1}{N} \sum_{i=1}^{N} A_i \sum_{t} \log \pi_\theta(a_{i,t} \mid s_{i,t}) \\
-\nabla_\theta L(\theta) &= \frac{1}{N} \sum_{i=1}^{N} A_i \sum_{t} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})
\end{aligned}
$$

Gradient descent on $L$ is gradient ascent on $J$.

---

## From the loss to autodiff

In code, we compute the **log-probs** and let autodiff handle the gradient:

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages.detach()).mean()
loss.backward()  # fills each parameter's .grad with ∇L
optimizer.step()  # θ ← θ - η ∇L
```

`loss.backward()` runs backpropagation: it computes $\partial L / \partial \theta$ for every parameter. 

`optimizer.step()` applies the update. 

The loss values themselves are often weird in RL (e.g. zero-centered distributions, rather than nice decreasing accuracy metrics of pretraining)

Most RL algorithms used today are variations on this computation.

---

## Computing log-probabilities (expanded)

The fundamental building block: per-token log-probabilities from the policy. 

```python
# Forward pass through the model
logits = model(input_ids).logits              # (B, L, vocab_size)

# Autoregressive shift: logit at position t predicts token at t+1
logits = logits[:, :-1, :]                    # (B, L-1, vocab_size)
labels = input_ids[:, 1:]                     # (B, L-1)
completion_mask = completion_mask[:, 1:]       # (B, L-1)

# Per-token log-probs
log_probs = logits.log_softmax(dim=-1)
token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (B, L-1)

# Per-sequence log-prob: sum over completion tokens
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
```

Everything in log-space to avoid numerical underflow from multiplying many small probabilities.

---

## The REINFORCE loss in code

The simplest policy gradient loss:

```python
# rewards: (B,) — one reward per sequence
# seq_log_probs: (B,) — sum of log-probs over completion tokens

# Baseline: average reward in the batch
baseline = rewards.mean()
advantages = rewards - baseline

# REINFORCE loss (negative because we minimize)
loss = -(advantages * seq_log_probs).mean()
```

TPositive advantage → increase probability. Negative → decrease. 

(Aside: How we reduce per-token losses to a scalar — `.mean()` here — turns out to matter more than you'd expect. More in the [book](https://rlhfbook.com/c/06-policy-gradients#loss-aggregation-tradeoffs) and [lecture video](https://www.youtube.com/watch?v=i-AIMpZHgeg&t=2704s).)

---

<!-- layout: section-break -->

## Challenges of distributed RL training

---

## On-policy vs. off-policy

Policy-gradient theory is built on the assumption of on-policy data -- from the current model weights in the training framework of the LLM. 

As models get bigger, this is hard because: sampling is expensive, sampling is dominated by long-trail generations, most people use separate systems for inference v training, moving large model weights around is slow, inference weights can be further quantized for speed relative to training weights, multi-cluster deployments, etc.

---

## Synchronous: Share generation and training

<!-- cite-right: noukhovitch2024asynchronous -->

The default implementation for policy-gradient algorithms is **on-policy** execution: the generations from the language model are scored before updating the model.

<!-- img-align: center -->

![](assets/async_v_synch_rl_1.png)

---

## Intermediate step: Separate GPUs, still on-policy

<!-- cite-right: noukhovitch2024asynchronous -->

Moving generation to an inference engine on its own GPUs makes it fast. To stay on-policy, each side waits for the other.

<!-- img-align: center -->

![](assets/async_v_synch_rl_2.png)

---

## Asynchronous: Off-policy training

<!-- cite-right: noukhovitch2024asynchronous -->

The common solution is to constantly run inference and training on separate GPU nodes. The model used for generation may be 1–N steps behind the training model.

<!-- img-align: center -->

![](assets/async_v_synch_rl_3.png)

This turns out to be mostly "fine" -- but it can be taken too far.

---

<!-- columns: 42/58 -->
## Synchronous vs. asynchronous training

<!-- cite-right: noukhovitch2024asynchronous -->

<div class="col-text-sm">

**Ideal / Theory (on-policy)**: generate → update → generate → update

Each batch of completions is scored and used for a short update window (one or a few epochs), then discarded.

**Reality (async)**: generation and training overlap on different GPU groups for better throughput. The model used for generation may be 1–N steps behind the training model.

**Tradeoff**: perfect on-policy is slow (GPUs idle during generation or training). Slight staleness (how off-policy the data is) is usually fine.

</div>

|||

![Synchronous training idles GPUs; separating generation and training helps, but on-policy requires waiting. Async (off-policy) overlaps both for full utilization.](assets/async_v_synch_rl.png)

---

## Asynchronous RL training

<!-- columns: 50/50 -->

Modern RL for LLMs splits compute into two groups:

- **Actors** (inference GPUs): generate completions using vLLM or similar
- **Learners** (training GPUs): compute policy gradient updates

A process management library (e.g., Ray) coordinates data flow between them. Model weights are synced periodically from learner → actor.

|||

![Distributed RL system with prompt and result queues between learner and actor GPUs.](assets/distributed-rl.png)

---


<!-- columns: 45/55 -->
<!-- valign: center -->
## The long-tail generations cause idle GPUs

<!-- cite-right: teamolmo2025olmo3 -->

With **static batching**, each actor generates a whole batch and returns it only when the longest sequence finishes.

- Wasted compute is (max length − mean length) / max length.
- Olmo 3 at a 32K limit: mean generation length 14,628, so up to **54%** of compute would be wasted.
- Agentic RL is worse: one trajectory makes 2 tool calls, another makes 200. Tool calls add additional lag and complexity.

|||

![Static batching. Pink: prefill, green: decoding, dark green: EOS, grey: idle until the batch finishes.](assets/olmo3-static-batching.png)

---


<!-- columns: 45/55 -->
<!-- valign: center -->
## Solution: Update inference model weights on the fly

<!-- cite-right: teamolmo2025olmo3 -->

<div class="col-text-sm">

| Olmo 3 RL infrastructure | Tokens/s |
|---|---:|
| Olmo 2 baseline | 881 |
| + continuous batching | 975 |
| + better threading | 1,358 |
| + inflight updates | **2,949** |

**Continuous batching:** start a new prompt as soon as a sequence finishes.

**Inflight updates:** load new weights without pausing generation or dropping the KV cache. The biggest speedup, but tokens in one completion now come from different weights.

</div>

|||

![Continuous batching backfills finished rows immediately (S5–S7), so no slot sits idle.](assets/olmo3-continuous-batching.png)

---


<!-- columns: 62/38 -->
<!-- valign: center -->
## Inflight updates: PipelineRL

The canonical reference for inflight weight updates is **PipelineRL** [@piche2025pipelinerl].

[arxiv.org/abs/2509.19128](https://arxiv.org/abs/2509.19128)

|||

![](assets/pipelinerl-paper-2025.png)

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

- **Staleness:** $\tilde{\theta} \neq \theta$, because the weights have moved since the last syn (on the trainer side).
- **Numerics:** $\tilde{f} \neq f$, because of different kernels, precision, parallelism, or a quantized KV cache (the cached attention keys and values for the prefix, stored in fewer bits) [@yao2025offpolicy; @marek2026score].


---

## Same tokens, different probabilities

<!-- cite-right: minimax2025minimax_m1 -->

<!-- img-align: center -->

![Training vs. inference probability for the same tokens in MiniMax-M1, before and after a fix. Agreement improves, but the two systems still differ.](assets/minimax-m1-logprob-mismatch.png)

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

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim q}\left[ w(a) \, A \sum_{t=1}^{T} \boldsymbol{\xi}_{a_t} \right], \qquad w(a) = \frac{p_\theta(a \mid x)}{q(a \mid x)}$$

<!-- step -->

But $w$ can take very large values on rare tokens, which inflates the variance of the gradient.

Truncated importance sampling [@ionides2008truncated] caps the per-token ratio at a constant $C$, trading a small bias for bounded variance in the policy gradient:

$$w_t = \frac{p(a_t \mid x, a_{<t})}{q(a_t \mid x, a_{<t})}, \qquad \tilde{w}_t = \min(w_t, C)$$

---

## Truncated importance sampling in code

In practice, LLM RL systems apply TIS as a per-token correction weight on the policy-gradient loss:

```python
# Shape: (B*G, L)
C = 2.0  # TIS cap

logratio = learner_logprobs - sampler_logprobs
logratio = logratio.clamp(-10.0, 10.0)              # numerical safety
tis_weight = torch.exp(logratio).clamp(max=C)        # one-sided truncation

# Use as a fixed correction weight on the per-token PG loss
per_token_pg_loss = per_token_pg_loss * tis_weight.detach()
```

TIS for the learner–sampler ratio has been adopted across major open-source RL frameworks (VeRL, TRL, OpenRLHF, SkyRL, OAT, and Open Instruct, which uses $C = 2$).

---

<!-- layout: section-break -->

## Score centering: managing train-inference mismatch

---

## Mismatch can collapse training

<!-- cite-right: marek2026score -->

<!-- img-align: center -->

![Qwen3-30B-A3B-Base trained with RL on math, with the sampler quantized more and more heavily from left to right. W/A is weights and activations; KV is the KV cache. With an FP8 sampler, plain policy gradient (PG) trains stably. With an FP4 KV cache, PG collapses within 200 steps. With INT8 weights and an INT4 KV cache, score centering (SC) reaches 30% accuracy, TIS 12%, and every other method ends below 5%.](assets/score-centering-30b-quantization.png)

---


## What is the score?

In statistics, the **score** is the gradient of the log-likelihood with respect to the parameters. For one token $v$ at one position:

$$\boldsymbol{\xi}_v = \nabla_\theta \log p_\theta(v \mid x, a_{<t})$$

- It is a vector the size of the parameters: the direction that makes token $v$ more likely.
- The policy gradient is an advantage-weighted sum of the scores of the sampled tokens:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ A \sum_{t=1}^{T} \boldsymbol{\xi}_{a_t} \right]$$

---

## Recall: The score has zero mean on-policy

<!-- cite-right: williams1992simple -->

At one position, the expected score under the learner, written as a sum:

$$\mathbb{E}_{v \sim p}[\boldsymbol{\xi}_v] = \sum_{v \in \mathcal{V}} p_v \, \nabla_\theta \log p_v$$

<!-- step -->

Apply the log-derivative identity, $p_v \nabla_\theta \log p_v = \nabla_\theta p_v$:

$$= \sum_{v \in \mathcal{V}} \nabla_\theta p_v$$

<!-- step -->

Pull the gradient out of the sum. Probabilities sum to one:

$$= \nabla_\theta \sum_{v \in \mathcal{V}} p_v = \nabla_\theta 1 = \mathbf{0}$$

<!-- step -->

This is why baselines are free: $\mathbb{E}[b(x) \, \boldsymbol{\xi}_{a_t}] = b(x) \cdot \mathbf{0}$. It relies on the expectation and the score using **the same distribution**.

---

## Off-policy, the identity breaks

Tokens now come from $q$, but the score is still computed through $p$. Define the expected score under the sampler:

$$\bar{\boldsymbol{\xi}} = \mathbb{E}_{v \sim q}[\boldsymbol{\xi}_v] = \sum_{v \in \mathcal{V}} q_v \, \boldsymbol{\xi}_v$$

<!-- step -->

Substitute the score from earlier, $\boldsymbol{\xi}_v = \nabla_\theta f_v - \sum_u p_u \nabla_\theta f_u$:

$$\bar{\boldsymbol{\xi}} = \sum_{v} q_v \nabla_\theta f_v - \left( \sum_{v} q_v \right) \sum_{u} p_u \nabla_\theta f_u$$

<!-- step -->

Since $\sum_v q_v = 1$, and renaming $u$ to $v$:

$$\bar{\boldsymbol{\xi}} = \sum_{v} q_v \nabla_\theta f_v - \sum_{v} p_v \nabla_\theta f_v = \sum_{v \in \mathcal{V}} (q_v - p_v) \, \nabla_\theta f_v$$

<!-- step -->

The expected score is zero when $q = p$. Otherwise, the probability gap sets its direction.

---

## A worked example with three tokens

Take the logits themselves as parameters, so $\nabla_\theta f_v$ is a unit vector and $\boldsymbol{\xi}_{a_t} = e_{a_t} - p$.

| | Token 1 | Token 2 | Token 3 |
|---|:---:|:---:|:---:|
| Learner $p$ | 0.50 | 0.30 | 0.20 |
| Sampler $q$ | 0.60 | 0.25 | 0.15 |
| Expected score $\bar{\boldsymbol{\xi}} = q - p$ | **+0.10** | **−0.05** | **−0.05** |

<!-- step -->

Suppose every completion earns the same advantage, $A = 1$. There is nothing to learn, since no token is better than another.

<!-- step -->

The expected update is still $1 \cdot \bar{\boldsymbol{\xi}} = (+0.10, -0.05, -0.05)$: the learner's logits move **toward the sampler**.

---

## Zooming in on one position

Start from the estimate, and move the sum outside the expectation:

$$\hat{g} = \mathbb{E}_{a \sim q}\left[ \sum_{t} A \, \boldsymbol{\xi}_{a_t} \right] = \sum_{t} \mathbb{E}_{a \sim q}\left[ A \, \boldsymbol{\xi}_{a_t} \right]$$

<!-- step -->

Sequences are sampled one token at a time, so split each term into the prefix, then the token and everything after it:

$$\hat{g} = \sum_{t} \mathbb{E}_{a_{<t} \sim q}\Big[ \mathbb{E}_{a_{\geq t} \sim q}\big[ A \, \boldsymbol{\xi}_{a_t} \big] \Big]$$

<!-- step -->

From here on, fix a prefix and study the inner term. We write it as $\mathbb{E}_q[A \, \boldsymbol{\xi}_{a_t}]$. With the prefix fixed, the token $a_t$ and the advantage $A$ are both still random.

---

## Drift and signal

Use the covariance identity, $\mathbb{E}[XY] = \mathbb{E}[X] \, \mathbb{E}[Y] + \text{Cov}(X, Y)$, starting on-policy:

$$
\begin{aligned}
\mathbb{E}_p[A \, \boldsymbol{\xi}_{a_t}] &= \mathbb{E}_p[A] \, \mathbb{E}_p[\boldsymbol{\xi}_{a_t}] + \text{Cov}_p(A, \boldsymbol{\xi}_{a_t}) \\
&= \mathbb{E}_p[A] \cdot \mathbf{0} + \text{Cov}_p(A, \boldsymbol{\xi}_{a_t}) \\
&= \text{Cov}_p(A, \boldsymbol{\xi}_{a_t})
\end{aligned}
$$

<!-- step -->

Now the same identity under the sampler, where $\mathbb{E}_q[\boldsymbol{\xi}_{a_t}] = \bar{\boldsymbol{\xi}}$:

$$
\mathbb{E}_q[A \, \boldsymbol{\xi}_{a_t}] = \mathbb{E}_q[A] \, \mathbb{E}_q[\boldsymbol{\xi}_{a_t}] + \text{Cov}_q(A, \boldsymbol{\xi}_{a_t}) = \underbrace{\mathbb{E}_q[A] \, \bar{\boldsymbol{\xi}}}_{\text{drift}} + \underbrace{\text{Cov}_q(A, \boldsymbol{\xi}_{a_t})}_{\text{signal}}
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

$$\nabla_\theta \text{CE}(q, p_\theta) = -\sum_{v \in \mathcal{V}} q_v \, \nabla_\theta \log p_v = -\bar{\boldsymbol{\xi}}$$

<!-- step -->

So the drift term, $\mathbb{E}_q[A] \, \bar{\boldsymbol{\xi}}$, is a step of **supervised distillation toward the sampler**, scaled by the mean advantage at that prefix:

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
- The bias does not average out with more samples, since $\bar{\boldsymbol{\xi}}$ is fixed at each prefix. It needs a direct correction.

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

$$\tilde{\boldsymbol{\xi}}_{a_t} = \boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}}$$

<!-- step -->

$\bar{\boldsymbol{\xi}}$ is a sum over the whole vocabulary at a fixed prefix, so it is a constant with respect to the sampled token:

$$\mathbb{E}_q[\tilde{\boldsymbol{\xi}}_{a_t}] = \mathbb{E}_q[\boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}}] = \mathbb{E}_q[\boldsymbol{\xi}_{a_t}] - \bar{\boldsymbol{\xi}} = \bar{\boldsymbol{\xi}} - \bar{\boldsymbol{\xi}} = \mathbf{0}$$

<!-- step -->

The centered score has zero mean under the sampler, which restores the property that made baselines free. When $q = p$, $\bar{\boldsymbol{\xi}} = \mathbf{0}$ and the correction does nothing.

---

## The drift cancels

Apply the covariance identity to the centered score:

$$
\begin{aligned}
\mathbb{E}_q[A \, \tilde{\boldsymbol{\xi}}_{a_t}] &= \mathbb{E}_q[A] \, \mathbb{E}_q[\tilde{\boldsymbol{\xi}}_{a_t}] + \text{Cov}_q(A, \tilde{\boldsymbol{\xi}}_{a_t}) \\
&= \mathbb{E}_q[A] \, \mathbb{E}_q[\boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}}] + \text{Cov}_q(A, \boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}}) \\
&= \mathbb{E}_q[A] \left( \mathbb{E}_q[\boldsymbol{\xi}_{a_t}] - \bar{\boldsymbol{\xi}} \right) + \text{Cov}_q(A, \boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}}) \\
&= \mathbb{E}_q[A] \left( \bar{\boldsymbol{\xi}} - \bar{\boldsymbol{\xi}} \right) + \text{Cov}_q(A, \boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}}) \\
&= \text{Cov}_q(A, \boldsymbol{\xi}_{a_t})
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
| On-policy | $\text{Cov}_p(A, \boldsymbol{\xi}_{a_t})$ | None |
| Vanilla, under TIM | $\mathbb{E}_q[A] \, \bar{\boldsymbol{\xi}} + \text{Cov}_q(A, \boldsymbol{\xi}_{a_t})$ | Yes |
| Exact importance sampling | $\text{Cov}_p(A, \boldsymbol{\xi}_{a_t})$, at the cost of variance | None |
| Clipped or masked importance sampling | Depends on the bound | Reintroduced |
| Score centering | $\text{Cov}_q(A, \boldsymbol{\xi}_{a_t})$ | None |

<!-- step -->

Drift-free is **not** the same as on-policy: the covariance is still measured under $q$, not $p$.

<!-- step -->

The two corrections are independent. The paper recommends composing score centering with importance sampling when the learner can move far from the sampler between syncs.

---

## Quantization vs. staleness

<!-- cite-right: marek2026score -->

<!-- img-align: center -->

![Qwen3-0.6B on Countdown with an INT8 sampler (left) and a sampler updated only every 64 steps (right). Under quantization, score centering alone and composed with TIS or MIS perform best. Under staleness, score centering alone is unstable, and composing it with TIS or MIS works best.](assets/score-centering-quant-stale.png)

---

## From the update to a loss

<!-- cite-right: marek2026score -->

We want a scalar whose gradient is $A \, (\boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}})$.

The first piece is already a gradient, $\boldsymbol{\xi}_{a_t} = \nabla_\theta \log p_{a_t}$. For the second:

$$\bar{\boldsymbol{\xi}} = \sum_{v \in \mathcal{V}} q_v \, \nabla_\theta \log p_v$$

<!-- step -->

The sampler's probabilities are logged values, so we mark them as constants with a stop-gradient and pull the gradient out:

$$\bar{\boldsymbol{\xi}} = \nabla_\theta \sum_{v \in \mathcal{V}} \text{sg}[q_v] \log p_v$$

<!-- step -->

Combine the two pieces, and multiply by $-A$:

$$L = -A \left( \log p_{a_t} - \sum_{v \in \mathcal{V}} \text{sg}[q_v] \log p_v \right), \qquad -\nabla_\theta L = A \, (\boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}}) = A \, \tilde{\boldsymbol{\xi}}_{a_t}$$

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

Computing $\bar{\boldsymbol{\xi}}$ exactly needs the sampler's next-token distribution at every position:

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

$$\bar{\boldsymbol{\xi}} \approx \sum_{v} \hat{q}_v \, \boldsymbol{\xi}_v = \sum_{v \in H} q_v \, \boldsymbol{\xi}_v + \rho \sum_{v \notin H} p_v \, \boldsymbol{\xi}_v$$

<!-- step -->

The zero-mean identity splits the same way, which rewrites the tail in terms of the head:

$$\mathbf{0} = \sum_{v} p_v \, \boldsymbol{\xi}_v = \sum_{v \in H} p_v \, \boldsymbol{\xi}_v + \sum_{v \notin H} p_v \, \boldsymbol{\xi}_v \quad \Rightarrow \quad \sum_{v \notin H} p_v \, \boldsymbol{\xi}_v = -\sum_{v \in H} p_v \, \boldsymbol{\xi}_v$$

<!-- step -->

Substitute, and collect the terms. Only the $k$ head tokens remain:

$$\bar{\boldsymbol{\xi}} \approx \sum_{v \in H} q_v \, \boldsymbol{\xi}_v - \rho \sum_{v \in H} p_v \, \boldsymbol{\xi}_v = \sum_{v \in H} (q_v - \rho \, p_v) \, \boldsymbol{\xi}_v$$

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

$$\boldsymbol{\xi}_v = e_v - p$$

<!-- step -->

Estimate $\bar{\boldsymbol{\xi}}$ by drawing 8 tokens from the sampler, $v_1, \dots, v_8 \sim q$:

$$\bar{\boldsymbol{\xi}} \approx \frac{1}{8} \sum_{i=1}^{8} \boldsymbol{\xi}_{v_i} = \frac{1}{8} \sum_{i=1}^{8} (e_{v_i} - p) = \frac{1}{8} \sum_{i=1}^{8} e_{v_i} - p$$

<!-- step -->

Subtract it from the score of the sampled token. The dense $-p$ cancels exactly:

$$\boldsymbol{\xi}_{a_t} - \bar{\boldsymbol{\xi}} \approx (e_{a_t} - p) - \left( \frac{1}{8} \sum_{i=1}^{8} e_{v_i} - p \right) = e_{a_t} - \frac{1}{8} \sum_{i=1}^{8} e_{v_i}$$

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
| Estimate of $\bar{\boldsymbol{\xi}}$ | Top-$k$ head, plus a tail model | 8 tokens drawn from the sampler |
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
- The zero-mean score behind baselines fails off-policy, which leaves a drift term, $\mathbb{E}_q[A] \, \bar{\boldsymbol{\xi}}$, next to the signal.
- Drift is distillation toward the sampler, and it compounds because the sampler keeps being refreshed from the learner.
- Score centering subtracts $\bar{\boldsymbol{\xi}}$ from the score, which cancels the drift exactly. It is one extra term in the loss, and it does not make the update fully on-policy.

---

<!-- columns: 38/62 -->
<!-- valign: center -->
## Thank you

This lecture is a slice of a full course on post-training, free online with the book:

- **Course:** 14 lectures, Q&As, and guest episodes at [rlhfbook.com/course](https://rlhfbook.com/course)
- **Book:** [rlhfbook.com](https://rlhfbook.com), also in print from [Manning](https://www.manning.com/books/reinforcement-learning-from-human-feedback)
- **Newsletter:** [interconnects.ai](https://www.interconnects.ai/)

nathan@natolambert.com

|||

![](assets/post-training-course-thumbnails.jpeg)

---


<!-- after: references -->
<!-- layout: section-break -->

## Extra slides

---


<!-- after: references -->
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


<!-- after: references -->
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
