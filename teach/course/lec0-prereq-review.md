---
title: "The ML Foundations of LLM Post-Training"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "ML Foundations of LLM Post-Training"
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #84A98C; }
  :root {
    --colloquium-progress-fill: #84A98C;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
    letter-spacing: 0;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# The ML Foundations of LLM Post-Training

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">From cross-entropy to preferences and reinforcement learning — a 20–30 minute refresher before Lecture 1.</p>

<p class="colloquium-title-note" style="font-size: 0.6em; opacity: 0.7;">This deck was drafted with assistance from GLM 5.2.</p>

---

## What this review covers

**No prior reinforcement learning is assumed.** We assume intro-ML comfort with losses, gradients, and probability, plus basic PyTorch tensor fluency for the implementation lectures (Lecture 4 onward). This deck refreshes autoregressive language models and the handful of ideas the course *uses* without re-teaching them:

- **Language models** — autoregressive factorization, the LM head, softmax, cross-entropy, and the tensor-shape anatomy of one training example
- **Optimization** — gradient descent, the training loop, pretraining → SFT
- **Probability** — KL divergence (with entropy), sigmoid and pairwise likelihood
- **Reinforcement learning framing** — the MDP setup and the goal of RL; the course derives the algorithms themselves

This is a refresher, not a gate — chase what excites you and skip what you already know.

---

<!-- layout: section-break -->

## Language models

---

<!-- columns: 45/55 -->
## What a language model is

A language model learns the joint probability of a sequence of **tokens** (words / subwords) in an **autoregressive** manner — each prediction depends on the tokens before it.

Given $x = (x_1, \ldots, x_T)$, the model factorizes the sequence probability:

$$P_\theta(x) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1, \ldots, x_{t-1}).$$

Modern LMs (ChatGPT, Claude, Gemini) are **decoder-only Transformers** built on **self-attention** — each position attends only to itself and the positions before it.

|||

![Next-token prediction: a decoder-only language model predicts one token at a time from the prefix before it.](assets/pretraining_next_token_tikz.png)
<!-- cite-right: Vaswani2017AttentionIA -->

---

## The LM head

The transformer backbone outputs a **hidden state** vector per token. The **LM head** is a final linear projection from the hidden dimension back to the **vocabulary** (tokenizer space).

- Hidden state $\xrightarrow{\text{LM head}}$ **logits** (one score per vocab token)
- **softmax** over logits $\rightarrow$ a probability distribution over the next token
- Sampling / argmax from that distribution $\rightarrow$ the generated token

The same backbone can carry **different heads** for different jobs. In this course the first example is a **reward-model head** (Chapter 5) — same transformer, new head, new objective.

---

## Softmax & log-probabilities

**Softmax** turns logits $z$ into a distribution:

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.$$

In practice we almost always work in **log-space**. The log-probability of a sequence is the sum of per-token log-probs:

$$\log P_\theta(x) = \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}).$$

Why log-probs? **Numerical stability** (products of many small probabilities underflow), and sums are easier to differentiate than products.

---

## Anatomy of one LM training example

Lecture 4 jumps straight into shifting logits, gathering target log-probs, and masking. Here is the whole path once, up front:

```text
input_ids  ∈ ℕ^{B × L}          # token IDs, batch B, length L
   → model(input_ids).logits ∈ ℝ^{B × L × V}     # one vector per position, V = vocab
   → log_softmax + gather    ∈ ℝ^{B × (L-1)}     # log-prob of each observed token
   → (× completion_mask).sum ∈ ℝ^{B}             # one completion log-prob per sequence
```

```python
logits = model(input_ids).logits            # (B, L, V)
logits = logits[:, :-1, :]                  # shift: logit at t predicts token at t+1
labels = input_ids[:, 1:]                   # (B, L-1)
completion_mask = completion_mask[:, 1:]    # (B, L-1) — shift the mask to match labels
log_probs = logits.log_softmax(dim=-1)
token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)           # (B,)
```

The **one-token shift** is the autoregressive step made literal: position $t$'s logits predict token $t+1$. Sum the per-token log-probs and `loss.backward()` — autodiff turns that sum into the corresponding sum of per-token gradients.

---

## Three masks — don't confuse them

Masking is where silent bugs live. Three different masks do three different jobs:

- **Causal mask** (built into decoder attention) — position $t$ cannot attend to positions $> t$. This is what makes the model autoregressive; you never touch it in training code.
- **Attention / padding mask** — tells attention to ignore padding tokens (variable-length sequences batched together). Shape `(B, L)`.
- **Completion / loss mask** — `1` only on completion tokens; the loss is multiplied by it before summing. **Prompt tokens condition the prediction but contribute no loss.**

> Get the completion mask wrong and you either train on the prompt (wastes gradient on tokens you can't change) or on padding (trains on noise). Lecture 4 lists exactly these as the most common silent failures.

---

## Teacher forcing vs generation

How the sequence you score gets into the model determines what kind of training you're doing:

- **SFT** — score a **supplied** target under *teacher forcing*: the model always sees the correct prefix, one token off from the label.
- **RL** — **sample** a new sequence from the current policy, then score *that sampled* sequence. The model sees its own prefixes, which may be wrong.
- **Preference optimization** — score **supplied** chosen and rejected sequences (no sampling).

A small decoding review:

$$y_t \sim \mathrm{softmax}(z_t / T)$$

with $T$ a temperature (high $T$ → flatter, more random) and truncation methods (top-$k$, top-$p$) shaping the sample. The key distinction: **the model distribution** ($\pi_\theta$) vs **the sampling procedure** applied to it. Teacher-forced training never encounters the model's own errors; rollout training does — this is the root of *on-policy data*, *distribution shift*, and why RL infrastructure is different.

---

## Training an LM: cross-entropy

To fit the model we **maximize the likelihood** of the training data — equivalently, minimize the **negative log-likelihood (NLL)**, a.k.a. cross-entropy:

$$\mathcal{L}_{\text{LM}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right].$$

Read it simply: for each position, compare the model's predicted next-token distribution to the *true* token; penalize putting low probability on it.

"Predict the next token" **is** supervised learning — the label is just the next token in the sequence.

---

<!-- layout: section-break -->

## Optimization & fine-tuning

---

## Gradient descent & the training loop

Training repeats one step:

$$\theta \leftarrow \theta - \eta \,\nabla_\theta \mathcal{L}(\theta).$$

- $\theta$ — model parameters, $\eta$ — learning rate, $\nabla_\theta \mathcal{L}$ — gradient of the loss
- **AdamW** and related adaptive optimizers are standard for LM training (adaptive, momentum-based) rather than plain SGD
- Training is measured in optimizer **steps / tokens**; **epochs** are a useful unit for finite fine-tuning datasets. The learning-rate schedule matters as much as the rate itself

The basic backprop loop is shared across pretraining, mid-training, and fine-tuning — what differs is **masking, sampling, batching, data construction, and systems**, not the optimizer.

---

## Pretraining → mid-training → SFT

The boundary at the *end* of pretraining is fuzzy and worth naming:

- **Pretraining** — next-token cross-entropy on massive raw web/code data; **every** token contributes to the loss.
- **Mid-training** — the annealing / high-quality-data phase at the very end of pretraining, before any instruction data. Quality ramps up here.
- **SFT (supervised fine-tuning)** — still cross-entropy, but now on **instruction–response** pairs with a **completion mask**: only the response tokens contribute to the loss, the prompt merely conditions. The model learns to *respond*, not just continue text.

Modern post-training pipelines start from a **mid-trained** checkpoint, so this is where the course's story effectively begins.

---

<!-- layout: section-break -->

## Probability essentials

---

## KL divergence (and entropy)

The **Kullback–Leibler divergence** measures how one distribution differs from another:

$$\mathcal{D}_{\text{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}.$$

Three things to remember:

- **Asymmetric** — $\mathcal{D}_{\text{KL}}(p\|q) \neq \mathcal{D}_{\text{KL}}(q\|p)$; direction matters
- **Non-negative**, and $0$ only when $p = q$
- **The** central object of alignment: the RLHF KL penalty (Lectures 3–4) keeps the policy close to its SFT start, and the DPO loss (Lecture 6) is derived from a **KL-regularized reward-maximization** objective

> **Entropy** $H(p) = -\sum_i p_i \log p_i$ is the uncertainty of a distribution (uniform = high, peaked = low). It shows up here as a regularizer that keeps an RLHF'd policy from collapsing onto a single mode, and in Lecture 8's preference distributions.

You will see KL in essentially every alignment lecture.

---

## Sigmoid & pairwise likelihood

A preference ("response $y_w$ is better than $y_l$") is a **binary** outcome, so a score difference is squeezed through a **sigmoid** into a probability:

$$\sigma(z) = \frac{1}{1+e^{-z}}, \qquad P(y_w \succ y_l \mid x) = \sigma\!\left(r(x,y_w) - r(x,y_l)\right).$$

Three things to recognize:

- sigmoid converts a **score difference** into a binary probability
- only **relative** reward differences matter — shifting both rewards by a constant changes nothing
- $-\log \sigma(\cdot)$ is just **binary negative log-likelihood**

**Lecture 2 derives this Bradley–Terry model in full** to train a reward model; **Lecture 6 reuses the exact same $\sigma(\Delta r)$ shape** inside DPO. This slide is "recognize the pattern" — the derivations come later.

---

<!-- layout: section-break -->

## Reinforcement learning framing

---

## Language models as an MDP

Reinforcement learning studies an agent acting in a **Markov decision process** $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$: states, actions, a transition function, a reward, and a discount. Generation maps cleanly onto one:

| MDP concept | Language model |
|-------------|---------------|
| State $s_t$ | Prompt + tokens so far: $(x, y_{<t})$ |
| Action $a_t$ | Next token $y_t$ |
| Transition $P$ | Deterministic: append the token |
| Policy $\pi_\theta(a_t \mid s_t)$ | LM next-token distribution |
| Reward $r$ | Usually **terminal**: RM score or verifier output |
| Discount $\gamma$ | Typically $1.0$ (no discounting) |

The policy is the LM, factorized over tokens:

$$\pi_\theta(y \mid x) = \prod_t \pi_\theta(y_t \mid x, y_{<t}).$$

> **The supervision is usually response-level, but the gradient and KL bookkeeping are token-level.** That single distinction explains a large fraction of the notation the course uses later — far more useful than memorizing the bare five-tuple.

**Lecture 3 develops policy gradients, baselines, and PPO/GRPO from this setup; Lecture 4 turns them into code.** More on the algorithms there.

---

## The goal of RL, and why for language models

The goal of RL is to choose policy parameters $\theta$ that maximize **expected reward** under the policy itself:

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(x, y)].$$

A human preference (or a verifier) is a single scalar at the *end* of a response:

- **Sparse** — one label per whole response, not per token
- **Delayed** — the quality of a token depends on the whole completion
- **No token-level target** — and ordinary gradients can't pass back through the discrete sampled token

So there is no cross-entropy target for "preferred." Policy gradients optimize the expected evaluator score anyway — first by *learning* that signal as a reward model, then by *optimizing* it. That is the whole motivation for RLHF.

> **Lecture 3 derives the policy-gradient math; Lecture 4 turns it into code.** This deck stops at the framing.

---

## These tools become post-training

Everything in this deck is the basic machinery — **probabilities, losses, gradients, and sampling**. Post-training is what you get when you point that machinery at a pretrained model with richer feedback than raw next-token prediction:

- **Demonstrations** ("produce this response") → supervised fine-tuning
- **Preferences** ("make this response more likely than that one") → reward modeling or preference optimization (DPO)
- **Rewards** ("sample responses that score highly") → reinforcement learning (PPO, GRPO, RLVR)
- **Constraints** ("don't drift too far from where you started") → regularization, especially the KL penalty

Different data and different objectives; the same underlying goal:

> **Change the model's distribution over responses toward behavior we want.**

That is what the rest of this course is about.

---

## You are ready if…

You can do all of the following (or know which one to come back to):

- **Shift and gather** — explain why `logits[:, :-1]` predicts `input_ids[:, 1:]` and what `gather` does
- **Name the three masks** — causal, attention/padding, completion/loss — and which one the loss multiplies by
- **Distinguish teacher forcing from rollout** — and say why RL infrastructure is different
- **Read $\mathcal{D}_{\text{KL}}(p \| q)$** — asymmetric, non-negative, and central to alignment
- **Read a sigmoid score difference** — $P(y_w \succ y_l) = \sigma(r_w - r_l)$, and know only the *difference* matters
- **State the LM-as-MDP mapping** — and the response-level-vs-token-level takeaway

If any felt rusty:

- **RL** — Sutton & Barto, *Reinforcement Learning: An Introduction*; David Silver's UCL course; UC Berkeley CS285
- **Language models from scratch** — Sebastian Raschka, *Build a Large Language Model (From Scratch)*
- **The book's own reference** — Appendix A (Definitions) covers the language-modeling overview, KL, and RL vocabulary used throughout

Go down rabbit holes, skip around, and chase what excites you — the course is built for that.
