---
title: "Prerequisite Review"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "Prerequisite Review"
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

# Prerequisite Review

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">A 20–30 minute refresher on language models, optimization, probability, and RL — before Lecture 1.</p>

---

## What this review covers

This course assumes the basics of language modeling / NLP and an intro ML course. The lectures *use* the following ideas without re-teaching them. This deck rebuilds the four pillars you'll lean on:

- **Language models** — what an LM is, the LM head, softmax, cross-entropy
- **Optimization** — gradient descent, the training loop, pretraining → mid-training → SFT
- **Probability** — entropy, KL divergence, the log-derivative trick
- **Reinforcement learning** — policies, returns, policy gradients, why RL for LMs

You do **not** need prior RL or language-modeling background to start the course — this is a refresher, not a gate.

---

<!-- layout: section-break -->

## Language models

---

<!-- columns: 45/55 -->
## What a language model is

A language model learns the joint probability of a sequence of **tokens** (words / subwords) in an **autoregressive** manner — each prediction depends on the tokens before it.

Given $x = (x_1, \ldots, x_T)$, the model factorizes the sequence probability:

$$P_\theta(x) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1, \ldots, x_{t-1}).$$

Modern LMs (ChatGPT, Claude, Gemini) are **decoder-only Transformers** built on **self-attention**.

|||

![The original Transformer architecture diagram. 2017.](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

## The LM head

The transformer backbone outputs a vector per token in an internal **embedding space**. The **LM head** is a final linear projection from that embedding space back to the **vocabulary** (tokenizer space).

- Embedding $\xrightarrow{\text{LM head}}$ **logits** (one score per vocab token)
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
- In practice everyone uses **Adam** (adaptive, momentum-based) rather than plain SGD
- One **step** per batch; sweep the dataset in **epochs**; the learning-rate schedule matters as much as the rate itself

The machinery is identical across pretraining, mid-training, and fine-tuning — what changes is the **data** and the **objective**, not the optimizer.

---

## Pretraining → mid-training → SFT

The boundary at the *end* of pretraining is fuzzy and worth naming:

- **Pretraining** — next-token cross-entropy on massive raw web/code data.
- **Mid-training** — the annealing / high-quality-data phase at the very end of pretraining: "mid-training like annealing / high-quality end of pretraining web data." Quality ramps up here before any instruction data.
- **SFT (supervised fine-tuning)** — still cross-entropy, but now on **instruction–response** pairs: the model learns to *respond*, not just continue text.

Modern post-training pipelines start from a **mid-trained** checkpoint, so this is where the course's story effectively begins.

---

<!-- layout: section-break -->

## Probability essentials

---

## Entropy

The **entropy** of a distribution $p$ measures its uncertainty / average surprisal:

$$H(p) = -\sum_i p_i \log p_i.$$

- Uniform distributions have **high** entropy; peaked ones have **low** entropy
- Entropy shows up in this course when we talk about preference distributions (Lecture 8) and as a regularizer that keeps an RLHF'd policy from collapsing onto a single mode

---

## KL divergence

The **Kullback–Leibler divergence** measures how one distribution differs from another:

$$\mathcal{D}_{\text{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}.$$

Three things to remember:

- **Asymmetric** — $\mathcal{D}_{\text{KL}}(p\|q) \neq \mathcal{D}_{\text{KL}}(q\|p)$; direction matters
- **Non-negative**, and $0$ only when $p = q$
- **The** central object of alignment: the RLHF KL penalty (Lectures 3–4) keeps the policy close to its SFT start, and the DPO loss (Lecture 6) is derived from a KL-bound objective

You will see KL in essentially every alignment lecture.

---

## Expectations & the log-derivative trick

Many RL quantities are **expectations** under the model's own distribution, e.g. $\mathbb{E}_{p_\theta}[f]$. We need their *gradients*, but we can't differentiate through a sample.

The **log-derivative trick** (a.k.a. score-function / REINFORCE estimator) sidesteps this:

$$\nabla_\theta \,\mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}\!\left[\,f \,\nabla_\theta \log p_\theta\,\right].$$

Intuition: $\nabla_\theta \log p_\theta$ is the *score* — it points in the direction that increases the model's log-probability of a sample. Weight it by how good the sample was, and you get a usable gradient. This is the bridge into policy gradients.

---

<!-- layout: section-break -->

## Reinforcement learning basics

---

## RL framing

Reinforcement learning is about learning to **act** by maximizing accumulated **reward**.

- **Agent** takes **actions** in an **environment**; receives **rewards**
- **Policy** $\pi(a \mid s)$ — a distribution over actions given state
- **Return** $G_t$ — discounted future reward from time $t$
- **Value** $V^\pi(s)$ — expected return from state $s$ under $\pi$

The hard part is **credit assignment**: rewards are sparse and delayed, so which action *caused* the payoff?
<!-- cite-right: sutton2018reinforcement -->

---

## Policy gradients / REINFORCE

Instead of learning a value function, **directly optimize the policy**. The policy-gradient theorem (REINFORCE) gives a gradient of expected return:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\, R(\tau)\, \nabla_\theta \log \pi_\theta(\tau)\,\right].$$

Read it with slide 13 in hand: $R(\tau)$ is "how good was this trajectory," $\nabla_\theta \log \pi_\theta(\tau)$ is "nudge the policy to make this trajectory more likely." Together: **make good trajectories more probable**.

This is exactly what Lecture 3 builds on — the course just specializes $\pi$ to a language model and $R$ to a reward model.
<!-- cite-right: williams1992simple -->

---

## Why RL for language models?

A human preference ("response A is better than B") is:

- **Sparse** — one label per whole response, not per token
- **Non-differentiable** — there's no cross-entropy target for "preferred"
- **Delayed** — the quality of a token depends on the whole completion

So we can't write human feedback as a loss and backprop. RL lets us **optimize a reward signal we can't differentiate through** — first by *learning* that signal as a reward model, then by *optimizing* it with policy gradients. That is the whole motivation for RLHF.

---

## Where to go deeper

You're now ready for **Lecture 1**. If any pillar felt rusty:

- **RL** — Sutton & Barto, *Reinforcement Learning: An Introduction*; David Silver's UCL course; UC Berkeley CS285
- **Language models from scratch** — Sebastian Raschka, *Build a Large Language Model (From Scratch)*
- **The book's own reference** — Appendix A (Definitions) covers the language-modeling overview, KL, and RL vocabulary used throughout

Go down rabbit holes, skip around, and chase what excites you — the course is built for that.
