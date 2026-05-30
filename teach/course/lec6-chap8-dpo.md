---
title: "Lecture 6: Direct Preference Optimization"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "Lecture 6"
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

<!-- Source note: build with `make teach`, which copies assets/ into the output. A single-file `colloquium build -o ...` does NOT copy assets/, so the meme + displacement images 404 in that standalone build. -->

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 6: Direct Preference Optimization

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 8 — direct alignment, derived one step at a time</p>

---

<!-- columns: 50/50 -->
## What we are doing today

We will **derive Direct Preference Optimization (DPO) from scratch**, never skipping a step.

The promise of DPO:

- No separate reward model
- No reinforcement learning loop
- Just a single, directly-differentiable loss on preference pairs

|||

The plan, in four movements:

1. Solve the RLHF objective for its **optimal policy**
2. Invert that to write the **reward in terms of the policy**
3. Plug it into the **Bradley-Terry** preference model
4. Read off the **DPO loss** and its **gradient**

*Rafailov et al., 2023 — "Your Language Model is Secretly a Reward Model."*

---

<!-- columns: 48/52 -->
## Where DPO sits in the pipeline

Classic RLHF is three moving parts:

1. Collect human preference pairs
2. Train a reward model $r_\phi(x,y)$
3. Optimize the policy against $r_\phi$ with RL (e.g. PPO), under a KL penalty

|||

DPO collapses steps 2 and 3 into **one supervised-style loss**.

The key realization we will prove:

> The optimal RLHF policy and the reward model are two views of the *same* object. If we know one, we know the other in closed form.

So we can train the policy *directly* on preferences.

---

<!-- layout: section-break -->
<!-- align: center -->

## Movement 1 — Solving the RLHF objective for the optimal policy

---

<!-- rows: 40/60 -->
<!-- title: center -->
## The objective we are optimizing

$$
\max_{\pi}\;
\mathbb{E}_{x \sim \mathcal{D}}\,\mathbb{E}_{y \sim \pi(y\mid x)}
\underbrace{\big[\,r(x, y)\,\big]}_{\text{maximize reward}}
\;-\;
\underbrace{\beta\, \mathcal{D}_{\text{KL}}\big(\pi(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big)}_{\text{stay close to the reference model}}
$$

===

- The expectation $\mathbb{E}_{y\sim\pi}$ applies to the reward term (we must *sample* to estimate it).
- The KL term is an **analytic** expression — no sampling needed.
- $\beta$ trades off reward against drift from $\pi_{\text{ref}}$.

Goal: find the exact $\pi$ that maximizes this.

---

<!-- columns: 55/45 -->
## Step 1 — fold the KL into the expectation

Recall the definition of the KL divergence:

$$
\mathcal{D}_{\text{KL}}(\pi \,\|\, \pi_{\text{ref}})
= \mathbb{E}_{y\sim\pi}\!\left[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\right]
$$

Both terms now share the **same** expectation $\mathbb{E}_{y\sim\pi}$, so we combine them:

$$
\max_{\pi}\;\mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi(y\mid x)}
\left[\, r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \,\right]
$$

|||

**What changed**

- Wrote the KL as an expectation over $y\sim\pi$.
- Merged the two terms under one expectation.

Now everything lives inside a single bracket.

---

<!-- columns: 55/45 -->
## Step 2 — flip to a minimization

Multiply the objective by $-1$. Maximizing a quantity is the same as minimizing its negative, so $\max \to \min$ and **every** term in the bracket flips sign:

$$
\min_{\pi}\;\mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi(y\mid x)}
\left[\, \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - r(x,y) \,\right]
$$

|||

**What changed**

- $\times(-1)$ applied to the whole objective.
- $\max \to \min$: the $r$ term becomes $-r$ and the log term becomes $+\beta\log\frac{\pi}{\pi_{\text{ref}}}$.

Nothing has been rescaled yet — only the sign and the direction of optimization.

---

<!-- columns: 55/45 -->
## Step 3 — divide by $\beta$

Divide the whole objective by $\beta > 0$. Both terms pick up a factor of $\tfrac{1}{\beta}$, and the minimizer is unchanged:

$$
\min_{\pi}\;\mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi(y\mid x)}
\left[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \frac{1}{\beta}\, r(x,y) \,\right]
$$

|||

**What changed**

- $\div\,\beta$: $\beta\log(\cdot) \to \log(\cdot)$ and $r \to \tfrac{1}{\beta} r$ — the coefficient hits **both** terms.
- Dividing by a positive constant does not move the minimizer.

We now want to recognize this bracket as a **distance** we can drive to zero.

---

<!-- columns: 55/45 -->
## Step 4 — introduce the partition function

Define, for each prompt $x$:

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y\mid x)\,\exp\!\left(\frac{1}{\beta}\, r(x,y)\right)
$$

|||

**Why?**

$Z(x)$ is exactly the normalizer that turns

$$
\pi_{\text{ref}}(y\mid x)\,\exp\!\left(\tfrac{1}{\beta} r(x,y)\right)
$$

into a **valid probability distribution** over $y$.

Note: $Z(x)$ depends on $x$ and $r$ — **but not on $\pi$**. Remember this.

---

<!-- columns: 52/48 -->
## Step 5 — complete the expression with $Z(x)$

Take the bracket from Step 3 and add $0 = \log Z(x) - \log Z(x)$, then regroup:

$$
\begin{aligned}
&\;\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \frac{1}{\beta} r(x,y) \\[4pt]
&= \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \frac{1}{\beta} r(x,y) + \log Z(x) - \log Z(x) \\[4pt]
&= \log\frac{\pi(y\mid x)}{\tfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\left(\tfrac{1}{\beta} r(x,y)\right)} - \log Z(x)
\end{aligned}
$$

|||

**What changed**

- Added and subtracted $\log Z(x)$ (a legal $+0$).
- Used $\log a + \log b = \log(ab)$ and $\frac{1}{\beta}r = \log\exp(\frac{1}{\beta}r)$ to pull everything into **one log-ratio**.

The denominator is now the normalized distribution from Step 3.

---

<!-- columns: 55/45 -->
## Step 6 — recognize a KL divergence

Substituting back, the objective becomes:

$$
\min_{\pi}\;\mathbb{E}_{x\sim\mathcal{D}}
\left[
\mathcal{D}_{\text{KL}}\!\left(\pi(y\mid x)\,\Big\|\,
\tfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\left(\tfrac{1}{\beta} r(x,y)\right)\right)
- \log Z(x)
\right]
$$

|||

**What changed**

- The inner expectation is now literally $\mathbb{E}_{y\sim\pi}\!\left[\log\frac{\pi}{q}\right]$ — a **proper KL divergence**, because the denominator $q$ is a valid distribution (thanks to $Z$).

We are minimizing a KL plus a term that does not involve $\pi$.

---

<!-- rows: 45/55 -->
<!-- title: center -->
## Step 7 — Gibbs' inequality gives the optimal policy

$$
\min_{\pi}\;\mathbb{E}_{x\sim\mathcal{D}}
\big[\, \underbrace{\mathcal{D}_{\text{KL}}(\pi \,\|\, q)}_{\geq 0,\ =0 \text{ iff } \pi=q} - \underbrace{\log Z(x)}_{\text{independent of }\pi} \,\big]
$$

===

- $\log Z(x)$ does not depend on $\pi$, so it is a constant for the minimization — ignore it.
- A KL divergence is $\geq 0$ and equals $0$ **only when the two distributions are identical** (Gibbs' inequality).
- Therefore the minimizer sets $\pi$ equal to $q$.

---

<!-- align: center -->
<!-- valign: center -->
## Result: the optimal RLHF policy

```box
title: Optimal policy
tone: accent
content: |
  The policy that solves the KL-constrained RLHF objective is, in closed form:
```

$$
\pi^{*}(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\left(\frac{1}{\beta}\, r(x,y)\right)
$$

We can write $\pi^{*}$ exactly — but $Z(x)$ is a sum over **all possible responses**, so we cannot compute it directly. The next movement makes that problem disappear.

---

<!-- layout: section-break -->
<!-- align: center -->

## Movement 2 — Inverting: the reward in terms of the policy

---

<!-- columns: 52/48 -->
## Step 8 — solve the optimal policy for the reward

Start from $\pi^{*}$ and take $\log$ of both sides, then isolate $r^{*}$:

$$
\begin{aligned}
\log \pi^{*}(y\mid x)
&= \log\!\left(\tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big)\right) \\[4pt]
&= -\log Z(x) + \log \pi_{\text{ref}}(y\mid x) + \frac{1}{\beta}\, r^{*}(x,y)
\end{aligned}
$$

|||

**What changed**

- Took $\log$ of both sides.
- Expanded with $\log(abc) = \log a + \log b + \log c$.

Everything is linear in the logs now — easy to rearrange for $r^{*}$.

---

<!-- columns: 52/48 -->
## Step 9 — rearrange for the implicit reward

Move terms across and multiply by $\beta$:

$$
\begin{aligned}
\frac{1}{\beta}\, r^{*}(x,y) &= \log \pi^{*}(y\mid x) - \log \pi_{\text{ref}}(y\mid x) + \log Z(x) \\[6pt]
r^{*}(x,y) &= \beta \log \frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)
\end{aligned}
$$

|||

**What changed**

- Isolated the reward term.
- Multiplied through by $\beta$ and combined the two logs into a **log-ratio**.

---

<!-- align: center -->
<!-- valign: center -->
## Result: your language model is secretly a reward model

```box
title: Implicit reward
tone: accent
content: |
  Any policy implies a reward, expressed entirely through its own probabilities:
```

$$
r^{*}(x, y) = \beta \log \frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)
$$

- The reward is a **log-ratio** between the trained policy and the reference, scaled by $\beta$.
- The ugly $\beta \log Z(x)$ term depends only on $x$ — keep an eye on it.

---

<!-- layout: section-break -->
<!-- align: center -->

## Movement 3 — Substituting into Bradley-Terry

---

<!-- rows: 45/55 -->
<!-- title: center -->
## Step 10 — recall the Bradley-Terry model

From the reward-modeling chapter, the probability that response $y_1$ is preferred over $y_2$:

$$
p^{*}(y_1 \succ y_2 \mid x)
= \frac{\exp\big(r^{*}(x, y_1)\big)}{\exp\big(r^{*}(x, y_1)\big) + \exp\big(r^{*}(x, y_2)\big)}
$$

===

- This is a softmax over two reward scores.
- We will now substitute our **implicit reward** from Movement 2 — and watch the intractable $Z(x)$ vanish.

---

## Step 11 — substitute the implicit reward

Replace each $r^{*}(x, y_i)$ with $\beta \log \frac{\pi^{*}(y_i\mid x)}{\pi_{\text{ref}}(y_i\mid x)} + \beta \log Z(x)$:

$$
p^{*}(y_1 \succ y_2 \mid x)
= \frac{\exp\!\left(\beta \log \frac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} + \beta \log Z(x)\right)}
{\exp\!\left(\beta \log \frac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} + \beta \log Z(x)\right) + \exp\!\left(\beta \log \frac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)} + \beta \log Z(x)\right)}
$$

Every exponential carries the **same** $\beta \log Z(x)$ term. That is about to save us.

---

<!-- columns: 55/45 -->
## Step 12 — the partition function cancels

Use $e^{a+b} = e^{a} e^{b}$ to factor out $\exp(\beta \log Z(x)) = Z(x)^{\beta}$ from every term. It appears in the numerator and in **both** denominator terms, so it cancels:

$$
p^{*}(y_1 \succ y_2 \mid x)
= \frac{\exp\!\left(\beta \log \frac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\right)}
{\exp\!\left(\beta \log \frac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\right) + \exp\!\left(\beta \log \frac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}\right)}
$$

|||

**Why this matters**

- $Z(x)$ was the one term we could not compute.
- Because it is identical across responses, the **pairwise** comparison erases it.
- No reward model, no partition function — only policy probabilities remain.

---

<!-- columns: 55/45 -->
## Step 13 — rewrite as a sigmoid

Divide numerator and denominator by the numerator, then use $\sigma(z) = \frac{1}{1 + e^{-z}}$:

$$
\begin{aligned}
p^{*}(y_1 \succ y_2 \mid x)
&= \frac{1}{1 + \exp\!\left(\beta \log \frac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)} - \beta \log \frac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\right)} \\[6pt]
&= \sigma\!\left(\beta \log \frac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} - \beta \log \frac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}\right)
\end{aligned}
$$

|||

**What changed**

- Divided top and bottom by the first exponential (top $\to 1$).
- Matched the denominator to the sigmoid form.

The preference probability is now a **sigmoid of the difference of two log-ratios**.

---

<!-- layout: section-break -->
<!-- align: center -->

## Movement 4 — The DPO loss and its gradient

---

<!-- rows: 45/55 -->
<!-- title: center -->
## Step 14 — the DPO loss

Fit by **maximum likelihood** on preference pairs $(x, y_c, y_r)$ — i.e. minimize the negative log-likelihood of the chosen response winning:

$$
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}})
= -\,\mathbb{E}_{(x, y_c, y_r)\sim\mathcal{D}}
\left[ \log \sigma\!\left( \beta \log \frac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta \log \frac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)} \right) \right]
$$

===

- $y_c$ = chosen, $y_r$ = rejected. We replaced the abstract $\pi^{*}$ with the trainable $\pi_{\theta}$.
- The loss falls when the chosen log-ratio exceeds the rejected log-ratio.
- **Directly differentiable** — no sampling, no reward model, no RL loop.

---

<!-- columns: 50/50 -->
## Reading the loss

$$
\beta \log \frac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)}
\;-\;
\beta \log \frac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}
$$

|||

- **First term**: how much the policy raised the probability of the *chosen* response vs. the reference.
- **Second term**: the same for the *rejected* response.
- The loss rewards widening the gap between them.
- $\beta$ controls how hard we push relative to staying near $\pi_{\text{ref}}$.

---

## Step 15 — the gradient

Differentiating the loss (using $\sigma' = \sigma(1-\sigma)$ and $\sigma(-z) = 1 - \sigma(z)$) gives:

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}}
= -\,\beta\, \mathbb{E}_{(x, y_c, y_r)\sim\mathcal{D}}
\Big[\, w \cdot \big( \nabla_{\theta}\log \pi_{\theta}(y_c\mid x) - \nabla_{\theta}\log \pi_{\theta}(y_r\mid x) \big) \Big]
$$

$$
\text{where}\quad w = \sigma\!\big(\, r_{\theta}(x, y_r) - r_{\theta}(x, y_c) \,\big),
\qquad r_{\theta}(x, y) = \beta \log \tfrac{\pi_{\theta}(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$

---

<!-- columns: 45/55 -->
## What the gradient does

$$
-\,\beta\, \mathbb{E}\Big[\, \underbrace{w}_{\text{how wrong}} \cdot \big( \underbrace{\nabla_{\theta}\log \pi_{\theta}(y_c) - \nabla_{\theta}\log \pi_{\theta}(y_r)}_{\text{up on chosen, down on rejected}} \big) \Big]
$$

|||

- **The weight $w$** runs from 0 to 1 and is **larger when the model is more wrong** — when it currently ranks the rejected response above the chosen.
- **The bracket** pushes up the likelihood of $y_c$ and pushes down the likelihood of $y_r$.
- **$\beta$** scales the step, balancing correct ordering against drift from $\pi_{\text{ref}}$.

---

<!-- rows: 50/50 -->
<!-- title: center -->
## Recap: the whole derivation on one slide

$$
\begin{aligned}
\textbf{Objective} \;&:\; \max_{\pi}\, \mathbb{E}\big[r(x,y)\big] - \beta\, \mathcal{D}_{\text{KL}}(\pi \,\|\, \pi_{\text{ref}}) \\[4pt]
\Rightarrow\ \textbf{Optimal policy} \;&:\; \pi^{*}(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac{1}{\beta} r(x,y)\big) \\[4pt]
\Rightarrow\ \textbf{Implicit reward} \;&:\; r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x) \\[4pt]
\Rightarrow\ \textbf{Bradley-Terry} \;&:\; p^{*}(y_1 \succ y_2) = \sigma\!\big(r^{*}(x,y_1) - r^{*}(x,y_2)\big)\ \ (Z \text{ cancels}) \\[4pt]
\Rightarrow\ \textbf{DPO loss} \;&:\; -\log \sigma\!\big(\beta \log \tfrac{\pi_{\theta}(y_c)}{\pi_{\text{ref}}(y_c)} - \beta \log \tfrac{\pi_{\theta}(y_r)}{\pi_{\text{ref}}(y_r)}\big)
\end{aligned}
$$

===

The reward model never had to be built — it was hiding inside the policy the whole time.

---

<!-- layout: section-break -->
<!-- align: center -->

## Weaknesses, variants, and practice

---

<!-- columns: 55/45 -->
## A subtle failure: the chosen probability can fall

The DPO loss only cares about the **margin** between the chosen and rejected log-ratios — not their absolute values.

So the model can lower the loss by pushing the *rejected* probability down **faster** than the chosen, even while the **chosen probability also falls**.

|||

![Sketch of preference displacement in DPO.](assets/dpo_displacement.png)

- Called **preference displacement**; posited to push probability toward unaddressed, off-distribution behaviors.
- A reason practitioners add an SFT term on the chosen response, or use fixes like Cal-DPO / AlphaPO.

---

## The $\beta$ parameter and the static KL

$\beta$ sets the strength of the KL constraint relative to reward maximization:

- **Large $\beta$** → policy stays close to $\pi_{\text{ref}}$; it barely moves.
- **Small $\beta$** → policy is free to deviate; it can **over-optimize**.

Crucially, DPO's KL is **static**: it steps directly to the *optimal* solution implied by the dataset and the chosen $\beta$. Online RL instead takes steps based on freshly sampled batches.

---

<!-- columns: 50/50 -->
## A zoo of direct alignment algorithms

Each variant tweaks the loss to fix a limitation — usually a one-line change.

- **IPO** — softens the preference probability away from Bradley-Terry to curb overfitting.
- **cDPO / ODPO** — assume label noise / require a margin offset, so not all pairs count equally.
- **ORPO** — adds an odds-ratio pull toward the chosen response and **drops the reference model**.
- **SimPO** — length-normalizes the reward, also reference-free.

|||

**SimPO loss**

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\tfrac{\beta}{|y_w|}\log \pi_\theta(y_w\mid x) - \tfrac{\beta}{|y_l|}\log \pi_\theta(y_l\mid x) - \gamma\right)\right]
$$

- $\tfrac{1}{|y|}$ normalizes by length; $\gamma$ is a target margin; no $\pi_{\text{ref}}$.
- The algorithm matters **far less** than the base model and the data.

---

## Implementation is genuinely simple

No generation during training, no separate reward model — the heart of the loss is a few lines:

```python
# log-prob gaps for policy and frozen reference
pi_logratios  = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# positive when policy shifts mass toward the chosen completion
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

**Tip:** $\pi_{\text{ref}}$ is frozen, so precompute and cache its log-probs to cut peak memory ~50%. Reference code: `code/direct_alignment/`.

---

## DAAs work with synthetic preference data

Direct alignment needs *feedback*, not necessarily *human* feedback — AI feedback works just as well.

- Most modern DPO uses preferences labeled by a strong model. **UltraFeedback** was the first such dataset; Tülu 3, SmolLM 3, and Olmo 3 followed.
- **On-policy data** (some completions from the model you are tuning) helps the contrastive loss optimize the right token space.
- **Delta Learning**: later work argues the *gap* between chosen and rejected matters more than which models produced them (e.g. Qwen3-32B chosen vs Qwen3-0.6B rejected).

Watch for judge biases — frontier labelers favor longer, self-similar outputs.

---

<!-- columns: 50/50 -->
## DPO vs. RL: offline vs. online

**DPO and other DAAs are offline**

- Train on a fixed dataset collected ahead of time.
- Simpler, more stable, fast to iterate on data.
- Limited by the **coverage** of that dataset — a slightly lower performance ceiling.

|||

**PPO / policy gradient is online**

- Generate fresh completions during training, score with a reward model.
- Can explore new regions → often higher peak performance.
- More compute, more moving parts (four models in memory).

**Middle ground:** *online / iterative DPO* regenerates responses and relabels during training.

---

<!-- columns: 50/50 -->
## Takeaways

- DPO is **not** "just supervised fine-tuning on chosen responses." It optimizes the *same* KL-constrained RLHF objective — exactly, in closed form.
- The optimal policy and the reward model are **two views of one object**; the log-ratio $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ *is* the implicit reward.
- The intractable partition function $Z(x)$ cancels because preferences are **pairwise**.

|||

**Why people care**

- One loss, one model, no sampling loop — far simpler to implement than PPO.
- $\beta$ is often easier to tune than in online RL, but the best value depends on the model and the data.
- Because data is offline, DPO solves for the policy implied by *that* dataset and *that* $\beta$ — a core difference from online policy-gradient methods.

![When DPO was released it sparked a fierce debate about how to best do preference learning. Meme credit: Tom Goldstein.](assets/dpo_meme.jpeg)
