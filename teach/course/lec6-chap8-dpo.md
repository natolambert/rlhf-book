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
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only — leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
---

<!-- Source note: build with `make teach`, which copies assets/ into the output. A single-file `colloquium build -o ...` does NOT copy assets/, so the meme + displacement images 404 in that standalone build. -->
<!-- Reveal convention: each derivation step is a run of duplicate slides with identical title; one aligned line (with its reason) is appended per slide so equations build up in place and the layout never recenters. Every equation from chapter 8 is shown. -->

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 6: Direct Preference Optimization

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 8 on Direct Alignment Algorithms.</p>

---

<!-- animate: bullets -->
## Why DPO was -- and is -- such a big deal

The Direct Preference Optimization paper [@rafailov2024direct] was a breakthrough in the accessibility of RLHF and post-training research.
Some context:
- DPO paper was released in May of 2023, when many groups were struggling to get open-source replications of RLHF pipelines going. Most "aligned" models were just SFT.
- It took until about the fall of 2023 for people to figure out the right data and settings for DPO to work
- Zephyr-Beta, Tülu 2, and other models that fall opened the floodgates of post-training research. Iterating on DPO methods in 2024 felt as exciting as RL methods in 2025 or tool-use methods in 2026.
- **Terminology note:** I call the class of algorithms that operate like DPO on static preference data Direct Alignment Algorithms (DAAs). The name didn't stick much, but we'll see there are a ton of variants!

---

<!-- img-align: center -->
<!-- valign: center -->
## The DPO debate (DPO v RL)

![When DPO was released it sparked a fierce debate about how best to do RLHF and preference learning. Meme credit: Tom Goldstein.](assets/dpo_meme.jpeg)

---

<!-- columns: 50/50 -->
## This lecture

We will **derive Direct Preference Optimization (DPO) from scratch**, then look at how it is used in practice.

The promise of DPO:

- No separate reward model
- No reinforcement learning loop
- Just a single, directly-differentiable loss on preference pairs

|||

The plan, in four steps:

1. Solve the RLHF objective for its **optimal policy**
2. Invert that to write the **reward in terms of the policy**
3. Plug it into the **Bradley-Terry** preference model
4. Read off the **DPO loss** and its **gradient**

*Rafailov et al., 2023 — "Your Language Model is Secretly a Reward Model"* [@rafailov2024direct].

---

<!-- rows: 60/40 -->
## Recall: Where DPO sits in the pipeline

<!-- row-columns: 48/52 -->
Classic RLHF is three moving parts:

1. Collect human preference pairs
2. Train a reward model $r_\phi(x,y)$
3. Optimize the policy against $r_\phi$ with RL (e.g. PPO), under a KL penalty

|||

DPO collapses steps 2 and 3 into **one supervised-style loss**.

The key realization we will prove:

> The optimal RLHF policy and the reward model are two views of the *same* object. If we know one, we know the other in closed form.

So we can train the policy *directly* on preferences.

===

![Modern post-training runs many rounds of feedback and optimization; DPO is one of the interchangeable optimizers in that loop.](assets/rlhf-complex.png)

---

<!-- columns: 55/45 -->
## DPO at a glance: where we're headed

The whole derivation lands on two objects.

**Optimal policy** — the RLHF solution in closed form:

$$ \pi^{*}(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac{1}{\beta} r(x,y)\big) $$

**DPO loss** — what you actually train:

$$ \mathcal{L}_{\text{DPO}} = -\mathbb{E}\Big[\log\sigma\big(\beta\log\tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta\log\tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big)\Big] $$

|||

In practice, that loss is just a few lines:

```python
pi_logratios  = policy_chosen_logps  - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits)
```

Everything between here and the implementation is *why* this is the right loss.

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO Derivation P1/4: Deriving the optimal policy

---

<!-- valign: top -->
<!-- align: center -->
## Start with the RLHF optimization problem

$$
\begin{aligned}
 & \max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}\big[\,r(x,y)\,\big] - \beta\,\mathcal{D}_{\text{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big) && \text{the RLHF objective}
\end{aligned}
$$

We want to find the $\pi$ that solves this equation! But, without RL.

---

<!-- valign: top -->
<!-- title: center -->
## Fold the KL into the expectation

$$
\begin{aligned}
 & \max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}\big[\,r(x,y)\,\big] - \beta\,\mathcal{D}_{\text{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big) && \text{the RLHF objective}\\[6pt]
={}& \max_{\pi}\ \mathbb{E}\Big[\, r(x,y) - \beta\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \,\Big] && \text{write KL}=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi}\big[\log\tfrac{\pi}{\pi_{\text{ref}}}\big]
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Fold the KL into the expectation

$$
\begin{aligned}
 & \max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}\big[\,r(x,y)\,\big] - \beta\,\mathcal{D}_{\text{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big) && \text{the RLHF objective}\\[6pt]
={}& \max_{\pi}\ \mathbb{E}\Big[\, r(x,y) - \beta\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \,\Big] && \text{write KL}=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi}\big[\log\tfrac{\pi}{\pi_{\text{ref}}}\big]\\[6pt]
={}& \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{split into two terms}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Flip to a minimization & clean-up

$$
\begin{aligned}
 & \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{where we left off}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Flip to a minimization & clean-up

$$
\begin{aligned}
 & \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{where we left off}\\[6pt]
={}& \min_{\pi}\Big( -\mathbb{E}[r(x,y)] + \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{multiply by } -1:\ \max\to\min
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Flip to a minimization & clean-up

$$
\begin{aligned}
 & \max_{\pi}\Big( \mathbb{E}[r(x,y)] - \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{where we left off}\\[6pt]
={}& \min_{\pi}\Big( -\mathbb{E}[r(x,y)] + \beta\,\mathbb{E}\big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\big] \Big) && \text{multiply by } -1:\ \max\to\min\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\Big[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\Big] && \text{divide by }\beta\text{, recombine}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- align: center -->
<!-- animate: bullets -->
## Fold the optimization target into one log-ratio

Pick up where we left off and fold the bracket into a single log-ratio, using $\tfrac{1}{\beta} r = \log e^{\,r/\beta}$:

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\Big[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\Big] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]
\end{aligned}
$$

- $\pi$ is now measured against $\pi_{\text{ref}}\,e^{r/\beta}$: the **reference reweighted by reward** — each response's probability is multiplied by $e^{r/\beta}$, which is larger the higher its reward ($\beta$ sets how aggressive the reweighting is).
- If that target were a probability distribution $q$, this $\min_{\pi}$ would be a **KL divergence**, $\mathbb{E}_{y\sim\pi}\!\big[\log\tfrac{\pi}{q}\big] = \mathcal{D}_{\text{KL}}(\pi\,\|\,q)$ — minimized *exactly*, in closed form ([Gibbs' inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality)).

---

<!-- valign: top -->
<!-- align: center -->
## Introduce the partition function $Z(x)$

We are minimizing $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$.

Why do we want $\pi$ to match that denominator?

---

<!-- valign: top -->
<!-- title: center -->
## Introduce the partition function $Z(x)$

We are minimizing $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$.

Why do we want $\pi$ to match that denominator? Because $\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big]$ is a **KL divergence** $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$ — it is $\ge 0$, and equals $0$ **only when $\pi = q$**. So the minimizer is whatever distribution $q$ sits in the denominator.

---

<!-- valign: top -->
<!-- title: center -->
## Introduce the partition function $Z(x)$

We are minimizing $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$.

Why do we want $\pi$ to match that denominator? Because $\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big]$ is a **KL divergence** $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$ — it is $\ge 0$, and equals $0$ **only when $\pi = q$**. So the minimizer is whatever distribution $q$ sits in the denominator.

The catch: our denominator $\pi_{\text{ref}}\,e^{r/\beta}$ is **not a distribution** — summed over $y$ it does not equal $1$. 

---

<!-- valign: top -->
<!-- title: center -->
## Introduce the partition function $Z(x)$

We are minimizing $\min_{\pi}\ \mathbb{E}_{y\sim\pi}\Big[\, \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}} \,\Big]$.

Why do we want $\pi$ to match that denominator? Because $\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big]$ is a **KL divergence** $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$ — it is $\ge 0$, and equals $0$ **only when $\pi = q$**. So the minimizer is whatever distribution $q$ sits in the denominator.

The catch: our denominator $\pi_{\text{ref}}\,e^{r/\beta}$ is **not a distribution** — summed over $y$ it does not equal $1$. 
We normalize it with the partition function:

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r(x,y)\big)
\qquad\text{(depends on $x$ and $r$, not on $\pi$)}
$$

<!-- animate: bullets -->
- Now $q(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,e^{\,r/\beta}$ **is** a valid distribution.
- The objective becomes a genuine $\mathcal{D}_{\text{KL}}(\pi\,\|\,q)$ — minimized exactly at $\pi = q$ (made rigorous with [Gibbs' inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality) shortly).
- Next we massage the inside of the objective so $Z(x)$ appears.

---

<!-- valign: top -->
<!-- title: center -->
## Revisiting the objective with the partition function $Z(x)$

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Revisiting the objective with the partition function $Z(x)$

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) + \log Z(x) - \log Z(x) \,\right] && \text{add } 0 = \log Z - \log Z
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Revisiting the objective with the partition function $Z(x)$

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) + \log Z(x) - \log Z(x) \,\right] && \text{add } 0 = \log Z - \log Z\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \big( \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \log Z(x) \big) - \log Z(x) - \tfrac{1}{\beta} r(x,y) \,\right] && \text{group terms}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Revisiting the objective with the partition function $Z(x)$

$$
\begin{aligned}
 & \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) \,\right] && \text{where we left off}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \tfrac{1}{\beta} r(x,y) + \log Z(x) - \log Z(x) \,\right] && \text{add } 0 = \log Z - \log Z\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \big( \log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \log Z(x) \big) - \log Z(x) - \tfrac{1}{\beta} r(x,y) \,\right] && \text{group terms}\\[6pt]
={}& \min_{\pi}\ \mathbb{E}\left[\, \log\frac{\pi(y\mid x)}{\tfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)} - \log Z(x) - \tfrac{1}{\beta} r(x,y) \,\right] && \log a + \log b = \log ab
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Turn the objective into a KL divergence

Fold $\tfrac{1}{\beta} r = \log e^{\,r/\beta}$ into the denominator of the first term in the objective (again, log rules) and name it $q(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}$ — a valid distribution over $y$. The objective is:

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi}\Big[\, \log\tfrac{\pi(y\mid x)}{q(y\mid x)} - \log Z(x) \,\Big]
$$

<!-- step -->

$\log Z(x)$ does not depend on $y$, so pull it out of the inner expectation:

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\Big[\, \mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi(y\mid x)}{q(y\mid x)}\big] - \log Z(x) \,\Big]
$$

<!-- step -->

The inner expectation is exactly a **KL divergence**, $\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi}{q}\big] = \mathcal{D}_{\text{KL}}(\pi\,\|\,q)$:

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\big[\, \mathcal{D}_{\text{KL}}\big(\pi(y\mid x)\,\|\,q(y\mid x)\big) - \log Z(x) \,\big]
$$

---

<!-- valign: top -->
<!-- title: center -->
## Apply Gibbs' inequality to read off $\pi^{*}$

Where we landed -- the objective is now a KL plus a $\pi$-independent constant:

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\big[\, \mathcal{D}_{\text{KL}}\big(\pi(y\mid x)\,\|\,q(y\mid x)\big) - \log Z(x) \,\big]
$$

- $\log Z(x)$ does not depend on $\pi$ — only the KL term can be optimized.
- A KL is $\ge 0$ and equals $0$ **only when $\pi = q$** ([Gibbs' inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality)), so the minimizer is $\pi^{*} = q$:

$$
\boxed{\ \ \pi^{*}(y\mid x) = q(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r(x,y)\big)\ \ }
$$

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO Derivation P2/4: Recovering the reward from the policy (needed to implement it)

---

<!-- valign: top -->
<!-- title: center -->
## Invert $\pi^{*}$ for the implicit reward

$$
\begin{aligned}
 & \pi^{*}(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big)  && 
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Invert $\pi^{*}$ for the implicit reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Invert $\pi^{*}$ for the implicit reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}\\[6pt]
 & \log \pi^{*}(y\mid x) = -\log Z(x) + \log \pi_{\text{ref}}(y\mid x) + \tfrac{1}{\beta} r^{*}(x,y) && \log abc = \log a+\log b+\log c
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Invert $\pi^{*}$ for the implicit reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}\\[6pt]
 & \log \pi^{*}(y\mid x) = -\log Z(x) + \log \pi_{\text{ref}}(y\mid x) + \tfrac{1}{\beta} r^{*}(x,y) && \log abc = \log a+\log b+\log c\\[6pt]
 & \tfrac{1}{\beta} r^{*}(x,y) = \log \pi^{*}(y\mid x) - \log \pi_{\text{ref}}(y\mid x) + \log Z(x) && \text{rearrange for } r^{*}
\end{aligned}
$$

---

<!-- valign: top -->
<!-- title: center -->
## Invert $\pi^{*}$ for the implicit reward

$$
\begin{aligned}
 & \log \pi^{*}(y\mid x) = \log\!\Big( \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(\tfrac{1}{\beta} r^{*}(x,y)\big) \Big) && \text{take $\log$ of both sides}\\[6pt]
 & \log \pi^{*}(y\mid x) = -\log Z(x) + \log \pi_{\text{ref}}(y\mid x) + \tfrac{1}{\beta} r^{*}(x,y) && \log abc = \log a+\log b+\log c\\[6pt]
 & \tfrac{1}{\beta} r^{*}(x,y) = \log \pi^{*}(y\mid x) - \log \pi_{\text{ref}}(y\mid x) + \log Z(x) && \text{rearrange for } r^{*}\\[6pt]
 & r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x) && \text{multiply by }\beta
\end{aligned}
$$

$$
\boxed{\ \ r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)\ \ }\qquad\text{\small your LM is secretly a reward model}
$$

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO Derivation P3/4: Connecting to Bradley-Terry preference (what our alignment data has looked like) 

---

<!-- valign: top -->
<!-- title: center -->
## Recall the Bradley-Terry model

A preference between two responses is a softmax over their rewards:

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{\exp\!\big(r^{*}(x,y_1)\big)}{\exp\!\big(r^{*}(x,y_1)\big) + \exp\!\big(r^{*}(x,y_2)\big)}
$$

Now substitute the implicit reward $r^{*}(x,y) = \beta \log \tfrac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$.

*New here? See [Lecture 2](https://rlhfbook.com/teach/course/lec2-chap4-5-9/) and [Chapter 5: Reward Modeling](https://rlhfbook.com/c/05-reward-models.html) for the Bradley-Terry model.*

---

<!-- valign: top -->
<!-- align: center -->
## Substitute the reward, then cancel $Z(x)$

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} + \beta \log Z(x)\big)}{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} + \beta \log Z(x)\big) + \exp\!\big(\beta \log \tfrac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)} + \beta \log Z(x)\big)}
$$

- Every term carries the same factor $\exp(\beta \log Z(x)) = Z(x)^{\beta}$.

<!-- step -->

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\big)}{\exp\!\big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\big) + \exp\!\big(\beta \log \tfrac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}\big)}
$$

- Decompose $e^{a+b}=e^a e^b$ and cancel the shared $Z(x)^{\beta}$ — the **intractable term is gone**.

---

<!-- valign: top -->
<!-- title: center -->
## Divide through to a sigmoid

Write $\Delta_i = \beta \log \tfrac{\pi^{*}(y_i\mid x)}{\pi_{\text{ref}}(y_i\mid x)}$, so the cancelled form is $p^{*} = \dfrac{e^{\Delta_1}}{e^{\Delta_1} + e^{\Delta_2}}$. Multiply numerator and denominator by $e^{-\Delta_1}$:

$$
p^{*}(y_1 \succ y_2 \mid x) = \frac{e^{\Delta_1}\,e^{-\Delta_1}}{e^{\Delta_1}\,e^{-\Delta_1} + e^{\Delta_2}\,e^{-\Delta_1}} = \frac{1}{1 + e^{\,\Delta_2 - \Delta_1}}
$$

- The numerator becomes $1$; the denominator holds an exponential of the **difference** $\Delta_2 - \Delta_1$.

<!-- step -->

With $\sigma(z) = \tfrac{1}{1+e^{-z}}$, this is a sigmoid ($\Delta_i = \beta \log \tfrac{\pi^{*}(y_i\mid x)}{\pi_{\text{ref}}(y_i\mid x)}$):

$$
\boxed{\ \ p^{*}(y_1 \succ y_2 \mid x) = \sigma\!\Big(\beta \log \tfrac{\pi^{*}(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)} - \beta \log \tfrac{\pi^{*}(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}\Big)\ \ }
$$

- A **sigmoid of the difference of two log-ratios** — Bradley-Terry, with the policy in place of the reward model.

---

<!-- layout: section-break -->
<!-- align: center -->

## DPO Derivation P4/4: The loss function and gradient

---

<!-- valign: top -->
<!-- title: center -->
## The DPO loss

Minimize the negative log-likelihood of the observed preferences ($y_c \succ y_r$) -- this is making the probability more likely:

$$
\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\,\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\big[ \log p(y_c \succ y_r \mid x) \big]
$$

<!-- step -->

Plug in the sigmoid form of $p$, with the trainable policy $\pi_{\theta}$ as the implicit reward:

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\Big[ \log \sigma\big(\beta \log \tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta \log \tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big) \Big]
$$

This is directly differentiable — **no reward model, no sampling, no RL loop.**

---

<!-- valign: top -->
<!-- title: center -->
## The DPO gradient

Start by taking the gradient of the previous loss function.

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\nabla_{\theta}\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \log \sigma\big(\beta \log \tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta \log \tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big) \Big]
$$

Let $u = \big(\beta \log \tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \beta \log \tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\big)$ — the term inside $\sigma$.

<!-- step -->

Chain rule, starting with $\tfrac{d}{dz}\log\sigma(z) = \tfrac{\sigma'(z)}{\sigma(z)}$ as the inner operation:

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \tfrac{\sigma'(u)}{\sigma(u)}\,\nabla_{\theta} u \Big]
$$

<!-- step -->

Substitute $\sigma'(u) = \sigma(u)\big(1-\sigma(u)\big)$ and cancel $\sigma(u)$:

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \tfrac{\sigma(u)\big(1-\sigma(u)\big)}{\sigma(u)}\,\nabla_{\theta} u \Big] = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \big(1-\sigma(u)\big)\,\nabla_{\theta} u \Big]
$$

---

<!-- valign: top -->
<!-- title: center -->
## The DPO gradient

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \big(1-\sigma(u)\big)\,\nabla_{\theta} u \Big]
$$

The reflection identity $1-\sigma(u) = \sigma(-u)$ flips the argument, giving the compact form:

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \sigma(-u)\,\nabla_{\theta} u \Big]
$$

<!-- step -->

Now differentiate $u$, where $-u = \beta\log\tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)} - \beta\log\tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)}$. Only $\pi_{\theta}$ depends on $\theta$ — $\pi_{\text{ref}}$ is frozen, so its $\log$ terms drop:

$$
\nabla_{\theta} u = \beta\big[\nabla_{\theta}\log\pi_{\theta}(y_c\mid x) - \nabla_{\theta}\log\pi_{\theta}(y_r\mid x)\big]
$$

<!-- step -->

Substitute $\nabla_{\theta} u$ and write $\sigma(-u)$ out in full:

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}} = -\,\beta\,\mathbb{E}_{(x,y_c,y_r)}\Big[ \sigma\big(\beta\log\tfrac{\pi_{\theta}(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)} - \beta\log\tfrac{\pi_{\theta}(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)}\big)\,\big[\nabla_{\theta}\log\pi_{\theta}(y_c\mid x) - \nabla_{\theta}\log\pi_{\theta}(y_r\mid x)\big] \Big]
$$

---

<!-- valign: top -->
<!-- title: center -->
<!-- animate: bullets -->
## The DPO gradient

$$
\begin{aligned}
\nabla_{\theta}\mathcal{L}_{\text{DPO}} &= -\,\beta\, \mathbb{E}_{(x,y_c,y_r)}\Big[\, \underbrace{w}_{\text{how wrong}} \cdot \big( \nabla_{\theta}\log\pi_{\theta}(y_c\mid x) - \nabla_{\theta}\log\pi_{\theta}(y_r\mid x) \big)\,\Big] \\[6pt]
&\hspace{6em}\text{where}\quad w = \sigma\big(r_\theta(x,y_r) - r_\theta(x,y_c)\big)
\end{aligned}
$$

- **Weight $w \in (0,1)$** is larger when the model is *more wrong* — when it ranks the rejected response above the chosen the loss is higher.
- **The delta-gradient bracket** raises the likelihood of $y_c$ and lowers that of $y_r$.
- **$\beta$** scales the step, trading correct ordering against drift from $\pi_{\text{ref}}$ (downstream of the KL penalty in the RLHF objective).

---

<!-- rows: 55/45 -->
<!-- title: center -->
## Recap: the DPO derivation

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

## DPO weaknesses, variants, and implementation details

---

<!-- columns: 55/45 -->
## A subtle risk: the chosen probability can fall

The DPO loss only cares about the **margin** between the chosen and rejected log-ratios — not their absolute values.
So the model can lower the loss by pushing the *rejected* probability down **faster** than the chosen, even while the **chosen probability also falls**.

Mediated through the partition function $Z(x)$ in the derivation.

- Called **likelihood displacement** [@razin2024unintentional] [@ren2024learning]; posited to push probability toward unaddressed, off-distribution behaviors.
- A reason some practitioners add an SFT term on the chosen response, or use fixes like Cal-DPO [@xiao2024cal] / AlphaPO [@gupta2025alphapo].


|||

![Sketch of likelihood displacement in DPO.](assets/dpo_displacement.png)

<!-- step -->

**In a real run:** [Olmo 1B DPO](https://wandb.ai/rlhf-book/core/runs/fzy8k8go) — the chosen/rejected reward margin widens, yet the *chosen* log-prob itself can still drift down.


---

## The $\beta$ parameter and the static KL

$\beta$ sets the strength of the KL constraint relative to reward maximization:

- **Large $\beta$** → policy stays close to $\pi_{\text{ref}}$; it barely moves.
- **Small $\beta$** → policy is free to deviate; it can **over-optimize**.

Crucially, DPO's final KL distance is **static**: it steps directly to the *optimal* solution implied by the dataset and the chosen $\beta$. 
Online RL instead takes steps based on freshly sampled batches and a per-sample KL penalty. Some RL runs even include dynamically adjusted KL controllers.

---

<!-- valign: top -->
## A zoo of direct alignment algorithms

Each variant tweaks the loss to fix a limitation — often a one-line change. I started calling all the variants Direct Alignment Algorithms (DAAs). Two to start:

<!-- step -->

- **Identity Preference Optimization (IPO)** [@azar2024general] — softens the preference probability away from Bradley-Terry to curb overfitting.

$$ \mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x,y_w,y_l)}\!\left[\left(\log\tfrac{\pi_\theta(y_w\mid x)\,\pi_{\text{ref}}(y_l\mid x)}{\pi_\theta(y_l\mid x)\,\pi_{\text{ref}}(y_w\mid x)} - \tfrac{1}{2\beta}\right)^2\right] $$

<!-- step -->

- **Conservative DPO (cDPO)** [@rafailov2024direct] (in the original DPO paper) — assumes the preference labels are noisy (flipped with probability $\varepsilon$) and softens the loss target accordingly.

<!-- step -->

- **Offset DPO (ODPO)** [@amini2024direct] — requires the chosen to beat the rejected by a margin offset, scaled by how strong the preference is.

---

<!-- valign: top -->
## A zoo of direct alignment algorithms

Two more that **drop the reference model** entirely:

<!-- step -->

- **Odds Ratio Preference Optimization (ORPO)** [@hong2024reference] — adds an odds-ratio pull toward the chosen response, folded directly into the SFT loss. The odds ratio uses only $\pi_\theta$, which allows dropping the reference model.

$$ \mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} - \lambda\,\mathbb{E}\!\left[\log \sigma\!\left(\log \tfrac{\text{odds}_\theta(y_w\mid x)}{\text{odds}_\theta(y_l\mid x)}\right)\right], \quad \text{odds}_\theta(y) = \tfrac{\pi_\theta(y)}{1-\pi_\theta(y)} $$

<!-- step -->

- **Simple Preference Optimization (SimPO)** [@meng2025simpo] — length-normalizes the reward into an average log-prob, with a target margin $\gamma$. $\tfrac{1}{|y|}$ normalizes by length; $\gamma$ is a target margin; no $\pi_{\text{ref}}$.

$$ \mathcal{L}_{\text{SimPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\tfrac{\beta}{|y_w|}\log \pi_\theta(y_w\mid x) - \tfrac{\beta}{|y_l|}\log \pi_\theta(y_l\mid x) - \gamma\right)\right] $$

<!-- step -->

The algorithm matters **far less** than the base model and the data. Still, many papers continued to make minor algorithmic tweaks. Many more exist than were on these slides.

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

<!-- animate: bullets -->
## DAAs work with synthetic preference data

These algorithms need *feedback* data, not necessarily *human* feedback data — AI feedback works just as well.

- Most modern DPO uses preferences labeled by a strong model. **UltraFeedback** [@cui2023ultrafeedback] was the first prominent *open* synthetic preference dataset and helped kickstart the DPO era; Tülu 3 [@lambert2024t], SmolLM 2 [@allal2025smollm2], and others followed with larger synthetic-feedback recipes.
- **On-policy data** (some completions from the model you are tuning) helps the contrastive loss optimize the right token space within a complex post-training recipe (studied this specifically in Tülu 3).
- **Delta Learning** [@geng2025the]: later work argues the *gap* between chosen and rejected matters more than which models produced them (e.g. Qwen3-32B chosen vs Qwen3-0.6B rejected). Used this in Olmo 3!

---

<!-- columns: 50/50 -->
## DPO vs. RL: offline vs. online

**DPO and other DAAs**

- Train on a fixed dataset collected ahead of time.
- Simpler, more stable, fast to iterate on data.
- Limited by the **coverage** of that dataset — a slightly lower performance ceiling.

|||

**PPO / policy gradient is online**

- Generate fresh completions during training, score with a reward model.
- Can explore new regions → often higher peak performance [@ivison2024unpacking] [@xu2024dpo] [@tajwar2024preference].
- More compute, more moving parts (four models in memory), more engineering complexity.

---

## Where does this leave us today?

If so many models have used DPO well and it's so simple, why does it seem like it comes up so infrequently -- especially at the frontier?

---

<!-- columns: 52/48 -->
## Hypotheses for DPO's role today

- **Olmo 3** (Nov 2025) / **SmolLM 3** (Jul 2025) -- used DPO over reasoning traces after SFT to boost performance in a simple pipeline.
- **NVIDIA Nemotron 3** (Dec 2025) -- *Mixed Preference Optimization*: DPO + Binary Classifier Optimization in one offline stage, over data scored by a generative reward model.
- **Liquid AI LFM2** (Nov 2025) -- length-normalized DPO on semi-online data, before a final RLVR pass.

|||

---

<!-- columns: 52/48 -->
## Hypotheses for DPO's role today

- **Olmo 3** (Nov 2025) / **SmolLM 3** (Jul 2025) -- used DPO over reasoning traces after SFT to boost performance in a simple pipeline.
- **NVIDIA Nemotron 3** (Dec 2025) -- *Mixed Preference Optimization*: DPO + Binary Classifier Optimization in one offline stage, over data scored by a generative reward model.
- **Liquid AI LFM2** (Nov 2025) -- length-normalized DPO on semi-online data, before a final RLVR pass.

|||

**How I see things:**

- DPO works well in wonky, distillation-heavy recipes like Olmo -- e.g. a model with a spikier distribution, with training data from many teacher models.
- DPO still works in other settings, but most labs have the engineering resources to do other things with higher peak performance.
- TLDR: DPO is a path to a good/solid model, but not to the best model. And DPO may do less in *cleaner* training recipes.