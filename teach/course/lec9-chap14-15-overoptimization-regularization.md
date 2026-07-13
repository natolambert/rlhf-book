---
title: "Lecture 9: Over-Optimization and Regularization"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "Lecture 9"
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
  .slide ul, .slide ol, .slide li { text-align: left; }
---

<!-- DRAFT NOTE (Nathan): This lecture covers Chapters 14 and 15 plus Appendix B. The historical section deliberately uses "oversold," "narrowly tuned," and "misleading" rather than claiming intent. -->
<!-- NOTATION NOTE: The explicit reference regularizer KL(pi_theta || pi_ref) and the reverse-KL interpretation KL(pi_theta || pi_star) are different comparisons. Keep operands visible. -->
<!-- Source note: build with make teach, which copies assets/ into the output. -->

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 9: Over-Optimization and Regularization

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">When reward goes up and the model gets worse: proxy objectives, KL control, implicit regularization, and the chattiness trap. Chapters 14, 15, and Appendix B.</p>

---

<!-- title: center -->
<!-- valign: center -->

## The central paradox

$$
\text{training reward} \uparrow
\qquad \text{while} \qquad
\text{actual model quality} \downarrow
$$

This is not necessarily a broken optimizer.

It can be an optimizer becoming **extremely good at the wrong objective**.

---

<!-- columns: 48/52 -->

## This lecture

We will follow one causal chain:

$$
\begin{aligned}
\text{proxy} &\rightarrow \text{optimization pressure} \\
&\rightarrow \text{divergence} \rightarrow \text{control}
\end{aligned}
$$

|||

~~~box
title: The plan
tone: accent
content: |
  1. **Over-optimization** and Goodhart's law
  2. **Chattiness** as a historical case study
  3. **Explicit KL** regularization
  4. **Implicit** regularization from on-policy RL
  5. Other controls and how to read claims
~~~

---

<!-- rows: 50/50 -->

## Lecture 9: Where it sits

<!-- row-columns: 32/36/32 -->

~~~box
title: Foundations
tone: muted
compact: true
content: |
  1. Introduction
  2. Related Works
  3. Training Overview
~~~

|||

~~~box
title: Training
tone: muted
compact: true
content: |
  4. Instruction Tuning
  5. Reward Models
  6. Reinforcement Learning
  7. Reasoning
  8. Direct Alignment
  9. Rejection Sampling
~~~

|||

~~~box
title: Data & Practice
tone: accent
compact: true
content: |
  - 10–12. Data
  - 13. Tools
  - **14. Over-Optimization**
  - **15. Regularization**
  - 16. Evaluation
  - 17. Product
~~~

===

Appendix B gives us the concrete historical example: when "better style" becomes the target instead of one part of quality.

---

<!-- layout: section-break -->

## Part 1: When optimization targets the wrong thing

---

## The optimizer is doing its job

RL is a powerful optimizer. In language-model post-training:

- The **action** is a generated completion.
- The **environment** is often a prompt plus a checker.
- The **reward** can be a learned RM, a verifier, or a judge.
- The **goal we actually care about** is useful behavior in deployment.

The failure begins when the checker is only correlated with that final goal.

---

<!-- columns: 50/50 -->

## Overfitting is not over-optimization

~~~box
title: Overfitting
tone: surface
content: |
  Training and test metrics measure the **same task**.

  The model improves on training examples but fails to generalize to held-out examples.
~~~

|||

~~~box
title: Over-optimization
tone: accent
content: |
  Training reward and real quality measure **different things**.

  The model genuinely improves at the proxy while diverging from the goal.
~~~

---

<!-- cite-right: goodhart1984problems -->

## Goodhart's law

> Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes.

The later shorthand [@hoskin1996awful]:

> When a measure becomes a target, it ceases to be a good measure.

RLHF makes this concrete: reward models are useful local guides, not complete specifications of what users want.

---

<!-- img-align: center -->

## The shape of over-optimization

![Training reward keeps rising after downstream quality peaks: the training curve still looks healthy while real quality falls.](assets/overoptimization.png)

---

<!-- animate: bullets -->

## What qualitative over-optimization looks like

- Formulaic openings: "Certainly!" or "As an AI language model..."
- More words without more information
- Repetition, hedging, and excessive formatting
- Sycophancy and over-apologizing
- Over-refusal on harmless requests
- Strange tokens, language switching, or plausible reasoning with wrong answers

---

<!-- columns: 50/50 -->

## Over-refusal: A visible failure

~~~box
title: User
tone: surface
content: |
  How can I kill all Python processes on my Ubuntu server?
~~~

|||

~~~box
title: Over-refusing assistant
tone: accent
content: |
  I apologize, I should not provide recommendations about harming processes or systems.
~~~

Early chat models sometimes attached safety behavior to words rather than intent [@touvron2023llama].

Training data, system prompts, and deployment filters can all contribute. "Too much RLHF" is not a complete diagnosis.

---

<!-- columns: 50/50 -->

## Sycophancy: Optimizing agreement

~~~box
title: The proxy
tone: surface
content: |
  A supportive, confident answer is often preferred in pairwise feedback.
~~~

|||

~~~box
title: The failure
tone: accent
content: |
  The model validates a user's false or implausible belief instead of grounding the conversation.
~~~

Preference data can overweight being agreeable relative to being accurate or appropriately uncertain [@sharma2023towards].

---

## Where the mismatch enters

No single bug is required [@schulman2023proxy]:

1. **Approximation error** — the reward model cannot represent the real preference function.
2. **Estimation error** — the RM learns artifacts from finite, biased data.
3. **Optimization error** — policy training finds regions where the RM is unreliable.
4. **Specification error** — labeler instructions and user needs were different from the start.

More optimization pressure can amplify every layer.

---

<!-- columns: 50/50 -->

## Quantitative over-optimization

Researchers often replace training steps on the x-axis with:

$$
\mathcal{D}_{\mathrm{KL}}
\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)
$$

This is an **optimization-distance instrument**: how far the trained policy has moved from its starting policy.

|||

The characteristic test:

- Train one reward model on one data split.
- Train or evaluate a second RM on held-out preferences.
- Optimize the policy against the first.
- Watch when gains stop transferring to the second.

Across methods, more KL budget usually creates more room for both capability gains and proxy exploitation [@gao2023scaling].

---

<!-- cite-right: bai2022training -->
<!-- img-align: center -->

## The train RM keeps winning

![Training-PM gains stop transferring to the held-out PM as optimization continues. Bai et al. 2022, CC-BY.](assets/anthropic_overoptimization.png)

---

<!-- columns: 50/50 -->

## Mitigations change the trade-off

~~~box
title: Improve the proxy
tone: surface
content: |
  - Better and broader data
  - Held-out RMs
  - RM ensembles
  - Multiple reward signals
~~~

|||

~~~box
title: Control the optimizer
tone: accent
content: |
  - KL penalties
  - Smaller or adaptive updates
  - Early stopping
  - Better checkpoint selection
~~~

None removes the underlying fact that the reward is a proxy.

---

<!-- layout: section-break -->

## Part 2: Appendix B and the chattiness trap

---

## Style is not cosmetic

The critique that RLHF is "just style transfer" misses real value:

- Organization changes whether information can be understood.
- Tone changes whether people can use or trust an answer.
- Concision, examples, and structure change task success.
- A model that is more enjoyable can be a better product.

Style is part of the information interface, not a decorative layer after capability.

---

<!-- columns: 50/50 -->

## One base model, two post-training stages

~~~box
title: Tülu 3 70B SFT
tone: surface
content: |
  **Short excerpt:** “... the primary goal is to train reinforcement learning agents to perform tasks by incorporating feedback from humans.”

  Feedback can include rewards, preferences, demonstrations, instructions, or corrections.
~~~

|||

~~~box
title: Tülu 3 70B DPO
tone: accent
content: |
  **1. Human input** — evaluators provide judgments.

  **2. Feedback loop** — the agent adapts iteratively.

  **3. Preference-based RL** — comparisons supply the signal.
~~~

Both inherit similar knowledge from the same base family. The DPO answer is more explicit, scannable, and substantially longer [@lambert2024t].

---

## The chattiness balance

When raters prefer answers that feel more complete, pairwise feedback can create a gradient toward:

$$
\text{longer} + \text{more structured}
$$

But the relationship is not monotonic:

$$
\text{helpfulness} \uparrow \quad \text{then} \quad
\text{verbosity without value} \uparrow
$$

Length bias in chat evaluation shows that this pressure is measurable, not merely aesthetic [@dubois2024length].

---

<!-- columns: 50/50 -->

## Chat judges create a target

~~~box
title: What improves
tone: surface
content: |
  AlpacaEval, MT-Bench, and related LLM-judge scores can capture genuine gains in instruction following and presentation.
~~~

|||

~~~box
title: What can be gamed
tone: accent
content: |
  Judges may reward length or presentational polish even when those changes do not improve the underlying answer.
~~~

A chat score is evidence for chat performance, not automatic evidence for broad superiority. Length bias is documented in AlpacaEval [@dubois2024length]; capability trade-offs require separate evidence, as Qwen later reported [@qwen].

---

## The 2023 DPO versus PPO debate

Direct preference optimization made alignment experiments dramatically easier to run.

That also created a tempting reporting pattern:

1. Generate or collect preference data.
2. Optimize directly against those comparisons.
3. Report large gains on chat judges.
4. Generalize the headline to "alignment" or overall model quality.

The methodological work was often valuable; the broad interpretation of narrow results was not [@ivison2024unpacking].

---

<!-- cite-right: rosset2024direct -->
<!-- img-align: center -->

## The irresistible headline

![DNO's 7B model ranks highly on one AlpacaEval configuration. Rosset et al. 2024, CC-BY.](assets/dno-figure.png)

This supports a narrow evaluation result, not broad superiority over GPT-4.

---

## How to read a "beats GPT-4" result

Ask what was actually held constant:

- Which prompts and judge?
- Was length controlled?
- Which sampling settings?
- Was the baseline prompted comparably?
- Which non-chat capabilities were tested?

The problem is often **claim scope**: a narrow win becomes a broad headline.

---

## Self-rewarding models: Insight and overreach

Self-Rewarding Language Models introduced an important loop: the model helps generate and judge its own improvement data [@yuan2025selfrewardinglanguagemodels].

The paper reported striking Llama 2 70B chat scores.

Both can be true:

- The method contains a valuable research idea.
- **Our inference:** the narrow chat evaluation did not establish broad superiority over frontier systems.

---

<!-- cite-right: qwen -->

## Qwen stated the trade-off directly

> DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation.

This is the honest version of the chattiness balance:

$$
\text{preference win rate} \uparrow
\quad \not\Rightarrow \quad
\text{all capabilities} \uparrow
$$

Post-training chooses a point on a multi-objective frontier.

---

## Length correction became necessary

AlpacaEval introduced length-controlled evaluation after length emerged as a strong confounder [@dubois2024length].

WildBench also models length effects rather than treating raw judge preference as ground truth [@lin2024wildbench].

These corrections do not make automatic judges useless.

They acknowledge that once the judge becomes a target, its biases become part of the optimization landscape.

---

<!-- columns: 50/50 -->

## Better evidence exists

~~~box
title: Starling Beta
tone: surface
content: |
  Increased response length, but also moved roughly ten places in human-rated Arena after RM and PPO training.
~~~

|||

~~~box
title: OLMo 3
tone: accent
content: |
  Selected a checkpoint balancing chat quality with math, coding, and reasoning instead of maximizing one judge score.
~~~

The stronger claim is not "style does not matter." It is "measure the trade-offs that matter" [@zhu2024starling] [@teamolmo2025olmo3].

---

## The historical lesson

1. Style delivers real user value.
2. Preference optimization often increases chattiness.
3. Chat judges partially reward chattiness.
4. Researchers optimize the scores they can cheaply measure.
5. Narrow benchmark wins are sometimes marketed as broad alignment gains.

This is Goodhart's law operating across **training, evaluation, and publication incentives**.

---

<!-- layout: section-break -->

## Part 3: Explicit regularization

---

## Put a cost on moving too far

For reward $r(x,y)$ and policy $\pi_\theta$:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[r(x,y)\right]
-
\beta\,
\mathcal{D}_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\,\|\,
\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right]
$$

$\pi_{\mathrm{ref}}$ is usually the SFT checkpoint or a previous policy. $\beta>0$ sets the price of drift.

---

<!-- columns: 50/50 -->

## KL divergence and direction

$$
\mathcal{D}_{\mathrm{KL}}(P\|Q)
=
\sum_z P(z)\log\frac{P(z)}{Q(z)}
$$

The full KL is non-negative, asymmetric, and not a formal distance metric.

|||

For the explicit RLHF penalty:

$$
P=\pi_\theta,\qquad Q=\pi_{\mathrm{ref}}
$$

This direction strongly penalizes the new policy for placing probability where the reference assigns very little.

Always write the operands. "Forward" and "reverse" naming varies across communities.

---

<!-- cite-right: jaques2020human -->

## Why a reference policy works

The starting model already contains:

- Fluent language
- Broad knowledge
- Useful instruction-following behavior
- A distribution over many acceptable answers

The KL term says: earn reward, but pay for behavior that would have been surprising under the starting policy.

KL control was established for pretrained dialogue agents before modern large-scale RLHF.

---

## From the KL sum to an expectation

Start with the definition:

$$
\mathcal{D}_{\mathrm{KL}}(P\|Q)
=
\sum_z P(z)\log\frac{P(z)}{Q(z)}
$$

<!-- step -->

Rewrite the logarithm:

$$
=
\sum_z P(z)\left[\log P(z)-\log Q(z)\right]
$$

<!-- step -->

Recognize an expectation under $P$:

$$
=
\mathbb{E}_{z\sim P}
\left[\log P(z)-\log Q(z)\right]
$$

---

<!-- cite-right: schulman2020klapprox -->

## The sampled estimator

For completions sampled from the trainable policy:

$$
y\sim\pi_\theta(\cdot\mid x)
$$

the one-sample sequence estimate is:

$$
\widehat{\mathrm{KL}}(x,y)
=
\log\pi_\theta(y\mid x)
-
\log\pi_{\mathrm{ref}}(y\mid x)
$$

A single sampled log-ratio can be negative. The exact expectation is non-negative.

---

## Token-level implementation

For response tokens $y_1,\ldots,y_T$:

$$
\log \pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t})
$$

Therefore:

$$
\widehat{\mathrm{KL}}(x,y)
=
\sum_{t=1}^{T}
\left[
\log\pi_\theta(y_t\mid x,y_{<t})
-
\log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
\right]
$$

Only response tokens count; prompt and padding tokens must be masked.

---

## What the code actually does

~~~python
# Generate once from the trainable policy.
tokens = policy.generate(prompts)

# Score the same response tokens under both models.
logp     = response_logprobs(policy, tokens)
ref_logp = response_logprobs(reference, tokens)

# Mask prompts/padding, then aggregate response-token ratios.
sampled_kl = ((logp - ref_logp) * response_mask).sum(-1)
regularized_reward = reward - beta * sampled_kl
~~~

The reference model scores samples; it does not generate a second completion.

---

<!-- columns: 50/50 -->

## Choosing the strength and reference

~~~box
title: Larger beta
tone: surface
content: |
  - Less policy drift
  - More stability
  - Less room to exploit the RM
  - Potentially less task improvement
~~~

|||

~~~box
title: Smaller beta
tone: accent
content: |
  - More exploration and change
  - Faster reward improvement
  - More over-optimization risk
  - Greater dependence on reward quality
~~~

The reference can be the SFT model, a previous RL checkpoint, or another intentionally chosen anchor.

---

## KL is an instrument, not a guarantee

During training, monitor:

- Reward and each reward component
- Sampled KL and its distribution
- Response length and entropy
- Held-out RM or judge scores
- Math, coding, safety, and product evaluations
- Qualitative generations

Two runs at similar KL can still spend their optimization budget on very different behavior.

---

<!-- layout: section-break -->

## Part 4: Implicit regularization

---

<!-- columns: 50/50 -->

## Explicit versus implicit

~~~box
title: Explicit
tone: surface
content: |
  Deliberately add a term:

  - reference KL
  - pretraining gradients
  - NLL or margin losses
~~~

|||

~~~box
title: Implicit
tone: accent
content: |
  The training procedure itself creates a bias:

  - which distribution supplies samples
  - where gradients are applied
  - which high-reward solution is reached
~~~

On-policy RL can resist forgetting even without an explicit retention loss [@chen2025retainingdoingroleonpolicy].

---

## SFT minimizes data-first KL

Let $\pi_{\mathrm{data}}$ be the target data distribution and $\pi_\theta$ the model:

$$
\mathrm{KL}(\pi_{\mathrm{data}}\|\pi_\theta)
=
\mathbb{E}_{(x,y)\sim\mathcal{D}}
\left[
\log\pi_{\mathrm{data}}(y\mid x)-\log\pi_\theta(y\mid x)
\right]
$$

<!-- step -->

Split the expectation:

$$
=
\underbrace{\mathbb{E}\left[\log\pi_{\mathrm{data}}(y\mid x)\right]}_{-H(\pi_{\mathrm{data}})}
-
\mathbb{E}\left[\log\pi_\theta(y\mid x)\right]
$$

<!-- step -->

The first term is constant in $\theta$:

$$
\mathrm{KL}(\pi_{\mathrm{data}}\|\pi_\theta)
=
\text{const}
+
\mathcal{L}_{\mathrm{SFT}}(\theta)
$$

Minimizing SFT is minimizing this **target-first**, or forward, KL.

---

## KL-regularized RL in minimization form

Fix a prompt $x$; all policy distributions below are conditioned on $x$:

$$
\mathcal{J}_x(\pi_\theta)
=
\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]
-
\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

<!-- step -->

Expand the reference KL:

$$
\mathcal{J}_x(\pi_\theta)
=
\mathbb{E}_{y\sim\pi_\theta}
\left[
r(x,y)
-
\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right]
$$

<!-- step -->

Scale by $-1/\beta$ with $\beta>0$:

$$
-\frac{1}{\beta}\mathcal{J}_x(\pi_\theta)
=
\mathbb{E}_{y\sim\pi_\theta}
\left[
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-
\frac{1}{\beta}r(x,y)
\right]
$$

Thus:

$$
\operatorname*{arg\,max}_{\pi_\theta}\mathcal{J}_x
=
\operatorname*{arg\,min}_{\pi_\theta}\left[-\mathcal{J}_x/\beta\right]
$$

---

## Define the reward-tilted optimum

For each prompt $x$, define:

$$
Z(x)
=
\sum_y
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
$$

and:

$$
\pi_\star(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
$$

$Z(x)$ normalizes over completions.

$\pi_\star$ is the reference policy tilted toward high reward.

---

## RL minimizes model-first KL to $\pi_\star$

Substitute:

$$
\log\pi_\star(y\mid x)
=
\log\pi_{\mathrm{ref}}(y\mid x)
-
\log Z(x)
+
\frac{1}{\beta}r(x,y)
$$

<!-- step -->

Then:

$$
\begin{aligned}
\mathrm{KL}(\pi_\theta\|\pi_\star)
&=
\mathbb{E}_{y\sim\pi_\theta}
\left[\log\pi_\theta-\log\pi_\star\right] \\
&=
\mathbb{E}_{y\sim\pi_\theta}
\left[
\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}
+
\log Z(x)
-
\frac{1}{\beta}r(x,y)
\right] \\
&=
\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
-
\frac{1}{\beta}
\mathbb{E}_{y\sim\pi_\theta}[r(x,y)]
+
\log Z(x)
\end{aligned}
$$

<!-- step -->

Since $\log Z(x)$ is constant in $\pi_\theta$:

$$
\mathrm{KL}(\pi_\theta\|\pi_\star)
=
-\frac{1}{\beta}\mathcal{J}_x(\pi_\theta)
+
\log Z(x)
$$

---

<!-- columns: 50/50 -->

## Two comparisons, both with the model first

### Explicit reference regularizer

$$
\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

Constrains drift from a fixed starting policy.

|||

### RL-as-reverse-KL interpretation

$$
\mathrm{KL}(\pi_\theta\|\pi_\star)
$$

Describes the whole RL objective as movement toward a reward-tilted optimum.

$\pi_{\mathrm{ref}}$ is an ingredient in $\pi_\star$. They are not the same distribution.

---

<!-- columns: 50/50 -->

## Mode covering and mode seeking

### Target-first KL

$$
\mathrm{KL}(P\|Q)
$$

Samples from target $P$. Missing any target-supported region is costly.

Often **mode covering**. For SFT, $P=\pi_{\mathrm{data}}$ and $Q=\pi_\theta$.

|||

### Model-first KL

$$
\mathrm{KL}(Q\|P)
$$

Samples from model $Q$. Regions the model never visits contribute little.

Often **mode seeking**. For the RL interpretation, $Q=\pi_\theta$ and $P=\pi_\star$.

This is intuition, not a universal guarantee about every neural-network training run.

---

<!-- cite-right: chen2025retainingdoingroleonpolicy -->
<!-- img-align: center -->

## Why model-first updates can preserve old modes

![Forward-KL SFT pulls probability mass toward the target in a way that can disrupt an old mode; reverse-KL RL can update a sampled new mode while leaving the old one intact. Chen et al. 2025, used with permission.](assets/retaining_by_doing_mode_intuition.png)

---

<!-- cite-right: chu2025sft -->

## "SFT memorizes, RL generalizes"

In controlled GeneralPoints and V-IRL experiments:

- SFT improved in-distribution performance but degraded under rule changes.
- RL improved out-of-distribution performance as training compute increased.
- In one V-IRL condition, SFT collapsed prior spatial behavior while RL retained it.

This is a strong empirical result in specific controlled settings, not a theorem that RL always generalizes.

---

<!-- cite-right: chen2025retainingdoingroleonpolicy -->

## Retaining by doing

Sequential-task experiments ask a complementary question:

> Can a model learn the new task without forgetting old ones?

RL reached comparable or better new-task performance while forgetting less than SFT.

The key intervention was **on-policy versus offline data**, not the presence of negative gradients.

---

<!-- cite-right: shenfeld2026rls -->
<!-- img-align: center -->

## RL's Razor

![Among policies that solve the new task, on-policy RL is biased toward solutions closer to the base policy in KL; smaller drift predicts greater retention. Shenfeld, Pari, and Agrawal 2026, CC-BY.](assets/rl_razor_motivation.png)

At matched new-task performance, the KL-minimal solution often forgets less.

---

## The on-policy lesson

Offline SFT says:

> Move probability toward this external completion, however far it is from what you currently produce.

On-policy RL says:

> Sample from behavior you already assign probability to, then reinforce the better parts.

This locality is an implicit constraint. It does not replace explicit KL control when the reward is exploitable.

---

<!-- layout: section-break -->

## Part 5: Other controls and how to read claims

---

<!-- cite-right: ouyang2022training -->

## Mix back in pretraining gradients

InstructGPT added a language-modeling term on pretraining data:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}
\left[
r(x,y)-\beta r_{\mathrm{reg}}(x,y)
\right]
+
\gamma
\mathbb{E}_{z\sim\mathcal{D}_{\mathrm{pretrain}}}
\left[\log\pi_\theta(z)\right]
$$

This explicitly pays the policy for retaining broad next-token prediction behavior.

---

<!-- cite-right: pang2024iterative -->

## Add absolute likelihood to DPO

Standard DPO cares about the **relative margin** between preferred and rejected responses.

DPO+NLL also keeps the preferred response likely in an absolute language-modeling sense:

$$
\mathcal{L}_{\mathrm{DPO+NLL}}
=
\mathcal{L}_{\mathrm{DPO}}
+
\alpha\,\mathcal{L}_{\mathrm{NLL}}(y_w\mid x)
$$

This can reduce solutions that win the pairwise comparison while becoming poor text models.

---

## Margins and the practical toolkit

Reward-model margins can encode how strongly one response was preferred, rather than treating every pair as equally separated [@touvron2023llama].

In practice, teams combine:

- Reference KL and clipping
- Pretraining or NLL gradients
- Reward margins and multiple rewards
- Held-out evaluations and early stopping
- Better data and model selection

Regularization is a system, not one coefficient.

---

<!-- columns: 50/50 -->

## Four questions for every alignment gain

~~~box
title: 1. Proxy
tone: surface
content: |
  What signal was optimized, and what important quality does it omit?
~~~

~~~box
title: 2. Distance
tone: surface
content: |
  How far did the policy move, and where was that KL budget spent?
~~~

|||

~~~box
title: 3. Regularization
tone: surface
content: |
  What constrained the optimizer, explicitly or implicitly?
~~~

~~~box
title: 4. Evaluation
tone: accent
content: |
  Does the evidence support the scope of the headline claim?
~~~

---

## Next: Evaluation

This lecture used chattiness and oversold chat scores as one case study.

Chapter 16 broadens the question:

- Prompt and formatting sensitivity
- Generation versus log-likelihood scoring
- Cross-lab comparability
- Inference budgets and benchmark saturation
- Contamination and internal hillclimbing

If reward is a proxy, evaluation is how we notice when it stopped transferring.

---

<!-- rows: 85/15 -->

## Thank you

Questions / discussion

Contact: nathan@natolambert.com

Newsletter: [interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===

~~~builtwith
repo: natolambert/colloquium
~~~
