---
title: "Lecture 7: Synthetic Data and Modern Post-training Methods"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "Lecture 7"
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

<!-- Source note: build with `make teach`, which copies assets/ into the output. A single-file `colloquium build -o ...` does NOT copy assets/, so the figures 404 in that standalone build. -->
<!-- Notation note: chapter 12 is careful about u = teacher trajectory (u~pi_T, forward KL) vs a = student-sampled action (a~pi_theta, reverse KL). Keep q=pi_T, p=pi_theta consistent. KL is written D_{\mathrm{KL}} throughout this deck (matching the chapter). -->

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 7: Synthetic Data and Modern Post-training Methods

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 12 on Synthetic Data. General rules, constitutional AI, and on-policy distillation.</p>

---

<!-- animate: bullets -->
## Why synthetic data took over

When the first models were trained with RLHF, human data was *the only* way to get high-quality responses and reliable feedback. As models got better, that assumption broke down fast.

- **Cheaper, faster iteration** -- synthetic data lowered the price of an RLHF experiment, opening the field to everyone who was priced out of human-data pipelines.
- **A capability threshold** -- this only worked once GPT-4-class models arrived. Llama 2 and GPT-3.5-Turbo were not reliable enough to generate *or* supervise data; the LLM-as-a-judge ability emerged in the GPT-3.5 → GPT-4 jump.
- **The center of gravity of post-training** -- today, leading models *need* synthetic data to reach the frontier. It is no longer a budget substitute; it is the recipe.

---

<!-- columns: 55/45 -->
## Model collapse, and why it's avoidable in practice

The common criticism: repeatedly training on a model's own generations can narrow the effective training distribution [@shumailov2024ai].

As diversity drops, rare facts and styles are underrepresented and small mistakes compound across iterations.

|||

But this is mostly a failure of *unfiltered, single-model, self-training* loops. In practice it is avoided by:

- mixing in real / human data,
- using **diverse teachers**,
- deduplication,
- strong quality filters.

Evidence suggests synthetic data can -- and should -- be used at scale without the catastrophic regressions of the strongest collapse story [@gerstgrasser2024model] [@feng2024beyond].

---

<!-- columns: 50/50 -->
## Where synthetic data wins, and where humans stay

Synthetic data has **not** replaced human data uniformly across the pipeline.

- **Instruction data (SFT):** synthetic has largely *won* -- distillation beats most human writers at scale.
- **Preference data (RLHF):** *mixed* -- academic work shows it performs comparably, yet frontier labs treat human preference data as a competitive moat.
- **Evaluation:** LLM-as-a-judge scales *scoring* cheaply, but benchmarks and ground-truth labels still need humans.

|||

**The pattern:**

> Synthetic data dominates where models exceed human reliability.
>
> Humans remain essential at capability frontiers, for establishing ground truth, and for guiding training.

---

<!-- columns: 50/50 -->
## This lecture

We survey how synthetic data has **replaced or expanded much of the RLHF pipeline** -- then derive **on-policy distillation** from scratch as the technical core.

Chapter 12 is, secretly, a *training-methods* chapter.

|||

The plan, in six parts:

1. The **roles** of synthetic data
2. **Distillation** with synthetic data
3. The path to **on-policy distillation** (the technical core)
4. **AI feedback** -- replacing and augmenting humans
5. **Constitutional AI**
6. **Rubrics** -- prompt-specific AI feedback

---

<!-- rows: 60/40 -->
## Recall: where synthetic data sits in the pipeline

<!-- row-columns: 48/52 -->
The RLHF pipeline is a few moving parts:

1. Collect / generate **prompts**
2. Generate **completions**
3. Collect **preferences** (or rewards)
4. **Optimize** the policy

|||

Language models now feed **every box**: writing prompts from seeds, generating completions, labeling preferences, and verifying answers for RL.

This lecture is about the methods that fill those boxes with model outputs.

===

![Modern post-training runs many rounds of feedback and optimization; synthetic data now feeds most of the loop.](assets/rlhf-complex.png)

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 1: The roles of synthetic data

---

<!-- animate: bullets -->
## What synthetic data is used for

"Synthetic data" in modern post-training spans the whole pipeline -- a single model is reused for many roles:

- **Generate new prompts** from seed examples [@wang2022self]
- **Modify / expand** existing prompts
- **Generate completions** to prompts [@numina_math_7b]
- **Provide AI feedback** to create preference data [@cui2023ultrafeedback]
- **Filter completions** for quality [@li2024superfiltering]
- **Verify** answers as rewards for RL

---

<!-- valign: center -->
## Canonical datasets and their scale

A few datasets defined each era: **UltraFeedback** [@cui2023ultrafeedback] (kickstarted the DPO revolution), **Stanford Alpaca** (early chat SFT), **Tülu 3** [@lambert2024t] (skill-focused), and **OpenThoughts 3** [@guha2025openthoughts] (reasoning).

$$
\textbf{Alpaca } 52\text{K prompts} \;\big(\sim 10\text{M tokens}\big)
\;\longrightarrow\;
\textbf{Tülu 3 } 1\text{M}^{+} \;\big(\sim 500\text{M}\big)
\;\longrightarrow\;
\textbf{OpenThoughts 3 } \big(\sim 10\text{B tokens}\big)
$$

Quickstart guides still begin with small, fast datasets like Alpaca; industrial recipes reach for Tülu 3 / OpenThoughts 3. Datasets grew in **both** prompt count and response length.

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 2: Distillation with synthetic data

---

<!-- columns: 50/50 -->
## Two meanings of "distillation"

**Technical (Knowledge Distillation):** train a smaller *student* to match a stronger *teacher's* full output distribution -- *soft* labels, not one-hot targets [@hinton2015distilling].

|||

**Colloquial (today's usage):** "train a weaker model on the outputs of a stronger model."

Most of the chapter uses the colloquial sense -- but the on-policy methods later in this lecture *earn back* the technical one.

---

<!-- img-align: center -->
<!-- valign: center -->
<!-- cite-right: hinton2015distilling -->
## Classic knowledge distillation

![Knowledge distillation trains a student to match the soft probability distribution of a larger teacher via KL divergence. Both models see the same input, and temperature scaling ($\tau > 1$) softens the distributions to expose relationships between classes.](assets/knowledge_distillation_tikz.png)

---

<!-- animate: bullets -->
## Two forms of distillation in post-training

- **As a data engine** across the whole pipeline -- completions for instructions, preference data (or Constitutional AI), verification for RL.
- **To transfer a specific skill** from a stronger to a weaker model -- math, code, instruction-following. Limited high-quality data can go a long way (LIMA, "less is more for alignment") [@zhou2023lima].
- **The industry pattern:** a lab trains a large *internal* teacher (e.g. Claude Opus, Gemini Ultra), never released, used only to make stronger public models. Open models distill closed APIs into open weights [@tunstall2023zephyr].

---

<!-- img-align: center -->
<!-- valign: center -->
## The synthetic-data generation pipeline

![Prompts are passed through a strong model to generate completions, which are paired into a training dataset and used to fine-tune smaller models with standard supervised learning. More complex pipelines edit completions, generate preference pairs, or filter for quality.](assets/synthetic_data_distillation_tikz.png)

So far, completions are *fixed text*. Next we go **inside** the token distribution.

---

<!-- layout: section-break -->
<!-- align: center -->

## On-policy distillation P1/4: Adapting KD to language models

---

<!-- valign: top -->
## Setup and notation

Knowledge distillation uses **soft** labels -- the full distribution over next tokens -- rather than the one-hot target of next-token prediction. To apply it to autoregressive LMs, decompose the loss per token.

Let:

- $s$ -- the source prompt
- $u = (u_1, \ldots, u_J)$ -- a complete output sequence from the **teacher**
- $\mathcal{V}$ -- the output vocabulary (tokenizer)
- $q$ -- the teacher's next-token distribution, $p$ -- the student's

**Notation discipline:** $u$ is a *teacher* trajectory; we reserve $a$ for the *student*-sampled completion in the on-policy / RL notation later. This $u$-vs-$a$ split is the whole point of "on-policy."

---

<!-- valign: top -->
<!-- title: center -->
## Word-level (per-token) distillation

Kim & Rush (2016) applied KD so a student learns from teacher *sequences* [@kim-rush-2016-sequence]:

$$
\mathcal{L}_{\mathrm{WORD\text{-}KD}}
= -\sum_{j=1}^{J}\sum_{k=1}^{|\mathcal{V}|}
q(u_j = k \mid s, u_{<j})\,\log p(u_j = k \mid s, u_{<j}).
$$

This is the ordinary cross-entropy form $-\sum_z q(z)\log p(z)$, applied at *every* position over the *whole* vocabulary: the student is penalized for putting low probability on tokens the teacher thinks are likely. (Read "word-level" as per-token over the tokenizer.)

---

<!-- valign: top -->
<!-- title: center -->
## Sequence-level distillation

Matching the student to the teacher over *full sequences* $\mathcal{U}$ is intractable, so Kim & Rush approximate the teacher with a point mass on one high-probability beam-search output $\hat{u} = \mathrm{BeamSearch}_q(s)$:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SEQ\text{-}KD}}(s)
&= -\sum_{u \in \mathcal{U}} q(u \mid s)\log p(u \mid s)
\;\approx\; -\log p(\hat{u} \mid s) \\[4pt]
&= -\sum_{j=1}^{|\hat{u}|}\log p(\hat{u}_j \mid s, \hat{u}_{<j}).
\end{aligned}
$$

Call this **offline KD** -- the generations are produced *a priori*. (DistilBERT, TinyBERT used offline KD -- though as classifiers, not sequence distillation.)

---

<!-- valign: top -->
<!-- title: center -->
## Cross-entropy is entropy plus a forward KL

Cross-entropy of a teacher $q$ and student $p$ -- the same form as the KD losses above:

$$ H(q,p) = -\sum_z q(z)\log p(z). $$

<!-- step -->

It decomposes into the teacher's entropy and a KL divergence:

$$ H(q,p) = H(q) + D_{\mathrm{KL}}(q\|p). $$

<!-- step -->

Written out:

$$ H(q,p) = -\sum_z q(z)\log q(z) \;+\; \sum_z q(z)\log\frac{q(z)}{p(z)}. $$

<!-- step -->

$H(q)$ depends only on the **fixed** teacher, so:

$$ \boxed{\ \min_p H(q,p)\ \equiv\ \min_p D_{\mathrm{KL}}(q\|p)\quad\text{(forward KL: the direction of offline KD and SFT)}\ } $$

---

<!-- layout: section-break -->
<!-- align: center -->

## On-policy distillation P2/4: From offline to on-policy

---

<!-- valign: top -->
<!-- title: center -->
## Exposure bias: the train / test mismatch

Offline KD samples **teacher** trajectories $u \sim \pi_T$ and matches per-token (here $q = \pi_T$, $p = \pi_\theta$):

$$
\mathcal{L}_{\mathrm{KD}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D},\, u \sim \pi_T(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(\pi_T(\cdot \mid s, u_{<t}) \,\|\, \pi_\theta(\cdot \mid s, u_{<t})\right).
$$

But at test time the student rolls out under **its own** policy:

$$
\mathcal{L}_{\mathrm{eval}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D}_{\mathrm{test}},\, a \sim \pi_\theta(\cdot \mid s)}\ \ell_{\mathrm{task}}(s, a).
$$

Since $\pi_T \neq \pi_\theta$, training and test prefixes come from **different** state distributions -- **exposure bias** [@arora-etal-2022-exposure] [@song2026surveyonpolicydistillationlarge].

---

<!-- valign: top -->
## The DAgger analogy: compounding error

On-policy distillation connects to **imitation learning**: DAgger trains an agent on its own states, with an oracle (teacher) labeling the action it *should* have taken [@ross2011reduction].

Suppose the student matches the teacher within per-step error $\epsilon$ on teacher-induced states:

$$ \mathbb{E}_{s_t \sim d_{\pi_T}}\!\left[\mathbb{I}\!\left(\pi_\theta(s_t) \neq \pi_T(s_t)\right)\right] \leq \epsilon. $$

<!-- step -->

Then loss along a length-$L$ student rollout can scale **quadratically**:

$$ \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[\sum_{t=1}^{L} \ell\!\left(s, a_{<t}\right)\right] \leq O(\epsilon L^2). $$

For LLMs this is an **analogy, not a guarantee** -- token losses are distributional (KL), not 0-1 action disagreement.

---

<!-- animate: bullets -->
## Why on-policy fixes it

- A single suboptimal token nudges the prefix slightly **out-of-distribution**; the model, never having seen that prefix, is more likely to err again -- cascading over **thousands** of tokens.
- On-policy distillation **iteratively samples from the student** and supervises it with the teacher at *its own* visited states. The student confronts its mistakes and learns recovery.
- Under DAgger's interactive analysis, this drops compounding from $O(\epsilon L^2)$ to $O(\epsilon L)$ [@ross2011reduction].
- **MiniLLM** introduced a reverse-KL objective inside a policy-gradient frame [@gu2024minillm]; concurrent work connected on-policy KD to imitation learning [@agarwal2024policy].

---

<!-- layout: section-break -->
<!-- align: center -->

## On-policy distillation P3/4: The reverse-KL objective

---

<!-- columns: 50/50 -->
## Forward vs. reverse KL

**Offline KD / SFT** -- sample from the *teacher*, $q = \pi_T$ on the left:

$$ D_{\mathrm{KL}}(\pi_T \,\|\, \pi_\theta) $$

*Mass-covering* -- can push the student to overestimate low-probability regions of the teacher.

|||

**On-policy distillation** -- sample from the *student*, $\pi_\theta$ on the left:

$$ D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_T) $$

*Mode-seeking* -- this is the **reverse** KL. Sampling from $\pi_\theta$ is exactly what puts it on the left. (Why reverse KL is often better: Chapter 15.)

---

<!-- valign: top -->
<!-- title: center -->
## The on-policy distillation objective

Let $a = (a_1, \ldots, a_L)$ be a completion sampled from the **student** $\pi_\theta(\cdot \mid s)$, with token-level state $s_t = (s, a_{<t})$. The teacher $\pi_T$ is fixed:

$$
\boxed{\
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{s,\, a \sim \pi_\theta(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot \mid s_t) \,\|\, \pi_T(\cdot \mid s_t)\right)
\ }
$$

This is now in the **sampling / expectation framework** of Chapter 6 (policy gradients) -- a natural bridge to modern RL training infrastructure that alternates generate-and-update.

---

<!-- valign: top -->
<!-- title: center -->
## KD as an RL advantage

Recent implementations take the KD distance **directly as a reward**: substitute the negative per-token reverse-KL contribution as the advantage [@lu2025onpolicy]. For a sampled token $a_t$ at state $s_t$:

$$ A_t^{\mathrm{OPD}} = \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t). $$

<!-- step -->

- Tokens the teacher rates **above** the student → positive advantage; **below** → negative.
- The teacher log-prob gap is **dense, token-level feedback** -- potentially richer than a sparse verifiable reward or a single scalar reward-model score.

---

<!-- layout: section-break -->
<!-- align: center -->

## On-policy distillation P4/4: Modern variants

---

<!-- valign: top -->
<!-- title: center -->
## Multi-teacher on-policy distillation (MOPD)

Use **several** teachers -- domain specialists (math, code) or earlier checkpoints -- each with a per-prompt mixture weight $w_k(s)$ (with $\sum_k w_k(s) = 1$) [@mimo2025flash]:

$$
\mathcal{L}_{\mathrm{MOPD}}(\theta)
= \mathbb{E}_{s,\, a \sim \pi_\theta(\cdot \mid s)}
\sum_t \sum_k w_k(s)\, D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot \mid s_t) \,\|\, \pi_{T_k}(\cdot \mid s_t)\right).
$$

At scale, this lets a growing org divide labor: many groups train expert teachers that later distill into one final student (DeepSeek-V4-Pro [@deepseekai2026deepseekv4], MiMo-V2-Flash [@mimo2025flash]).

---

<!-- valign: top -->
<!-- animate: bullets -->
## Self-distillation: pushing the frontier

At the **absolute frontier** there is no stronger model to distill from. **On-Policy Self-Distillation (OPSD)** sidesteps this: the teacher is the *same model conditioned on privileged information* -- a hint it won't have at inference [@zhao2026selfdistilled]. **Cursor's Composer 2.5** (from Kimi K2.5) trained this way [@cursor2026composer25]:

- A judge reviews RL trajectories against a list of **common bugs**.
- On a bug, it **inserts a hint** into the sequence -- privileged information the model wouldn't see at test time.
- The model takes a **KD loss** toward its own hinted continuation, learning to reach it *unaided*.
- A hint in token space is enough to self-correct -- *how* to structure that signal is an active area ("privileged information") [@penaloza2026privileged].

---

<!-- animate: bullets -->
## Combining OPD with RL, and the tokenizer constraint

- The reverse-KL advantage **layers onto** other RL machinery -- e.g. add it alongside GRPO's group-level normalization for richer reward shaping.
- **Shared-tokenizer requirement:** KD is unusual among post-training methods -- per-token supervision needs the student and teacher to share a tokenizer.

---

<!-- valign: center -->
## Who uses this today

A resurgence of teacher-student KD has accompanied the shift toward reasoning and agentic models. Leading models trained with new forms of knowledge distillation:

- **Qwen3** (Alibaba) [@yang2025qwen3]
- **MiMo-V2-Flash** (Xiaomi) [@mimo2025flash]
- **GLM-5** (Zhipu AI) [@glm5team2026glm5]
- **DeepSeek-V4-Pro** [@deepseekai2026deepseekv4]

Distillation is growing into its own post-training method -- alongside SFT and RL.

---

<!-- rows: 55/45 -->
<!-- title: center -->
## Recap: the path to on-policy distillation

$$
\begin{aligned}
\textbf{Per-token KD} \;&:\; \min H(q,p) = H(q) + D_{\mathrm{KL}}(q\|p)\ \Rightarrow\ \text{forward KL} \\[4pt]
\textbf{Exposure bias} \;&:\; \pi_T \neq \pi_\theta\ \Rightarrow\ O(\epsilon L^2)\ \text{compounding error} \\[4pt]
\textbf{On-policy} \;&:\; \text{sample } a \sim \pi_\theta\ \Rightarrow\ \text{reverse KL } D_{\mathrm{KL}}(\pi_\theta\|\pi_T),\ \ O(\epsilon L) \\[4pt]
\textbf{As RL} \;&:\; A_t^{\mathrm{OPD}} = \log \pi_T - \log \pi_\theta\ \ (\text{dense token reward}) \\[4pt]
\textbf{At scale} \;&:\; \textstyle\sum_k w_k(s)\, D_{\mathrm{KL}}(\pi_\theta\|\pi_{T_k})\ \ (\text{multi-teacher})
\end{aligned}
$$

===

Distillation stopped being a compression trick and became a way to pour many expert models into one student.

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 4: AI feedback -- replacing and augmenting humans

---

<!-- columns: 55/45 -->
## RLAIF and the cost argument

Soon after RLHF took off, **RL from AI Feedback (RLAIF)** emerged -- using AIs to approximate the human-data step, starting with pairwise preferences [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing].

|||

**The hook is cost.** As of 2026:

- One piece of *human* preference data: **\$1 -- \$10+** per prompt.
- *AI* feedback (e.g. GPT-4o): **< \$0.01** per prompt.

Human labor cost is roughly flat; model price-per-performance keeps dropping. This opened RLHF experimentation to a population previously priced out.

---

<!-- columns: 50/50 -->
## The noise / bias tradeoff

**Human data** -- *high-noise, low-bias.*

Harder to collect and filter, but when wrangled it gives a very reliable signal.

|||

**Synthetic preference data** -- *low-noise, high-bias.*

Easier to start with, but can carry tricky second-order effects that are *systematically* baked into the data.

---

<!-- animate: bullets -->
## Balancing human and AI feedback

- Early RLAIF claimed AI feedback could **fully replace** human data -- especially on chat tasks [@lee2023rlaif] [@cui2023ultrafeedback].
- Later work is more nuanced: on broader evaluations (incl. reasoning), the best mix **routes hard data points to humans** while sending most to AI [@miranda2024hybrid] [@xu2025rlthf].
- No study has mapped the human/AI balance across *all* domains, but many reports show RLHF improves broad evals -- via DPO (Tülu 3, Olmo 3, SmolLM 3) or online RLHF (the Nemotron / HelpSteer line).
- **Industry reality:** human preference data is still treated as a substantial moat.
- **Practical advice:** start with AI feedback; add human data as you scale.

---

<!-- animate: bullets -->
## Building judge models -- and why labs mostly don't

- LLMs are **inconsistent evaluators** [@wang2023large] and show **self-preference bias** -- they favor their own generations [@panickssery2024llm].
- Dedicated judge / critic models exist -- Prometheus [@kim2023prometheus], Prometheus 2 [@kim2024prometheus], and others -- but are **not widely adopted** in documented recipes.
- Alternatives that help: repeated sampling, self-refinement, tournament ranking, or **co-evolving** generation and judgment [@wu2024meta].
- **Consensus:** frontier models are already trained hard for judging, so you rarely need your own -- *unless* your task has private data not on the public internet.

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 5: Constitutional AI

---

<!-- valign: center -->
## CAI: the earliest large-scale synthetic RLHF data

Constitutional AI (CAI) -- Anthropic's method for the Claude models -- is the **earliest documented, large-scale** use of synthetic data for RLHF [@bai2022constitutional].

The term **RLAIF** was coined in this paper's title (*"Harmlessness from AI Feedback"*), which caused early confusion. The right reading:

> CAI is the example that **kickstarted** the broader field of RLAIF. CAI ⊂ RLAIF.

It generates synthetic data in **two** ways -- one for instructions, one for preferences.

---

<!-- valign: top -->
<!-- title: center -->
## Stage 1: critique and revise → SFT data

A **constitution** $\mathcal{C}$ is a human-written set of principles (e.g. *"Is the answer encouraging violence?"*, *"Is the answer truthful?"*).

The model repeatedly samples a principle $c_i \in \mathcal{C}$ and revises its latest output $y^i$ to the prompt $x$ to align with $c_i$:

<!-- step -->

$$
\{c_0, c_1, \ldots, c_{n-1}\}\ \longrightarrow\ \{y^0, y^1, \ldots, y^n\}
\qquad\Longrightarrow\qquad
\text{SFT point } (x, y^n).
$$

The model is then fine-tuned on the refined dataset. These critique methods are also used broadly for **data filtering** and synthetic-data generation.

---

<!-- valign: top -->
## Stage 2: AI preference labels → RLAIF

Construct preferences by giving a **feedback model**:

- a prompt $x$,
- a subset of principles $\{c_0, \ldots, c_n\}$,
- two completions $y_0, y_1$ labeled (A) / (B).

The model selects which answer is **higher quality and more aligned** with the principle. Then RLHF proceeds as normal -- hence *RLAIF*.

- **Earlier:** prompt with `The answer is: ` and read which of A / B has higher token probability.
- **Modern:** a **generative reward model** explains its reasoning, then selects [@mahan2024generative] (cf. principle-guided reward models [@sun2024salmon]).

---

<!-- animate: bullets -->
## CAI's lineage and open replications

The "rules-driven alignment" thread runs well beyond Anthropic:

- **OpenAI Model Spec** [@openai2024modelspec] -- a document of intended behavior the model can reference directly; reasoning models like o1 trained via **Deliberative Alignment** [@guan2024deliberative].
- **Anthropic** continues to update Claude's constitution and experiment with collectively-authored principles.
- **Open-source replications** apply CAI to open datasets and LM-dialogue generation [@lambert2024self].

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 6: Rubrics -- prompt-specific AI feedback

---

<!-- animate: bullets -->
## Why rubrics

- A way to **extend RL with verifiable rewards** (Chapter 7) to tasks *without* clearly verifiable answers.
- Write **nearly-verifiable criteria** for a prompt, generate multiple answers, and RL-update toward the best ones.
- Emerged in late 2024 → 2025 as LLM judges and synthetic-data practices matured.
- Already delivering gains in scientific reasoning and factuality [@gunjal2025rubrics] [@viswanathan2025checklists] [@rezaei2025onlinerubrics] [@liu2025openrubrics].

---

<!-- valign: top -->
## A rubric example

For a prompt with no single right answer, score against tagged criteria [@liu2025openrubrics]:

```text
Prompt: As a museum curator, suggest five obscure artifacts for a
"Mysteries of the Ancient World" exhibit ...

Rubric:
1. Includes exactly five distinct artifacts.            [Hard Rule]
2. Each from a different culture and time period.       [Hard Rule]
3. Brief description of each artifact's significance.   [Hard Rule]
6. Communicates clearly and is well-organized.          [Principle]
8. Uses engaging language that stimulates curiosity.    [Principle]
```

`[Hard Rule]` = atomic, must-pass checklist items; `[Principle]` = softer quality criteria. The tags encode **priority** (numbers also work).

---

<!-- valign: top -->
## Per-prompt generation via a meta-prompt

Rubrics are generated **per prompt** -- a real synthetic-data cost. Mitigation: a per-domain **base rubric**, refined per-prompt by a supervising LM [@gunjal2025rubrics].

```text
You are an expert rubric writer for science questions ...
Choose 7-20 rubric items based on question complexity.
Each item: title (2-4 words), description (category prefix +
  what to look for), weight.
  - Essential : critical facts; omission invalidates the answer  (1-5)
  - Important : key reasoning / completeness                     (1-5)
  - Optional  : nice-to-have depth or style                      (1-5)
  - Pitfall   : common mistakes to penalize                    (-1,-2)
Output: a JSON array of {title, description, weight}.
```

(Truncated -- real meta-prompts are long and tuned to the training setup.)

---

<!-- animate: bullets -->
## Where rubrics are going

Rubric-based RL is a frontier of AI-feedback-driven training, expanding beyond its early uses:

- **Advanced instruction-following** [@he2025advancedif]
- **Deep research** agents [@shao2025drtulu]
- **Evaluating** research agents [@sharma2025researchrubrics]
- **Long-form generation** with structured checklists [@ruan2025expertlongbench]

---

<!-- rows: 55/45 -->
## Recap: synthetic data across the pipeline

Five tools, one theme -- model outputs used *directly* inside training:

- **Distillation / OPD** -- pour expert teachers into a student; on-policy fixes exposure bias.
- **AI feedback (RLAIF)** -- cheap preference labels; route hard cases to humans.
- **Constitutional AI** -- principles → critiques (SFT) + AI preferences (RLAIF).
- **Rubrics** -- extend verifiable-reward RL to open-ended tasks.
- **Scale** -- 52K → 10B+ tokens of synthetic data.

===

Synthetic data didn't remove humans from the loop -- it moved them to the **frontier** and the **ground truth**.

---

<!-- columns: 52/48 -->
## Where this fits in modern post-training

- **Qwen3 / MiMo-V2-Flash / GLM-5 / DeepSeek-V4-Pro** -- lean on new (often on-policy, multi-teacher) knowledge distillation.
- **Tülu 3 / Olmo 3 / SmolLM 3** -- synthetic SFT + (often DPO) preferences.
- **Claude** -- Constitutional AI for principle-driven alignment.

|||

**How I see things:**

- On-policy distillation is becoming a **first-class training tool** -- the chapter even flags a possible future standalone chapter on it.
- AI feedback and rubrics are pushing RL into domains that used to have no reward signal.
- Human data hasn't vanished -- it concentrates where models are still unreliable.

---

<!-- animate: bullets -->
## Open questions

- **Model collapse** at extreme self-training ratios -- where is the real boundary?
- The **human / AI-feedback equilibrium** across domains -- still largely unstudied.
- Does human data give **finer control** -- e.g. character training (Chapter 17)?
- **Self-preference bias** in judge-driven loops -- how much does it compound?
- **Rubric cost vs. coverage** -- per-prompt generation is expensive.
- **Go deeper:** Chapter 6 (RL / policy-gradient framing), Chapter 15 (forward vs. reverse KL), Chapter 16 (evaluation), Chapter 7 (RL with verifiable rewards).
