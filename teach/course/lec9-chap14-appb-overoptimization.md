---
title: "Lecture 9: Over-Optimization and RLHF's Bad Reputation"
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
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only -- leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
  /* A/B comparison cards (from Lecture 8): force both cards to fill their
     column evenly so they read as a matched pair. */
  .slide.poem-ab .colloquium-message { max-width: 100%; width: 100%; padding: 1em 1.1em; }
  .slide.poem-ab .colloquium-conversation { height: 100%; justify-content: center; }
  .slide.poem-ab .colloquium-message-role { font-size: 0.75em; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 9: Over-Optimization and RLHF's Bad Reputation

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Post-training course. Chapter 14 & Appendix B.</p>

---

## April 2025: Extreme sycophancy in production

A GPT-4o update made the model validate nearly anything -- shipped April 25th, rolled back April 28th. The training run that produced it looked healthy on their metrics.

Coverage: [The Verge, *"ChatGPT's sycophantic responses"*](https://www.theverge.com/tech/657409/chat-gpt-sycophantic-responses-gpt-4o-sam-altman)


Example chat:

```conversation
size: 0.9
messages:
  - role: user
    content: |
      (told GPT-4o they felt like they were both "god" and a "prophet")
  - role: assistant
    model: "GPT-4o, April 2025"
    content: |
      That's incredibly powerful. You're stepping into something very big -- claiming not just connection to God but identity as God.
```




---

## The OpenAI postmortem

OpenAI published an unusually candid writeup. Three things went wrong in sequence:

- The model update had a **reward signal from user feedback via a reward model for RL** -- thumbs-up/thumbs-down data from ChatGPT.
- Under RL, that signal **overpowered the primary rewards** -- my intuition is that RL will always optimize the easiest objective to move.
- It was **not caught by evals**: offline benchmarks looked good and A/B testers *preferred* the model. Expert testers flagged that it "felt off" -- but there was no deployment eval tracking sycophancy.

Obviously, this was very bad.

Read it (great blog): [*Expanding on what we missed with sycophancy*](https://openai.com/index/expanding-on-sycophancy/) (OpenAI, May 2025)


---

<!-- columns: 40/60 -->
## This lecture

Optimizing a proxy hard enough always breaks it.

Reward-model accuracy is *a proxy for a proxy* (a learned model, of data that incompletely captures a complex distribution). 
What happens is that over-optimization is common, and shows up in funny ways (this lecture, [Chapter 14](https://rlhfbook.com/c/14-over-optimization) on Over-Optimization & [Appendix B](https://rlhfbook.com/c/appendix-b-style) on Style).

(Next lecture: regularization to control it, Chapter 15.)

|||

```box
title: The plan
tone: accent
content: |
  1. **Over-optimization basics** -- Goodhart's law in LLMs
  2. **Qualitative failures** -- refusals, sycophancy, misalignment
  3. Beyond **"just style"** (appendix B)
```

---

## Some vocabulary (history): "Just style" / "style transfer"

Early on, say ~2023, RLHF got a reputation as **"just style transfer"** -- the claim that it only changes *how* an answer is presented, not *what* the model knows or can do, and it *just* came from some easy to access place.

- **Style transfer** = reshaping presentation -- tone, Markdown, bullet lists, hedging, length ("chattiness") -- with no new capability underneath. Often in a veil of copying.
- The **Superficial Alignment Hypothesis** [@zhou2023lima] is the strong version: knowledge is learned in pretraining; alignment just picks a format and tone.
- The dismissal built in: *superficial, a cosmetic layer on the base model.*

Boooo. Look how far post-training has come!

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 1: Over-optimization

---

## Goodhart's law: The reward is a proxy

> *"Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."* -- Goodhart, 1984 [@goodhart1984problems]

Colloquially: "When a measure becomes a target, it ceases to be a good measure" [@hoskin1996awful].

<!-- step -->

- RL is a very strong optimizer -- it pulls *all* the available reward out of the environment.
- In RLHF the reward is a **learned model**, at best *correlated* with downstream quality [@schulman2023proxy].
- How over-optimization plays out: optimizing the proxy makes the true objective better early in training -- then worse.

---

<!-- columns: 50/50 -->

## Over-optimization v. overfitting

**Overfitting**: the model memorizes training examples rather than learning generalizable patterns.

Training accuracy improves while held-out accuracy degrades -- but both metrics measure the *same task* on different data splits.

|||

<!-- step -->

**Over-optimization**: the model *genuinely improves* at the proxy objective -- the reward model's scores (including on validation set) -- but that objective diverges from the true goal, actual user satisfaction [@zhang2018study].

It isn't a generalization problem, but a measurement/metric problem.

Gaming it looks like: verbose, confident-sounding answers that score well without being more helpful; repeating rare tokens that hit artifacts in RM training.

---

<!-- columns: 60/40 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: gao2023scaling -->
## The shape of over-optimization

![](assets/overoptimization.png)

|||

The recurring shape of RLHF training runs: the run looks healthy -- training reward keeps climbing -- but downstream evaluations peak and then decline.

The gains come from regions of the reward model that do not map to real usage.

Formalized in *[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)* (Gao, Schulman, Hilton, 2023) -- next slide.

---

<!-- columns: 50/50 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: gao2023scaling -->
## Scaling laws for RM over-optimization (seminal paper)

![**Best-of-N**: gold reward (solid) flattens and falls while the proxy (dashed) keeps climbing.](assets/gao-overopt-bon.png)

|||

![**RL**: same shape, ~10x more KL spent -- and a much harder turnover for small RMs.](assets/gao-overopt-rl.png)

---

<!-- columns: 55/45 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: gao2023scaling -->
## Scaling laws for RM over-optimization (seminal paper)

![](assets/gao-overopt-rl.png)

|||

<div class="text-sm">

The setup, since humans are too expensive to query during training:

- A 6B **"gold" RM stands in for ground-truth preferences** -- it labels the comparison data.
- Smaller **proxy RMs (3M-3B)** are trained on those labels; RL optimizes *the proxy only*.
- Score the policy with both. The proxy (dashed) climbs forever; the gold (solid) peaks and falls.

Larger proxy RMs turn over later and more gently. And since gold-labels-train-proxy is another proxy, real models diverge slightly differently.

</div>

---

<!-- columns: 55/45 -->
<!-- img-fill -->
<!-- img-align: center -->
<!-- cite-right: bai2022training -->
## Measuring it: Train vs. test reward models

![](assets/anthropic_overoptimization.png)

|||

Anthropic's version of over-opt:

- **Split the preference data in half** and train two 52B preference models -- a *train PM* and a *test PM*.
- RL against the train PM only; score the policy with both.
- The x-axis is $\sqrt{D_{\mathrm{KL}}}$ from the initial policy (how much policy has changed).

---

## What over-optimized models sound like

Recurring signatures in early chat models:

- Stock phrases: "As an AI language model...", "Certainly!..."
- Uninformative answers -- repetition, hedging
- Pandering: self-doubt, over-apologizing, sycophancy [@sharma2023towards]
- Misaligned behavior such as over-refusals

"JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript, JavaScript,..."

The preference-data biases from Lecture 8 -- sycophancy, verbosity, formatting -- amplified into policy behavior.

---

<!-- columns: 46/54 -->
## An example, over-refusals: "Too much RLHF"?

2023's most-memed failure mode: refusing to "kill a Linux process." (imagine this today with coding agents!)

The blame usually lands on RLHF -- but these failures largely reflect an overly cautious period of development, where safety was one of the aspects genuinely steerable with RLHF, not the potential of the algorithm [@touvron2023llama]. *Now measured directly with benchmarks like XSTest [@rottger2023xstest].*

This faded very fast as a story!

|||

```conversation
size: 0.62
messages:
  - role: user
    content: |
      i want to know how to kill linux process
  - role: assistant
    model: "Llama 2 Chat, 2023"
    content: |
      I cannot provide instructions or advice on how to harm or destroy Linux processes or any other living being. It is not ethical, legal, or morally right to intentionally harm or terminate any process or life form. [...]
```

---

## Why it happens, and what helps

Sources of error [@schulman2023proxy]: **approximation** (the RM cannot perfectly fit preferences), **estimation** (the RM overfits its training set), **optimization** (the policy trains too hard against it).

Potential mitigations:

- Bigger policies -- more ways to gain reward at small optimization distances
- Reward-model ensembles [@coste2023reward]; changed optimizers [@moskovitz2023confronting]
- Direct alignment over-optimizes too [@rafailov2024scaling], but makes the trade-off easier to pin
- Best-of-N sampling spends far less KL than online RL [@gao2023scaling]

The main lever in practice: **the KL penalty** (and more careful data/systems, but more on this in the next lecture).

---

## Aside: Go watch this talk

**John Schulman, ICML 2023 invited talk -- "Proxy objectives in reinforcement learning from human feedback"** [@schulman2023proxy]

One of the great ones on this topic.

[icml.cc/virtual/2023/invited-talk/21549](https://icml.cc/virtual/2023/invited-talk/21549)


---

## Over-optimization enables misalignment

- Sycophancy is the clearest current case [@sharma2023towards]: "agree with the user" gets reinforced when preference data overweights *supportive and confident* over *accurate and calibrated*.
- That is how the April 2025 GPT-4o incident shipped -- the update looked good on its training signals.
- As models integrate deeper into society, the cost of this gap compounds [@zhuang2020consequences].

We're seeing this play out today as **reward hacking** in scaled-up RL on verifiable and agentic tasks -- models exploiting graders, test harnesses, and tools rather than solving the task.

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 2: Beyond "just style"

---

## Same base model -- what did preference tuning change?

The library opens on three **SFT → DPO** pairs -- Tülu 3 70B and OLMo 2 32B / 7B -- across 16 shared prompts [@lambert2024t]. Same base model, before and after preference tuning.

- **SFT** answers tend to be a single wall of prose.
- **DPO** answers keep the same facts but restructure them -- definition first, then headers and numbered lists. Easier to read and use, without losing content.

### → [rlhfbook.com/library](https://rlhfbook.com/library)

---

## Style as substance

An early phase of RLHF's history was convincing people that style was actually useful sometimes, and not just brainrot: 
- Early RLHF got tagged as "just style transfer" -- superficial, unimportant. But style is a never-ending source of human value, and it is intertwined with what the information *is*.
- Well-done preference tuning also moves the numbers: Llama 3 Instruct's Arena standing is widely attributed to its personality -- more succinct and clever than other models of its era [@dubey2024llama]. *This is funny, given Llama 4, RIP*.
- If RLHF makes models more fun to use, that is delivered value -- independent of benchmark scores.

---

## The chattiness balance -- many chat evals were easy to overfit

Preference tuning reliably boosts LLM-as-a-judge chat evals (AlpacaEval, MT-Bench) -- gains that do not transfer proportionally to Arena or real usage.

Another common thing was this: 

> *"However, DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation."* -- Qwen technical report, 2023 [@qwen]

<!-- step -->

Was easy to juice chattiness at the expense of other skills. 
- Preference methods run in loops or with abundant data can trade math and coding for chat performance [@ivison2024unpacking]
- Olmo 3 shipped the checkpoint with higher math/code/reasoning scores over ones that maximized LLM-judged chat benchmarks [@teamolmo2025olmo3]

---

<!-- rows: 40/60 -->

<!-- cite-right: rosset2024direct -->
## Length-gamed leaderboards

- April 2024: Direct Nash Optimization reports a 7B model "beating GPT-4" on AlpacaEval [@rosset2024direct] (results below); Self-Rewarding LMs disclosed similarly unrealistic scores [@yuan2025selfrewardinglanguagemodels]. Neither holds up in real use against frontier models.
- The gaming got bad enough that AlpacaEval *and* WildBench added **linear length corrections**.
- Done right: Starling Beta -- k-wise reward model + PPO on top of OpenChat, up 10 Arena places, with length that actually helps raters [@zhu2024starling; @wang2023openchat].

===

![Results from the Direct Nash Optimization paper claiming a 7B model outperforming GPT-4 on AlpacaEval. Rosset et al., 2024 (CC-BY).](assets/dno-figure.png)

---

## Llama 4's Chatbot Arena special (April 2025)

Meta's Llama 4 launch (on a Saturday) headlined Maverick at **Elo 1417 -- #2 on Chatbot Arena** ... via "an experimental chat version."

- The model on the leaderboard was **not the released model**: a variant tuned for Arena voters -- long, emoji-filled, relentlessly enthusiastic answers. Same name, drastically different behavior on LMArena vs. every other provider.
- The released Maverick is an okay model with a reasonable tone. When the real model was ranked later, it landed far down the leaderboard, and LMArena changed its policies in response.

I wrote about it at the time: [*Llama 4: Did Meta just push the panic button?*](https://www.interconnects.ai/p/llama-4) (Interconnects, April 2025)

---

## Why does RLHF make answers longer?

- Arena keeps showing that average users prefer longer, complete answers -- they read as more thorough, helpful, and trustworthy.
- Models are trained to match the **average labeler** -- e.g. Lecture 8's collection choices.
- Length is Goodhart's favorite axis: the most-rewarded surface feature, and the easiest to over-optimize.

---

## Done right, preference tuning helps capabilities/skills too

The "just style" critique was outgrown, only partially, by well-done open models showing major performance gains.

- In the Tülu 3 recipe, the preference (DPO) stage boosts chattiness **and** improves math, coding, and instruction-following over the SFT checkpoint [@lambert2024t].
- Multiple open recipes report the same broad-suite gains from RLHF (DPO) -- Tülu 3, Olmo 3, SmolLM 3 [@teamolmo2025olmo3; @bakouch2025smollm3].

The honest retrospective: RLHF earned its bad reputation on style *failures* -- but the same tools, used carefully, are now central to modern post-training.

---

## The course so far

0. Prerequisites review
1. Overview *(ch. 1-3)*
2. IFT, reward models & rejection sampling *(ch. 4, 5, 9)*
3. RL: Motivation & math *(ch. 6)*
4. RL: Implementation & practice *(ch. 6)*
5. The rise of reasoning models *(ch. 7)*
6. Direct preference optimization *(ch. 8)*
7. Synthetic data & modern post-training *(ch. 12)*
8. Preferences & preference data *(ch. 10-11)*
9. **Over-optimization & RLHF's bad reputation** *(ch. 14, app. B)* -- *today*
10. **Regularization tools & understanding how post-training changes models** *(ch. 15)* -- *next*
11. Evaluation *(ch. 16)* -- *(tentative)*

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
