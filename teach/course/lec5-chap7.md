---
title: "Lecture 5: Reasoning Training & Inference-Time Scaling"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com"
  center: "Lecture 5"
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

# Lecture 5: Reasoning Training & Inference-Time Scaling

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 7</p>

---

<!-- rows: 50/50 -->
## Lecture 5: Reasoning training & inference-time scaling

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
  6. Reinforcement Learning
  7. **Reasoning**
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
tone: muted
compact: true
content: |
  A. Key Definitions
  B. Style Benchmarks
  C. References
```

|||

```box
title: Lectures
tone: surface
compact: true
content: |
  1. Overview (Ch. 1-3)
  2. IFT, RM, RS (Ch. 4,5,9)
  3. RL Theory (Ch. 6 pt 1)
  4. RL Practice (Ch. 6 pt 2)
  **5. Reasoning (Ch. 7)**
```

---

## From RL to reasoning

Lectures 3-4 covered the **math and implementation** of policy gradient RL for language models: PPO, GRPO, loss aggregation, async training.

This lecture: **where those algorithms go when you scale them up on verifiable problems** -- and the wave of models that resulted.

Two parts:

1. **The reasoning model landscape** -- key 2025 models, grouped by what they teach
2. **What changes in the recipe** -- implementation decisions that differ from standard RLHF RL

---

## What this lecture covers

```box
title: Lecture outline
tone: accent
content: |
  1. **RLVR recap** -- RLHF vs RLVR scoring, the feedback loop
  2. **The reasoning model landscape** -- key models grouped by what they teach us
  3. **What changes in the recipe** -- implementation decisions that differ from standard RLHF RL
  4. **Looking ahead** -- where reasoning training is going
```

---

## The LeCun cake

At NeurIPS 2016, Yann LeCun introduced the cake metaphor:

> If intelligence is a cake, the bulk of the cake is unsupervised learning, the icing on the cake is supervised learning, and the cherry on the cake is reinforcement learning.

With modern language models, the analogy is complete:

- **Self-supervised learning** on internet data = the bulk of the cake
- **Supervised fine-tuning** for instructions = the icing
- **Reinforcement learning** (RLHF, then RLVR) = the cherry on top

---

## The LeCun cake

At NeurIPS 2016, Yann LeCun introduced the cake metaphor:

> If intelligence is a cake, the bulk of the cake is unsupervised learning, the icing on the cake is supervised learning, and the cherry on the cake is reinforcement learning.

With modern language models, the analogy is complete:

- **Self-supervised learning** on internet data = the bulk of the cake
- **Supervised fine-tuning** for instructions = the icing
- **Reinforcement learning** (RLHF, then RLVR) = the cherry on top

Reasoning models graduated RL from **cherry-on-top to a load-bearing component** of the training stack.

---

<!-- columns: 50/50 -->
## RLHF vs RLVR: How reward changes everything

**RLHF** -- subjective scoring:

> *Explain opportunity cost in economics.*
>
> Scoring requires judging clarity, accuracy, completeness -- all learned preferences with no definitive answer.

|||

**RLVR** -- verifiable scoring:

> *What is the sum of all primes < 20?*
>
> `extracted_answer == 77` → Reward = 1
>
> *Write `fib(n)` returning the nth Fibonacci number.*
>
> `assert fib(10) == 55` → All tests pass → Reward = 1

No learned reward model needed.

---

## The RLVR feedback loop

<!-- cite-right: lambert2024t -->
<!-- img-align: center -->

![RLVR in the form of an RL feedback loop. Instead of a reward model, a verification function scores completions.](assets/rlvr-system.png)

The RL algorithms (PPO, GRPO) are the same as lectures 3-4. The key change: **reward comes from a verification function**, not a learned model.

---

<!-- layout: section-break -->

## The reasoning model landscape

---

## How to read the model table

25+ reasoning model reports landed in 2025 alone. Rather than chronological, we group by **what each model teaches us**:

- **The pioneer** -- DeepSeek R1 cracked open the door
- **The replicators** -- Open-Reasoner-Zero, Phi-4: is the recipe reproducible?
- **End-to-end pipelines** -- MiMo, OLMo 3: pretraining → post-training as one system
- **Toggleable reasoning** -- Llama-Nemotron, Qwen 3: reasoning as a product mode
- **Stability and scale** -- Skywork OR-1, MiniMax-M1, Magistral: making long RL runs work

---

<!-- columns: 55/45 -->
## DeepSeek R1: The catalyst

<!-- cite-right: guo2025deepseek -->

The anchor release for the open reasoning wave.

**R1-Zero**: Pure RL on a base model. No SFT warm-start. Showed that large-scale RL *alone* can induce chain-of-thought reasoning.

**The full R1 recipe**: Cold-start SFT → large-scale RL → distillation of smaller models.

Open weights, 671B MoE.

|||

![Training reward curve for DeepSeek R1-Zero showing emergent reasoning behavior.](assets/deepseek-r1-zero-figure1-training.png)

---

## DeepSeek R1: What it taught us

<!-- cite-right: guo2025deepseek -->

**R1-Zero** proved that RL alone produces reasoning behavior:

- Emergent self-verification and backtracking
- Thinking tokens appear without being taught
- Strong math/code gains

---

## DeepSeek R1: What it taught us

<!-- cite-right: guo2025deepseek -->

**R1-Zero** proved that RL alone produces reasoning behavior:

- Emergent self-verification and backtracking
- Thinking tokens appear without being taught
- Strong math/code gains

But R1-Zero also had problems:

- Language mixing mid-reasoning
- Poor formatting and readability
- Inconsistent output structure

The full R1 recipe re-introduced cold-start SFT to fix these, then scaled RL further. Also released distilled smaller models -- distillation as an alternative path to RL.

---

## What thinking tokens look like

<!-- cite-right: guo2025deepseek -->

**DeepSeek V3** (standard model) responds directly:

> *Goldfish Dreams* -- In a bowl of glass so clear and round, / A tiny goldfish swims without a sound...

**DeepSeek R1** (reasoning model) first generates `<thinking>` tokens:

> `<thinking>` Okay, the user wants a short poem about a goldfish. Let me brainstorm goldfish-related imagery. Goldfish are often in bowls, so maybe start with the setting. Words like "gilded bowl" or "crystal globe"... Need to check rhythm and flow. Avoid clichés but keep it accessible. `</thinking>`
>
> *Gilded Dreams in Liquid Glass* -- A gilded bowl, a crystal globe, / Where tiny tides in silence flow...

For harder problems, thinking can be **thousands of tokens** before answering.

---

## Open-Reasoner-Zero: The minimalist replication

<!-- cite-right: hu2025openreasonerzero -->

If DeepSeek R1 proved the concept, Open-Reasoner-Zero proved it was **reproducible**.

- Fully open: model, data, and code
- Vanilla PPO with GAE ($\lambda=1, \gamma=1$) and simple rule-based rewards
- No KL penalty
- Showed the recipe is not a DeepSeek-specific trick

One of the clearest "minimalism wins" results. Start here if you want to understand the basic recipe.

---

## Phi-4: Small model, careful recipe

<!-- cite-right: abdin2025phi4 -->

14B parameters (Microsoft). Excels at STEM reasoning despite small size.

Key lesson: **model quality and data curation can compensate for scale**.

- Curated set of "teachable" prompts and synthetic reasoning demonstrations
- Short phase of outcome-based RL after SFT
- Uses offline difficulty filtering and majority voting at inference

The best small-model argument in the reasoning table.

---

## MiMo: End-to-end reasoning pipeline

<!-- cite-right: xia2025mimo -->

Xiaomi controls the **entire pipeline** from pretraining through post-training.

Key lesson: **pretraining data choices dramatically affect RL headroom**.

- Three-stage data mixing during pretraining (25T tokens)
- Multi-Token Prediction (MTP) during pretraining
- Multi-domain RL to prevent over-optimization on a single task type

"MiMo is the best rebuttal to the idea that reasoning is just a late-stage RL patch."

---

## Llama-Nemotron: Toggleable reasoning

<!-- cite-right: bercovich2025llamanemotron -->

Multi-size models with a **system prompt toggle** for thinking on/off.

- Not every query needs 10K thinking tokens
- Open weights AND data
- Uses online difficulty curriculum and length-controlled RL training

The practical UX insight: reasoning should be a **dial, not a switch**.

---

## Toggleable reasoning is becoming standard

Many models now support reasoning on/off:

- **Llama-Nemotron** [@bercovich2025llamanemotron]: system prompt toggle
- **Qwen 3** [@yang2025qwen3]: `/think` and `/no_think` modes + thinking budget
- **K2-V2**: low / medium / high reasoning effort
- **GLM-4.5**: thinking vs direct response modes

Training this requires either length-controlled RL or multi-stage SFT with both thinking and non-thinking demonstrations.

This is a **UX-driven training decision** -- not just about capability.

---

## Qwen 3: The R1 recipe, refined

<!-- cite-right: yang2025qwen3 -->

Applies a similar recipe to R1 but across the Qwen model family.

- Unified thinking/non-thinking mode with an explicit **thinking budget mechanism**
- Both dense and MoE variants
- Strong multilingual performance

Key contribution: made controllable reasoning a **mainstream product abstraction** rather than a niche research demo.

---

## MiniMax-M1: Scale meets architecture

<!-- cite-right: minimax2025minimaxm1scalingtesttimecompute -->

456B total / 45.9B active MoE with **Lightning Attention**. Native 1M context.

- CISPO: clips importance sampling weights rather than token updates
- Releases both **40K** and **80K thinking budget** checkpoints
- Shows how architecture changes (hybrid attention) interact with reasoning training

"Once reasoning traces get very long, attention design and RL design become inseparable."

---

## Skywork OR-1: Fighting entropy collapse

<!-- cite-right: he2025skyworkor1 -->

The best "stability and ablations" paper in the table.

- Studies **entropy dynamics** during long-CoT RL training
- Argues that avoiding premature entropy collapse is critical for final performance
- Fully open: weights, data, AND code

"The paper to cite when someone says the high-level recipe is enough by itself." Stability engineering matters as much as the algorithm.

---

## Magistral: Ground-up RL infrastructure

<!-- cite-right: mistral2025magistral -->

Pure RL on Mistral 3 (no SFT reasoning data).

- No KL penalty
- Relaxed clipping (DAPO-style) [@yu2025dapo]
- Format rewards + language consistency rewards
- Batch-level loss normalization
- Multilingual chain-of-thought

Where reasoning stopped being a "clever replication exercise" and became an **infrastructure discipline**.

---

<!-- columns: 40/60 -->
## OLMo 3 Think: The fully open reasoning model

<!-- cite-right: teamolmo2025olmo3 -->

The most comprehensive open documentation of a reasoning model.

Releases: stages, checkpoints, data, infrastructure, hyperparameters.

"If you want to study how reasoning training actually works, this is the model."

|||

![OLMo 3 Think RL training curves showing extended RL training.](assets/olmo31-extended-rl.jpeg)

---

## Empirical lessons from OLMo

<!-- cite-right: teamolmo2025olmo3 -->

- **DPO is a better RL starting point than SFT alone** -- preference tuning raises pass@K before RL converts it to pass@1
- **Mixed-domain RL prevents over-optimization** -- training only on instruction following helps IFEval but harms general chat
- **Midtraining determines RL ceiling** -- reasoning-poor pretraining data prevents the model from learning backtracking and self-verification

---

## Empirical lessons from OLMo

<!-- cite-right: teamolmo2025olmo3 -->

- **DPO is a better RL starting point than SFT alone** -- preference tuning raises pass@K before RL converts it to pass@1
- **Mixed-domain RL prevents over-optimization** -- training only on instruction following helps IFEval but harms general chat
- **Midtraining determines RL ceiling** -- reasoning-poor pretraining data prevents the model from learning backtracking and self-verification
- **Zero-gradient filtering**: skip prompt groups where all completions succeed or all fail
- **Performance was still improving** when training ended -- the run wasn't saturated

---

## DeepSeek V3.2: Where it all converges

<!-- cite-right: deepseekai2025v32 -->

Closes the loop on the R1 story at the end of 2025.

- Sparse attention (DSA) for efficiency
- RL framework upgrades
- Data synthesis for agentic/reasoning performance
- Open weights

Represents the endpoint: reasoning, agentic capabilities, and efficiency all integrated into one model.

---

## What the landscape tells us

- **Algorithm is table stakes**: Most models use GRPO or close variants -- the differentiator is systems engineering and data
- **Open weights is the norm**: Nearly all models release weights; open *process* (data, code, checkpoints) is rarer and more valuable

---

## What the landscape tells us

- **Algorithm is table stakes**: Most models use GRPO or close variants -- the differentiator is systems engineering and data
- **Open weights is the norm**: Nearly all models release weights; open *process* (data, code, checkpoints) is rarer and more valuable
- **Reasoning toggle is becoming standard**: Users and developers want controllable thinking, not always-on long CoT
- **Agentic absorption**: Later models (Kimi K2, GLM-4.5, DeepSeek V3.2) blend reasoning with tool use and agentic behavior -- reasoning is becoming a substrate, not a product category

---

## The research that came before

The ideas behind RLVR aren't new -- they were explored before o1/R1 made them mainstream:

- **STaR** [@zelikman2022star] and **Quiet-STaR** [@Zelikman2024QuietSTaRLM]: self-taught reasoning with ground-truth rewards (2022-2024)
- **TRICE** [@hoffman2023training]: MCMC-inspired optimization for reasoning traces
- **VinePPO** [@VinePPO]: PPO with binary math rewards on GSM8K/MATH
- **Tulu 3** [@lambert2024t]: PPO for math correctness while maintaining broad capabilities

The difference: these didn't scale to the same factor, or sacrificed general performance for specialized gains.

---

## Why does RL work now?

- **Stability is solved** (enough): RL training is still harder than SFT, but the tooling and recipes are mature enough for widespread adoption
- **Open-source tooling**: TRL, Open Instruct [@lambert2024t], veRL [@sheng2024hybridflow], OpenRLHF [@hu2024openrlhf]

---

## Why does RL work now?

- **Stability is solved** (enough): RL training is still harder than SFT, but the tooling and recipes are mature enough for widespread adoption
- **Open-source tooling**: TRL, Open Instruct [@lambert2024t], veRL [@sheng2024hybridflow], OpenRLHF [@hu2024openrlhf]
- **Base models are good enough**: Multiple sources suggest RL reasoning training only became viable with models from ~2024 onwards -- a capability floor was needed
- **Verifiable domains provide clean signal**: Math and code give unambiguous rewards, avoiding the reward hacking problems of RLHF

---

<!-- layout: section-break -->

## What changes in the RL recipe

---

<!-- columns: 50/50 -->
## Standard RLHF RL vs RLVR for reasoning

**RLHF RL (lectures 3-4)**:

- Reward model scores
- KL penalty to reference model
- Short completions (~100-500 tokens)
- Hundreds of RL steps
- PPO or GRPO

|||

**RLVR for reasoning**:

- Verification function (binary or partial credit)
- Often **no KL penalty**
- Completions can be **1K-30K tokens**
- Thousands of RL steps
- GRPO variants dominate

Same core gradient -- different engineering around it.

---

## Offline difficulty filtering

The model can only learn from problems where there is a **gradient signal**.

- If pass rate is **0%**: all completions fail → advantages are all equal → zero gradient
- If pass rate is **100%**: all completions succeed → same problem
- Sweet spot: **20-80% pass rate** per prompt

Recipe: sample $N$ completions per prompt before training, keep prompts in the productive range.

Used by Seed-Thinking 1.5 [@seed2025seed], Open-Reasoner-Zero [@hu2025openreasonerzero], Phi-4 [@abdin2025phi4], MiMo [@xia2025mimo], Skywork OR-1 [@he2025skyworkor1].

---

## Online filtering and difficulty curriculum

Offline filtering is a snapshot -- the model improves during training, shifting the difficulty distribution.

Solutions:

- **Per-batch online filtering**: Skip prompts that are now too easy or too hard
- **Difficulty schedules**: Save harder problems for later in training
- **Dynamic resampling**: Re-evaluate difficulty periodically

Used by Kimi 1.5 [@team2025kimi], Magistral [@mistral2025magistral], Llama-Nemotron [@bercovich2025llamanemotron], MiMo [@xia2025mimo].

---

## Zero-gradient filtering in practice

<!-- cite-right: teamolmo2025olmo3 -->

A more precise version used in OLMo 3 Think:

Within each batch, skip any prompt group where **all** $G$ completions succeed **or** all fail.

- Advantage = 0 for every completion in that group → zero gradient
- "Free" -- no extra sampling needed, just discard before the gradient step

Combined with **active sampling**: resample to fill the batch with non-zero-gradient groups, maintaining the target batch size.

---

## Removing the KL penalty

In RLHF (lectures 3-4): KL penalty prevents the policy from drifting too far from the reference model. **Essential** when reward models can be gamed.

In RLVR: rewards are **ground truth** (not a learned proxy), so over-optimization is less of a risk.

Removing KL allows the model to **explore more freely** during long training runs, discovering novel reasoning strategies the reference model never exhibited.

Used by Magistral [@mistral2025magistral], Open-Reasoner-Zero [@hu2025openreasonerzero], Skywork OR-1 [@he2025skyworkor1].

---

## Relaxed and asymmetric clipping

Standard PPO/GRPO uses symmetric clipping:

$$\text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)$$

**DAPO** [@yu2025dapo] and related variants propose **asymmetric clipping** -- wider on the upside to encourage exploration of new reasoning behaviors.

This matters more for reasoning because the action space is larger and the model needs to **discover** novel strategies, not just refine known ones.

Used by Magistral [@mistral2025magistral], INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained].

---

## Format and language consistency rewards

Beyond binary correctness, many models add small **auxiliary rewards**:

**Format rewards**: Encourage `<think>...</think>` before answers, penalize malformed reasoning blocks. Makes answer extraction, tooling, and distillation much easier.

**Language consistency**: Penalize language switching mid-reasoning. Common in multilingual models where the model might reason in English but answer in Chinese (or vice versa).

These are not about correctness -- they're about making reasoning **predictable and usable**.

Used by DeepSeek R1 [@guo2025deepseek], Magistral [@mistral2025magistral], Skywork OR-1 [@he2025skyworkor1].

---

## Length penalties and overthinking

Without intervention, RL-trained models generate **longer and longer** reasoning traces. Not always useful -- "overthinking" wastes compute.

Mitigation strategies:

- **Progressive length extension** (Kimi 1.5 [@team2025kimi]): gradually increase the target length during training
- **Small length penalty** (INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained]): penalize excessive trace length throughout
- **Overlong filtering**: discard completions that exceed a threshold for throughput

Goal: teach the model to reason **efficiently**, not just verbosely.

---

## Loss normalization: Group vs batch

Recall from lecture 4: loss aggregation strategy matters.

- **Standard GRPO**: normalizes advantages within each prompt group

$$\hat{A}_i = \frac{R_i - \mu_G}{\sigma_G}$$

- **Batch-level normalization**: normalizes across the entire batch -- avoids per-group biases when groups have very different difficulty levels
- **Token-level vs sequence-level**: normalizing loss by total tokens across the batch reduces length bias (Dr. GRPO [@liu2025understanding])

Used by Magistral [@mistral2025magistral], MiMo [@xia2025mimo].

---

<!-- columns: 55/45 -->
## The infrastructure bottleneck

<!-- cite-right: teamolmo2025olmo3 -->

Reasoning completions are **long and variable** in length.

Result: inference (rollout generation) dominates training time.

From OLMo 3:

- Learner GPUs sit idle **~75%** of the time
- **5-14x** more compute for inference than training
- Static batching wastes **up to 54%** of compute

|||

![Distributed RL training infrastructure.](assets/distributed-rl.png)

---

## Off-policy and asynchronous updates

As completions get longer, synchronous rollout-then-train becomes **wasteful**.

Moving to async:

- **Actors** generate completions continuously
- **Learner** consumes them as available
- Trade-off: data is slightly stale (off-policy), but throughput increases dramatically

Partial-to-full async used by Seed-Thinking 1.5 [@seed2025seed], INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained], and others.

This is the "algorithm to systems" shift -- **keeping the GPUs busy** matters as much as the loss function.

---

## Parallel test-time compute scaling

Combining answers from multiple parallel rollouts improves over a single rollout.

- **Majority voting**: Sample $N$, take the most common answer
- **Scoring model**: Use a learned selector to pick the best answer
- **Best-of-N**: Score with a reward model or verifier, take the highest

Used at inference by DeepSeek R1 [@guo2025deepseek], Phi-4 [@abdin2025phi4].

---

<!-- columns: 50/50 -->
## RL training vs inference-time scaling

<!-- cite-right: openai2024o1 -->

![Train-time compute scaling: more RL training improves the policy.](assets/o1-train-time.png)

|||

![Test-time compute scaling: more inference compute improves performance.](assets/o1-test-time.png)

Both axes show log-linear performance gains. RL training **shifts the curve**; inference-time scaling **moves along it**. They are complementary.

---

## Summary: RLVR recipe changes vs RLHF

| Decision | RLHF RL (Lec 3-4) | RLVR for reasoning |
|:---------|:-------------------|:-------------------|
| Reward signal | Learned RM | Verification function |
| KL penalty | Essential | Often removed |
| Clipping | Symmetric | Asymmetric / relaxed |
| Completion length | ~100-500 tokens | ~1K-30K tokens |
| Difficulty filtering | Rarely | Standard practice |
| Loss normalization | Per-group | Per-group or per-batch |
| Training duration | ~100s of steps | ~1000s of steps |
| Infrastructure | Synchronous OK | Async near-mandatory |

---

## Cross-model empirical findings

Three results that appeared independently across multiple teams:

- **Text-only reasoning boosts multimodal performance**: MiMo-VL and Magistral [@mistral2025magistral] found that text-only reasoning RL *after* multimodal training improves vision tasks
- **Mixed-domain RL prevents over-optimization**: Training on math alone leads to degradation on general chat; mixing in code and instruction following is safer [@teamolmo2025olmo3]
- **Midtraining determines RL ceiling**: How much math/code is in pretraining data sets the upper bound on what RL can achieve [@xia2025mimo]

---

<!-- layout: section-break -->

## Looking ahead

---

## The expanding scope of RLVR

RLVR started with math and code because they have the **strongest automatic feedback loops**: symbolic equivalence, unit tests, compilation.

It is expanding to:

- **Precise instruction following**: Verifiable constraints (length, format, inclusion/exclusion rules)
- **Agentic tasks**: Did the agent complete the task in the environment?
- **Quality preservation**: LM-judge signals to maintain general capabilities during reasoning RL

"The core to progress on RLVR is having a variety and depth of verifiable problems."

---

## Open questions

- Is RL training **discovering** new capabilities, or **eliciting** what pretraining already learned?
- How far can reasoning training go without better pretraining data?
- Will agentic RL (tool use + reasoning) require fundamentally different recipes?
- Can we systematically study the scaling properties of RL for reasoning? [@khatri2025art]

---

## Lecture summary

1. **RLVR** -- verification functions replace reward models; same RL algorithms, different signal
2. **The landscape** -- 25+ models in 2025; DeepSeek R1 pioneered, the community rapidly iterated
3. **Key recipe changes** -- difficulty filtering, no KL, relaxed clipping, format rewards, async infrastructure
4. **Cross-cutting patterns** -- toggleable reasoning, algorithm-to-systems shift, open weights vs open process
5. **The cake metaphor** -- RL moved from cherry on top to load-bearing component

---

<!-- rows: 50/50 -->
## Resources

<!-- row-columns: 50/50 -->

```box
title: Book & Course
tone: accent
compact: true
content: |
  - rlhfbook.com — Chapter 7
  - Course slides & recordings
  - GitHub: natolambert/rlhf-book
```

|||

```box
title: Key Papers
tone: surface
compact: true
content: |
  - DeepSeek R1
  - OLMo 3 Think
  - DAPO
  - Tulu 3
```

===

<!-- row-columns: 50/50 -->

```box
title: Codebases
tone: surface
compact: true
content: |
  - TRL (Hugging Face)
  - Open Instruct (Ai2)
  - veRL (Bytedance)
  - OpenRLHF
```

|||

```box
title: Further Reading
tone: surface
compact: true
content: |
  - Skywork OR-1
  - Magistral
  - Open-Reasoner-Zero
  - OpenThoughts
```

---

## Course outline

1. Introduction & Training Overview -- Chapters 1-3
2. IFT, Reward Models, Rejection Sampling -- Chapters 4, 5, 9
3. RL Theory -- Chapter 6 (Part 1)
4. RL Implementation & Practice -- Chapter 6 (Part 2)
5. **Reasoning -- Chapter 7**
6. Direct Alignment Algorithms -- Chapter 8
7. ...

---

<!-- rows: 85/15 -->
## Thank you

Questions and discussion welcome.

**Nathan Lambert**

rlhfbook.com | interconnects.ai

===

<div class="text-xs" style="text-align: center; opacity: 0.5;">
Built with <a href="https://github.com/natolambert/colloquium">colloquium</a>
</div>
