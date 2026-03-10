---
title: "An introduction to reinforcement learning from human feedback and post-training"
author: "Nathan Lambert"
date: "March 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com"
  center: ""
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# An introduction to reinforcement learning from human feedback and post-training

<div class="colloquium-title-eyebrow">SALA 2026</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
<p>Quito, Ecuador<br>11 March 2026</p>
</div>

<p class="colloquium-title-note">A cursory overview of RLHF, RLVR, and modern post-training recipes for language models.</p>

---

<!-- columns: 45/55 -->
## What is a language model?

Core properties:
- A language model assigns probabilities to text.
- Chunks of words are broken down as **tokens**, which are the internal representation of the model.
- Given previous tokens, it predicts the next token. Repeating this produces a completion one step at a time (this is called **autoregressive**).

|||

![The original model architecture diagram for the Transformer. 2017.](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## What is a (modern) language model?

Modern language models:
- Have billions to trillions of parameters.
- Largely downstream of The Transformer architecture, which popularized the use of the **self-attention** mechanism along with fully-dense layers.
- Predict and work over much more than text: Gemini and ChatGPT work with images, audio, and video.

|||

![The original model architecture diagram for the Transformer. 2017.](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2017: The Transformer is born

- **2017:** the Transformer is born

|||

![The original model architecture diagram for the Transformer. 2017.](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2018: GPT-1, ELMo, and BERT

- 2017: the Transformer is born
- **2018:** GPT-1, ELMo, and BERT released

|||

![The language model architecture for GPT-1. 2018.](assets/gpt-1.png)
<!-- cite-left: radford2018gpt, peters2018elmo, devlin2018bert -->

---

<!-- rows: 30/70 -->
## 2019: GPT-2 and scaling laws

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- **2019:** GPT-2 and scaling laws

===

![The famous scaling laws plots. 2020.](assets/scaling-laws.png)
<!-- cite-right: radford2019gpt2, kaplan2020scaling -->

---

<!-- columns: 45/55 -->
## 2020: GPT-3 surprising capabilities

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- **2020:** GPT-3 surprising capabilities

|||

![GPT-3 was known for exapnding the idea of in-context learning and few-shot prompting. Screenshot from the paper.](assets/few-shot.png)
<!-- cite-left: brown2020gpt3 -->

---

<!-- columns: 45/55 -->
## 2021: Stochastic Parrots

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- 2020: GPT-3 surprising capabilities
- **2021:** Stochastic Parrots

|||

![](assets/stochastic-parrots.png)
<!-- cite-right: bender2021stochastic -->

---

<!-- columns: 45/55 -->
## 2022: ChatGPT

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- 2020: GPT-3 surprising capabilities
- 2021: Stochastic Parrots
- **2022:** ChatGPT

|||

![](assets/chatgpt.webp)
<!-- cite-right: openai2022chatgpt -->

---

<!-- columns: 45/55 -->
## 2023: GPT-4 and frontier-scale

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- 2020: GPT-3 surprising capabilities
- 2021: Stochastic Parrots
- 2022: ChatGPT
- **2023:** GPT-4 and frontier-scale

|||

![An image where Nvidia CEO Jensen Huang supposedly leaked that GPT-4 was an ~2T parameter MoE model.](assets/jensen-gpt4.jpeg)
<!-- cite-right: openai2023gpt4 -->

---

<!-- columns: 45/55 -->
## 2024: o1 and reasoning models

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- 2020: GPT-3 surprising capabilities
- 2021: Stochastic Parrots
- 2022: ChatGPT
- 2023: GPT-4 and frontier-scale
- **2024:** o1 and reasoning models

|||

![The famous test-time scaling plot from OpenAI's o1 announcement.](assets/o1-test-time.png)
<!-- cite-right: openai2024o1 -->

---

<!-- columns: 45/55 -->
## 2025: o3, Claude Code, and agents

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- 2020: GPT-3 surprising capabilities
- 2021: Stochastic Parrots
- 2022: ChatGPT
- 2023: GPT-4 and frontier-scale
- 2024: o1 and reasoning models
- **2025:** o3, Claude Code, and agents

|||

![](assets/claude-code.png)
<!-- cite-right: openai2025o3, anthropic2025claudecode, openai2025agents -->

---

<!-- columns: 50/50 -->
## Pretraining: next-token prediction

- Train on a trillions of tokens of text from the web, books, code, and documents
  - Models are often trained on 5-50+ trillion tokens
  - 1T of text tokens is about 3-5 TB of data
  - Labs gather and filter 10-20X more data than is used for the model
  - Total data funnel targetted for models is on the order of petabytes
- Objective: predict the next token in each sequence
- Result: Incredible, flexible, useful models

|||

![](assets/pretraining_next_token_tikz.png)

---

## A base model completes text

After pretraining we are left wtih a glorified autocomplete model, for example:^[Base models are also becoming more flexible through midtraining and better data mixtures.]

<div class="colloquium-spacer-md"></div>

```conversation
messages:
  - role: user
    content: "The president of the United States in 2006 was"
  - role: assistant
    model: "Llama 3.1 405B Base"
    content: "George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator in 2006..."
```

---

## Post-training makes it answer like a chatbot

The earliest forms of modern post-trained (or RLHF-tuned) models shifted the continuation format to always conforming to the "answering a question style."
An example of what early conversational models looked like is below:

<div class="colloquium-spacer-md"></div>

```conversation
messages:
  - role: user
    content: "The president of the United States in 2006 was"
  - role: assistant
    model: "Tülu 3 405B"
    content: "George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009."
```

---

<!-- columns: 40/60 -->
## ChatGPT was when RLHF made the models even easier to use

Model's responses evolved quickly to have:
- Better **format**: direct, conversational answers
- Better **style**: helpful, concise, markdown, etc.
- Better **product**: people could actually use it every day

|||

```conversation
size: 0.775
messages:
  - role: user
    content: "I'm giving a talk on RLHF tomorrow. Can you help me structure it?"
  - role: assistant
    content: |
      Absolutely — here's a simple structure:

      **1. Start with the basics**
      - What a language model is
      - Why next-token prediction is not enough

      **2. Explain post-training**
      - Instruction tuning
      - Preference tuning / RLHF
      - RLVR for verifiable tasks

      **3. End with why it matters**
      - Better products like ChatGPT
      - Better reasoning in models like o1
      ...
```

---
<!-- layout: section-break -->

## So what is reinforcement learning from human feedback (RLHF) anyways?

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## Which is the better backflip?

![](assets/christiano-backflip-human.webp)

|||

![](assets/christiano-backflip-reward.webp)

---

<!-- cite-right: christiano2017, ziegler2019fine, ouyang2022training, bai2022constitutional -->
## Why did people make RLHF?

- Many objectives are easy for humans to **judge**, but hard to write as an exact reward function
- In language models, what we want is often implicit: **follow intent**, be **helpful**, be **harmless**
- Pretraining optimizes **next-token prediction**, not assistant behavior
- Preference comparisons turn those human judgments into a scalable training signal

RLHF lets us optimize for behavior we can **evaluate**, even when we cannot easily **specify** the reward.

---

<!-- columns: 40/60 -->
<!-- cite-right: christiano2017 -->
## RLHF before language models

- **TAMER** (Knox & Stone, 2008) — humans score agent actions to learn a reward
- **Christiano et al. 2017** — RLHF on Atari trajectory preferences
- **Ziegler et al. 2019** — first RLHF on language models

|||

![A recreation of the system diagram from Christiano et al. 2017.](assets/rlhf_schematic_tikz.png)

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## Left: Human feedback; Right: Hand-designed reward function

![](assets/christiano-backflip-human.webp)

|||

![](assets/christiano-backflip-reward.webp)

---

<!-- cite-right: sutton2018reinforcement -->
<!-- columns: 65/35 -->

## Classical RL

A reinforcement learning problem is often written as a **Markov Decision Process (MDP)**:
- state space $\mathcal{S}$, action space $\mathcal{A}$
- transition dynamics $P(s_{t+1}\mid s_t, a_t)$
- reward function $r(s_t, a_t)$ and discount $\gamma$
- optimize cumulative return over a trajectory

$$\text{MDP } (\mathcal{S}, \mathcal{A}, P, r, \gamma)$$

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

|||

![](assets/rl.png)

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## Classical RL vs. RLHF

<div class="text-sm">

**Classical RL**
- Agent takes actions $a_t$ in an environment with states $s_t$ 
- Reward is a known function $r(s_t, a_t)$ from the environment per step
- Optimize cumulative return over a trajectory (total steps $T$)

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

<div class="colloquium-spacer-md"></div>

**RLHF**
- No environment — prompts sampled from a dataset
- Reward is **learned** from human preferences (a proxy)
- **Response-level** reward (bandit-style, not per-token)
- Regularized with **KL penalty** to stay close to the base model

$$J(\pi) = \mathbb{E}\left[ r_\theta(x, y) \right] - \beta \, D_{\text{KL}}\!\left(\pi \| \pi_{\text{ref}}\right)$$

</div>

|||

![](assets/rlhf.png)

---

<!-- columns: 50/50 -->

## Reinforcement learning with *Verifiable* rewards

Apply the same RL algorithms to LLMs just on if the answer was right. No need to train a reward model:
- E.g. Math: check the final answer.  
  Code: run the tests.
- No learned reward model — **no proxy objective**
- Enables scaling RL compute on reasoning tasks
- Unlocked **inference time scaling**: Spending more compute at generation time per problem increases performance log-linearlly w.r.t. compute
- RLVR was named by **Tülu 3** [@lambert2024t] and popularized by **DeepSeek R1** [@guo2025deepseek]

|||

![](assets/rlvr-system.png)

---

## Comparing classical RL vs. LLM RLHF and RLVR

| | Classical RL | RLHF | RLVR |
|---|---|---|---|
| **Reward** | Environment | Learned (proxy) | Verifiable (exact) |
| **State transitions** | Yes | No | No |
| **Reward granularity** | Per-step | Per-response | Per-response |
| **Primary challenge** | Explor-Exploit Trade-off | Over-optimization | Task generalization |
| **Example** | CartPole | Chat style tuning | Math reasoning |

---


<!-- rows: 60/40 -->
## The path to modern RLHF

![](assets/rlhf_timeline_tikz.png)

===

<!-- row-columns: 50/50 -->

- **Ziegler 2019** [@ziegler2019fine] — first RLHF on language models
- **InstructGPT** [@ouyang2022training] — the canonical RLHF recipe behind ChatGPT
- **Constitutional AI** [@bai2022constitutional] — Introduced early methods for AI feedback in Claude

|||

- **DPO** [@rafailov2024direct] — direct preference optimization (DPO) without a reward model
- **Llama 3** [@dubey2024llama] and **Tülu 3** [@lambert2024t] — modern multi-stage recipes
- **DeepSeek R1** [@guo2025deepseek] — popularized RLVR

---

<!-- valign: center -->
<!-- cite-right: ouyang2022training -->

## InstructGPT's 3-step RLHF recipe


![The 3-step RLHF process figure from InstructGPT, which became "the standard" approach to RLHF for a few years.](assets/instructgpt.jpg)


---

<!-- columns: 45/55 -->
<!-- cite-right: ouyang2022training -->
## Step 1/3: Instruction fine-tuning (IFT)

The foundation of post-training. Also called **Supervised Fine-tuning (SFT)**:
- Start from a pretrained language model
- Collect demonstrations of *desired* assistant behavior
- Train with standard supervised learning on prompt-response pairs.  
  (different batch size, learning rate, etc.)
- Model can now answer questions.  
  Easy to use IFT to quickly adapt base model to many domains.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=
- \sum_{(x, y^\star)} \sum_{t=1}^{|y^\star|}
\log \pi_\theta \!\left(y^\star_t \mid x, y^\star_{<t}\right)
$$

|||

```conversation
size: 0.9
messages:
  - role: system
    content: "You are a helpful, harmless assistant. A system message like this can be used to steer the model to specific persona's or behaviors."
  - role: user
    content: "Write me a short poem about an optimistic goldfish."
  - role: assistant
    content: "Bright little goldfish\nFinds a sunrise in each wave\nSmall bowl, endless hope"
```

---

<!-- columns: 45/55 -->
<!-- cite-right: christiano2017, ouyang2022training -->
## Step 2/3: Reward modeling

Overview:
- Collect **comparisons** between two model outputs for the same prompt
- RLHF gets its name from collecting *human* feedback between completions, but today much of it is AI feedback
- Train a reward model $r_\phi(x, y)$ to score preferred completions higher

|||


The probability model says a response should win when it gets a higher reward score:

$$
P(y_w \succ y_l \mid x)
=
\sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

Training then minimizes the negative log-likelihood of the preferred response beating the rejected one:

$$
\mathcal{L}_{\mathrm{RM}}(\phi)
=
- \log \sigma \!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)
$$

Notation:
- $x$ is the prompt
- $y_w$ is the **winning** response
- $y_l$ is the **losing** response
- $r_\phi(x, y)$ is the trained reward model

---

<!-- columns: 50/50 -->
<!-- cite-right: ouyang2022training -->
<!-- footnotes: right -->
## Step 3/3: RL against the reward model

Where everything comes together (and RLHF gets its name):
- Sample a batch of prompts $x_i$ from the dataset $\mathcal{D}$
- Generate completions $y_i \sim \pi_\theta(\cdot \mid x_i)$ from the model being trained
- Score them with the reward model $r_\phi(x_i, y_i)$
- Add a **KL penalty** so the policy stays close to the SFT/reference model.^[KL divergence measures how much the current policy differs from the reference model. For discrete outputs, $D_{\mathrm{KL}}(\pi \,\|\, \pi_{\mathrm{ref}})=\mathbb{E}_{y \sim \pi}\!\left[\log \pi(y \mid x)-\log \pi_{\mathrm{ref}}(y \mid x)\right]$. People often colloquially call this the “KL distance” between the models, even though it is not a true metric.]
- Update the policy with a policy-graident RL algorithm (Proximal Policy Optimization, PPO in InstructGPT & ChatGPT)

$$
J(\pi)
=
\mathbb{E}\!\left[r_\phi(x, y)\right]
- \beta D_{\mathrm{KL}}\!\left(\pi \,\|\, \pi_{\mathrm{ref}}\right)
$$

|||

![](assets/rlhf-overview.png)

---

<!-- rows: 35/65 -->
<!-- title: center -->

## The RLHF objective, unpacked

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
\underbrace{r_\phi(x, y)}_{\text{maximize the reward}}
- \underbrace{\beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)}_{\text{but don't change the model too much}}
$$


===


<!-- row-columns: 50/50 -->

<div style="text-align: left;">

<!-- Prompts $x$ come from a dataset, not an environment.

The policy $\pi_\theta(y \mid x)$ generates a full response $y$, and the reward model $r_\phi(x, y)$ scores whether humans would like that response. -->

</div>

|||

<div style="text-align: left;">

The reference model $\pi_{\mathrm{ref}}$ keeps the policy anchored to the SFT model.

$D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)$ measures how far the new policy moves from that reference on prompt $x$.

$\beta$ controls the tradeoff between **improving behavior** and **staying close** to what the model already knows.

</div>

---

<!-- rows: 27/73 -->
<!-- cite-right: rafailov2024direct -->
## What if we optimize this more directly?

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
r_\phi(x, y)
- \beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)
$$

===

<!-- row-columns: 60/40 -->

**Direct Preference Optimization (DPO)**

- Derived the gradient toward the optimal solution, $\pi^*$ to the above equation 
- Eliminated the need for a separate reward model (via training an implicit one)
- Train directly on preferred ($y_w$) vs. rejected ($y_l$) responses to a prompt ($x$)

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
- \log \sigma \!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right)
$$

|||

---

<!-- rows: 27/73 -->
<!-- cite-right: rafailov2024direct -->
## What if we optimize this more directly?

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
r_\phi(x, y)
- \beta D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot \mid x)\right)
$$

===

<!-- row-columns: 60/40 -->

**Direct Preference Optimization (DPO)**

- Derived the gradient toward the optimal solution, $\pi^*$ to the above equation 
- Eliminated the need for a separate reward model (via training an implicit one)
- Train directly on preferred ($y_w$) vs. rejected ($y_l$) responses to a prompt ($x$)

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
- \log \sigma \!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right)
$$

|||

```box
title: DPO became very popular as it is
tone: accent
content: |
  - Far simpler to implement
  - Far cheaper to run
  - Achieves ~80% or more of the final performance
  - I used it to build models like Zephyr-Beta, Tülu 2/3, Olmo 2/3, etc.
```

---

<!-- valign: center -->
## How training recipes have evolved

| | InstructGPT (2022) | Tülu 3 (2024) | DeepSeek R1 (2025) |
|---|---|---|---|
| **Instruction data** | ~10K | ~1M | 100K+ |
| **Preference data** | ~100K | ~1M | On-policy |
| **RL stage** | ~100K prompts | ~10K (RLVR) | N/A |

An overall trend is to use far more compute across all the stages, but shifting more to RLVR.

---

<!-- rows: 60/40 -->
## The early days: InstructGPT

![](assets/rlhf-basic.png)

===

Early on, RLHF has a well-documented, simple enough approach.
- **InstructGPT** made the classic three-stage recipe canonical:
  SFT, reward modeling, then RL against the reward model. *OpenAI even hinted that the original ChatGPT even used this!*
- This became the intellectual template for much of modern post-training.

---

<!-- rows: 50/50 -->
## From RLHF to post-training

![](assets/rlhf-complex.png)

===

What began as an "RLHF" recipe evolved into a complex series of steps to get the final, best model (e.g. Nemotron 4 340B, Llama 3.1).
- Modern systems keep the same core idea of using multiple optimizers with different strengths and weaknesses, but add more stages, more data, and more filtering.
- This trend has only continued, and recipes eb and flow, as tools like RLVR and model merging change the scope of what is doable in different ways.

---

## From RLHF to "post-training"

As time has passed since ChatGPT, the field as gone through multiple distinct phases (roughly):
1. 2023: Simple SFT for better chatbots and reproducing RLHF fundamentals (Alpca, Vicuna, etc.)
2. 2024: DPO dominates open models and training stages expand (Zephyr-beta, Tülu 2, etc.)
3. 2025: RLVR, complex recipes (Tülu 3, Olmo 3, Nemotron 3, R1, etc.)
4. 2026: Agentic training, multi-turn RL, etc.

---

## From RLHF to "post-training"

As time has passed since ChatGPT, the field as gone through multiple distinct phases (roughly):
1. 2023: Simple SFT for better chatbots and reproducing RLHF fundamentals (Alpca, Vicuna, etc.)
2. **2024: DPO dominates open models and training stages expand** (Zephyr-beta, Tülu 2, etc.)
3. 2025: RLVR, complex recipes (Tülu 3, Olmo 3, Nemotron 3, R1, etc.)
4. 2026: Agentic training, multi-turn RL, etc.

Within 2024 the field shifted its focus to post-training, as training stages evolved beyond the InstructGPT-style recipe, DPO proliferated, and largely RLHF was viewed as one tool (that you may not even need).

---

<!-- columns: 50/50 -->
## An intuition for post-training
<!-- cite-right: zhou2023lima -->

RLHF's reputation was that its contributions are minor on the final language models.

> "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users."

*LIMA: Less Is More for Alignment* (2023)


|||

---


<!-- columns: 50/50 -->
## An intuition for post-training
<!-- cite-right: zhou2023lima,muennighoff2024olmoe,ai2_olmoe_ios_2025 -->

RLHF's reputation was that its contributions are minor on the final language models.

> "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users."

*LIMA: Less Is More for Alignment* (2023)

|||

Sometimes this view of alignment (or RLHF) teaching "format" made people think that post-training only made minor changes to the model. This would describe finetuning as "*just style transfer*."

The base model trained on trillions of tokens of web text has seen and learned from an extremely broad set of examples.
The model at this stage contains far more latent capability than early post-training recipes were able to expose.

The question is: How does post-training interact with these?

---

<!-- columns: 50/50 -->
## An intuition for post-training
<!-- cite-right: zhou2023lima,muennighoff2024olmoe,ai2_olmoe_ios_2025 -->

RLHF's reputation was that its contributions are minor on the final language models.

An example, **OLMoE** — same base model family, updated only post-training:
- [`OLMoE-1B-7B-0924-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct) (Sep. 2024): **38.44** avg. eval score
- [`OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) (Jan. 2025): **45.62** avg. eval score

Base models determine the *ceiling*. Post-training's job has been to **reach it**.

---

<!-- columns: 50/50 -->
## An intuition for post-training
<!-- cite-right: zhou2023lima,vergarabrowne2026operationalising, -->

RLHF's reputation was that its contributions are minor on the final language models.

> "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users."

*LIMA: Less Is More for Alignment* (2023)

> "The superficial alignment hypothesis (SAH) posits that large language models learn most of their knowledge during pre-training, and that post-training merely surfaces this knowledge."

*Operationalising the Superficial Alignment Hypothesis via Task Complexity* (2026)

|||

The second paper, 3 years later, matches my intuition for post-training. 

```box
title: I call this the **Elicitation Theory** of post-training, where we're trying to pull out the most useful knowledge of the model.
tone: accent
content: |
  (TODO make boxes accept empty content)
```

---


<!-- layout: section-break -->

## Beyond ellicitation: The scaling RL era of post-training

---

## o1 scaling post-training

<!-- img-align: center -->
<!-- cite-right: openai2024o1 -->

![](assets/o1.webp)

---

## o1: Train-time scaling

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->

text goes here

|||

![](assets/o1-train-time.png)


---

## o1: Test-time scaling

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->
text goes here


|||

![](assets/o1-test-time.png)

---

## The scale of post-training

- **DeepSeek R1**: RL used ~147K H800 GPU hours (~5% of total training)
- Individual ablation runs: **10–100K GPU hours**
- The trend: **more compute going to post-training every year**

---



## The post-ChatGPT acceleration

- **Early 2023**: Alpaca era — limited data, impressive but narrow
- **Late 2023**: DPO — direct alignment without a reward model
- **2024**: Complex multi-stage recipes (Llama 3, Tülu 3)
- **2025**: Reasoning via RL (DeepSeek R1, o1)

---

## The classic RLHF pipeline

1. **Instruction fine-tuning** — teach Q&A format from examples
2. **Reward model training** — learn a scoring function from human preferences
3. **RL optimization** — optimize the model against the reward model

---


## Where things are heading

- RLHF and RLVR are **complementary** — style vs. capabilities
- Modern recipes use **both** in sequence
- The boundary between them is blurring (generative reward models, self-correction)

---

## Beyond elicitation?

- Maybe post-training does more than just **extract** existing ability
- Long RL runs may reshape how models **reason**, not just how they respond
- Open question: when does scaling RL create **new capabilities**?

---

<!-- rows: 50/50 -->
## This talk is ~lecture 1 of a larger course

<!-- row-columns: 34/33/33 -->

```box
title: Introductions (this talk)
tone: accent
compact: true
content: |
  1. Introduction
  2. Key Related Works
  3. Training Overview
```

|||

```box
title: Core Training Pipeline
tone: muted
compact: true
content: |
  4. Instruction Tuning
  5. Reward Models
  6. Reinforcement Learning
  7. Reasoning
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

<!-- row-columns: 34/33/33 -->

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
tone: surface
compact: true
content: |
  - A. Definitions  
  - B. Style & Information  
  - C. Practical Issues  
```

|||

Full lecture slides coming to [rlhfbook.com/slides](https://rlhfbook.com//slides) and YouTube @natolambert!

---

<!-- rows: 85/15 -->
## Thank you

Sorry I could not make it in person!

Contact: nathan@natolambert.com

Newsletter: [interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===


```builtwith
repo: natolambert/colloquium
```
