---
title: "An Introduction to Reinforcement Learning from Human Feedback and Post-training"
author: "Nathan Lambert"
date: "March 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
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

# An Introduction to Reinforcement Learning from Human Feedback and Post-training

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

![transformer original diagram](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## What is a (modern) language model?

Modern language models:
- Have billions to trillions of parameters.
- Largely downstream of The Transformer architecture, which popularized the use of the **self-attention** mechanism along with fully-dense layers.
- Predict and work over much more than text: Gemini and ChatGPT work with images, audio, and video.

|||

![transformer original diagram](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2017: the Transformer is born

- **2017:** the Transformer is born

|||

![transformer original diagram](assets/transformer.webp)
<!-- cite-right: Vaswani2017AttentionIA -->

---

<!-- columns: 45/55 -->
## 2018: GPT-1, ELMo, and BERT

- 2017: the Transformer is born
- **2018:** GPT-1, ELMo, and BERT released

|||

![GPT-1](assets/gpt-1.png)
<!-- cite-right: radford2018gpt, peters2018elmo, devlin2018bert -->

---

<!-- rows: 30/70 -->
## 2019: GPT-2 and scaling laws

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- **2019:** GPT-2 and scaling laws

===

![Scaling laws](assets/scaling-laws.png)
<!-- cite-right: radford2019gpt2, kaplan2020scaling -->

---

<!-- columns: 45/55 -->
## 2020: GPT-3 surprising capabilities

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- **2020:** GPT-3 surprising capabilities

|||

![GPT-3 few-shot prompting](assets/few-shot.png)
<!-- cite-right: brown2020gpt3 -->

---

<!-- columns: 45/55 -->
## 2021: Stochastic Parrots

- 2017: the Transformer is born
- 2018: GPT-1, ELMo, and BERT released
- 2019: GPT-2 and scaling laws
- 2020: GPT-3 surprising capabilities
- **2021:** Stochastic Parrots

|||

![Stochastic parrots](assets/stochastic-parrots.png)
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

![ChatGPT](assets/chatgpt.webp)
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

![GPT-4](assets/jensen-gpt4.jpeg)
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

![o1 test-time scaling](assets/o1-test-time.png)
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

![Claude Code](assets/claude-code.png)
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

![Next-token prediction pretraining example](assets/pretraining_next_token_tikz.png)

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

## So what is Reinforcement Learning from Human Feedback (RLHF) anyways?

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## Which is the better backflip?

![Backflip trained from human preferences](assets/christiano-backflip-human.webp)

|||

![Backflip trained with a hand-designed reward](assets/christiano-backflip-reward.webp)

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
## RLHF Before Language Models

- **TAMER** (Knox & Stone, 2008) — humans score agent actions to learn a reward
- **Christiano et al. 2017** — RLHF on Atari trajectory preferences
- **Ziegler et al. 2019** — first RLHF on language models

|||

![Christiano et al. 2017 RLHF overview](assets/rlhf_schematic_tikz.png)

---

<!-- columns: 50/50 -->
<!-- cite-right: christiano2017 -->
## Left: Human Feedback; Right: Hand-design Reward Function

![Backflip trained from human preferences](assets/christiano-backflip-human.webp)

|||

![Backflip trained with a hand-designed reward](assets/christiano-backflip-reward.webp)

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

![Classical RL basics](assets/rl.png)

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

![RLHF basic system](assets/rlhf.png)

---

<!-- columns: 50/50 -->

## Reinforcement Learning with *Verifiable* Rewards

Apply the same RL algorithms to LLMs just on if the answer was right. No need to train a reward model:
- E.g. Math: check the final answer.  
  Code: run the tests.
- No learned reward model — **no proxy objective**
- Enables scaling RL compute on reasoning tasks
- Unlocked **inference time scaling**: Spending more compute at generation time per problem increases performance log-linearlly w.r.t. compute
- RLVR was named by **Tülu 3** [@lambert2024t] and popularized by **DeepSeek R1** [@guo2025deepseek]

|||

![RLVR system](assets/rlvr-system.png)

---

## Comparing classical RL vs LLM RLHF and RLVR

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

![RLHF timeline](assets/rlhf_timeline_tikz.png)

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

## InstructGPT's 3 Step RLHF Recipe


![InstructGPT](assets/instructgpt.jpg)


---

<!-- columns: 45/55 -->
<!-- cite-right: ouyang2022training -->
## Step 1/3: Instruction Fine-tuning (IFT)

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
## Step 3/3 RL against the reward model

Where everything comes together (and RLHF gets its name):
- Sample a batch of prompts $x_i$ from the dataset $\mathcal{D}$
- Generate completions $y_i \sim \pi_\theta(\cdot \mid x_i)$ from the model being trained
- Score them with the reward model $r_\phi(x_i, y_i)$
- Add a **KL penalty** so the policy stays close to the SFT/reference model.^[KL divergence measures how much the current policy differs from the reference model. For discrete outputs, $D_{\mathrm{KL}}(\pi \,\|\, \pi_{\mathrm{ref}})=\mathbb{E}_{y \sim \pi}\!\left[\log \pi(y \mid x)-\log \pi_{\mathrm{ref}}(y \mid x)\right]$. People often colloquially call this the “KL distance” between the models, even though it is not a true metric.]
- Update the policy with PPO in InstructGPT

$$
J(\pi)
=
\mathbb{E}\!\left[r_\phi(x, y)\right]
- \beta D_{\mathrm{KL}}\!\left(\pi \,\|\, \pi_{\mathrm{ref}}\right)
$$

|||

![RLHF training loop](assets/rlhf-overview.png)

---

<!-- rows: 48/52 -->
<!-- cite-right: ouyang2022training -->
## The RLHF objective, unpacked

<div style="text-align: center;">

$$
\max_{\pi} \;
\mathbb{E}_{x \sim D,\; y \sim \pi(\cdot \mid x)}
\underbrace{r_\phi(x, y)}_{\text{maximize the reward}}
- \underbrace{\beta \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}}_{\text{but don't change the model too much}}
$$

</div>

===

<!-- row-columns: 50/50 -->

Prompts $x$ come from a dataset, not an environment.

The policy $\pi_\theta(y \mid x)$ generates a full response $y$, and the reward model $r_\phi(x, y)$ scores whether humans would like that response.

|||

The reference model $\pi_{\mathrm{ref}}$ keeps the policy anchored to the SFT model.

$\beta$ controls the tradeoff between **improving behavior** and **staying close** to what the model already knows.

InstructGPT's answer was:
- a **learned reward model**
- **PPO** as the RL optimizer

---

<!-- columns: 45/55 -->
<!-- cite-right: rafailov2024direct, ouyang2022training -->
## What if we optimize this more directly?

- PPO works, but it is an **online RL loop**
- We must keep sampling completions from the current policy
- We must tune RL hyperparameters and KL penalties carefully
- This led people to ask if the preference data itself could define the objective

$$
\text{Can we learn directly from } (x, y_w, y_l)
\text{ without a separate RL stage?}
$$

|||

**Direct Preference Optimization (DPO)**

- Start from the same KL-regularized RLHF objective
- Eliminate the explicit reward model
- Train directly on preferred vs. rejected responses

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
- \log \sigma \!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right)
$$

---

<!-- rows: 45/55 -->
## How Training Recipes Have Evolved

| | InstructGPT (2022) | Tülu 3 (2024) | DeepSeek R1 (2025) |
|---|---|---|---|
| **IFT data** | ~10K | ~1M | 100K+ |
| **Preference data** | ~100K | ~1M | On-policy |
| **RL stage** | ~100K prompts | ~10K (RLVR) | "Until convergence" |

More data, more preference signals, more RL compute.

===

![Simple RLHF recipe](assets/rlhf-basic.png)

---

<!-- rows: 45/55 -->
## How Training Recipes Have Evolved

| | InstructGPT (2022) | Tülu 3 (2024) | DeepSeek R1 (2025) |
|---|---|---|---|
| **IFT data** | ~10K | ~1M | 100K+ |
| **Preference data** | ~100K | ~1M | On-policy |
| **RL stage** | ~100K prompts | ~10K (RLVR) | "Until convergence" |

More data, more preference signals, more RL compute.

===

![Modern multi-stage recipe](assets/rlhf-complex.png)

---

## The Elicitation Theory

Post-training **extracts latent potential** from the base model

**OLMoE** — same base model, updated only post-training:
- Version 1: **35** benchmark average
- Version 2: **48** benchmark average

Base models determine the *ceiling*. Post-training's job is to **reach it**.

---

## From RLHF to post-training

- The classic **3-step RLHF recipe** became the intellectual center of modern post-training
- Even when recipes changed, people still thought in terms of:
  - instruction tuning
  - a reward / preference signal
  - policy improvement
- Modern post-training extends, simplifies, or scales that template

Placeholder: connect InstructGPT-style RLHF to DPO, RLVR, and reasoning-oriented RL.

---

## o1 scaling post training

<!-- img-align: center -->
<!-- cite-right: openai2024o1 -->

![o1 AIME accuracy](assets/o1.webp)

---

## o1: train-time scaling

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->

text goes here

|||

![o1 AIME accuracy during training](assets/o1-train-time.png)


---

## o1: test-time scaling asd asds 

<!-- columns: 2 -->
<!-- cite-right: openai2024o1 -->
text goes here


|||

![o1 AIME accuracy at test time](assets/o1-test-time.png)

---

## The Scale of Post-Training

- **DeepSeek R1**: RL used ~147K H800 GPU hours (~5% of total training)
- Individual ablation runs: **10–100K GPU hours**
- The trend: **more compute going to post-training every year**

---

<!-- layout: section-break -->

## How We Got Here

---


## The Post-ChatGPT Acceleration

- **Early 2023**: Alpaca era — limited data, impressive but narrow
- **Late 2023**: DPO — direct alignment without a reward model
- **2024**: Complex multi-stage recipes (Llama 3, Tülu 3)
- **2025**: Reasoning via RL (DeepSeek R1, o1)

---

## The Classic RLHF Pipeline

1. **Instruction fine-tuning** — teach Q&A format from examples
2. **Reward model training** — learn a scoring function from human preferences
3. **RL optimization** — optimize the model against the reward model

---


## Where Things Are Heading

- RLHF and RLVR are **complementary** — style vs. capabilities
- Modern recipes use **both** in sequence
- The boundary between them is blurring (generative reward models, self-correction)

---

## Beyond elicitation?

- Maybe post-training does more than just **extract** existing ability
- Long RL runs may reshape how models **reason**, not just how they respond
- Open question: when does scaling RL create **new capabilities**?

---

<!-- rows: 85/15 -->
## Thank You

**rlhfbook.com**

===

<!-- row-columns: 65/35 -->

|||

```builtwith
repo: natolambert/colloquium
```
