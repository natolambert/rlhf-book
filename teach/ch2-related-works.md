---
title: "Chapter 2: Key Related Works"
author: "Nathan Lambert"
date: "2026"
---

# Key Related Works

A Brief History of RLHF

Nathan Lambert

---

## Why History Matters

- RLHF and its related methods are **very new**
- Many procedures were formalized only recently
- The field is **rapidly evolving** — expect uncertainty and change
- Seminal papers were often for applications **totally distinct** from modern LLMs
- For comprehensive surveys: Wirth et al. 2017, Casper et al. 2023, Kaufmann et al. 2025

<!-- notes: This is intentionally focused on recent work that led to ChatGPT, not a comprehensive review -->

---

<!-- layout: section-break -->

## Origins to 2018: RL on Preferences

---

## Early Foundations

**TAMER** (Knox & Stone, 2008)
- First approach similar to modern RLHF
- Humans iteratively scored agent actions to learn a reward model

**COACH** (MacGlashan et al., 2017)
- Actor-critic where human feedback (positive and negative) tunes the advantage function

**Deep TAMER** (Warnell et al., 2018)
- Extended TAMER with neural networks

---

## The Primary Reference: Christiano et al. 2017

Applied RLHF to preferences between **trajectories in Atari games**

Key insight: Humans choosing between trajectories can be **more effective** than directly interacting with the environment

The core RLHF loop:
1. Agent generates trajectory segments
2. Humans compare pairs of segments
3. Reward predictor trained from comparisons (asynchronously)
4. Agent maximizes predicted reward

<!-- notes: This followed DeepMind's seminal DQN work. The shift was from environment rewards to learned preference rewards. -->

---

## Transition to Alignment

**Ibarz et al. 2018**: Expanded with more direct reward modeling approaches

**Leike et al. 2018**: Proposed reward models as a method for studying **alignment**, not just solving RL problems

This reframing — from tool to alignment technique — set the stage for RLHF's role in language models.

---

<!-- layout: section-break -->

## 2019–2022: RLHF Meets Language Models

---

## The First RLHF on Language Models

**Ziegler et al. 2019** — *Fine-Tuning Language Models from Human Preferences*

- Many canonical terms formalized in this paper:
  - **Reward models**
  - **KL distances**
  - **Feedback diagrams**
- Only the evaluation tasks and capabilities differed from today's methods
- This was the template that everything built on

---

## Key Applications (2020–2022)

| Paper | Application |
|-------|-------------|
| Stiennon et al. 2020 | Text summarization |
| Wu et al. 2021 | Recursive summarization of books |
| Nakano et al. 2021 (WebGPT) | Browser-assisted question answering |
| Menick et al. 2022 (GopherCite) | Supporting answers with citations |
| Glaese et al. 2022 (Sparrow) | General dialogue |
| **Ouyang et al. 2022 (InstructGPT)** | **Instruction following — precursor to ChatGPT** |

---

## Foundational Research (2021–2022)

Three seminal papers defined key areas for RLHF's future:

1. **Reward model over-optimization** (Gao, Schulman, Hilton 2023)
   - RL optimizers can over-fit to preference-trained models
   - Established scaling laws for this phenomenon

2. **LMs as alignment laboratories** (Askell et al. 2021)
   - Language models as a general area for alignment study

3. **Red teaming** (Ganguli et al. 2022)
   - Process of assessing safety of RLHF-trained models

Early open-source tools emerged: **TRL** (von Werra et al.), **TrlX** (Havrilla et al.)

---

<!-- layout: section-break -->

## 2023 to Present: The ChatGPT Era

---

## ChatGPT and the RLHF Spotlight

OpenAI's ChatGPT announcement was explicit:

> "We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup."

This single sentence made RLHF the most discussed technique in AI research overnight.

---

## RLHF in Leading Models

RLHF is well-documented in:

- **Anthropic**: Constitutional AI for Claude (Bai et al. 2022)
- **Meta**: Llama 2 (2023) and Llama 3 (2024)
- **Nvidia**: Nemotron 4 340B (2024)
- **Ai2**: Tülu 3 (2024)
- **OpenAI**: GPT series, o1 reasoning models

Companies that embraced RLHF early ended up winning out.

---

## The Expanding Frontier

RLHF is growing into **Preference Fine-tuning (PreFT)**, with new applications:

- **Process reward models** for intermediate reasoning steps (Lightman et al. 2024)
- **Direct Preference Optimization** and variants (Rafailov et al. 2023)
- **Execution feedback** from code or math (Kumar et al. 2024, Singh et al. 2023)
- **Reasoning models** inspired by OpenAI's o1 (2024)
- **Self-correction** via RL (Kumar et al. 2025)

---

## Timeline Summary

| Era | Key Development |
|-----|----------------|
| **2008** | TAMER — first human-in-the-loop reward learning |
| **2017** | Christiano et al. — RLHF on Atari |
| **2019** | Ziegler et al. — first RLHF on language models |
| **2022** | InstructGPT — RLHF for instruction following |
| **2022** | ChatGPT — RLHF goes mainstream |
| **2023** | DPO — direct alignment without reward model |
| **2024** | Tülu 3, Llama 3 — complex multi-stage recipes |
| **2024–25** | DeepSeek R1, o1 — reasoning via RL |

---

<!-- layout: title -->

# Key Takeaway

RLHF went from Atari game preferences (2017) to the core technique behind ChatGPT (2022) in just **five years**

The field is still young and rapidly evolving
