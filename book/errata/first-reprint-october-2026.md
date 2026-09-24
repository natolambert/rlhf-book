# RLHF Book — first reprint typesetting changes

October 2026 · Print manuscript only · Source lines refer to the current supplied LaTeX files.

Cumulative changes from the original print source. Apply each replacement at the listed locations.

## Chapter 1 — `Chapter1.tex`

- L41: `true object` → `true objective`.
- L48: `RLHF and related models` → `RLHF and related methods`.
- L167: `markdown formatting` → `Markdown formatting`.
- L244: `ChatBotArena` → `Chatbot Arena`.
- L295: `MT Bench` → `MT-Bench`.
- L303: `OLMo 3.1` → `Olmo 3.1`.
- L312: `Deep-\linebreak[4]Mind merging with Google) or being started` → `Deep-\linebreak[4]Mind merging with Google Brain or new labs being started)`.
- L321: `better, lower, learning rate` → `better, lower learning rate`.

## Chapter 2 — `Chapter2.tex`

- L73: `Nvidia's Nemotron` → `NVIDIA's Nemotron`.

## Chapter 3 — `Chapter3.tex`

- L240: `100,000 pairwise prompts` → `100,000 prompts with pairwise completions`.
- L268: `ChatBotArena` → `Chatbot Arena`.

## Chapter 4 — `Chapter4.tex`

- L251: `OLMo 3` → `Olmo 3`.

## Chapter 5 — `Chapter5.tex`

- L308: Replace the two-class-head description with a small head that outputs a scalar logit at every token.
- L318: `r \in {0,1}` → `r \in \{0,1\}` (restore visible set braces).
- L410: `continues to use` → `continues to be inspired by` (Cobbe et al.'s original definition).
- L448: Remove the extra closing `)` from the PRM predictor definition.
- L450: `HuggingFace's TRL` → `Hugging Face's TRL`.

## Chapter 6 — `Chapter6.tex`

- L71: Return definition: `R_{t+1}`, `R_{t+2}`, `R_{t+k+1}` → `r_t`, `r_{t+1}`, `r_{t+k}`.
- L80: `The return definition can also be estimated as` → `The return can also be written recursively as`.
- L85: Recursive return: `R_{t+1}` → `r_t`.
- Lines 157, 215, 278, 288, 393, 408, 487: Trajectory expectations: `\tau \sim \pi_\theta` → `\tau \sim p_\theta` (7 occurrences; retain existing subscript braces).
- L221: Monte Carlo sampling: `\tau_i \sim \pi_\theta` → `\tau_i \sim p_\theta`.
- L332: Vanilla policy gradient: `R_t` → `G_t`.
- L332: Place `G_t` before `\nabla_\theta` in the vanilla policy-gradient summand (equation 6.20).
- L1103: Policy-ratio description: `reference model` → `old policy that generated the batch`.

## Chapter 7 — `Chapter7.tex`

- Lines 426, 475, 486: `OLMo 3` → `Olmo 3` (4 occurrences; twice on L426).
- L450: `Phi 4` → `Phi-4`.
- L470: `GPT-OSS` → `gpt-oss`.

## Chapter 11 — `Chapter11.tex`

- L57: `ChatBotArena` → `Chatbot Arena`.
- L273: `Nvidia GPUs` → `NVIDIA GPUs`.
- L300: `good or great` → `good and great` (both occurrences in the sentence).

## Chapter 12 — `Chapter12.tex`

- L1: `\chapter{Synthetic data}` → `\chapter{Synthetic data \& distillation}`.
- L201: `train a weaker version of itself` → `improve its own performance`.
- L223: `Nvidia's work` → `NVIDIA's work`.
- L248: `CriticLLM` → `CritiqueLLM`.

## Chapter 14 — `Chapter14.tex`

- L163: `LlamaGuard` → `Llama Guard`.

## Chapter 15 — `Chapter15.tex`

- L347: Section heading: `Other types of regularization` → `Other tools to control optimization`.
- L349: `These two examples that follow` → `These examples that follow`.
- L351: Subsection heading: `Pretraining gradients` → `Pretraining gradients in RL`.
- L373: Add the missing subsection heading `Next-token accuracy in DPO` above the existing DPO/NLL text.
- L402: Subsection heading: `Margin-based regularization` → `Margin-based regularization in reward modeling`.

## Chapter 16 — `Chapter16.tex`

- L14: `SWE-Bench` → `SWE-bench`.
- L200: Remove incorrect FLAN expansion
- L295: `SWE-Bench-Verified` → `SWE-bench Verified`.

## Chapter 17 — `Chapter17.tex`

- L31: `ChatBotArena` → `Chatbot Arena`.

## `Appendix_B.tex`

- L24: `heavy markdown use` → `heavy Markdown use`.
- L160: `MT Bench` → `MT-Bench`.

## `Brief_Lines.txt`

- L15: `\numberline {12}{Synthetic data}` → `\numberline {12}{Synthetic data \& distillation}`.

## `TOC_Lines.txt`

- L235: `\numberline {12}{Synthetic data}` → `\numberline {12}{Synthetic data \& distillation}`.

## `RLHF_Bib.bib`

- L1033: `title={Chatbot arena: An open platform for evaluating llms by human preference},` → `title={{Chatbot Arena}: An Open Platform for Evaluating {LLMs} by Human Preference},`.
- L1318: `title={Judging llm-as-a-judge with mt-bench and chatbot arena},` → `title={Judging {LLM}-as-a-Judge with {MT-Bench} and {Chatbot Arena}},`.
- L1639: `Kimi k1. 5` → `Kimi k1.5`.
- L1681: `LLM Trainin` → `LLM Training`.

## `References.tex`

- Lines 163, 552, 1404, 1627, 2370, 3053, 3384, 3402: `OLMo 3` → `Olmo 3` (8 occurrences).
- Lines 647, 2918: `chatbot arena` → `Chatbot Arena` (both “Judging LLM-as-a-judge…” entries).
- Lines 2032, 3028: `Chatbot arena` → `Chatbot Arena` (both “Chatbot arena: An open platform…” entries).
