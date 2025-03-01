---
prev-chapter: "Direct Alignment"
prev-url: "12-direct-alignment.html"
page-title: Constitutional AI and AI Feedback
next-chapter: "Reasoning Models"
next-url: "14-reasoning.html"
---

# [Incomplete] Constitutional AI and AI Feedback

RL from AI Feedback (RLAIF) is a larger set of techniques for using AI to augment or generate feedback data, including pairwise preferences [@lee2023rlaif]  [@sharma2024critical] [@castricato2024suppressing].

## Trade-offs

1. Human data is high-noise and low-bias,
2. Synthetic preference data is low-noise and high-bias,

Results in many academic results showing how one can substitute AI preference data in RLHF workflows and achieve strong evaluation scores, but shows how the literature of RLHF is separated from industrial best practices.

## Constitutional AI

### Summary

The method of Constitutional AI (CAI), which Anthropic uses extensively in their Claude models, is earliest, large-scale use of synthetic data for RLHF training. 
Constitutional AI has two uses of synthetic data:

1. Critiques of instruction-tune data to follow a set of principles like “Is the answer encouraging violence” or “Is the answer truthful.” When the model generates answers to questions, it checks the answer against the list of principles in the constitution, refining the answer over time. Then, they fine-tune the model on this resulting dataset.
2. Generates pairwise preference data by using a language model to answer which completion was better, given the context of a random principle from the constitution (similar to this paper for principle-guided reward models). Then, RLHF proceeds as normal with synthetic data, hence the RLAIF name.

### Mathematical Formulation

By employing a human-written set of principles, which they term a *constitution*, Bai et al. 2022 use a separate LLM to generate artificial preference and instruction data used for fine-tuning [@bai2022constitutional].
A constitution $\mathcal{C}$ is a set of written principles indicating specific aspects to focus on during a critique phase.
The instruction data is curated by repeatedly sampling a principle $c_i \in \mathcal{C}$ and asking the model to revise its latest output $y^i$ to the prompt $x$ to align with $c_i$. 
This yields a series of instruction variants $\{y^0, y^1, \cdots, y^n\}$ from the principles  $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$ used for critique.
The final data point is the prompt $x$ together with the final completion $y^n$, for some $n$. 

The preference data
%, the focus of this paper, 
is constructed in a similar, yet simpler way by using a subset of principles from $\mathcal{C}$ as context for a feedback model.
The feedback model is presented with a prompt $x$, a set of principles $\{c_0, \cdots, c_n\}$, and two completions $y_0$ and $y_1$ labeled as answers (A) and (B) from a previous RLHF dataset.
The feedback models' probability of outputting either (A) or (B) is recorded as a training sample for the reward model