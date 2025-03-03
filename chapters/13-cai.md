---
prev-chapter: "Direct Alignment"
prev-url: "12-direct-alignment.html"
page-title: Constitutional AI & AI Feedback
next-chapter: "Reasoning Models"
next-url: "14-reasoning.html"
---

# [Incomplete] Constitutional AI & AI Feedback

RL from AI Feedback (RLAIF) is a larger set of techniques for using AI to augment or generate feedback data, including pairwise preferences [@lee2023rlaif]  [@sharma2024critical] [@castricato2024suppressing].
There are many motivations to using RLAIF to either entirely replace human feedback or augment it. 
AI models are far cheaper than humans, with a single piece of human preference data costing on the order of $1 or higher (or even above $10 per prompt), AI feedback with a frontier AI model, such as GPT-4o costs less than $0.01. 
This cost differences opens the market of experimentation with RLHF methods to an entire population of people previously priced out.
Other than price, AI feedback introduces different *tradeoffs* on performance than human feedback, which are still being investigated.
The peak performance for AI feedback is at least in the same ballpark of human data on skill-based evaluations, but it is not studied if human data allows finer control of the models in real-world product settings or for newer training methods such as character training.

The term RLAIF was introduced in Anthropic's work *Constitutional AI: Harmlessness from AI Feedback* [@bai2022constitutional], which resulted in initial confusion in the AI community over the relationship between the methods.
Since the release of the Constitutional AI (CAI) paper and the formalization of RLAIF, RLAIF has become a default method within the post-training and RLHF literatures -- there are far more examples than one can easily enumerate.
The relationship should be understood as CAI was the example that kickstarted the broader field of RLAIF.


## Trade-offs

A rule of thumb for the difference between human data and AI feedback data is as follows:

1. Human data is high-noise and low-bias,
2. Synthetic preference data is low-noise and high-bias,

Results in many academic results showing how one can substitute AI preference data in RLHF workflows and achieve strong evaluation scores, but shows how the literature of RLHF is separated from industrial best practices.

## Constitutional AI

The method of Constitutional AI (CAI), which Anthropic uses extensively in their Claude models, is earliest, large-scale use of synthetic data for RLHF training. 
Constitutional AI has two uses of synthetic data:

1. Critiques of instruction-tune data to follow a set of principles like “Is the answer encouraging violence” or “Is the answer truthful.” When the model generates answers to questions, it checks the answer against the list of principles in the constitution, refining the answer over time. Then, they fine-tune the model on this resulting dataset.
2. Generates pairwise preference data by using a language model to answer which completion was better, given the context of a random principle from the constitution (similar to this paper for principle-guided reward models). Then, RLHF proceeds as normal with synthetic data, hence the RLAIF name.

Largely, CAI is known for the second half above, the preference data, but the methods introduced for instruction data are used in general data filtering and synthetic data generation methods across post-training.

### Mathematical Formulation

By employing a human-written set of principles, which they term a *constitution*, Bai et al. 2022 use a separate LLM to generate artificial preference and instruction data used for fine-tuning [@bai2022constitutional].
A constitution $\mathcal{C}$ is a set of written principles indicating specific aspects to focus on during a critique phase.
The instruction data is curated by repeatedly sampling a principle $c_i \in \mathcal{C}$ and asking the model to revise its latest output $y^i$ to the prompt $x$ to align with $c_i$. 
This yields a series of instruction variants $\{y^0, y^1, \cdots, y^n\}$ from the principles  $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$ used for critique.
The final data point is the prompt $x$ together with the final completion $y^n$, for some $n$. 

The preference data is constructed in a similar, yet simpler way by using a subset of principles from $\mathcal{C}$ as context for a feedback model.
The feedback model is presented with a prompt $x$, a set of principles $\{c_0, \cdots, c_n\}$, and two completions $y_0$ and $y_1$ labeled as answers (A) and (B) from a previous RLHF dataset.
The feedback models' probability of outputting either (A) or (B) is recorded as a training sample for the reward model

## Specific LLMs for Judgement

Multiple models have been released with the goal of substituting for frontier models as a data labeling tool, such as Shepherd [@wang2023shepherd], Prometheus [@kim2023prometheus], and Prometheus 2 [@kim2024prometheus] but they are widely adopted.


## Further Reading

TODO search for references of constitutional AI outside of Anthropic
- OpenAI mentioned in their announcement of the Model Spec that they are investigating training models referencing it, which would be similar, then deliberative alignment
- Cite Claude [@Anthropic2023ClaudesConstitution], Collective CAI [@ganguli2023], and other Anthropic work
- TODO cite SDSD [@lambert2024self] & HuggingFace's mini replication [@Huang2024cai], see SDSD's related work
- Other work has used principle-driven preferences or feedback with different optimization methods.
[@sun2023principledriven] uses principles as context for the reward models, which was used to train the Dromedary models [@sun2024salmon].
[@glaese2022improving] uses principles to improve the accuracy of human judgements in the RLHF process.


More: https://chatgpt.com/share/67c335c2-fb3c-8005-832d-ccbaf3ca64f8