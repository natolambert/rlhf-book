<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "Preference Data"
prev-url: "11-preference-data"
page-title: Synthetic Data
search-title: "Chapter 12: Synthetic Data"
next-chapter: "Tool Use"
next-url: "13-tools"
---

# Synthetic Data

Reinforcement learning from *human feedback* is deeply rooted in the idea of keeping a human influence on the models we are building.
When the first models were trained successfully with RLHF, human data was *the only* viable way to improve the models in this way.

Humans were the only way to create high enough quality responses to questions for training. 
Humans were the only way to collect reliable and specific feedback data to train reward models.

As AI models got better, this assumption rapidly broke down.
The possibility of synthetic data, which is far cheaper and easier to iterate on, enabled the proliferation from RLHF being the center of attention to the idea of a broader "post-training" shaping the models.
This chapter provides a cursory overview of how and why synthetic data is replacing or expanding many pieces of the RLHF pipeline.

One common criticism of synthetic data is **model collapse** -- the idea that repeatedly training on a model’s own generations can progressively narrow the effective training distribution [@shumailov2024ai].
As diversity drops, rare facts and styles are underrepresented, and small mistakes can be amplified across iterations, leading to worse generalization.
In practice, these failures are most associated with self-training on unfiltered, repetitive, single-model outputs; mixing in real/human data, using diverse teachers, deduplication, and strong quality filters largely avoids the collapse regime.
For today’s frontier training pipelines, evidence suggests synthetic data can, and should, be used at scale without the catastrophic regressions implied by the strongest versions of the collapse story [@gerstgrasser2024model] [@feng2024beyond].

The leading models **need synthetic data** to reach the best performance.
Synthetic data in modern post-training encompasses many pieces of training -- language models are used to generate new training prompts from seed examples [@wang2022self], modify existing prompts, generate completions to prompts [@numina_math_7b], provide AI feedback to create preference data [@cui2023ultrafeedback], filter completions [@li2024superfiltering], and much more.
Synthetic data is key to post-training.

The ability for synthetic data to be impactful to this extent emerged with GPT-4 class models.
With early language models, such as Llama 2 and GPT-3.5-Turbo, the models were not reliable enough in generating or supervising data pipelines.
Within 1-2 years, language models were far superior to humans for generating answers.
In the transition from GPT-3.5 to GPT-4 class models, the ability for models to perform LLM-as-a-judge tasks also emerged.
GPT-4 or better models are far more robust and consistent in generating feedback or scores with respect to a piece of content.

Through the years since ChatGPT's release at the end of 2022, we've seen numerous, impactful synthetic datasets -- some include: UltraFeedback [@cui2023ultrafeedback], the first prominent synthetic preference dataset that kickstarted the DPO revolution, or Stanford Alpaca, one of the first chat-style fine-tuning datasets, in 2023, skill-focused (e.g. math, code, instruction-following) synthetic datasets in Tülu 3 [@lambert2024t], or OpenThoughts 3 and many other synthetic reasoning datasets in 2025 for training thinking models [@guha2025openthoughts].
Most of the canonical references for getting started with industry-grade post-training today involve datasets like Tülu 3 or OpenThoughts 3 above, where quickstart guides often start with smaller, simpler datasets like Alpaca due to far faster training.

A large change is also related to dataset size, where fine-tuning datasets have grown in the number of prompts, where Alpaca is 52K, OpenThoughts and Tülu 3 are 1M+ samples, and in the length of responses.
Longer responses and more prompts results in the Alpaca dataset being on the order of 10M training tokens, where Tülu is 50X larger at about 500M, and OpenThoughts 3 is bigger still at the order of 10B tokens.

Throughout this transition, synthetic data has not replaced human data uniformly across the pipeline. 
For **instruction data (SFT)**, synthetic generation has largely won -- distillation from stronger models now produces higher quality completions than most human writers can provide at scale (with some exceptions in the hardest frontier reasoning problems).
For **preference data in RLHF**, the picture is more mixed: academic work shows synthetic preference data performs comparably, yet frontier labs still treat human preference data as a competitive moat. 
For **evaluation**, the split takes a different flavor: LLM-as-a-judge scales the *scoring* of model outputs cost-effectively, but the underlying benchmarks and ground-truth labels still require human creation. 
The pattern is that synthetic data dominates where models exceed human reliability, while humans remain essential at capability frontiers, for establishing ground truth, and for guiding training.

## Distillation with Synthetic Data

The term distillation has been the most powerful form of discussion around the role of synthetic data in language models.
Distillation as a term comes from a technical definition of teacher-student Knowledge Distillation (KD) from the deep learning literature [@hinton2015distilling].

![Traditional knowledge distillation trains a smaller student model to match the soft probability distribution of a larger teacher model using KL divergence loss. Both models process the same input simultaneously, and temperature scaling ($\tau > 1$) softens the distributions to reveal more information about class relationships.](images/knowledge_distillation_tikz.png){#fig:knowledge-distillation}

Distillation colloquially refers to using the outputs from a stronger model to train a smaller model.

![Synthetic data generation in LLM post-training: prompts are passed through a strong model to generate completions, which are paired to create a training dataset. This dataset is then used to fine-tune smaller models via standard supervised learning. More complex pipelines may involve multiple models editing completions, generating preference pairs, or filtering for quality.](images/synthetic_data_distillation_tikz.png){#fig:synthetic-data-generation}
In post-training, this general notion of distillation takes two common forms:

1. As a data engine to use across wide swaths of the post-training process: Completions for instructions, preference data (or Constitutional AI), or verification for RL.
2. To transfer specific skills from a stronger model to a weaker model, which is often done for specific skills such as mathematical reasoning or coding.

The first strategy has grown in popularity as language models evolved to be more reliable than humans at writing answers to a variety of tasks.
GPT-4 class models expanded the scope of this to use distillation of stronger models for complex tasks such as math and code (as mentioned above).
Here, distillation motivates having a model suite where often a laboratory will train a large internal model, such as Claude Opus or Gemini Ultra, which is not released publicly and just used internally to make stronger models.
With open models, common practice is to distill training data from closed API models into smaller, openly available weights [@tunstall2023zephyr].
Within this, curating high-quality prompts and filtering responses from the teacher model is crucial to maximize performance.

Transferring specific skills into smaller language models uses the same principles of distillation -- get the best data possible for training.
Here, many papers have studied using limited datasets from stronger models to improve alignment [@zhou2023lima], mathematical reasoning [@shridhar2023distilling] [@hsieh2023distilling], and test-time scaling [@muennighoff2025s1].

The synthetic-data methods in the rest of this chapter are all ways of crafting data recipes that use language-model outputs directly inside training pipelines.

## The Path to On-Policy Distillation

While distillation generally has become a standard approach for post-training language models, a resurgence of interest in the specific sub-area of teacher-student knowledge distillation has accompanied the shift of post-training recipes towards reasoning and agentic models.
Examples of leading models trained with new forms of knowledge distillation include Alibaba's Qwen3 [@yang2025qwen3], Xiaomi's MiMo-V2-Flash [@mimo2025flash], Zhipu AI's GLM-5 [@glm5team2026glm5], and DeepSeek-V4-Pro [@deepseekai2026deepseekv4].

Distillation belongs in this chapter because many modern uses of synthetic data in post-training are, in practice, distillation-inspired pipelines: a stronger model produces labels, completions, logits, critiques, or other supervision, and a student model is trained on that signal.
At the same time, the technical literature on distillation is growing into its own set of post-training methods, especially as on-policy and self-distillation recipes become more common.
For now, we cover it here as part of the synthetic-data toolkit, but future versions of this book may warrant a dedicated chapter on distillation as a training tool alongside instruction finetuning, reinforcement learning, etc.

The original literature introduced it specifically as a way to train a *student* model from an already trained, stronger, and/or bigger *teacher* network [@hinton2015distilling].
KD is known as a technique that uses *soft* training labels, as opposed to the one-hot labels used in standard objectives like next-token prediction with cross-entropy loss.
The objectives over soft labels look at the distribution over all possible next tokens or predictions, rather than just whether or not the single predicted token was correct, and train the student distribution to match the teacher distribution.

KD generally can be applied to any deep learning problem, e.g. predicting a single class of an input.
In order to apply it specifically to the autoregressive style of language models, the loss can be decomposed to make a per-token distribution-matching loss.
In 2016, Kim & Rush applied KD to have a student model learn from *sequences* generated by a teacher model [@kim-rush-2016-sequence].

Let $s$ be the source sentence or prompt, $u = (u_1,\ldots,u_J)$ be a complete output sequence from the teacher model, $\mathcal{V}$ be the output vocabulary (possible tokens in the tokenizer), $q$ be the teacher distribution over next-tokens, and $p$ be the student distribution.
We use $u$ here as a neutral symbol for a complete teacher output sequence, reserving $a$ for the student-sampled completion/action sequence in the on-policy/RL notation below.
Note that their paper calls this word-level distillation, but for modern language models this is best read as per-token distribution matching over the tokenizer vocabulary, since the paper predates modern sub-word tokenizers:

$$
\mathcal{L}_{\mathrm{WORD-KD}}
= -\sum_{j=1}^{J}\sum_{k=1}^{|\mathcal{V}|}
q(u_j = k \mid s, u_{<j})\log p(u_j = k \mid s, u_{<j}).
$$ {#eq:word_kd}

This has the ordinary cross-entropy form $-\sum_z q(z)\log p(z)$.
At each position $j$, the teacher distribution $q$ assigns probability to every possible next token $k \in \mathcal{V}$, and the student is penalized when its distribution $p$ puts low probability on tokens the teacher considers likely.

Sequence-level distillation instead treats $\mathcal{U}$ as the space of possible output sequences and matches the student to the teacher distribution over full sequences.
Because the sum over all complete sequences $u \in \mathcal{U}$ is intractable, requiring summing over an exponential number of potential sequences, Kim & Rush approximate the teacher distribution over sequences with a point mass on a single high-probability teacher output $\hat{u}$.
Here $\hat{u}$ is a sequence produced by beam search with the teacher model, so $\hat{u} = \mathrm{BeamSearch}_q(s) \approx \arg\max_{u \in \mathcal{U}} q(u \mid s)$:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SEQ-KD}}(s)
= -\sum_{u \in \mathcal{U}} q(u \mid s)\log p(u \mid s)
\approx -\log p(\hat{u} \mid s) \\
= -\sum_{j=1}^{|\hat{u}|}\log p(\hat{u}_j \mid s, \hat{u}_{<j}).
\end{aligned}
$$ {#eq:sequence_kd}

As we transition to the popular variants of KD with modern models, we'll refer to this style of training as *offline* KD -- as in the generations for training the student model are generated a priori.

Before proceeding, two connections are useful.

First, there was a series of popular models trained with offline KD, such as the classifiers DistilBERT [@sanh2019distilbert] and TinyBERT [@jiao2020tinybert], which combined other improvements in language models with offline distillation (notably, not *sequence* distillation because these encoder models were not distilled for multi-token autoregressive prediction).

Second, we can make the connection to the thorough coverage of Kullback-Leibler (KL) divergence in Chapter 15, because the cross-entropy objective used above is closely related to KL divergence.
For a teacher distribution $q$ and student distribution $p$, cross-entropy is defined as

$$
H(q,p) = -\sum_z q(z)\log p(z).
$$ {#eq:kd_cross_entropy}

This has the same form as @eq:word_kd and the first term of @eq:sequence_kd.
Cross-entropy also can be decomposed into the entropy of the teacher distribution and a KL divergence:

$$
\begin{aligned}
H(q,p)
&= H(q) + D_{\mathrm{KL}}(q\|p) \\
&= -\sum_z q(z)\log q(z)
+ \sum_z q(z)\log\frac{q(z)}{p(z)}.
\end{aligned}
$$ {#eq:kd_forward_kl}

The first term, $H(q)$, only depends on the teacher.
Thus, when the teacher is fixed and the source of training data, minimizing cross-entropy is equivalent to minimizing the forward KL, $D_{\mathrm{KL}}(q\|p)$, from teacher to student.
This is the KL direction used by offline KD and SFT-like training.

These *offline* KD algorithms had a few limitations that motivated on-policy variants.
The offline nature of the learning meant that the student models could suffer from a distribution mismatch between the teacher model and sequences generated by the student at inference time.
For example, the forward KL objective can push student models to overestimate low-probability regions of the teacher distribution.
Together, these issues were an opening for *on-policy* distillation (OPD).

This train-test gap is known as **exposure bias** [@song2026surveyonpolicydistillationlarge].
Offline KD samples teacher trajectories $u \sim \pi_T(\cdot \mid s)$ and minimizes the per-token KL on the resulting prefixes,

$$
\mathcal{L}_{\mathrm{train}}(\theta)
= \mathbb{E}_{u \sim \pi_T(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(\pi_T(\cdot \mid s, u_{<t}) \;\|\; \pi_\theta(\cdot \mid s, u_{<t})\right).
$$ {#eq:exposure_train}

At inference the student instead rolls out under its own policy, so the quantity that actually matters is the expected task loss along *its own* trajectories,

$$
\mathcal{L}_{\mathrm{test}}(\theta)
= \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\,\mathcal{L}_{\mathrm{task}}(s, a).
$$ {#eq:exposure_test}

Exposure bias is the direct consequence of the inequality $\pi_T(\cdot \mid s) \neq \pi_\theta(\cdot \mid s)$: the prefixes $(s, u_{<t})$ visited during training and the prefixes $(s, a_{<t})$ visited at test time are drawn from different state-visitation distributions, so the student is supervised on a set of states distinct from those it acts on.

The core shift to on-policy distillation is the idea that we can tweak the optimization by sampling from the student model and measuring its distance to the teacher distribution, rather than sampling from the teacher model.
MiniLLM noted the need to shift to a reverse KL optimization (we explain intuitively why this target can be better in Chapter 15) and proposed using KD loss functions within an online policy-gradient RL framework [@gu2024minillm].
Other concurrent work [@agarwal2024policy] showed the promise of on-policy KD and connected the iterative process of generating from the student and grading with a teacher to imitation-learning work from the RL literature.
To make the connection, one such imitation-learning algorithm, DAgger, iteratively trains an agent that acts in the world with its learned policy and is given feedback from an oracle policy on what action it should have taken, which can then be used to update its policy [@ross2011reduction].

The cost of this gap can be quantified through imitation-learning bounds.
Suppose the student matches the teacher within an expected per-step error $\epsilon$ on the training distribution,

$$
\mathbb{E}_{u \sim \pi_T(\cdot \mid s)}\!\left[\mathbb{I}\!\left(\pi_\theta(\cdot \mid s, u_{<t}) \neq \pi_T(\cdot \mid s, u_{<t})\right)\right] \leq \epsilon.
$$ {#eq:dagger_perstep}

The DAgger analysis [@ross2011reduction] shows that the expected loss accumulated along a length-$L$ trajectory sampled from the student then scales quadratically in $L$ [@song2026surveyonpolicydistillationlarge]:

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[\sum_{t=1}^{L} \ell\!\left(s, a_{<t}\right)\right] \leq O(\epsilon L^2).
$$ {#eq:dagger_trajectory}

This $O(\epsilon L^2)$ compounding is especially pronounced for modern LLMs, which routinely generate sequences spanning thousands of tokens.
A single suboptimal token shifts the prefix slightly out-of-distribution, and the model, having never seen this perturbed prefix, is more likely to err again, leading to degraded or hallucinatory text.
On-policy distillation resolves this by replacing the teacher-rollout expectation in @eq:dagger_perstep with a student-rollout expectation: the student confronts its own mistakes, receives teacher feedback on the specific out-of-distribution states it visits, and learns recovery behaviors, reducing the compounding from $O(\epsilon L^2)$ to $O(\epsilon L)$.

For on-policy distillation, let $s$ be a prompt, $a = (a_1,\ldots,a_L)$ be a completion sampled from the current student policy $\pi_\theta(\cdot \mid s)$, and let $s_t = (s, a_{<t})$ be the token-level state at step $t$.
The teacher policy $\pi_T$ is fixed, so the objective compares the student's next-token distribution to the teacher's distribution on states induced by the student.
Because the expectation samples from $\pi_\theta$ and the student distribution is on the left side of $D_{\mathrm{KL}}(\pi_\theta \| \pi_T)$, this is a reverse-KL objective:

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_T(\cdot \mid s_t)\right).
$$ {#eq:opd_reverse_kl}

Here, we have shifted to the expectation notation, as used extensively in Chapter 6, which covers the fundamental RL policy-gradient algorithms, as the optimization is solved by sampling trajectories and numerically estimating the gradient.
This shift to the sampling framework acts as a natural transition to modern LLM training infrastructure with RL, which is designed to rapidly alternate between generating tokens from the current policy being trained and taking learning updates.

In fact, recent implementations of OPD take this integration of KD with RL a step further, where the KD distance is taken directly as a reward signal within the RL optimization.
A canonical implementation is to substitute the negative per-token contribution to the reverse KL distance as the advantage within an RL algorithm [@lu2025onpolicy].
For a sampled token $a_t$ at state $s_t$, the token-level log-probability gap can be written as an advantage-like signal:

$$
A_t^{\mathrm{OPD}}
= \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t).
$$ {#eq:opd_kl_advantage}

Using the negative per-token KL contribution turns minimization into a maximization signal: sampled tokens the teacher rates above the student receive positive advantage, and tokens the teacher rates below the student receive negative advantage.
The teacher log-prob gap acts like dense token-level feedback, providing potentially more useful learning feedback than the sparse verifiable rewards or reward model outputs.

This setup can even be expanded further, where multiple teacher models are used to teach one final model.
These teachers can be specific specialist models, e.g. for a domain such as math or code, or a previous, intermediate training checkpoint.
For each teacher, a contribution weight can be chosen per prompt or task type in the training batch, in order to create Multi-Teacher On-Policy Distillation (MOPD) [@mimo2025flash].
For multiple teachers, let $\pi_{T_k}$ be teacher $k$ and let $w_k(s)$ be its prompt-dependent mixture weight (with $\sum_k w_k(s) = 1$) within the reverse KL loss:

$$
\mathcal{L}_{\mathrm{MOPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t \sum_k w_k(s) D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_{T_k}(\cdot \mid s_t)\right).
$$ {#eq:mopd_objective}

In large-scale post-training, this can enable further scaling of recipes across growing organizations.
Multiple groups can work on high-quality expert models, which can serve as teacher models down the line for the final student model, as done for [@deepseekai2026deepseekv4] and [@mimo2025flash].

There are many ways to combine OPD with other areas investigated in this book, such as using the reverse KL as advantage in addition to other forms of advantage computation, such as GRPO's group-level normalization, which enables more complex reward shaping.
KD methods are unusual among post-training methods because they often require the student and teacher to share a tokenizer, since the supervision can be per-token feedback from another LLM.
Extended approaches, such as On-Policy Self-Distillation (OPSD), have a language model verify a completion either itself or with external tools to act as a teacher with privileged information, so it can train a weaker version of itself [@zhao2026selfdistilled].

## AI Feedback

Soon after the explosion of growth in RLHF, RL from AI Feedback (RLAIF) emerged as an alternative approach where AIs could approximate the human data piece of the pipeline and accelerate experimentation or progress.
AI feedback, generally, is a larger set of techniques for using AI to augment or generate data explaining the quality of a certain input (which can be used in different training approaches or evaluations), which started with pairwise preferences [@lee2023rlaif]  [@sharma2024critical] [@castricato2024suppressing].
There are many motivations to using RLAIF to either entirely replace human feedback or augment it. 
Within the RLHF process, AI feedback is known most for its role within the preference data collection and the related reward model training phase (of which constitutional AI is a certain type of implementation).
In this chapter, we focus on the general AI feedback and this specific way of using it in the RLHF training pipeline, and we cover more ways of understanding or using synthetic data later in this book.

As AI feedback matured, its applications expanded beyond simply replacing human preference labels. 
The same LLM-as-a-judge infrastructure that enabled cheaper preference data collection also enabled scalable evaluation (see Chapter 16), and more recently, rubric-based rewards that extend RL training to domains without verifiable answers -- a frontier explored later in this chapter.

### Balancing AI and Human Feedback Data

AI models are far cheaper than humans at generating a specific quantity of feedback, with a single piece of human preference data costing as of writing this on the order of $1 or higher (or even above $10 per prompt), AI feedback with a frontier AI model, such as GPT-4o costs less than $0.01. 
Beyond this, the cost of human labor is remaining roughly constant, while the performance of leading models at these tasks continues to increase while price-per-performance decreases.
This cost difference opens the market of experimentation with RLHF methods to an entire population of people previously priced out.

Other than price, AI feedback introduces different *tradeoffs* on performance than human feedback, which are still being investigated in the broader literature.
AI feedback is far more predominant in its role in evaluation of the language models that we are training, as its low price allows it to be used across a variety of large-scale tasks where the cost (or time delay) in human data would be impractical.
All of these topics are deeply intertwined -- AI feedback data will never fully replace human data, even for evaluation, and the quantity of AI feedback for evaluation will far outperform training because far more people are evaluating than training models.

The exact domains and applications -- i.e. chat, safety, reasoning, mathematics, etc. -- where AI feedback data outperforms human data is not completely established. 
Some early work in RLAIF shows that AI feedback can completely replace human data, touting it as an effective replacement [@lee2023rlaif] and especially when evaluated solely on chat tasks [@cui2023ultrafeedback] [@yuan2025selfrewardinglanguagemodels]. 
Early literature studying RLHF after ChatGPT had narrow evaluation suites focused on the "alignment" of models that act as helpful assistants across a variety of domains (discussed further in Chapter 17).
Later work takes a more nuanced picture, where the optimal equilibrium on a broader evaluation set, e.g. including some reasoning tasks, involves routing a set of challenging data-points to accurately label to humans, while most of the data is sent for AI feedback [@miranda2024hybrid] [@xu2025rlthf].
While there are not focused studies on the balance between human and AI feedback data for RLHF across broader domains, there are many technical reports that show RLHF generally can improve these broad suite of evaluations, some that use DPO, such as Ai2's Tülu 3 [@lambert2024t] & Olmo 3 [@teamolmo2025olmo3], or HuggingFace's SmolLM 3 [@bakouch2025smollm3], and others that use online RLHF pipelines, such as Nvidia's work that uses a mix of human preference data from Scale AI and LLM-based feedback (through the helpsteer line of work [@wang2024helpsteer] [@wang2024helpsteer2] [@wang2024helpsteer2p] [@wang2025helpsteer3]): Nemotron Nano 3 [@nvidia2025nemotron3nano], Nemotron-Cascade [@wang2025nemotron], or Llama-Nemotron reasoning models [@bercovich2025llamanemotron].

Overall, where AI feedback and related methods are obviously extremely useful to the field, it is clear that human data has not been completely replaced by these cheaper alternatives.
Many hypotheses exist, but it is not studied if human data allows finer control of the models in real-world product settings or for newer training methods such as character training (an emerging set of techniques that allow you to precisely control the personality of a model, covered in Chapter 17).
For those getting started, AI feedback should be the first attempt, but for pipelines that're scaling to larger operations the eventual transition to include human feedback is likely.

The term RLAIF was introduced in Anthropic's work *Constitutional AI: Harmlessness from AI Feedback* [@bai2022constitutional], which resulted in initial confusion in the AI community over the relationship between the two methods in the title of the paper (Constitutional AI and AI Feedback).
Since the release of the Constitutional AI (CAI) paper and the formalization of RLAIF, RLAIF has become a default method within the post-training and RLHF literatures -- there are far more examples than one can easily enumerate.
The relationship should be understood as CAI was the example that kickstarted the broader field of RLAIF.

A rule of thumb for the difference between human data and AI feedback data is as follows:

1. Human data is high-noise and low-bias. This means that collection and filtering of the data can be harder, but when wrangled it'll provide a very reliable signal.
2. Synthetic preference data is low-noise and high-bias. This means that AI feedback data will be easier to start with, but can have tricky, unintended second-order effects on the model that are systematically represented in the data.

This book highlights many academic results showing how one can substitute AI preference data in RLHF workflows and achieve strong evaluation scores [@miranda2024hybrid], but broader industry trends show how the literature of RLHF is separated from more opaque, best practices.
Across industry, human data is often seen as a substantial moat and a major technical advantage.

### Building Specific LLMs for Judgement

As RLAIF methods have become more prevalent, many have wondered if we should be using the same models for generating responses as those for generating critiques or ratings.
Specifically, the calibration of the LLM-as-a-judge used has come into question.
Several works have shown that LLMs are inconsistent evaluators [@wang2023large] and prefer their own responses over responses from other models (coined self-preference bias) [@panickssery2024llm].

As a result of these biases, many have asked: Would a solution be to train a separate model just for this labeling task?
Multiple models have been released with the goal of substituting for frontier models as a data labeling tool, such as critic models Shepherd [@wang2023shepherd] and CriticLLM [@ke2023critiquellm] or models for evaluating response performance akin to Auto-J [@li2023generative], Prometheus [@kim2023prometheus], Prometheus 2 [@kim2024prometheus], or Prometheus-Vision [@lee2024prometheus] but they are not widely adopted in documented training recipes.
Some find scaling inference via repeated sampling [@brown2024large] [@zhao2025sample] [@kalra2025verdict], self-refinement [@madaan2023self], or tournament ranking [@pace2024west] provides a better estimate of the true judgement or higher-quality preference pairs.
Other calibration techniques co-evolve the generation and judgement capabilities of the model [@wu2024meta].
It is accepted that while biases exist, the leading language models are trained extensively for this task -- as its needed for both internal operations at AI labs and is used extensively by customers -- so it is generally not needed to train your own judge, unless your task involves substantial private information that is not exposed on the public internet.

## Constitutional AI

The method of Constitutional AI (CAI), which Anthropic uses in their Claude models, is the earliest documented, large-scale use of synthetic data for RLHF training. 
Constitutional AI involves generating synthetic data in two ways:

1. Critiques of instruction-tuned data to follow a set of principles like "Is the answer encouraging violence" or "Is the answer truthful." When the model generates answers to questions, it checks the answer against the list of principles in the constitution, refining the answer over time. Then, the model is fine-tuned on this resulting dataset.
2. Generates pairwise preference data by using a language model to answer which completion was better, given the context of a random principle from the constitution (similar to research for principle-guided reward models [@sun2024salmon]). Then, RLHF proceeds as normal with synthetic data, hence the RLAIF name.

Largely, CAI is known for the second half above, the preference data, but the methods introduced for instruction data are used in general data filtering and synthetic data generation methods across post-training.

CAI can be formalized as follows.

By employing a human-written set of principles, which they term a *constitution*, Bai et al. 2022 use a separate LLM to generate artificial preference and instruction data used for fine-tuning [@bai2022constitutional].
A constitution $\mathcal{C}$ is a set of written principles indicating specific aspects to focus on during a critique phase.
The instruction data is curated by repeatedly sampling a principle $c_i \in \mathcal{C}$ and asking the model to revise its latest output $y^i$ to the prompt $x$ to align with $c_i$. 
This yields a series of instruction variants $\{y^0, y^1, \cdots, y^n\}$ from the principles  $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$ used for critique.
The final data point is the prompt $x$ together with the final completion $y^n$, for some $n$. 

The preference data is constructed in a similar, yet simpler way by using a subset of principles from $\mathcal{C}$ as context for a feedback model.
The feedback model is presented with a prompt $x$, a set of principles $\{c_0, \cdots, c_n\}$, and two completions $y_0$ and $y_1$ labeled as answers (A) and (B) from a previous RLHF dataset.
The new datapoint is generated by having a language model select which output (A) or (B) is both higher quality and more aligned with the stated principle.
In earlier models this could be done by prompting the model with `The answer is: `, and then looking at which logit (A or B) had a higher probability, but more commonly is now handled by a model that'll explain its reasoning and then select an answer -- commonly referred to as a type of generative reward model [@mahan2024generative].

### Further Reading on CAI

There are many related research directions and extensions of Constitutional AI, but few of them have been documented as clear improvements in RLHF and post-training recipes.

- OpenAI has released a Model Spec [@openai2024modelspec], which is a document stating the intended behavior for their models, and stated that they are exploring methods for alignment where the model references the document directly (which could be seen as a close peer to CAI). OpenAI has continued and trained their reasoning models such as o1 with a method called Deliberative Alignment [@guan2024deliberative] to align the model while referencing these safety or behavior policies.
- Anthropic has continued to use CAI in their model training, updating the constitution Claude uses [@Anthropic2023ClaudesConstitution] and experimenting with how population collectives converge on principles for models and how that changes model behavior when they create principles on their own and then share them with Anthropic to train the models [@ganguli2023].
- The open-source community has explored replications of CAI applied to open datasets [@Huang2024cai] and for explorations into creating dialogue data between LMs [@lambert2024self].
- Other work has used principle-driven preferences or feedback with different optimization methods.
Sun et al. 2023 [@sun2023principledriven] uses principles as context for the reward models, which was used to train the Dromedary models [@sun2024salmon].
Glaese et al. 2022 [@glaese2022improving] uses principles to improve the accuracy of human judgments in the RLHF process.
Liu et al. 2025 [@liu2025inference] train a reward model to generate its own principles at inference time, and use these to deliver a final score.
Franken et al. 2024 [@franken2024self] formulate principle-following as a mutual information maximization problem that the pretrained model can learn with no labels.

## Rubrics: Prompt-Specific AI Feedback for Training

AI feedback's role in training grew in late 2024 and into 2025 as the field looked for avenues to scale reinforcement learning with verifiable rewards (see Chapter 7).
The idea of rubrics emerged as a way to get nearly-verifiable criteria for prompts that do not have clearly verifiable answers. 
This would allow a model to try to generate multiple answers to a problem and update (with RL) towards the best answers.
This idea is closely related to other methods discussed in this chapter, and likely began functioning as the LLM judges and synthetic data practices improved across the industry.
Now, RL with rubrics as rewards is established in providing meaningful improvements across skills such as scientific reasoning or factuality [@gunjal2025rubrics; @viswanathan2025checklists; @rezaei2025onlinerubrics; @liu2025openrubrics].

An example rubric is shown below with its associated prompt [@liu2025openrubrics]:
```text
**Prompt**: As a museum curator, can you suggest five obscure artifacts that would be perfect for a "Mysteries of the Ancient World" exhibit? Each artifact should come from a different culture and time period, with a brief description of their historical significance and mysterious origins. These artifacts should leave visitors wondering about the secrets and lost knowledge of our past. Thank you for your expertise in bringing this exhibit to life.

** Rubric**: 
1. The response includes exactly five distinct artifacts as requested. [Hard Rule] 
2. The response ensures each artifact originates from a different culture and time period. [Hard Rule] 
3. The response provides a brief description of each artifact's historical significance. [Hard Rule] 
4. The response provides a brief description of each artifact's mysterious origins or unexplained aspects. [Hard Rule] 
5. The response conveys a sense of intrigue and mystery that aligns with the theme of the exhibit. [Hard Rule] 
6. The response clearly and accurately communicates information in a well-organized and coherent manner. [Principle] 
7. The response demonstrates precision and clarity by avoiding unnecessary or irrelevant details. [Principle] 
8. The response uses informative and engaging language that stimulates curiosity and critical thinking. [Principle] 
9. The response shows thoughtful selection by ensuring each example contributes uniquely to the overall theme without redundancy. [Principle] 
10. The response maintains consistency in style and format to enhance readability and comprehension. [Principle]
```

The `[Hard Rule]` and `[Principle]` are specific tags to denote the priority of a certain piece of feedback. Other methods of indicating importance can be used, such as simple priority numbers.

Rubric generation is generally done per-prompt in the training data, which accumulates meaningful synthetic data costs in preparation.
To alleviate this, a general rubric is often applied as a starting point per-domain, and then the fine-grained rubric scores per-prompt are assigned by a supervising language model to guide the feedback for training.
An example prompt to generate a rubric for a science task is shown below [@gunjal2025rubrics]:

```text
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. 
Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one of these domains. 
Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. 
Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7-20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. 

For example:
- Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
before the event equals the total mechanical energy after the event.
- Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
energy plus initial potential energy equals final kinetic energy plus final potential energy.
- Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
energy, to illustrate how energy shifts within the system.
- Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
conservation of mechanical energy.

3. weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
- If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.
Now, given the question and reference_answer, generate the rubric as described. 
The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance.
```

Another, simpler example follows as [@rezaei2025onlinerubrics]:

```text
SYSTEM:
You generate evaluation rubrics for grading an assistant's response to a user prompt.

Rubric design rules:
- Each criterion must be atomic (one thing), objective as possible, and written so a grader can apply it consistently.
- Avoid redundant/overlapping criteria; prefer criteria that partition different failure modes.
- Make criteria self-contained (don't rely on unstated context).
- Include an importance weight for each criterion.

Output format (JSON only):
{
  "initial_reasoning": "<brief rationale for what matters for this prompt>",
  "rubrics": [
    {
      "reasoning": "<why this criterion matters>",
      "criterion": "<clear, testable criterion>",
      "weight": <integer 1-10>
    },
    ...
  ]
}

USER:
User prompt:
{prompt}

Generate the rubric JSON now.
```

As you can see, the prompts can be very detailed and are tuned to the training setup.

Rubrics with RL training are going to continue to evolve beyond their early applications to instruction following [@he2025advancedif], deep research [@shao2025drtulu], evaluating deep research agents [@sharma2025researchrubrics], or long-form generation [@ruan2025expertlongbench].
