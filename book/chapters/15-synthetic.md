---
prev-chapter: "Tool Use & Function Calling"
prev-url: "14.5-tools"
page-title: Synthetic Data & Distillation
next-chapter: "Evaluation & Prompting"
next-url: "16-evaluation"
---

# Synthetic Data & Distillation

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

Through the years since ChatGPT's release at the end of 2022, we've seen numerous, impactful synthetic datasets -- some include: UltraFeedback [@cui2023ultrafeedback], the first prominent synthetic preference dataset that kickstarted the DPO revolution, or Stanford Alpaca, one of the first chat-style fine-tuning datasets, in 2023, skill-focused (e.g. math, code, instruction-following), synthetic datasets in Tülu 3 [@lambert2024t], or OpenThoughts 3 and many other synthetic reasoning datasets in 2025 for training thinking models [@guha2025openthoughts].
Most of the canonical references for getting started with industry-grade post-training today involve datasets like Tülu 3 or OpenThoughts 3 above, where quickstart guides often start with smaller, simpler datasets like Alpaca due to far faster training.

A large change is also related to dataset size, where fine-tuning datasets have grown in the number of prompts, where Alpaca is 52K, OpenThoughts and Tülu 3 are 1M+ samples, and in the length of responses.
Longer responses and more prompts results in the Alpaca dataset being on the order of 10M training tokens, where Tülu is 50X larger at about 500M, and OpenThoughts 3 is bigger still at the order of 10B tokens.

Throughout this transition, synthetic data has not replaced human data uniformly across the pipeline. 
For **instruction data (SFT)**, synthetic generation has largely won —- distillation from stronger models now produces higher quality completions than most human writers can provide at scale (with some exception in the hardest, frontier reasoning problems). 
For **preference data in RLHF**, the picture is more mixed: academic work shows synthetic preference data performs comparably, yet frontier labs still treat human preference data as a competitive moat. 
For **evaluation**, the split takes a different flavor: LLM-as-a-judge scales the *scoring* of model outputs cost-effectively, but the underlying benchmarks and ground-truth labels still require human creation. 
The pattern is that synthetic data dominates where models exceed human reliability, while humans remain essential at capability frontiers, for establishing ground truth, and for guiding training.

The term distillation has been the most powerful form of discussion around the role of synthetic data in language models.
Distillation as a term comes from a technical definition of teacher-student knowledge distillation from the deep learning literature [@hinton2015distilling].

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

## On-Policy Distillation

The synthetic data distillation described above is an **offline** method: the teacher generates data once, and the student trains on that fixed dataset.
This approach is simple and doesn't require a live teacher during training, but it suffers from a fundamental problem -- **train-test mismatch**.
The student only sees examples of perfect teacher behavior during training.
At test time, when the student makes a mistake, it enters a distribution of states it never encountered during training and may not know how to recover.

**On-policy distillation** addresses this mismatch by sampling outputs from the student model itself, then getting feedback from the teacher on those samples [@agarwal2024policy].
Rather than learning only from perfect teacher demonstrations, the student learns how to recover from its own mistakes.
This approach is inspired by the DAgger (Dataset Aggregation) algorithm from imitation learning in robotics [@pmlr-v15-ross11a], which showed that training on the learner's own trajectory distribution leads to better generalization than training on expert demonstrations alone.

![On-policy distillation (GKD): prompts are fed to the student model which samples its own outputs. Both student and teacher compute logits on these student-generated sequences. The reverse KL divergence $D_{KL}(\pi_\theta \| \pi_T)$ is minimized, providing mode-seeking behavior where the student learns to match the teacher's preferred outputs. Unlike offline distillation where students only see perfect teacher demonstrations, on-policy distillation exposes the student to its own imperfect generations, addressing train-test mismatch and enabling recovery from mistakes.](images/on_policy_distillation_tikz.png){#fig:on-policy-distillation}

Beyond train-test mismatch, offline distillation also suffers from **model underspecification**: the student is often not expressive enough to fit the teacher's full distribution.
Forward KL (MLE) training can lead to unnatural student-generated samples because the student tries to cover all teacher modes, even those it cannot represent well.

The key insight connecting on-policy distillation to RL is through the lens of KL divergence direction.
Standard synthetic data distillation minimizes the **forward KL divergence** $D_{\text{KL}}(\pi_{\text{teacher}} \| \pi_{\text{student}})$, which is mode-covering -- the student tries to place probability mass everywhere the teacher does.
On-policy distillation instead minimizes the **reverse KL divergence** $D_{\text{KL}}(\pi_{\text{student}} \| \pi_{\text{teacher}})$, which is mode-seeking -- the student focuses on producing high-quality outputs that match the teacher's preferred modes.
An alternative is **Jensen-Shannon Divergence (JSD)**, which provides a symmetric middle ground between forward and reverse KL.

$$\mathcal{L}_{\text{on-policy}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ D_{\text{KL}}\left(\pi_\theta(\cdot|x, y_{<t}) \| \pi_{\text{teacher}}(\cdot|x, y_{<t})\right) \right]$$ {#eq:on-policy-distillation}

The practical implementation is remarkably elegant: take any RLHF training framework, remove the reward term, and swap the KL anchor from the reference policy to the teacher model.
In standard RLHF, the objective is $\max_\theta \mathbb{E}[r(x,y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$.
In on-policy distillation, we set $r(x,y) = 0$ and replace $\pi_{\text{ref}}$ with $\pi_{\text{teacher}}$, yielding a pure KL minimization toward the teacher.
This means on-policy distillation can leverage all the infrastructure built for RLHF -- the same sampling procedures, the same optimization algorithms, and the same distributed training setups.

The two KL directions lead to different output characteristics.
Forward KL (offline distillation) tends to produce more diverse but potentially lower-quality outputs, as the student must cover all modes of the teacher distribution.
Reverse KL (on-policy distillation) produces higher-quality but less diverse outputs, as the student can focus on the teacher's strongest modes.
In practice, a mixture of both objectives often works well.

On-policy distillation can also be combined with reward maximization for simultaneous distillation and reinforcement learning:

$$\mathcal{L}_{\text{combined}}(\theta) = \mathbb{E}_{x, y \sim \pi_\theta} \left[ r(x, y) \right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{teacher}})$$ {#eq:combined-distillation-rl}

This formulation uses the teacher as both a source of knowledge to distill and a regularizer that prevents the student from deviating too far during RL.
The approach was used in Gemini's post-training, demonstrating its effectiveness at scale.

For reasoning model distillation (e.g., distilling chain-of-thought capabilities), synthetic data distillation gets approximately 80-90% of the way to teacher performance, but on-policy methods with logit-level feedback can close the remaining gap.
The additional compute investment is worthwhile because distillation is done once but the resulting model is served billions of times -- meaningful capability improvements compound over the model's deployment lifetime.
