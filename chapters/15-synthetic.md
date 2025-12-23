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

Humans were the only way to create high enough quality responses to questions to train on them. 
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

Through the years since ChatGPT's release at the end of 2022, we've seen numerous, impactful synthetic datasets -- some include: UltraFeedback [@cui2023ultrafeedback], the first prominent synthetic preference dataset that kickstarted the DPO revolution, or Stanford Alpaca, one of the first chat-style finetuning datasets, in 2023, skill-focused (e.g. math, code, instruction-following), synthetic datasets in Tülu 3 [@lambert2024t], or OpenThoughts 3 and many other synthetic reasoning datasets in 2025 for training thinking models [@guha2025openthoughts].
Most of the canonical references for getting started with industry-grade post-training today involve datasets like Tülu 3 or OpenThoughts 3 above, where quickstart guides often start with smaller, simpler datasets like Alpaca due to far faster training.

A large change is also down to size, where finetuning datasets have grown in the number of prompts, where Alpaca is 52K, OpenThoughts and Tülu 3 are 1M+ samples, and in the length of responses.
Longer responses and more prompts results in the Alpaca dataset being on the order of 10M training tokens, where Tülu is 50X larger at about 500M, and OpenThoughts 3 is bigger still at the order of 10B tokens.

Throughout this transition in capabilities, the role of synthetic data has only grown in language model training. 
Otherwise, there are two clear areas where human data continues to be important. 

1. The role of human data continues to be at the fringe of capabilities in models -- humans must generate data where AI's do not yet have any ability. Once the first strong model exists, synthetic data proliferates.
2. Human preference data is still used in the leading models, even though academic work shows synthetic versions to perform just as well. The role of human preferences is still being established in the literature.

The term distillation has been the most powerful form of discussion around the role of synthetic data in language models. 
Distillation as a term comes from a technical definition of teacher-student knowledge distillation from the deep learning literature [@hinton2015distilling].

Distillation colloquially refers to using the outputs from a stronger model to train a smaller model.
In post-training, this general notion of distillation takes two common forms:

1. As a data engine to use across wide swaths of the post-training process: Completions for instructions, preference data (or Constitutional AI), or verification for RL.
2. To transfer specific skills from a stronger model to a weaker model, which is often done for specific skill such as mathematic reasoning or coding.

The first strategy has grown in popularity as language models evolved to be more reliable than humans at writing answers to a variety of tasks.
GPT-4 class models expanded the scope of this to use distillation of stronger models for complex tasks such as math and code (as mentioned above).
Here, distillation motivates having a model suite where often a laboratory will train a large internal model, such as Claude Opus or Gemini Ultra, which is not released publicly and just used internally to make stronger models.
With open models, common practice is to distill training data from closed API models into smaller, openly available weights [@tunstall2023zephyr].
Within this, curating high-quality prompts and filtering responses from the teacher model is crucial to maximize performance.

Transferring specific skills into smaller language models uses the same principles of distillation -- get the best data possible for training.
Here, many papers have studied using limited datasets from stronger models to improve alignment
[@zhou2023lima], mathematic reasoning [@shridhar2023distilling] [@hsieh2023distilling], 
 and test-time scaling [@muennighoff2025s1].
