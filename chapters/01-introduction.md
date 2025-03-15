---
prev-chapter: "Home"
prev-url: "https://rlhfbook.com/"
page-title: Introduction
next-chapter: "Key Related Works"
next-url: "02-related-works.html"
---

# Introduction

Reinforcement learning from Human Feedback (RLHF) is a technique used to incorporate human information into AI systems.
RLHF emerged primarily as a method to solve hard to specify problems.
Its early applications were often in control problems and other traditional domains for reinforcement learning (RL).
RLHF became most known through the release of ChatGPT and the subsequent rapid development of large language models (LLMs) and other foundation models.

The basic pipeline for RLHF involves three steps.
First, a language model that can follow user questions must be trained (see Chapter 9).
Second, human preference data must be collected for the training of a reward model of human preferences (see Chapter 7).
Finally, the language model can be optimized with a RL optimizer of choice, by sampling generations and rating them with respect to the reward model (see Chapter 3 and 11).
This book details key decisions and basic implementation examples for each step in this process.

RLHF has been applied to many domains successfully, with complexity increasing as the techniques have matured.
Early breakthrough experiments with RLHF were applied to deep reinforcement learning [@christiano2017deep], summarization [@stiennon2020learning], following instructions [@ouyang2022training], parsing web information for question answering [@nakano2021webgpt], and "alignment" [@bai2022training].

In modern language model training, RLHF is one component of post-training. 
Post-training is a more complete set of techniques and best-practices to make language models more useful for downstream tasks [@lambert2024t].
Post-training can be summarized as using three optimization methods:

1. Instruction / Supervised Finetuning (IFT/SFT), where we teach formatting and for base of instruction following abilities. This is largely about learning *features* in language.
2. Preference Finetuning (PreFT), where we align to human preferences (and get smaller bump in capabilities at the same time). This is largely about *style* of language and subtle human preferences that are hard to quantify. 
3. Reinforcement Finetuning (RFT). The newest type of post-training that boosts performance on verifiable domains.

This book focuses on the second area, **preference finetuning**, which has more complexity than instruction tuning and is far more established than Reinforcement Finetuning.
That being said, RLHF colloquially *is* what led to modern post-training.
Soon after the release of ChatGPT, RLHF encompassed all of post-training.
The foundations of RLHF involve far more than preferences alone and this book provides introductions to all the related topics.

## What Does RLHF Do?

The biggest question around RLHF, yet one that is still hard to answer, is "What does RLHF training offer models?"
The core role of this book, beyond teaching the techniques for doing RLHF, is to distill intuition as to *why* RLHF is crucial to modern AI models.
In recent years, language models shifted from academic experiments studied in the purview of benchmarks to general purpose technology.
RLHF is at the core of this transition.

The most compelling view of how RLHF works is to think of how *style* applies to interactions you have with language models.
The style, or format, of information presented is crucial to how it is learned.
This has always been the case for examples such as coursework, but is normally applied in the background and not considered directly.

Modern research has established RLHF as a general method to integrate subtle stylistic and related behavioral features into the models.
Compared to other techniques for post-training, such as instruction finetuning, RLHF generalizes far better across domains [@kirk2023understanding] [@chu2025sft] -- helping create effective general purpose models.

Intuitively, this can be seen in how the optimization techniques are applied. 
Instruction finetuning is training the model to predict the next certain token when the text preceding is close to examples it has seen.
It is optimizing the model to more regularly output specific features in text. This is a per-token update.

RLHF on the other hand tunes the responses on the response level rather than looking at the next token specifically.
Additionally, it is telling the model what a *better* response looks like, rather than a specific response it should learn.
RLHF also shows a model which type of response it should avoid, i.e. negative feedback. 
The training to achieve this is often called a *contrastive* loss function and is referenced throughout this book.

While this flexibility is a major advantage of RLHF, it comes with implementation challenges. 
Largely, these center on *how to control the optimization.* 
As we will cover in this book, implementing RLHF often requires training a reward model, of which best practices are not strongly established and depend on the area of application.
With this, the optimization itself is prone to *over-optimization* because our reward signal is at best a proxy objective, requiring regularization.
With these limitations, effective RLHF requires a strong starting point, so RLHF cannot be a solution to every problem alone and needs to be approached in a broader lens of post-training.

Due to this complexity, implementing RLHF is far more costly than simple instruction finetuning and can come with unexpected challenges such as length bias [@singhal2023long] [@park2024disentangling]. 
For projects where performance matters, RLHF is established as being crucial to achieving a strong finetuned model, but it is more expensive in compute, data costs, and time.

## How We Got Here

Why does this book make sense now? How much still will change?

Post-training, the craft of eliciting powerful behaviors from a raw pretrained language model, has gone through many seasons and moods since the release of ChatGPT that sparked the renewed interest in RLHF. 
In the era of Alpaca [@alpaca], Vicuna [@vicuna2023],  [@koala_blogpost_2023], and Dolly [@DatabricksBlog2023DollyV1], a limited number of human datapoints with extended synthetic data in the style of Self-Instruct were used to normally fine-tune the original LLaMA to get similar behavior to ChatGPT. 
The benchmark for these early models was fully vibes (and human evaluation) as we were all so captivated by the fact that these small models can have such impressive behaviors across domains. 
It was justified excitement.

Open post-training was moving faster, releasing more models, and making more noise than its closed counterparts. 
Companies were scrambling, e.g. DeepMind merging with Google or being started, and taking time to follow it up. 
There are phases of open recipes surging and then lagging behind.

The era following Alpaca et al., the first lag in open recipes, was one defined by skepticism and doubt on reinforcement learning from human feedback (RLHF), the technique OpenAI highlighted as crucial to the success of the first ChatGPT. 
Many companies doubted that they needed to do RLHF. 
A common phrase – “instruction tuning is enough for alignment” – was so popular then that it still holds heavy weight today despite heavy obvious pressures against it. 

This doubt of RLHF lasted, especially in the open where groups cannot afford data budgets on the order of \$100K to \$1M. 
The companies that embraced it early ended up winning out.
Anthropic published extensive research on RLHF through 2022 and is now argued to have the best post-training [@askell2021general] [@bai2022training] [@bai2022constitutional]. 
The delta between open groups, struggling to reproduce, or even knowing basic closed techniques, is a common theme.

The first shift in open alignment methods and post-training was the story of Direct Preference Optimization (DPO) [@rafailov2024direct]. 
The DPO paper, posted in May of 2023, didn’t have any clearly impactful models trained with it going through the fall of 2023. 
This changed with the releases of a few breakthrough DPO models – all contingent on finding a better, lower, learning rate. 
Zephyr-Beta [@tunstall2023zephyr], Tülu 2 [@ivison2023camels], and many other models showed that the DPO era of post-training had begun. 
Chris Manning literally thanked me for “saving DPO.” 
This is how fine the margins are on evolutions of best practices with leading labs being locked down. 
Open post-training was cruising again.

Preference-tuning was something you needed to do to meet the table stakes of releasing a good model since late 2023. 
The DPO era continued through 2024, in the form of never-ending variants on the algorithm, but we were very far into another slump in open recipes. 
Open post-training recipes had saturated the extent of knowledge and resources available.  
A year after Zephyr and Tulu 2, the same breakout dataset, UltraFeedback is arguably still state-of-the-art for preference tuning in open recipes [@cui2023ultrafeedback]. 

At the same time, the Llama 3.1 [@dubey2024llama] and Nemotron 4 340B [@adler2024nemotron] reports gave us substantive hints that large-scale post-training is much more complex and impactful. 
The closed labs are doing full post-training – a large multi-stage process of instruction tuning, RLHF, prompt design, etc. – where academic papers are just scratching the surface. 
Tülu 3 represented a comprehensive, open effort to build the foundation of future academic post-training research [@lambert2024t].

Today, post-training is a complex process involving the aforementioned training objectives applied in various orders in order to target specific capabilities.
This book is designed to give a platform to understand all of these techniques, and in coming years the best practices for how to interleave them will emerge.

The primary areas of innovation in post-training are now in reinforcement finetuning, reasoning training, and related ideas. 
This newer methods build extensively on the infrastructure and ideas of RLHF, but are evolving far faster.
This book is written to capture the first stable literature for RLHF after its initial period of rapid change.

## Scope of This Book

This book hopes to touch on each of the core steps of doing canonical RLHF implementations. 
It will not cover all the history of the components nor recent research methods, just techniques, problems, and trade-offs that have been proven to occur again and again.

### Chapter Summaries


This book has the following chapters:

#### Introductions

Reference material useful throughout the book.

1. Introduction: Overview of RLHF and what this book provides.
2. Seminal (Recent) Works: Key models and papers in the history of RLHF techniques.
3. Definitions: Mathematical definitions for RL, language modeling, and other ML techniques leveraged in this book.

#### Problem Setup & Context

Context for the big picture problem RLHF is trying to solve.

4. RLHF Training Overview: How the training objective for RLHF is designed and basics of understanding it.
5. What are preferences?: Why human preference data is needed to fuel and understand RLHF.
6. Preference Data: How preference data is collected for RLHF.

#### Optimization Tools

The suite of techniques used to optimize language models to align them to human preferences.
This is a serial presentation of the techniques one can use to solve the problems proposed in the previous chapters.

7. Reward Modeling: Training reward models from preference data that act as an optimization target for RL training (or for use in data filtering).
8. Regularization: Tools to constrain these optimization tools to effective regions of the parameter space.
9. Instruction Tuning: Adapting language models to the question-answer format.
10. Rejection Sampling: A basic technique for using a reward model with instruction tuning to align models.
11. Policy Gradients: The core RL techniques used to optimize reward models (and other signals) throughout RLHF.
12. Direct Alignment Algorithms: Algorithms that optimize the RLHF objective direction from pairwise preference data rather than learning a reward model first.

#### Advanced (TBD)

Newer RLHF techniques and discussions that are not clearly established, but are important to current generations of models.

13. Constitutional AI and AI Feedback
14. Reasoning and Reinforcement Finetuning
15. Synthetic Data
16. Evaluation

#### Open Questions (TBD)

Fundamental problems and discussions for the long-term evolution of how RLHF is used.

16. Over-optimization
17. Style and Information


### Target Audience

This book is intended for audiences with entry level experience with language modeling, reinforcement learning, and general machine learning. 
It will not have exhaustive documentation for all the techniques, but just those crucial to understanding RLHF.

### How to Use This Book

This book was largely created because there were no canonical references for important topics in the RLHF workflow.
The contributions of this book are supposed to give you the minimum knowledge needed to try a toy implementation or dive into the literature. 
This is *not* a comprehensive textbook, but rather a quick book for reminders and getting started.
Additionally, given the web-first nature of this book, it is expected that there are minor typos and somewhat random progressions -- please contribute by fixing bugs or suggesting important content on [GitHub](https://github.com/natolambert/rlhf-book).

### About the Author

Dr. Nathan Lambert is a RLHF researcher contributing to the open science of language model fine-tuning.
He has released many models trained with RLHF, their subsequent datasets, and training codebases in his time at the Allen Institute for AI (Ai2) and HuggingFace.
Examples include [Zephyr-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), [Tulu 2](https://huggingface.co/allenai/tulu-2-dpo-70b), [OLMo](https://huggingface.co/allenai/OLMo-7B-Instruct), [TRL](https://github.com/huggingface/trl), [Open Instruct](https://github.com/allenai/open-instruct), and many more. 
He has written extensively on RLHF, including [many blog posts](https://www.interconnects.ai/t/rlhf) and [academic papers](https://scholar.google.com/citations?hl=en&user=O4jW7BsAAAAJ&view_op=list_works&sortby=pubdate).

## Future of RLHF

With the investment in language modeling, many variations on the traditional RLHF methods emerged.
RLHF colloquially has become synonymous with multiple overlapping approaches. 
RLHF is a subset of preference fine-tuning (PreFT) techniques, including Direct Alignment Algorithms (See Chapter 12).
RLHF is the tool most associated with rapid progress in "post-training" of language models, which encompasses all training after the large-scale autoregressive training on primarily web data. 
This textbook is a broad overview of RLHF and its directly neighboring methods, such as instruction tuning and other implementation details needed to set up a model for RLHF training.

As more successes of fine-tuning language models with RL emerge, such as OpenAI's o1 reasoning models, RLHF will be seen as the bridge that enabled further investment of RL methods for fine-tuning large base models.

<!-- This is the first paragraph of the introduction chapter.

## First: Images

This is the first subsection. Please, admire the gloriousnes of this seagull:

![A cool seagull.](images/seagull.png)

A bigger seagull:

![A cool big seagull.](images/seagull.png){ width=320px }

## Second: Tables

This is the second subsection.


Please, check [First: Images] subsection.

Please, check [this](#first-images) subsection.

| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table.

## Third: Equations

Formula example: $\mu = \sum_{i=0}^{N} \frac{x_i}{N}$

Now, full size:

$$\mu = \sum_{i=0}^{N} \frac{x_i}{N}$$

And a code sample:

```rb
def hello_world
  puts "hello world!"
end

hello_world
```

Check these unicode characters: ǽß¢ð€đŋμ

## Fourth: Cross references

These cross references are disabled by default. To enable them, check the
_[Cross references](https://github.com/wikiti/pandoc-book-template#cross-references)_
section on the README.md file.

Here's a list of cross references:

- Check @fig:seagull.
- Check @tbl:table.
- Check @eq:equation.

![A cool seagull](images/seagull.png){#fig:seagull}

$$ y = mx + b $$ {#eq:equation}

| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table. {#tbl:table} -->
