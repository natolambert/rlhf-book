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

In modern language model training, RLHF is one component on post-training. 
Post-training is a more complete set of techniques and best-practices to make language models more useful for downstream tasks [@lambert2024t].
Post-training can be summarized as using three optimization methods:

1. Instruction / Supervised Finetuning (SFT),
2. Preference Finetuning (PreFT), and
3. Reinforcement Finetuning (RFT).

This book focused on the second area, **preference finetuning**, which has more complexity than instruction tuning and is far more established than Reinforcement Finetuning.

## Scope of This Book

This book hopes to touch on each of the core steps of doing canonical RLHF implementations. 
It will not cover all the history of the components nor recent research methods, just techniques, problems, and trade-offs that have been proven to occur again and again.

### Chapter Summaries

This book has the following chapters following this Introduction:

**Introductions**:

1. Introduction
2. What are preferences?: The philosophy and social sciences behind RLHF.
3. Optimization and RL: The problem formulation of RLHF.
4. Seminal (Recent) Works: The core works leading to and following ChatGPT.

**Problem Setup**:

1. Definitions: Mathematical reference.
2. Preference Data: Gathering human data of preferences.
3. Reward Modeling: Modeling human preferences for environment signal.
4. Regularization: Numerical tricks to stabilize and guide optimization.

**Optimization**:

1. Instruction Tuning: Fine-tuning models to follow instructions.
2. Rejection Sampling: Basic method for using a reward model to filter data.
3. Policy Gradients: Core RL methods used to perform RLHF.
4. Direct Alignment Algorithms: New PreFT algorithms that do not need RL.

**Advanced (TBD)**:

1. Constitutional AI
2. Synthetic Data
3. Evaluation
4. Reasoning and Reinforcement Finetuning

**Open Questions (TBD)**:

1. Over-optimization
2. Style

### Target Audience

This book is intended for audiences with entry level experience with language modeling, reinforcement learning, and general machine learning. 
It will not have exhaustive documentation for all the techniques, but just those crucial to understanding RLHF.

### How to Use This Book

This book was largely created because there were no canonical references for important topics in the RLHF workflow.
The contributions of this book as supposed to give you the minimum knowledge needed to try a toy implementation or dive into the literature. 
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
