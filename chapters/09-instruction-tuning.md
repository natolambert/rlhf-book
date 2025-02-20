---
prev-chapter: "Regularization"
prev-url: "08-regularization.html"
page-title: Instruction Finetuning
next-chapter: "Rejection Sampling"
next-url: "10-rejection-sampling.html"
---

# Instruction Finetuning

Early language models were only trained to predict the next tokens in a sequence and were not adapted to any specific tasks.
Around the release of GPT-3 [@brown2020language], language models were still primarily used via in-context learning where examples where shown to the model and then it was asked to complete a similar task.

This was the combination of two trends -- historically in the natural language processing (NLP) literature, models were trained for a specific task.
Here, as seen with one example where bigger models generalize better, multiple results showed how standardizing the approach of task data can enable dramatically different downstream performance.
Prominent examples of unifying the framework for tasks includes *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5 models) [@raffel2020exploring], *Finetuned Language Models Are Zero-Shot Learners*  (FLAN dataset)[@wei2021finetuned], *Multitask Prompted Training Enables Zero-Shot Task Generalization* (T0 models) [@sanh2021multitask], and *Cross-Task Generalization via Natural Language Crowdsourcing Instructions* (Natural Instructions dataset) [@mishra2021cross].
These insights led to the era of *finetuning* language models. 
Historically, until RLHF and related methods, all finetuning was **instruction finetuning** (IFT), also known as **supervised finetuning**.

Since, instruction finetuning, also called colloquially just *instruction tuning*, has matured and is standard practice across many language modeling pipelines.
At its core, IFT is the simplest method for adapting language models to a desired task.
It serves as the foundation for RLHF by preparing the model for a format of instructions that is known common, question-answering, and is the first tool used by those attempting to apply modern techniques to new domains.

## Chat templates and the structure of instructions

## Best practices of instruction tuning

Specific tasks, e.g. LIMA can work. In general, things like Tülu 2/3 work best
Self instruct for synthetic data
InstructGPT shows that RLHF began to transition IFT away from SOTA