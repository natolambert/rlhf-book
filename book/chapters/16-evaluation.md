<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "Regularization"
prev-url: "15-regularization"
page-title: Evaluation
search-title: "Chapter 16: Evaluation"
next-chapter: "Crafting Model Character and Products"
next-url: "17-product"
---

# Evaluation

*Evaluation* is the set of techniques used to understand the quality and influence of the training processes detailed in this book.
It is normally expressed through *benchmarks* -- examples of popular benchmarks include Massive Multitask Language Understanding (MMLU), Graduate-Level Google-Proof Q&A (GPQA), SWE-Bench (SWE is short for Software Engineering), and MATH -- which are discrete sets of questions or environments designed to measure a specific property of a model.
Evaluation is an ever-evolving approach; this chapter presents vignettes of popular evaluation regimes throughout the early history of RLHF so you can understand the details, the failure modes, and the common themes that will carry forward into the future of language modeling.

The key to understanding language model evaluation, particularly with post-training, is that the current popular evaluation regimes represent a reflection of the popular training best practices and goals.
Although challenging evaluations drive progress in language models to new areas, the majority of evaluation is designed around building useful signals for new models.

Evaluation for RLHF and post-training went through a few distinct phases in its early history:

1. **Early Chat Phase**: Early models trained with RLHF or preference tuning targeted evaluations focused on capturing the chat performance of a model, especially relative to known strong models such as GPT-4. Early examples include MT-Bench [@zheng2023judging], AlpacaEval [@dubois2024length], and Arena-Hard [@li2024crowdsourced]. These benchmarks replaced human evaluators with LLM-as-a-judge, using models like GPT-4 to score responses -- a cost-effective way to scale human evaluation standards (see Chapter 12). These approaches introduced their own challenges, particularly around length bias, where longer responses tend to receive higher scores regardless of quality. Appendix B covers how this susceptibility to gaming shaped RLHF's early reputation and led to length-controlled variants of these benchmarks. Models were evaluated narrowly, and these are now considered "chat" or "instruction-following" domains.
2. **Multiskill Era**: Over time, common practice established that RLHF can be used to improve more skills than just chat. For example, the Tülu evaluation suite included tasks on knowledge (MMLU [@hendrycks2020measuring], PopQA [@mallen2023llm_memorization], TruthfulQA [@lin2021truthfulqa]), reasoning (BigBenchHard [@suzgun2022challenging], DROP [@dua2019drop]), math (MATH [@hendrycksmath2021], GSM8K [@cobbe2021gsm8k]), coding (HumanEval [@chen2021codex], HumanEval+ [@evalplus]), instruction following [@zhou2023instructionfollowingevaluationlargelanguage], and safety (a composite of many evaluations). This reflects the domain in which post-training is embraced as a multifaceted solution beyond safety and chat.
3. **Reasoning and Tools**: The current era for post-training is defined by a focus on challenging reasoning and tool use problems. These include much harder knowledge-intensive tasks, such as GPQA Diamond [@rein2023gpqa] and Humanity's Last Exam [@phan2025hle], intricate software engineering tasks such as SWE-Bench+ [@aleithan2024swebenchplus] and LiveCodeBench [@jain2024livecodebench], and challenging math problems exemplified by recent American Invitational Mathematics Examination (AIME) contests.

Beyond this, new domains will evolve.
As AI becomes more of an industrialized field, the incentives of evaluation are shifting and becoming multistakeholder.
Since the release of ChatGPT, private evaluations such as the Scale Leaderboard [@scale2024seal], community-driven evaluations such as Arena [@chiang2024chatbot], and third-party evaluation companies such as ArtificialAnalysis and Epoch AI have proliferated.
Throughout this chapter, we will include details that map to how these evaluations were implemented and understood.

## Prompting Formatting

*Prompting* language models is a simple action in itself, and a fairly natural one, but it is also considered a craft or art that we can practice and refine [@schulhoff2024prompt].
A *prompt* is the way of structuring information and context for a language model.
For common interactions, the prompt is relatively basic.
For advanced scenarios, a well-crafted prompt will mean success or failure on a specific one-off use case.

When it comes to evaluation, prompting techniques can have a substantial effect on the performance of the model.
Some prompting techniques -- such as formatting, discussed later -- can make a model's performance drop from 60% to near 0.
Similarly, a change of prompt can help models learn better during training.
Colloquially, prompting a model well can give the subjective experience of using future models, unlocking performance outside of normal use.

The gains from prompting are generally smaller than in core areas like improving the data or training algorithms, but they can be substantial in the final product.
The bigger takeaway is that when training a strong, leading model, it is easier to break it and cause performance to plummet than to find a little more performance.

Prompting well with modern language models can involve preparing an entire report for the model to respond to (often with thousands of tokens of generated text).
This behavior is downstream of many changes in how language model performance has been measured and understood.

### Few-Shot Prompting and Log-Likelihood Scoring

Early language models were used only as intelligent autocomplete.
To use these models in a more open-ended way, multiple examples were shown to the model, followed by a prompt that was an incomplete phrase.
This was called *few-shot* or *in-context learning* [@brown2020language], and at the time instruction tuning or RLHF was not involved.
In the case of popular evaluations, it looked like this:

```text
You are a helpful assistant. Below are example interactions to guide your style:

### Example 1
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

### Example 2
User: "Who wrote the novel '1984'?"
Assistant: "George Orwell wrote '1984.'"

# Now continue the conversation using the same style.
User: "Can you explain what a neural network is?"
Assistant:
```

Here, there are multiple ways to evaluate an answer.
Consider a few-shot prompt for a question in the style of MMLU, where the model has to choose between multiple answers:

```text
Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4. What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous on a closed interval [a,b], then f must attain both a maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle's Theorem

Correct Answer:
```

To have a language model provide an answer here, we could either generate a token based on some sampling parameters and see if the answer was correct, A, B, C, or D (formatting like this was proposed in [@robinson2023leveraging]), or we could look at the log-probabilities of each token and mark the task as correct if the correct answer was more likely.

Let's dig into these evaluation details for a moment.
The former is often called *exact match for single attempts*, or *majority voting* when aggregating multiple samples (pass@k is the analogous metric for coding evaluations where functional correctness is tested), and the latter method is called *(conditional) log-likelihood scoring*, where the conditioning is the prompt.
The core difference is that sampling from the underlying probability distribution naturally adds randomness, and the log-probabilities that a model outputs over its tokens are static (when we ignore minor numerical differences).

Log-likelihood scoring has two potential implementations.
First, we could look at the probability of the letter (A) or the answer "The Mean Value Theorem."
Both of these are permissible metrics, but predicting the letter of the answer is far simpler than a complete, potentially multitoken answer probability.
Log-likelihood scoring is more common in pretraining evaluation, where models lack the question-and-answer format needed for exact match, whereas exact match is standard in post-training [@teamolmo2025olmo3].

Exact match has different problems, such as requiring rigid format suffixes (e.g., `The answer is:`) and using regular expressions to detect answers anywhere in generated text (e.g., looking for `(C)` or the answer string itself).
If the evaluation format does not match how the model generates, scores can plummet.
Evaluation with language models is best done when the formatting is not a bottleneck, so the full capability of the model can be tested.
Achieving format-agnostic evaluation takes substantial effort and tinkering to get right and is rare in practice.

Returning to the history of evaluation, regardless of the setting used, a common challenge with few-shot prompting is models not following the format, which is counted as an incorrect answer.
When designing an evaluation domain, the number of examples used in-context is often considered a design parameter and ranges from 3 to 8 or more.

### Chain-of-Thought Prompting

Within the evolution of few-shot prompting came the idea of including chain-of-thought examples for the model to follow.
This comes in the form of in-context examples that have written-out reasoning (this was later superseded by explicit prompting to generate reasoning steps) [@wei2022chain].
For example, here are two standard prompts:

```text
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...
```

And here is the same example with chain-of-thought prompting:

```text
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They ...
```

### Zero-Shot Instruction Following

Over time, as language models became stronger, they evolved to zero-shot evaluation, a.k.a. *zero-shot learners* [@wei2021finetuned].
The Finetuned Language Net (FLAN) showed that language models fine-tuned on specific tasks, as a precursor to modern instruction tuning, could generalize to zero-shot questions they were not trained on [@wei2021finetuned] (similar results are also found in T0 [@sanh2021multitask]).
This is the emergence of instruction fine-tuning (IFT), an important precursor to RLHF and post-training.
A zero-shot question looks like this:

```text
User: "What is the capital of France?"
Assistant:
```

From here (2022), the timeline begins to include key early RLHF works, such as InstructGPT.
The core capability and use-case shift that accompanied these models consisted of even more open-ended usage.
With this, evaluation with sampling from the model became increasingly popular, as it mirrors actual usage (technically, this could be referred to as generation-based exact-match evaluation, but it does not have a clear canonical term).
During this period and through recent time, after ChatGPT, some multiple-choice evaluations are still used in RLHF research, as any transition of common practice takes a meaningful amount of time, usually years, to unfold.
For this type of evaluation, it is done by setting the temperature to 0 and sampling the characters A, B, C, and D.

### Reasoning-Era Evaluation Prompts

With the rise of reasoning models at the end of 2024 and the beginning of 2025, a major change in model behavior was the addition of a long chain-of-thought (CoT) reasoning process before every answer.
These models no longer need to be prompted with the canonical phrase "think step by step," as proposed in [@kojima2022large].
This next evolution of evaluation practices is generation-based exact-match evaluation with CoT reasoning, and therefore almost always uses a temperature greater than 0 for best performance.

For example, in some setups, every question or category has specially designed prompts to help extract behavior from the model.
Tülu 3 was an early seminal paper that details some prompts used for CoT answering multiple-choice questions [@lambert2024t].
Following is an example prompt used for MMLU, which is one of the evaluations that transitioned from single-token answer sampling to long-form CoT with exact-match answer checking:

```text
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

This, especially when the models use special formatting to separate thinking tokens from answer tokens, necessitated the most recent major update to evaluation regimes.
Evaluation is moving to models being tested to respond in a generative manner with CoT prompting.

## Why Many External Evaluation Comparisons Are Unreliable

Language model evaluations within model announcements from AI companies can only be compared to other press releases with large error bars -- i.e. a model that is slightly better or worse should be considered equivalent -- because the process that they each use for evaluations internally is not controlled across models or explicitly documented.
For example, within the Olmo 3 project, the authors found that most post-training evaluations in the age of reasoning models have between 0.25- and 1.5-point standard deviations when the evaluation setup is held constant [@teamolmo2025olmo3]; bigger changes in scores can come from using different prompts or sampling parameters.
Labs hillclimb on evaluations during training to make models more useful, traditionally using a mix of training, development (validation sets), and held-out evaluation sets (test sets).
(*Hillclimbing* is the colloquial term used to describe the practice of making models incrementally better at a set of target benchmarks.)
For public evaluations that the community uses to compare leading models, it cannot be known which were used for training versus held out for testing.

As evaluation scores have become central components of corporate marketing schemes, their implementations within companies have drifted.
There are rumors of major AI labs using "custom prompts" for important evaluations like GSM8K and MATH.
These practices evolve rapidly.

Language model evaluation stacks are perceived as marketing because the evaluations have no hard source of truth.
Inside frontier labs, evaluation suites are being tuned to suit the labs' internal needs.
When results are shared, we get output in the form of the numbers a lab got for its models, but not all the inputs to that function.
The inputs are very sensitive configurations, and they're different at OpenAI, Meta, Anthropic, and Google.
It's even hard to guarantee reproducibility on fully open evaluation standards.
Focusing efforts on your own models is the only way to get evaluation techniques that are close to repeatable.

Another example of confusion when comparing evaluations from multiple laboratories is the addition of inference-time scaling to evaluation comparisons.
Inference-time scaling shows that models can improve in performance by using more tokens at inference.
Thus, controlling evaluation scores by the total number of tokens for inference is important, but it is not yet common practice.

Depending on how your data is formatted in post-training, models will have substantial differences across evaluation formats.
For example, two popular, open math datasets, NuminaMath [@li2024numinamath] and MetaMath [@yu2023metamath], conflict with each other in training due to small differences in how the answers are formatted: Numina puts the answer in `\boxed{XYZ}`, and MetaMath puts the answer after `The answer is: XYZ`.
So, training on both can make performance worse than training with just one.
Strong models are trained to be able to function with multiple formats, but they generally have one format that's the strongest.

In the end, we are left with a few key points on the state of evaluating closed models:

- We do not necessarily know or have the key test sets that labs are climbing on, so some evaluations are proxies.
- Inference of frontier models is becoming more complicated with special system prompts, special tokens, and so on, and we don't know how it impacts evaluations.
- We do not know all the formats and details used to numerically report closed evaluations.

All of these dynamics, along with the very rapid progress of AI models over the last few years, result in plots similar to the famous one in @fig:benchmark-saturation, showing the in-vogue benchmarks of each era being solved very quickly.
The common term to describe this dynamic at a per-benchmark level is *saturation*.
As each benchmark approaches 100%, a model's progress begins to slow because there are only harder (or, in many cases, mislabeled) data points remaining, which makes any given benchmark less reliable as a measure of training progress (or comparison between two models).

![Report from Epoch AI showing how major AI evaluations are rapidly saturated over time (*saturation* is when a given benchmark reaches full performance and models no longer have meaningful signal). License CC-BY.](images/benchmark-performance.jpeg){#fig:benchmark-saturation}

## How Labs Actually Use Evaluations Internally to Improve Models

Evaluation of frontier language models is every bit as much an art today as it is a science; prescribing exactly how different groups use evaluations to understand cutting-edge language models would require a book in its own right.

Different groups choose different evaluations to keep independent from their own training stacks: that is, making a given evaluation part of a true test set that is not looked at across model versions, but no one discloses which evaluations they choose.
For example, popular reasoning evaluations MATH and GSM8K both have training sets with prompts that can easily be used to improve performance.
Improving performance with the prompts from the same distribution is very different than generalizing to these tasks by training on general math data.

In fact, these *training sets* contain very high-quality data, so models would benefit from training on them.
If these companies are *not* using the corresponding evaluation as a core metric to track, training on the evaluation set could be a practical decision because high-quality data is a major limiting factor of model development.

Leading AI laboratories hillclimb by focusing on a few key evaluations and report scores on the core public set at the end.
The key point is that some of their evaluations for tracking progress, such as the datasets for cross-entropy loss predictions in scaling from the GPT-4 report [@achiam2023gpt], often are not public.

The post-training evaluations are heavily codependent on human evaluation.
Human evaluation for generative language models yields Elo rankings (popular in early Anthropic papers such as Constitutional AI), and human evaluation for reward models shows agreement.
These can also be obtained by serving two different models to users with an A/B testing window (as discussed in Chapter 11).

The limited set of evaluations the laboratories choose to focus on forms a close link between evaluation and training.
At one point, one evaluation of focus across much of the industry was MMLU.
GPQA was extremely popular during the emergence of reasoning models due to increased community focus on scientific capabilities.
Labs will change the evaluations to make them better suited to their needs, such as OpenAI releasing SWE-Bench-Verified [@openai2024swebench].
There are many more internal evaluations that each frontier lab has built or bought but the public does not have access to.

The key capability that improving evaluations internally has on downstream training is **improving the statistical power when comparing training runs**.
By changing evaluations, these labs reduce the noise on their prioritized signals to make more informed training decisions.

This is compounded by the sophistication of post-training in the modern language model training stacks.
Evaluating language models today involves a moderate-to-substantial amount of generated tokens per prompt (rather than just looking at log-probabilities of answers) and therefore compute spend.
It is accepted that frontier labs use small tricks to boost performance on many tasks: the most common explanation is one-off prompts for certain evaluations.

## Contamination

A major problem with current language model practices (not restricted to RLHF and post-training) is intentional or unintentional use of data from evaluation datasets in training.
This is called *dataset contamination* (a form of *data leakage*), and the practices to avoid it are called *decontamination*.
To decontaminate a dataset, we perform searches over the training and test datasets, looking for matches in *n*-gram overlap over words/subword tokens, or fixed-length character substring matching (e.g., 50 characters) [@singh2024evaluation].
There are many ways in which data can become contaminated, but the most common is scraping training data for multiple stages from the web.
Benchmarks are often listed on public web domains that are crawled, or users pass questions into models that can then end up in candidate training data for future models.

For example, during the decontamination of the evaluation suite for Tülu 3, the authors found that popular open datasets were contaminated with popular evaluations for RLHF [@lambert2024t].
These overlaps included UltraFeedback's contamination with TruthfulQA, Evol-CodeAlpaca's contamination with HumanEval, NuminaMath's contamination with MATH, and WildChat's contamination with safety evaluations.
These were found via 8-gram overlap from the training prompt to the exact prompts in the evaluation set.

In other cases, models are found to have been trained on data very close to the benchmarks, such as keeping the words of a math problem the same and changing the numbers.
This can result in unusual behavior in post-training regimes, such as benchmarks improving when models are trained with RL on random rewards -- a contrived setup that should increase performance only if a model has certain types of data contamination.
This sort of base-model contamination, when it cannot be proven exactly why the model behaves in certain ways, has been a substantial confounding variable in many early RLVR works on top of Qwen 2.5 and Qwen 3 base models [@shao2025spurious] [@wu2025reasoning].

To understand the contamination of models that do not disclose or release the training data, new versions of benchmarks are created with questions that are slightly perturbed from the original (e.g., for MATH [@huang2025math]), to see which models were trained to match the original format or questions.
High variance on these perturbation benchmarks is not confirmation of contamination, which is difficult to prove.
Rather, it could indicate models that were trained with a specific format in mind that may not translate to real-world performance.


## Tooling

There are many open source evaluation tools to choose from.
Some of them are as follows:

- Inspect AI from the UK Safety Institute [@inspectAI2024],
- Hugging Face's LightEval [@fourrier2023lighteval], which powered the Open LLM Leaderboard [@open-llm-leaderboard-v2],
- Eleuther AI's evaluation harness [@gao2023evalharness] built on top of the infrastructure from its GPT-Neo-X model (this contains a good GPT-3-era evaluation setup and configuration) [@gpt-neox-20b],
- AI2's library based on OLMES [@gu2024olmes],
- Stanford's Center for Research on Foundation Model's HELM [@liang2023helm],
- Mosaic's (now Databricks') Eval Gauntlet [@mosaicml2024gauntlet], and more.

## Summary

- Evaluation for post-training has evolved through distinct phases: early chat-focused benchmarks scored by LLM-as-a-judge (MT-Bench, AlpacaEval); multiskill suites covering knowledge, math, code, and safety; and the current era of challenging reasoning and tool-use tasks (GPQA, SWE-Bench, AIME).
- Prompting format dramatically affects benchmark scores: the field has progressed from few-shot prompting to zero-shot prompting to chain-of-thought generation, and mismatches between the evaluation format and how the model was trained can cause performance to drop to near zero.
- Evaluation comparisons across labs are unreliable because each uses different prompts, sampling parameters, and internal evaluation sets. Scores should be treated as rough indicators with meaningful error bars rather than precise rankings.
- Dataset contamination, where evaluation data leaks into training through web scraping or user queries, is a persistent confounding factor, and decontamination via *n*-gram overlap matching is now standard practice in responsible post-training pipelines.
