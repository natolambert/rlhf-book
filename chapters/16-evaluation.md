---
prev-chapter: "Synthetic Data & Distillation"
prev-url: "15-synthetic"
page-title: Evaluation
next-chapter: "Over Optimization"
next-url: "17-over-optimization"
---

# Evaluation

Evaluation is an ever evolving approach.
The key to understanding language model evaluation, particularly with post-training, is that the current popular evaluation regimes represent a reflection of the popular training best practices and goals.
While challenging evaluations drive progress in language models to new areas, the majority of evaluation is designed around building useful signals for new models.

In many ways, this chapter is designed to present vignettes of popular evaluation regimes throughout the early history of RLHF, so readers can understand the common themes, details, and failure modes.

Evaluation for RLHF and post-training has gone a few distinct phases in its early history:

1. **Early chat-phase**: Early models trained with RLHF or preference tuning targeted evaluations focused on capturing the chat performance of a model, especially relative to known strong models such as GPT-4. Early examples include MT-Bench [@zheng2023judging], AlpacaEval [@dubois2024length], and Arena-Hard [@li2024crowdsourced]. Models were evaluated narrowly and these are now considered as "chat" or "instruction following" domains.
2. **Multi-skill era**: Over time, common practice established that RLHF can be used to improve more skills than just chat. For example, the Tülu evaluation suite included tasks on knowledge (MMLU [@hendrycks2020measuring], PopQA [@mallen2023llm_memorization], TruthfulQA [@lin2021truthfulqa]), Reasoning (BigBenchHard [@suzgun2022challenging], DROP [@dua2019drop]), Math (MATH [@hendrycksmath2021], GSM8K [@cobbe2021gsm8k]), Coding (HumanEval [@chen2021codex], HumanEval+ [@evalplus]), Instruction Following [@zhou2023instructionfollowingevaluationlargelanguage], and Safety (a composite of many evaluations). This reflects the domain where post-training is embraced as a multi-faceted solution beyond safety and chat.
3. **Reasoning & tools**: The current era for post-training is defined by a focus on challenging reasoning and tool use problems. These include much harder knowledge-intensive tasks such as GPQA Diamond [@rein2023gpqa] and Humanity's Last Exam [@phan2025hle], intricate software engineering tasks such as SWE-Bench+ [@aleithan2024swebenchplus] and LiveCodeBench [@jain2024livecodebench], or challenging math problems exemplified by recent AIME contests.

Beyond this, new domains will evolve. 
As AI becomes more of an industrialized field, the incentives of evaluation are shifting and becoming multi-stakeholder.
Since the release of ChatGPT, private evaluations such as the Scale Leaderboard [@scale2024seal], community-driven evaluations such as ChatBotArena [@chiang2024chatbot], and third-party evaluation companies such as ArtificialAnalysis and Epoch AI have proliferated.
Throughout this chapter we will include details that map to how these evaluations were implemented and understood.

## Prompting Formatting: From Few-shot to Zero-shot to CoT

**Prompting** language models is primarily a verb, but it is also considered a craft or art that one can practice and/or train in general [@schulhoff2024prompt].
A prompt is the way of structuring information and context for a language model. 
For common interactions, the prompt is relatively basic.
For advanced scenarios, a well crafted prompt will mean success or failure on a specific one-off use-case.

When it comes to evaluation, prompting techniques can have a substantial impact on the performance of the model.
Some prompting techniques -- e.g. formatting discussed below -- can make a model's performance drop from 60% to near 0.
Similarly, a change of prompt can help models learn better during training. 
Colloquially, prompting a model well can give the subjective experience of using future models, unlocking performance outside of normal use.

Prompting well with modern language models can involve preparing an entire report for the model to respond to (often with 1000s of tokens of generated text). 
This behavior is downstream of many changes in how language model performance has been measured and understood.

Early language models were only used as intelligent autocomplete.
In order to use these models in a more open ended way, multiple examples were shown to the model and then a prompt that is an incomplete phrase. This was called few-shot or in-context learning [@brown2020language], and at the time instruction tuning or RLHF was not involved.
In the case of popular evaluations, this would look like:

```
# Few-Shot Prompt for a Question-Answering Task
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

Here, there are multiple ways to evaluate an answer. If we consider a question in the style of MMLU, where the model has to choose between multiple answers:

```
# Few-Shot Prompt

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

To extract an answer here one could either generate a token based on some sampling parameters and see if the answer is correct, A,B,C, or D (formatting above like this proposed in [@robinson2023leveraging]), or one could look at the probabilities of each token and mark the task as correct if the correct answer is more likely. 
This second method has two potential implementations -- first, one could look at the probability of the letter (A) or the answer "The Mean Value Theorem." 
Both of these are permissible metrics, but answer prediction is more common among probability base metrics.

A common challenge with few-shot prompting is that models will not follow the format, which is counted as an incorrect answer. 
When designing an evaluation domain, the number of examples used in-context is often considered a design parameter and ranges from 3 to 8 or more.

Within the evolution of few-shot prompting came the idea of including chain-of-thought examples for the model to follow.
This comes in the form of examples where the in-context examples have written out reasoning, such as below (which later was superseded by explicit prompting to generate reasoning steps) [@wei2022chain]:

```
# standard prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...

# chain of thought prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They..
```

Over time, as language models became stronger, they evolved to zero-shot evaluation, a.k.a. "zero-shot learners" [@wei2022finetuned].
The Finetuned Language Net (FLAN) showed that language models finetuned in specific tasks, as a precursor to modern instruction tuning, could generalize to zero-shot questions they were not trained on [@wei2022finetuned] (similar results are also found in T0 [@sanh2022multitask]).
This is the emergence of instruction finetuning (IFT), an important precursor to RLHF and post-training.
A zero shot question would look like:

```
User: "What is the capital of France?"
Assistant:
```

From here in 2022, the timeline begins to include key early RLHF works, such as InstructGPT.
The core capability and use-case shift that accompanied these models is even more open-ended usage.
With more open-ended usage, generative evaluation became increasingly popular as it mirrors actual usage.
In this period through recent years after ChatGPT, some multiple-choice evaluations were still used in RLHF research as a holdback to common practice.

With the rise of reasoning models at the end of 2024 and the beginning of 2025, a major change in model behavior was the addition of a long Chain-of-Thought (CoT) reasoning process before every answer.
These models no longer needed to be prompted with the canonical modification of "think step by step," as proposed in [@kojima2022large].

For example, for every question or category there are specially designed prompts to help extract behavior from the model.
Tülu 3 details some prompts used for CoT answering on multiple choice questions [@lambert2024t]:

```
Answer the following multiple-choice question by giving the correct answer letter in parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

This, especially when the models use special formatting to separate thinking tokens from answer tokens, necessitated the most recent major update to evaluation regimes.
Evaluation is moving to where the models are tested to respond in a generative manner with a chain of thought prompting.

## Using Evaluations vs. Observing Evaluations

![Report from Epoch AI showing how major AI evaluations are rapidly saturated over time. License CC-BY.](images/benchmark-performance.jpeg)

Language model evaluations done within companies can only be compared to their peers with large error bars because the process that they use for evaluations internally is not matched with external evaluations.
Internal evaluations are made to hillclimb on for training, as would be called a "training set" in traditional machine learning.
The public evaluations that the community uses to compare leading models cannot be known if they were within said training set or as unseen "test sets" or "validation sets."

As evaluation scores have become central components of corporate marketing schemes, their implementations within companies have drifted. 
There are rumors of major AI labs using "custom prompts" for important evaluations like GSM8k or MATH. 
These practices evolve rapidly.

Language model evaluation stacks are perceived as marketing because the evaluations have no hard source of truth. 
What is happening inside frontier labs is that evaluation suites are being tuned to suit their internal needs. 
When results are shared, we get output in the form of the numbers a lab got for their models, but not all the inputs to that function. 
The inputs are very sensitive configurations, and they're different at all of OpenAI, Meta, Anthropic, and Google. 
Even fully open evaluation standards are hard to guarantee reproducibility on. 
Focusing efforts on your own models is the only way to get close to repeatable evaluation techniques. 
There are good intentions underpinning the marketing, starting with the technical teams.

Evaluation of frontier language models is every bit as much an art today as it is a science.

Different groups choose different evaluations to maintain independence on, i.e. making them a true test set, but no one discloses which ones they choose. 
For example, popular reasoning evaluations MATH and GSM8k both have training sets with prompts that can easily be used to improve performance. 
Improving performance with the prompts from the same distribution is very different than generalizing to these tasks by training on general math data.

In fact, these *training sets* are very high quality data so models would benefit from training on them.
If these companies are *not* using the corresponding evaluation as a core metric to track, training on the evaluation set could be a practical decision as high-quality data is a major limiting factor of model development.

Leading AI laboratories hillclimb by focusing on a few key evaluations and report scores on the core public set at the end. 
The key point is that some of their evaluations for tracking progress, such as the datasets for cross-entropy loss predictions in scaling from the GPT-4 report [@achiam2023gpt], are often not public.

The post-training evaluations are heavily co-dependent on human evaluation. 
Human evaluation for generative language models yields Elo rankings (popular in early Anthropic papers, such as Constitutional AI), and human evaluation for reward models shows agreement.
These can also be obtained by serving two different models to users with an A/B testing window (as discussed in the chapter on Preference Data).

The limited set of evaluations they choose to focus on forms a close link between evaluation and training. 
At one point one evaluation of focus was MMLU. 
GPQA was one of choice during reasoning models' emergence. 
Labs will change the evaluations to make them better suited to their needs, such as OpenAI releasing SWE-Bench-Verified [@openai2024swebench]. 
There are many more internally the public does not have access to.

The key "capability" that improving evaluations internally has on downstream training is **improving the statistical power when comparing training runs**. 
By changing evaluations, these labs reduce the noise on their prioritized signals in order to make more informed training decisions.

This is compounded by the sophistication of post-training in the modern language model training stacks. 
Evaluating language models today involves a moderate amount of generating tokens (rather than just looking at log probabilities of answers). 
It is accepted that small tricks are used by frontier labs to boost performance on many tasks --- the most common explanation is one-off prompts for certain evaluations. 

Another example of confusion when comparing evaluations from multiple laboratories is the addition of inference-time scaling to evaluation comparisons.
Inference-time scaling shows that models can improve in performance by using more tokens at inference.
Thus, controlling evaluation scores by the total number of tokens for inference is important, but not yet common practice.

Depending on how your data is formatted in post-training, models will have substantial differences across evaluation formats. 
For example, two popular, open math datasets NuminaMath [@li2024numinamath] and MetaMath [@yu2023metamath] conflict with each other in training due to small differences in how the answers are formatted -- Numina puts the answer in `\boxed{XYZ}` and MetaMath puts the answer after `The answer is: XYZ` ---- training on both can make performance worse than with just one. 
Strong models are trained to be able to function with multiple formats, but they generally have a strongest format.

In the end we are left with a few key points on the state of evaluating closed models:

* We do not know or necessarily have the key test sets that labs are climbing on, so some evaluations are proxies.
* Inference of frontier models is becoming more complicated with special system prompts, special tokens, etc., and we don't know how it impacts evaluations, and
* We do not know all the formats and details used to numerically report the closed evaluations.

## Contamination

A major issue with current language model practices (i.e. not restricted to RLHF and post-training) is intentional or unintentional use of data from evaluation datasets in training.
This is called *dataset contamination* and respectively the practices to avoid it are *decontamination*. 
In order to decontaminate a dataset, one performs searches over the training and test datasets, looking for matches in n-grams (characters) or tokens [@singh2024evaluation].
There are many ways that data can become contaminated, but the most common is from scraping of training data for multiple stages from the web. 
Benchmarks are often listed on public web domains that are crawled, or users pass questions into models which can then end up in candidate training data for future models.

For example, during the decontamination of the evaluation suite for Tülu 3, the authors found that popular open datasets were contaminated with popular evaluations for RLHF [@lambert2024t]. 
These overlaps include: UltraFeedback's contamination with TruthfulQA, Evol-CodeAlpaca's contamination with HumanEval, NuminaMath's contamination with MATH, and WildChat's contamination with safety evaluations. 
These were found via 8-gram overlap from the training prompt to the exact prompts in the evaluation set.

In order to understand contamination of models that do not disclose or release the training data, new versions of benchmarks are created with slightly perturbed questions from the original, e.g. for MATH [@huang2025math], in order to see which models were trained to match the original format or questions.
High variance on these perturbation benchmarks is not confirmation of contamination, which is difficult to prove, but could indicate models that were trained with a specific format in mind that may not translate to real world performance.


## Tooling

There are many open-sourced evaluation tools for people to choose from. 
There's Inspect AI from the UK Safety Institute [@inspectAI2024], HuggingFace's LightEval [@fourrier2023lighteval] that powered the Open LLM Leaderboard [@open-llm-leaderboard-v2], Eleuther AI's evaluation harness [@gao2023evalharness] built on top of the infrastructure from their GPT-Neo-X model (around GPT-3 evaluation config) [@gpt-neox-20b], AI2's library based on OLMES [@gu2024olmes], Stanford's Center for Research on Foundation Model's HELM [@liang2023helm], Mosaic's (now Databricks') Eval Gauntlet [@mosaicml2024gauntlet], and more.
