---
prev-chapter: "Synthetic Data & Distillation"
prev-url: "15-synthetic.html"
next-chapter: "Over Optimization"
next-url: "17-over-optimization.html"
---

# [Incomplete] Evaluation & Prompting

Evaluation is an ever evolving approach.
The key to understanding language model evaluation, particularly with post-training, is that the current popular evaluation regimes represents a reflection of the popular training best practices and goals.
While challenging evaluations drive progress in language models to new areas, the majority of evaluation is designed around building useful signals for new models.

In many ways, this chapter is designed to present vignettes of popular evaluation regimes throughout the early history of RLHF, so readers can understand the common themes, details, and failure modes.

Evaluation for RLHF and post-training has gone a few distinct phases in its early history:

1. **Early chat-phase**: Early models trained with RLHF or preference tuning targeted evaluations focused on capturing the chat performance of a model, especially relative to known strong models such as GPT-4. Early examples include MT-Bench [@zheng2023judging], AlpacaEval [@dubois2024length], and Arena-Hard [@li2024crowdsourced]. Models were evaluated narrowly and these are now considered as "chat" or "instruction following" domains.
2. **Multi-skill era**: Over time, common practice established that RLHF can be used to improve more skills than just chat. For example, the Tülu evaluation suite included tasks on knowledge (MMLU [@hendrycks2020measuring], PopQA [@mallen2023llm_memorization], TruthfulQA [@lin2021truthfulqa]), Reasoning (BigBenchHard [@suzgun2022challenging], DROP [@dua2019drop]), Math (MATH [@hendrycksmath2021], GSM8K [@cobbe2021gsm8k]), Coding (HumanEval [@chen2021codex], HumanEval+ [@evalplus]), Instruction Following [@zhou2023instructionfollowingevaluationlargelanguage], and Safety (a composite of many evaluations). This reflects the domain where post-training is embraced as a multi-faceted solution beyond safety and chat.
3. **Reasoning & tools**: The current era for post-training is defined by a focus on challenging reasoning and tool use problems. These include much harder knowledge-intensive tasks such as GPQA Diamond [@rein2023gpqa] and Humanity's Last Exam [@phan2025hle], intricate software engineering tasks such as SWE-Bench+ [@aleithan2024swebenchplus] and LiveCodeBench [@jain2024livecodebench], or challenging math problems exemplified by recent AIME contests.

Beyond this, new domains will evolve. 
Throughout this chapter we will include details that map to how these evaluations were implemented and understood.

## Formatting as prompting: From Few-shot to Zero-shot Prompting to CoT

Early language models were only used as intelligent autocomplete.
In order to use these models in an more open ended way, multiple examples were shown to the model and then a prompt that is an incomplete phrase. This was called few-shot or in-context learning [@brown2020language], and at the time instruction tuning or RLHF was not involved.
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
(D) Rolle’s Theorem

Correct Answer:
```

To extract an answer here one could either generate a token based on some sampling parameters and see if the answer is correct, A,B,C, or D (formatting above like this proposed in [@robinson2023leveraging]), or one could look at the probabilities of each token and mark the task as correct if the correct answer is more likely. 
This second method has two potential implementations -- first, one could look at the probability of the letter (A) or the answer "The Mean Value Theorem." 
Both of these are permissible metrics, but answer prediction is more common among probability base metrics.

A common challenge with few-shot prompting is that models will not follow the format, which is counted as an incorrect answer. 
When designing an evaluation domain, the number of examples used in-context is often considered a design parameter and ranges from 3 to 8 or more.

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

With the rise of reasoning models at the end of 2024 and the beginning of 2025, a major change in model behavior was the addition of a long Chain-of-Thought (CoT [@wei2022chain]) reasoning process before every answer.

For example, for every prompt there can specially designed prompts to help extract behavior from the model.
Tülu 3 details some prompts used for CoT answering on multiple choice questions [@lambert2024t]:

```
Answer the following multiple-choice question by giving the correct answer letter in parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the response with “Therefore, the answer is (ANSWER_LETTER)” where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) …

Answer the above question and REMEMBER to finish your response with the exact phrase “Therefore, the answer is (ANSWER_LETTER)” where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

This, especially when the models use special formatting to separate thinking tokens from answer tokens, necessitated the most recent major update to evaluation regimes.
Evaluation is moving to where the models are tested to respond in a generative manner with a chain of thought prompting.

## Prompting

Prompting, i.e. crafting the correct query for  a model, is a crucial portion of using them as the models are evolving rapidly.

TODO expand

## Tooling

*Needs editing, taken from blog*

Language model evaluations done within companies can only be compared to their peers with large error bars. As evaluation scores have become central components of corporate marketing schemes, their implementations within companies have drifted. There are rumors of major AI labs using “custom prompts” for important evaluations like GSM8k or MATH. At the same time, OpenAI has released open-source code for configurations (and basic prompts) for their models.

It would seem that open-source language model builders should have an advantage in trust and comparability of models, but the open-source AI community also hasn’t converged on a single rubric for evaluating language models. The experience that motivated this article has happened every time — it is looking at an open weights model (such as Llama) and realizing we need to re-run evaluations on our own setup to get a sense of what is actually happening. This is very salient and hard to communicate to the general audience, where open means more reproducible. It should also mean the results are easier to interpret. What tools do we need for this?

In my article on “Big Tech’s LLM evals are just marketing,” I didn’t uncover the deeper reasons as to why can’t fully believe these evaluations. Language model evaluation stacks are perceived as marketing because the evaluations have no hard source of truth. What is happening inside frontier labs is that evaluation suites are being tuned to suit their internal needs. When results are shared, we get output in the form of the numbers a lab got for their models, but not all the inputs to that function. The inputs are very sensitive configurations, and they’re different at all of OpenAI, Meta, Anthropic, and Google. Even fully open evaluation standards are hard to guarantee reproducibility on. Focusing efforts on your own models is the only way to get close to repeatable evaluation techniques. There are good intentions underpinning the marketing, starting with the technical teams.

Evaluation of frontier language models is every bit as much an art today as it is a science.

Different groups choose different evaluations to maintain independence on, i.e. making them a true test set, but no one discloses which ones they choose. For example, popular reasoning evaluations MATH and GSM8k both have training sets with prompts that can easily be used to improve performance. Improving performance with the prompts from the same distribution is very different than generalizing to these tasks by training on general math data.

Labs like OpenAI hillclimb by focusing on a few key evaluations and report scores on the core public set at the end. The key point is that some of their evaluations for tracking progress, such as the datasets for cross-entropy loss predictions in scaling from the GPT-4 report, are often not public.


The post-training evaluations are heavily co-dependent on human evaluation. Human evaluation for generative language models yields Elo rankings (popular in early Anthropic papers, such as Constitutional AI), and human evaluation for reward models shows agreement.

The limited set of evaluations they choose to focus on forms a close link between evaluation and training. At one point one evaluation of focus was MMLU. Now, GPQA is likely. Labs will change the evaluations to make them better suited to their needs, such as OpenAI releasing SWE-Bench-Verified. There are more internally we don’t know about.

The key “capability” that improving evaluations internally has on downstream training is improving the statistical power when comparing training runs. By changing evaluations, these labs reduce the noise on their prioritized signals in order to make more informed training decisions.

This is compounded by the sophistication of post-training in the modern language model training stacks. Evaluating language models today involves a moderate amount of generating tokens (rather than just looking at log probabilities of answers). It is accepted that small tricks are used by frontier labs to boost performance on many tasks — the most common explanation is one-off prompts for certain evaluations. Does Claude use its special thinking tokens when being evaluated on reasoning benchmarks? I have no way of knowing.

Depending on how your data is formatted in post-training, models will have substantial differences. For example, two popular, open math datasets Numina and MetaMathQA conflict with each other in training due to small differences in how the answers are formatted — training on both can make performance worse than with just one. Meta’s models use a very specific format for MATH, Minerva, while this varies substantially with post-training decisions.

In the end we are left with a few key points on the state of evaluating closed models:

We do not know or necessarily have the key test sets that labs are climbing on, so some evaluations are proxies (or worse, blissfully cheated on by training on text on the public web).

Inference of frontier models is becoming more complicated with special system prompts, special tokens, etc., and we don’t know how it impacts evaluations, and

We do not know all the formats and details used to numerically report the closed evaluations.

There are a lot of asterisks to put on the state of closed evaluations. To get ahead of some comments, it is important to acknowledge that there are some very useful efforts by closed labs to communicate their evaluation efforts. 
Some include OpenAI’s prompts in simple-evals, OpenAI releasing entire evals to the community, such as MLE Bench, or Meta’s documentation on reproducing their results with a popular open-source eval tools.

TODO open tools here:

There are many open-sourced evaluation tools for people to choose from. There’s Inspect AI from the UK Safety Institute [@inspectAI2024], HuggingFace’s LightEval [@fourrier2023lighteval] that powers the LLM Leaderboard, Eleuther AI’s evaluation harness [@gao2023evalharness] built on top of the infrastructure from their GPT-Neo-X model (around GPT-3 evaluation config), AI2’s library based on OLMES [@gu2024olmes], Stanford’s Center for Research on Foundation Model’s HELM [@liang2023helm], Mosaic’s (now Databricks’) Eval Gauntlet [@mosaicml2024gauntlet], and surely more I did not mention.

## Contamination

TODO mini section

## Case Studies

Below are a few types of evaluations that are at the core of how RLHF and post-training is evolving today.

### ChatBotArena

*Needs editing, taken from blog*


ChatBotArena is the largest community evaluation tool for language models. The LMSYS team, which emerged early in the post-ChatGPT craze, works with most of the model providers to host all of the relevant models. If you’re looking to get to know how multiple models compare to each other, ChatBotArena is the place to start.

ChatBotArena casts language model evaluation through the wisdom of the crowd. For getting an initial ranking of how models stack up and how the models in the ecosystem are getting better, it has been and will remain crucial.

ChatBotArena does not represent a controlled nor interpretable experiment on language models.

When evaluating models to learn which are the best at extremely challenging tasks, distribution control, and careful feedback are necessary. For these reasons, ChatBotArena cannot definitively tell us which models are solving the hardest tasks facing language models. It does not measure how the best models are improving in clear ways. This type of transparency comes elsewhere.

For most of its existence, people correlated the general capabilities tested in ChatBotArena with a definitive ranking of which models can do the hardest things for me. This is not true. In both my personal experience reading data and what the community knows about the best models, the ChatBotArena ranking shows the strongest correlations with:

Certain stylistic outputs from language models, and

Language models that have high rates of complying with user requests.

Both of these have been open research problems in the last two years. Style is deeply intertwined with how information is received by the user and precisely refusing only the most harmful requests is a deeply challenging technical problem that both Meta (with Llama 2) and Anthropic (with earlier versions of Claude particularly) have gotten deeply criticized for.

Among closed labs, their styles have been greatly refined. All of Meta, OpenAI, and Anthropic have distinctive styles (admittedly, I haven’t used Google’s Gemini enough to know).

Meta’s AI is succinct and upbeat (something that has been discussed many times on the LocalLlama subreddit).

OpenAI’s style is the most robotic to me. It answers as an AI and contains a lot of information.

Claude’s style is intellectual, bordering on curious, and sometimes quick to refuse.

When ChatBotArena was founded, these styles were in flux. Now, they majorly shift the rankings depending on what people like. People seem to like what OpenAI and Meta put out.

There are clear reasons why OpenAI’s models top the charts on ChatBotArena. They were the originators of modern RLHF, have most clearly dictated their goals with RLHF, continue to publish innovative ideas in the space, and have always been ahead here. Most people just did not realize how important this was to evaluation until the launch of GPT-4o-mini. Culture impacts AI style.

### Private Leaderboards

Scale Leaderboard [@scale2024seal] etc