---
prev-chapter: "Synthetic Data & Distillation"
prev-url: "15-synthetic.html"
next-chapter: "Over Optimization"
next-url: "17-over-optimization.html"
---

# [Incomplete] Evaluation & Prompting

## How To Tell if RLHF is Working?

## Prompting

Prompting, i.e. crafting the correct query for  a model, is a crucial portion of using them as the models are evolving rapidly.

## Evaluation

### Formatting and Overview

### ChatBotArena

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

Scale Leaderboard etc