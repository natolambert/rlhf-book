---
prev-chapter: "The Nature of Preferences"
prev-url: "05-preferences.html"
next-chapter: "Reward Modeling"
next-url: "07-reward-models.html"
---

# Preference Data

## Collecting Preference Data

Getting the most out of human data involves iterative training of models, evolving and highly detailed data instructions, translating through data foundry businesses, and other challenges that add up. 
The process is difficult for new organizations trying to add human data to their pipelines. 
Given the sensitivity, processes that work and improve the models are extracted until the performance runs out.

## Rankings vs. Ratings




[@likert1932technique]

For example, a 5 point Likert scale would look like the following:

| A$>>$B | A$>$B | Tie | B$>$A | B$>>$A |
|:------:|:-----:|:-----:|:-----:|:------:|
| 1    | 2   | 3   | 4   | 5    |

Table: An example 5-wise Likert scale between two responses, A and B. {#tbl:likert5}

Some early RLHF for language modeling works uses an 8-step Likert scale with levels of preference between the two responses [@bai2022training]. 
An even scale removes the possibility of ties:

Here's a markdown table formatted as an 8-point Likert scale:

| A$>>>$B |     |     | A$>$B | B$>$A  |     |     | B$>>>$A |
|:-------:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-------:|
| 1     | 2   | 3   | 4   | 5    | 6   | 7   | 8     |

Table: An example 8-wise Likert scale between two responses, A and B. {#tbl:likert8}

In this case [@bai2022training], and in other works, this information is still reduced to a binary signal for the training of a reward model.





### Sourcing and Contracts

The first step is sourcing the vendor to provide data (or ones own annotators). 
Much like acquiring access to cutting-edge Nvidia GPUs, getting access to data providers is also a who-you-know game. If you have credibility in the AI ecosystem, the best data companies will want you on our books for public image and long-term growth options. Discounts are often also given on the first batches of data to get training teams hooked.

If you’re a new entrant in the space, you may have a hard time getting the data you need quickly. Getting the tail of interested buying parties that Scale AI had to turn away is an option for the new data startups. It’s likely their primary playbook to bootstrap revenue.

On multiple occasions, I’ve heard of data companies not delivering their data contracted to them without threatening legal or financial action. Others have listed companies I work with as customers for PR even though we never worked with them, saying they “didn’t know how that happened” when reaching out. There are plenty of potential bureaucratic or administrative snags through the process. For example, the default terms on the contracts often prohibit the open sourcing of artifacts after acquisition in some fine print.

Once a contract is settled the data buyer and data provider agree upon instructions for the task(s) purchased. There are intricate documents with extensive details, corner cases, and priorities for the data. A popular example of data instructions is the one that [OpenAI released for InstructGPT](https://docs.google.com/document/d/1MJCqDNjzD04UbcnVZ-LmeXJ04-TKEICDAepXyMCBUb8/edit#heading=h.21o5xkowgmpj).

An example interface is shown below from [@bai2022training]:

![Example preference data collection interface.](images/anthropic-interface.png){#fig:preference-interface width=600px .center}

Depending on the domains of interest in the data, timelines for when the data can be labeled or curated vary. High-demand areas like mathematical reasoning or coding must be locked into a schedule weeks out. Simple delays of data collection don’t always work — Scale AI et al. are managing their workforces like AI research labs manage the compute-intensive jobs on their clusters.

Once everything is agreed upon, the actual collection process is a high-stakes time for post-training teams. All the infrastructure, evaluation tools, and plans for how to use the data and make downstream decisions must be in place.

The data is delivered in weekly batches with more data coming later in the contract. For example, when we bought preference data for on-policy models we were training at HuggingFace, we had a 6 week delivery period. The first weeks were for further calibration and the later weeks were when we hoped to most improve our model.

![Overview of the multi-batch cycle for obtaining human preference data from a vendor.](images/pref-data-timeline.png){#fig:preferences width=600px .center}

The goal is that by week 4 or 5 we can see the data improving our model. This is something some frontier models have mentioned, such as the 14 stages in the Llama 2 data collection [@touvron2023llama], but it doesn’t always go well. At HuggingFace, trying to do this for the first time with human preferences, we didn’t have the RLHF preparedness to get meaningful bumps on our evaluations. The last weeks came and we were forced to continue to collect preference data generating from endpoints we weren’t confident in.

After the data is all in, there is plenty of time for learning and improving the model. Data acquisition through these vendors works best when viewed as an ongoing process of achieving a set goal. It requires iterative experimentation, high effort, and focus. It’s likely that millions of the dollars spent on these datasets are “wasted” and not used in the final models, but that is just the cost of doing business. Not many organizations have the bandwidth and expertise to make full use of human data of this style.

This experience, especially relative to the simplicity of synthetic data, makes me wonder how well these companies will be doing in the next decade.

Note that this section *does not* mirror the experience for buying human-written instruction data, where the process is less of a time crunch.

## Synthetic Preferences and LLM-as-a-judge

TODO

### Example Prompts

TODO Cite MT Bench [@zheng2023judging],[@huang2024empirical], including specialized models for LLM as a judge [@kim2023prometheus]

> Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.