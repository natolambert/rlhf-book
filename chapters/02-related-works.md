---
prev-chapter: "Introduction"
prev-url: "01-introduction.html"
page-title: Related Works
next-chapter: "Definitions & Background"
next-url: "03-setup.html"
---

# Key Related Works

In this chapter we detail the key papers and projects that got the RLHF field to where it is today.
This is not intended to be a comprehensive review on RLHF and the related fields, but rather a starting point and retelling of how we got to today.
It is intentionally focused on recent work that led to ChatGPT.
There is substantial further work in the RL literature on learning from preferences [@wirth2017survey]. 
For a more exhaustive list, you should use a proper survey paper [@kaufmann2023survey],[@casper2023open].

## Origins to 2018: RL on Preferences

The field has recently been popularized with the growth of Deep Reinforcement Learning and has grown into a broader study of the applications of LLMs from many large technology companies.
Still, many of the techniques used today are deeply related to core techniques from early literature on RL from preferences.

*TAMER: Training an Agent Manually via Evaluative Reinforcement,* Proposed a learned agent where humans provided scores on the actions taken iteratively to learn a reward model [@knox2008tamer]. Other concurrent or soon after work proposed an actor-critic algorithm, COACH, where human feedback (both positive and negative) is used to tune the advantage function [@macglashan2017interactive].

The primary reference, Christiano et al. 2017, is an application of RLHF applied to preferences between Atari trajectories [@christiano2017deep]. The work shows that humans choosing between trajectories can be more effective in some domains than directly interacting with the environment. This uses some clever conditions, but is impressive nonetheless.
This method was expanded upon with more direct reward modeling [@ibarz2018reward].
TAMER was adapted to deep learning with Deep TAMER just one year later [@warnell2018deep].

This era began to transition as reward models as a general notion were proposed as a method for studying alignment, rather than just a tool for solving RL problems [@leike2018scalable].

## 2019 to 2022: RL from Human Preferences on Language Models

Reinforcement learning from human feedback, also referred to regularly as reinforcement learning from human preferences in its early days, was quickly adopted by AI labs increasingly turning to scaling large language models.
A large portion of this work began between GPT-2, in 2018, and GPT-3, in 2020.
The earliest work in 2019, *Fine-Tuning Language Models from Human Preferences* has many striking similarities to modern work on RLHF [@ziegler2019fine]. Learning reward models, KL distances, feedback diagrams, etc -- just the evaluation tasks, and capabilities, were different.
From here, RLHF was applied to a variety of tasks.
The popular applications were the ones that worked at the time.
Important examples include general summarization [@stiennon2020learning], recursive summarization of books [@wu2021recursively], instruction following (InstructGPT) [@ouyang2022training], browser-assisted question-answering (WebGPT) [@nakano2021webgpt], supporting answers with citations (GopherCite) [@menick2022teaching], and general dialogue (Sparrow) [@glaese2022improving].

Aside from applications, a number of seminal papers defined key areas for the future of RLHF, including those on:

1. Reward model over-optimization [@gao2023scaling]: The ability for RL optimizers to over-fit to models trained on preference data,
2. Language models as a general area of study for alignment [@askell2021general], and
3. Red teaming [@ganguli2022red] -- the process of assessing safety of a language model.

Work continued on refining RLHF for application to chat models.
Anthropic continued to use it extensively for early versions of Claude [@bai2022training] and early RLHF open-source tools emerged [@ramamurthy2022reinforcement],[@havrilla-etal-2023-trlx],[@vonwerra2022trl].

## 2023 to Present: ChatGPT Era

The announcement of ChatGPT was very clear about the role of RLHF in its training [@openai2022chatgpt]:

> We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT⁠, but with slight differences in the data collection setup.

Since then RLHF has been used extensively in leading language models. 
It is well known to be used in Anthropic's Constitutional AI for Claude [@bai2022constitutional], Meta's Llama 2 [@touvron2023llama] and Llama 3 [@dubey2024llama], Nvidia's Nemotron [@adler2024nemotron], Ai2's Tülu 3 [@lambert2024t], and more.

Today, RLHF is growing into a broader field of preference fine-tuning (PreFT), including new applications such as process reward for intermediate reasoning steps [@lightman2023let], direct alignment algorithms inspired by Direct Preference Optimization (DPO) [@rafailov2024direct], learning from execution feedback from code or math [@kumar2024training],[@singh2023beyond], and other online reasoning methods inspired by OpenAI's o1 [@openai2024o1].
