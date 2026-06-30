<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "定义"
prev-url: "appendix-a-definitions"
page-title: "附录 B：超越 \"只是风格\""
search-title: "附录 B：超越 \"只是风格\""
meta-description: "RLHF 与后训练在语言模型中塑造的风格、校准和信息呈现问题。"
next-chapter: "实践问题"
next-url: "appendix-c-practical"
---

# 超越 "只是风格"

RLHF 的早期发展让它获得了 "只是风格迁移" 的名声；其他更尖锐的批评则集中在 RLHF 如何操纵输出中信息的呈现方式。
本附录解释为什么 style 是理解 RLHF 所提供价值的核心，以及为什么它会正向影响模型能力和用户体验。

RLHF 只关乎 style transfer 这一想法，出于两个原因限制了人们对 RLHF 的叙事。
第一个原因是，RLHF 被关联到了对模型而言很小、并不重要的变化。
当人们讨论 style transfer 时，他们通常不会把它描述成重要或令人兴奋的事情，而是认为它很表面。
然而，style 是人类价值的无尽来源。它解释了为什么重述故事可以产生新的畅销书，例如 [Sapiens](https://en.wikipedia.org/wiki/Sapiens:_A_Brief_History_of_Humankind)，也解释了为什么 style 是我们智识生态持续进步的基本部分。
Style 与信息本身是交织在一起的。

第二个原因是，许多人忽视了一个事实：做得好的 RLHF 会提升流行 LLM 评估上的分数。
我们已经在 Llama 3 上看到，不同 style 确实可以显著改善评估 [@dubey2024llama]。
Llama 3 Instruct 模型在 Arena 上得分极高，而公认的原因是它们拥有更有趣的个性：与同时代其他模型相比，它们更简洁，也更聪明。
不管许多 LLM 用户痴迷的 benchmark 分数如何，如果 RLHF 能让语言模型变得更有趣，这本身就是交付出的价值。

在本附录中，"chattiness" 这个词用于涵盖经过 RLHF 训练的模型回复长度增长这一现象，但它也包括大量使用 Markdown、表情符号，以及把答案组织成项目符号列表等技术。
为了让讨论更具体，下面展示同一个 Tülu 3 家族中两个模型的示例 completion。一个只做了指令微调，另一个通过 length-controlled DPO 做了偏好微调。
首先是指令微调模型对 prompt `What is reinforcement learning from human feedback?` 的 completion（具体模型是 [Llama 3.1 Tulu 3 70B SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)）[@lambert2024t]：

```text
Reinforcement learning from human feedback, also known as human-in-the-loop reinforcement learning, is a subfield of machine learning where the primary goal is to train reinforcement learning agents to perform tasks by incorporating feedback from humans. This feedback can come in various forms, such as binary rewards (e.g., right or wrong), scalar rewards, preferences between actions, demonstrations, natural language instructions, or corrections to the agent's behavior.

The idea behind reinforcement learning from human feedback is to leverage human intuition and decision-making abilities to guide the learning process of the agent. In traditional reinforcement learning, an agent learns to make decisions by interacting with an environment and receiving rewards or punishments based on the outcomes of those interactions. However, defining a suitable reward function can be challenging and time-consuming, often leading to reward shaping or reward hacking, where the agent finds unexpected and often unintended ways to maximize the reward.

By using human feedback, reinforcement learning can address some of these issues by allowing humans to directly influence the learning process. Human trainers can provide real-time feedback, scale their feedback to match the complexity of the task, and update their feedback based on the agent's progress. This approach can lead to more robust and generalized agents, as they are guided not by handcrafted reward functions but by the nuanced understanding and experience of human trainers.

However, reinforcement learning from human feedback faces several challenges, such as the high cost of obtaining human feedback, subjective and inconsistent feedback, scaling issues, and methods for efficient and effective integration of human feedback into learning algorithms. Despite these challenges, the research in this area has shown promising results in various applications, including robotics, video games, prosthetics control, natural language processing, and many more.
```

然后是来自 [Llama 3.1 Tulu 3 70B DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO) 的 post-RLHF completion：
```text
Reinforcement learning from human feedback (RLHF) is a subset of reinforcement learning (RL) where the rewards that guide the learning process are provided by humans, rather than a pre-specified numerical reward function. This is particularly useful in situations where it is hard or infeasible to design a suitable reward function, or where the desired performance involves subjective judgments that humans are uniquely equipped to provide.

In traditional reinforcement learning, an agent interacts with an environment. With each interaction, the agent performs an action, receives an observation, and gets a numerical reward that tells it how well it performed in that situation. Through many such interactions, the agent learns to maximize the cumulative reward.

In contrast, in reinforcement learning from human feedback:

1. **Human Input:** Instead of a predefined reward function, the feedback comes from human evaluators, who provide qualitative, subjective judgments about the agent's performance. This feedback can be in various forms, such as numerical ratings, binary preference signals, natural language feedback, or even demonstrations showing how to improve.

2. **Feedback Loop:** Human evaluators observe the agent's actions or outcomes and provide feedback in real-time (or close to real-time). The agent learns through this iterative feedback loop, gradually adapting its behavior to align more closely with human preferences.

3. **Reward Shaping:** Because raw human feedback is not always directly usable in learning algorithms due to variability and potential bias, a reward shaping process may be involved, whereby human feedback is converted into a more stable signal that guides learning.

4. **Preference-based RL:** One common approach within RLHF is preference-based reinforcement learning, where the agent is trained by comparing pairs of actions (or outcomes) and receiving human feedback on which one is better. From these pairwise comparisons, the agent infers a reward function that it can use to guide its actions.

5. **Inverse Reinforcement Learning (IRL):** Another technique used in RLHF is IRL, which aims to infer the underlying reward function that could explain human actions or decisions, and then use that reward function to train the agent.

**Applications:**

Reinforcement Learning from human feedback has found applications in:

- **Robotics:** Teaching robots to perform tasks that are difficult to specify with predefined reward functions.
- **Game playing:** Human feedback can guide AI agents to better emulate human-like strategies or behaviors.
- **Language and content generation:** For example, suggestively guiding AI models to generate more coherent, creative, or helpful content.
- **Recommendation systems:** Human feedback can help adjust and refine the recommendations provided to users.

**Challenges:**

Using human feedback introduces some specific challenges:

1. **Scalability:** Human feedback is time-consuming and expensive to collect, and it's hard to scale to complex or high-dimensional tasks.
2. **Bias and Noise:** Human feedback can be noisy, inconsistent, and biased, leading to suboptimal learning outcomes.
3. **Alignment:** Ensuring that the reinforcement learning agent aligns with the true preferences of the human evaluator(s) and not just the superficial feedback is difficult.
4. **Safety and Misuse:** There can be concerns about the safety and ethical implications of using potentially flawed human feedback, especially if the feedback process is corrupted or gamed.

Researchers are actively working on techniques to mitigate these challenges and improve RLHF approaches. One such approach is leveraging machine learning models to amplify the efficacy of the human feedback, for example, by learning to generalize from sparse human input or by synthesizing plausible feedback in situations where direct human input is not available.
```

DPO 模型的回复包含类似信息，因为两个模型都在同一个预训练基础模型上训练；但偏好微调添加了更多结构、格式，当然也增加了长度，而这些特征已经被反复证明与人类对语言模型的一般偏好相关。
更多类似示例，可以查看本书配套的 completions library：[https://www.rlhfbook.com/library](https://rlhfbook.com/library)。

## Chattiness 的平衡

无数人正在使用 RLHF 或偏好微调方法来提升 AlpacaEval 以及其他自动聊天排行榜上的分数。这些排行榜使用 LLM-as-a-judge 来近似衡量 agent 在简单对话任务中有多 helpful、harmless 和 honest。但 RLHF 在这里带来的巨大增益，并不会在 Arena 这类更难被 game 的评估上成比例地体现出来。
其中的张力在于，虽然 RLHF 方法能给这些模型带来可测量的提升，但这种训练并不总能转化为人们真正关心的性能。
随着 RLHF 文献的建立，大量模型使用相关方法来提升模型与 RLHF 的 "alignment"，但它们往往做得过头，并发布了从误导性到毫无意义不等的评估分数。

这些由 alignment 动机驱动的 RLHF 方法，如果做得正确，会让模型更容易使用，也更令人愉快。
这通常会伴随 MT Bench 或 AlpacaEval 等评估工具上的明确改进。

2023 年秋季，关于 direct preference optimization（DPO）及其相对于 proximal policy optimization（PPO）和其他基于 RL 的偏好微调方法的作用，出现了一轮辩论高潮。聊天评估与真实世界性能之间的平衡正是这场辩论的核心（关于这些取舍的更多技术讨论，见第 8 章、Ivison et al. 2024 [@ivison2024unpacking]，或[这个演讲](https://youtu.be/YJMCSVLRUNs)）。
问题在于，你也可以在反馈循环中，或在大量数据上使用 DPO 和 PPO 这类技术，从而为了换取聊天性能而严重伤害模型在数学、编程等其他任务上的表现。

在 DPO 与 PPO 之争大量涌现期间，许多论文给出了惊人的 benchmark，但没有带来有意义的采用。
如果这些论文发布了模型权重，它们在公共使用中也并不流行，因为这些模型在一般使用中并不稳健。
在 2023 年秋季或不久之后应用 RLHF 时，不可能让一个 70 亿参数模型的 aligned 版本在全面 benchmark 上真正击败 GPT-4。这类比较仍然成立：当时的小模型无法稳健击败最好的大型前沿模型。
这看起来很明显，但总会有论文声称取得了这类结果。
@fig:DNO 来自一篇名为 Direct Nash Optimization（DNO）的论文，它声称其模型在 2024 年 4 月的 7B 模型 AlpacaEval 上达到某种 state-of-the-art [@rosset2024direct]。
作为背景，DNO 是一种 batched、on-policy、*iterative* 的替代方案，用来替代 reward-model+PPO（经典 RLHF）或一次性 DPO。它把 alignment 表述为寻找相对于偏好 oracle 的 Nash equilibrium，从而直接优化成对偏好（win-rate gaps）。
当学术激励与受到整个社会高度关注的技术发生交汇时，这些挑战就会出现。

![Direct Nash Optimization (DNO) 论文中的结果，突出显示其小模型超过 GPT-4 等模型。Rosset et al. 2024。License CC-BY。](images/dno-figure.png){#fig:DNO width=550px}

即便是 2024 年 1 月的开创性论文 Self Rewarding Language Models [@yuan2025selfrewardinglanguagemodels]，也披露了 Llama 2 70B 上不现实地强的分数。
当然，在当时，70B 模型比 7B 模型更接近 GPT-4（正如我们在 2024 年令人印象深刻的 Llama 3 发布中看到的那样），但重要的是把模型现实与现代 RLHF 论文中的主张区分开。
这些模型被调到狭窄测试集上，在真实使用中无法很好地支撑它们宣称击败的那些远大得多的模型。
还有许多类似方法来来去去，既分享了有价值的洞见，也给出了夸大的结果，这让 RLHF 更难理解。

应用了 "funky RLHF" 的模型，一个常见症状是长度偏置。
这种现象变得如此常见，以至于 AlpacaEval 和 WildBench 等多个评估系统都加入了线性长度校正机制。
这修补了通过 chattiness doping 来 "击败 GPT-4" 或击败当时领先前沿模型的激励，并创造了一种更少游戏化的动态，使更短且有用的模型也能真正获胜。

无论如何，在文献中，仅仅为了 chattiness 而对齐聊天模型如今已经带有一点声誉成本，因为人们承认这些狭窄方法可能以其他方式伤害模型。
下面这段来自 2023 年原始 Alibaba Qwen 模型的说明，在早期 alignment 实验中已经被多次观察到，它夸大了 chattiness 与性能之间的取舍 [@qwen]。

> We pretrained the models with a large amount of data, and we post-trained the models with both supervised fine-tuning and direct preference optimization. However, DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation.

一个较早且做得不错的例子，是 2024 年 3 月的 Starling Beta [@zhu2024starling]。
这个模型是从另一个聊天模型 OpenChat [@wang2023openchat] 微调而来，而 OpenChat 实际上是由另一个完整组织训练的。
它的训练完全聚焦于 k-wise reward model training 和 PPO optimization，并让模型在 Arena 上前进了 10 个名次。
模型的平均回复长度增加了，但增加方式足够好，确实帮助了人类评分者。
后来的例子，如 Olmo 3，被记录为经历了大量聊天训练，但作者更偏好最终模型 checkpoint 在数学、编程和推理上得分更高，而不是选择在基于 LLM-as-a-judge 的聊天 benchmark 上最高的潜在 checkpoint [@teamolmo2025olmo3]。

一个自然问题是：为什么 RLHF 会让模型回复变长？
从根本上说，Arena 等评估已经告诉我们，相比简短回复，模型的普通用户常常喜欢更长、更完整的答案。
更长的答案会让快速评估它们的用户觉得更全面、更有帮助，甚至更可信。
这并不代表 *每个* 用户的偏好，但这些模型被训练来匹配许多数据标注者的平均偏好，所以 RLHF 往往会让模型更冗长。
