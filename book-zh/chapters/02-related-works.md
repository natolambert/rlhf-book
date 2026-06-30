<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "Introduction"
prev-url: "01-introduction"
page-title: RLHF 简史
search-title: "第 2 章：RLHF 简史"
meta-description: "RLHF、奖励建模、偏好学习和后训练语言模型背后的关键论文与历史里程碑。"
next-chapter: "Training Overview"
next-url: "03-training-overview"
lectures:
  - video: "https://www.youtube.com/watch?v=MMDNaeIFVy8&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "Lecture 0: Prerequisites"
  - video: "https://www.youtube.com/watch?v=o6l6tJQgUg4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "Lecture 1: Overview (Chapters 1–3)"
---

# RLHF 简史

RLHF 及其相关方法都非常新。
我们梳理这段历史，是为了说明这些流程被形式化的时间有多晚，以及其中有多少内容仍散见于学术文献中。
借此我们也想强调，RLHF 正在非常快速地演进；因此，本书会为若干方法保留不确定性，并预期围绕少数核心实践的一些细节会继续变化。
除此之外，这里列出的论文和方法也展示了 RLHF 流水线中的许多组成部分为何会形成今天的样子，因为其中一些奠基性论文原本面向的应用与现代语言模型完全不同。

本章将详细介绍把 RLHF 领域推进到今天位置的关键论文和项目。
这里并不试图对 RLHF 及相关领域做全面综述，而是给出一个起点，并复述我们如何走到今天。
它有意聚焦于近期那些通向 ChatGPT 的工作。
在强化学习文献中，关于从偏好中学习还有大量后续工作 [@wirth2017survey]。
如果需要更完整的列表，应参考正式的综述论文 [@kaufmann2023survey], [@casper2023open]。

![本章讨论的 RLHF 关键发展时间线，从早期基于偏好的强化学习工作，到 RLHF 在大语言模型中的采用。](images/rlhf_timeline.png){#fig:rlhf_timeline data-dark-src="images/rlhf_timeline-dark.png"}

## 起源至 2018 年：基于偏好的强化学习

随着深度强化学习的发展，这一领域近年逐渐受到关注，并扩展为许多大型科技公司研究 LLM 应用的更广泛方向。
不过，今天使用的许多技术都与早期基于偏好的强化学习文献中的核心技术密切相关。

最早提出与现代 RLHF 相似方法的论文之一是 *TAMER*。
*TAMER: Training an Agent Manually via Evaluative Reinforcement* 提出了一种方法：人类反复为智能体的动作打分，以学习一个奖励模型，再用该模型学习动作策略 [@knox2008tamer]。
同期或稍后的其他工作提出了 actor-critic 算法 COACH，其中使用人类反馈（包括正向和负向反馈）来调整优势函数 [@macglashan2017interactive]。

主要参考文献 Christiano et al. 2017 是将 RLHF 应用于 Atari 游戏中智能体轨迹偏好的一个例子 [@christiano2017deep]。
这项引入 RLHF 的工作紧随 DeepMind 在强化学习中关于 Deep Q-Networks (DQN) 的奠基性工作之后；后者表明，强化学习智能体可以从零开始学习并解决流行视频游戏。
这项工作表明，在某些领域，让人类在轨迹之间做选择可能比直接与环境交互更有效。它依赖一些巧妙的条件，但依然令人印象深刻。

![Christiano et al. (2017) 中的核心 RLHF 循环：奖励预测器通过轨迹片段比较异步训练，而智能体最大化预测奖励。](images/rlhf_schematic.png){#fig:rlhf_schematic width=66% data-dark-src="images/rlhf_schematic-dark.png"}

随后，这一方法通过更直接的奖励建模得到扩展 [@ibarz2018reward]；早期 RLHF 工作中对深度学习的采用，又在一年后由使用神经网络模型扩展 TAMER 的工作画上阶段性句点 [@warnell2018deep]。

这一时期开始发生转向：奖励模型作为一种一般概念，被提出用来研究对齐，而不只是解决强化学习问题的工具 [@leike2018scalable]。

## 2019 至 2022 年：语言模型上的基于人类偏好的强化学习

基于人类反馈的强化学习在早期也常被称为基于人类偏好的强化学习；随着 AI 实验室越来越多地转向扩展大语言模型，它很快被采纳。
这部分工作中有很大一部分开始于 2019 年的 GPT-2 与 2020 年的 GPT-3 之间。
2019 年最早的工作 *Fine-Tuning Language Models from Human Preferences* 与现代 RLHF 工作以及本书将覆盖的内容有许多显著相似之处 [@ziegler2019fine]。
许多标准术语，例如学习奖励模型、KL 距离、反馈图等，都在这篇论文中被形式化；不过，最终模型的评估任务及其能力与今天人们关注的内容并不相同。
从这里开始，RLHF 被应用到多种任务中。
重要例子包括通用摘要 [@stiennon2020learning]、书籍递归摘要 [@wu2021recursively]、指令跟随（InstructGPT）[@ouyang2022training]、浏览器辅助问答（WebGPT）[@nakano2021webgpt]、用引用支撑答案（GopherCite）[@menick2022teaching]，以及通用对话（Sparrow）[@glaese2022improving]。

除应用之外，一批奠基性论文定义了 RLHF 未来的关键方向，包括：

1. 奖励模型过度优化 [@gao2023scaling]：强化学习优化器可能过拟合在偏好数据上训练出来的模型；
2. 将语言模型作为对齐研究的一个一般领域 [@askell2021general]；以及
3. Red teaming [@ganguli2022red] -- 评估语言模型安全性的过程。

围绕将 RLHF 应用于聊天模型的细化工作仍在继续。
Anthropic 在 Claude 的早期版本中继续大量使用 RLHF [@bai2022training]，早期的 RLHF 开源工具也开始出现 [@ramamurthy2022reinforcement], [@havrilla-etal-2023-trlx], [@vonwerra2022trl]。

## 2023 年至今：ChatGPT 时代

ChatGPT 发布时非常明确地说明了 RLHF 在其训练中的作用 [@openai2022chatgpt]：

> We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup.

从那以后，RLHF 被广泛用于领先的语言模型。
众所周知，它被用于 Anthropic 面向 Claude 的 Constitutional AI [@bai2022constitutional]、Meta 的 Llama 2 [@touvron2023llama] 和 Llama 3 [@dubey2024llama]、Nvidia 的 Nemotron [@adler2024nemotron]、Ai2 的 Tülu 3 [@lambert2024t] 等模型。

今天，RLHF 正在发展为更广泛的偏好微调（PreFT）领域，其中包括一些新应用：用于中间推理步骤的过程奖励 [@lightman2023let]（第 5 章介绍）；受 Direct Preference Optimization (DPO) [@rafailov2024direct] 启发的直接对齐算法（第 8 章介绍）；从代码或数学的执行反馈中学习 [@kumar2024training], [@singh2023beyond]，以及受 OpenAI 的 o1 [@openai2024o1] 启发的其他在线推理方法（第 7 章介绍）。
