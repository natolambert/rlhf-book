<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "塑造模型性格与产品"
prev-url: "17-product"
page-title: "附录 A：定义"
search-title: "附录 A：定义"
meta-description: "RLHF、强化学习、语言模型和后训练术语的定义与背景。"
next-chapter: "超越 \"只是风格\""
next-url: "appendix-b-style"
---

# 定义

本附录汇总 RLHF 过程中经常使用的定义、符号和操作，并快速概述语言模型。语言模型是本书的主要应用背景。

## 语言建模概览

大多数现代语言模型都以自回归方式训练，用来学习 token 序列（词、子词或字符）的联合概率分布。
自回归的含义很简单：每一次下一个预测都依赖序列中此前的实体。
给定 token 序列 $x = (x_1, x_2, \ldots, x_T)$，模型把整个序列的概率分解为一组条件分布的乘积：

$$P_{\theta}(x) = \prod_{t=1}^{T} P_{\theta}(x_{t} \mid x_{1}, \ldots, x_{t-1}).$$ {#eq:llming}

为了拟合一个能准确预测这一分布的模型，目标通常是最大化当前模型对训练数据给出的似然。
为此，可以最小化负对数似然（negative log-likelihood, NLL）损失：

$$\mathcal{L}_{\text{LM}}(\theta)=-\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T}\log P_{\theta}\left(x_t \mid x_{<t}\right)\right]. $$ {#eq:nll}

实践中，通常会针对每个下一个 token 预测使用交叉熵损失，也就是比较序列中的真实 token 与模型预测结果。

语言模型有许多架构，不同架构在知识、速度和其他性能特征上有不同取舍。
包括 ChatGPT、Claude、Gemini 等在内的现代 LM，最常使用 **decoder-only Transformers** [@Vaswani2017AttentionIA]。
Transformer 的核心创新，是大量使用 **self-attention** [@Bahdanau2014NeuralMT] 机制，使模型能够直接关注上下文中的概念，并学习复杂映射。
本书会多次讨论给 Transformer 添加新的 head，或者修改语言建模（LM）head，尤其是在第 5 章介绍奖励模型时。
LM head 是最后的线性投影层，它把模型内部 embedding 空间映射到 tokenizer 空间，也就是词表。
本书会看到，语言模型的不同 "heads" 可以用于把模型微调到不同目的。在 RLHF 中，这最常发生在训练奖励模型时，第 5 章会重点说明。

## 机器学习

- **Kullback-Leibler (KL) divergence ($\mathcal{D}_{\text{KL}}(P || Q)$)**，也称 KL 散度，是衡量两个概率分布差异的指标。
对于定义在同一概率空间 $\mathcal{X}$ 上的离散概率分布 $P$ 和 $Q$，从 $Q$ 到 $P$ 的 KL 距离定义为：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:def_kl}


## 自然语言处理

- **Chosen Completion ($y_c$)**：在多个备选项中被选择或更受偏好的 completion，常记作 $y_{chosen}$。

- **Completion ($y$)**：语言模型响应 prompt 生成的输出文本。completion 常记作 $y\mid x$。奖励和其他值通常计算为 $r(y\mid x)$ 或 $P(y\mid x)$。

- **Policy ($\pi$)**：可能 completion 上的概率分布，由 $\theta$ 参数化：$\pi_\theta(y\mid x)$。

- **Preference Relation ($\succ$)**：表示一个 completion 比另一个更受偏好的符号，例如 $y_{chosen} \succ y_{rejected}$。例如，奖励模型会预测偏好关系的概率 $P(y_c \succ y_r \mid x)$。

- **Prompt ($x$)**：给语言模型的输入文本，用于生成回复或 completion。

- **Rejected Completion ($y_r$)**：成对比较设置中不受偏好的 completion。

## 强化学习

- **Action ($a$)**：agent 在环境中做出的决策或动作，常表示为 $a \in A$，其中 $A$ 是可能动作的集合。

- **Advantage Function ($A$)**：优势函数 $A(s,a)$ 衡量在状态 $s$ 下采取动作 $a$ 相对于平均动作的相对收益。它定义为 $A(s,a) = Q(s,a) - V(s)$。优势函数（以及价值函数）可以依赖某个具体策略，即 $A^\pi(s,a)$。

- **Discount Factor ($\gamma$)**：标量 $0 \le \gamma < 1$，在 return 中对未来奖励进行指数衰减，用于权衡即时收益和长期收益，并保证无限时域求和收敛。有时不使用折扣，这等价于 $\gamma=1$。

- **Expectation of Reward Optimization**：RL 中的主要目标，即最大化期望累计奖励：

  $$\max_{\theta} \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$ {#eq:expect_reward_opt}

  其中 $\rho_\pi$ 是策略 $\pi$ 下的状态分布，$\gamma$ 是折扣因子。

- **Finite Horizon Reward ($J(\pi_\theta)$)**：由 $\theta$ 参数化的策略 $\pi_\theta$ 的有限时域期望折扣 return，定义为：

  $$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$ {#eq:finite_horizon_return}

  其中 $\tau \sim \pi_\theta$ 表示按照策略 $\pi_\theta$ 采样得到的轨迹，$T$ 是有限时域长度。

- **On-policy**：在 RLHF 中，尤其是在 RL 与直接对齐算法的讨论中，**on-policy** 数据经常出现。在 RL 文献中，on-policy 意味着数据 *精确地* 由 agent 的当前形式生成；但在一般偏好调优文献中，on-policy 被扩展为来自该版本模型的生成结果，例如在运行任何偏好微调之前的指令微调 checkpoint。在这个语境中，off-policy 可以指由任何其他用于后训练的语言模型生成的数据。

- **Policy ($\pi$)**，在 RLHF 中也称 **policy model**：在 RL 中，策略是 agent 在给定状态下决定采取哪个动作的策略或规则：$\pi(a\mid s)$。

- **Policy-conditioned Values ($[]^{\pi(\cdot)}$)**：在 RL 推导和实现中，理论和实践的关键组成部分是收集以特定策略为条件的数据或值。本书会在更简单的价值函数记号（$V,A,Q,G$）和它们对应的特定策略条件值（$V^\pi,A^\pi,Q^\pi$）之间切换。期望值计算中同样关键的是从数据 $d$ 中采样，而 $d$ 也以特定策略为条件，即 $d_\pi$。例如，在估计 $\mathbb{E}_{s\sim d_\pi,\,a\sim\pi(\cdot\mid s)}\!\left[A^\pi(s,a)\right]$ 时，有 $s \sim d_\pi$ 且 $a \sim \pi(\cdot\mid s)$。

- **Q-Function ($Q$)**：估计在给定状态下采取某个具体动作之后的期望累计奖励的函数：$Q(s,a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]$。

- **Reward ($r$)**：表示某个动作或状态可取程度的标量值，通常记作 $r$。

- **State ($s$)**：环境当前的配置或情形，通常记作 $s \in S$，其中 $S$ 是状态空间。

- **Trajectory ($\tau$)**：轨迹 $\tau$ 是 agent 经历的一系列状态、动作和奖励：$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$。

- **Trajectory Distribution ($(\tau\mid\pi)$)**：策略 $\pi$ 下某条轨迹的概率为 $P(\tau\mid\pi) = p(s_0)\prod_{t=0}^T \pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)$，其中 $p(s_0)$ 是先验状态分布，$p(s_{t+1}\mid s_t,a_t)$ 是转移概率。

- **Value Function ($V$)**：估计从给定状态出发的期望累计奖励的函数：$V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$。

## 仅 RLHF 使用

- **Reference Model ($\pi_{\text{ref}}$)**：RLHF 中保存下来的一组参数，其输出用于正则化优化过程。

## 扩展术语表

- **Chain-of-Thought (CoT)**：Chain-of-thought 是语言模型的一种特定行为，即模型被引导为以逐步分解问题的形式工作。最早的版本来自 prompt "Let's think step-by-step" [@wei2022chain]。

- **Distillation**：蒸馏是训练 AI 模型的一组通用实践，其中一个模型在更强模型的输出上训练。这是一种合成数据，已知可以产生较强的小模型。大多数模型会通过开放权重模型的 license，或仅可通过 API 访问模型的服务条款，明确关于蒸馏的规则。如今，distillation 这个词也承载了机器学习文献中的一个特定技术定义。

- **In-context Learning (ICL)**：这里的 in-context 指语言模型上下文窗口中的任何信息。通常，这是添加到 prompt 中的信息。最简单的 in-context learning 形式，是在 prompt 前加入形式相似的示例。更高级的版本可以学习针对特定用例应纳入哪些信息。

- **(Teacher-student) Knowledge Distillation**：从特定 teacher 到 student 模型的知识蒸馏，是上面所述蒸馏的一种具体类型，也是这个术语的来源。它是一种具体的深度学习方法，其中神经网络损失被修改为从 teacher 模型在多个潜在 token/logits 上的 log-probabilities 中学习，而不是直接从一个被选择的输出中学习 [@hinton2015distilling]。使用 Knowledge Distillation 训练的现代模型系列包括 Gemma 2 [@team2024gemma] 或 Gemma 3。对于语言建模设置，下一个 token 损失函数可以修改如下 [@agarwal2024policy]，其中 student 模型 $P_\theta$ 从 teacher 分布 $P_\phi$ 中学习：

$$\mathcal{L}_{\text{KD}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T} P_{\phi}(x_t \mid x_{<t}) \log P_{\theta}(x_t \mid x_{<t})\right]. $$ {#eq:knowledge_distillation}

- **Synthetic Data**：指任何由另一个 AI 系统输出、并用于训练 AI 模型的数据。这可以包括从模型开放式 prompt 生成的文本，也可以包括模型对已有内容的改写。
