<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "奖励建模"
prev-url: "05-reward-models"
page-title: 强化学习
search-title: "第 6 章：强化学习"
meta-description: "用于 RLHF 和大语言模型后训练的策略梯度方法，包括 PPO、REINFORCE、RLOO、GRPO 及其实现细节。"
next-chapter: "推理与推理时扩展"
next-url: "07-reasoning"
lectures:
  - video: "https://www.youtube.com/watch?v=K_Sj_-1BUMM&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=4"
    label: "第 3 讲：理解用于大语言模型 RL 的策略梯度算法"
  - video: "https://www.youtube.com/watch?v=i-AIMpZHgeg&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=5"
    label: "第 4 讲：实现用于大语言模型的 RL 算法"
---

# 强化学习

在 RLHF 流程中，强化学习算法会根据奖励模型给出的反馈，逐步更新模型权重。
策略（policy）就是正在训练的模型；它会针对训练集中的 prompt 生成补全，然后由奖励模型打分，强化学习优化器再基于这些信息执行梯度更新（概览见 @fig:rlhf-overview）。
本章解释多种算法的数学形式与权衡：这些算法都试图从奖励模型给 on-policy 数据提供的信号中学习。
这些算法通常会运行许多个 epoch，往往在更大的 prompt 集合上处理成千上万乃至数百万个 batch，并在每个 batch 之间执行梯度更新。

## 强化学习在 RLHF 中的作用

让 RLHF 在语言模型中流行起来的算法，是策略梯度强化学习算法。
这些算法，例如 Proximal Policy Optimization (PPO)、Group Relative Policy Optimization (GRPO) 和 REINFORCE，会使用最近生成的样本来更新模型，而不是像 Deep Q-Networks (DQN) 这类算法那样把分数存入 replay buffer；后者曾用于 AlphaGo 等广为人知的项目。
本节会介绍策略梯度算法的基础，以及它们如何用于现代 RLHF 框架。

从机器学习角度看，本节是 RLHF 流程中复杂度最高的部分。
不过，与大多数现代 AI 模型一样，决定其成功的最大因素仍然是输入到流程中的数据。

![RLHF 训练循环概览。来自数据集的 prompt 被传入调优后的策略，策略生成一个补全。奖励模型对该补全打分，同时冻结的初始模型（通常是 RL 之前的指令微调模型）在同一文本上计算 log probability，用于计算 KL 惩罚，以防止模型过度漂移。合并后的奖励信号随后驱动一次针对策略参数的强化学习更新。](images/rlhf-overview.png){#fig:rlhf-overview}

当 RLHF 随 ChatGPT 进入大众视野时，人们大体知道它使用了 PPO 的某种变体，许多早期工作也建立在这一点之上。
随着时间推移，多个研究项目展示了 REINFORCE 风格算法的潜力 [@ahmadian2024back] [@wang2024helpsteer2p]。这些算法相较 PPO 更简单：不需要单独的价值模型（节省内存，因此减少所需 GPU 数量），优势估计也更简单（不需要 Generalized Advantage Estimation, GAE；GAE 是策略梯度算法中用于降低方差的优势计算方法）。
后来还出现了更多算法，包括 Group Relative Policy Optimization，它在推理任务中特别流行；但总体而言，这些算法中的许多都可以调节以适配特定任务。
本章会覆盖核心策略梯度设定，以及上面提到的三类算法，因为它们在建立规范化 RLHF 文献中处于中心位置。

在最简单的形式中，RLHF 的 RL 阶段需要两个模型：一个策略（正在训练的模型），以及一个对其输出打分的奖励模型（上一章已经介绍）。
RL 之前的策略副本会作为 reference model，用来计算 KL 惩罚；该模型是冻结的，也就是说不会由自动微分引擎的梯度更新。
本章介绍的最复杂算法 PPO 还会加入第四个模型：一个学习得到的价值函数，用来估计 action 中每个 token 有多好；它同样是一个会在训练中更新的大语言模型。
本章中的算法主要差异在于它们如何估计称为优势（advantage）的量，也就是衡量模型当前 action（补全）相对于平均水平有多好；以及它们如何约束策略更新，让优化在数值上保持稳定。
@fig:rlhf-overview 展示了这一 RLHF 过程的视觉概览（其中没有价值模型）。

符号定义请参见问题设定章节。

*本章使用强化学习文献中的 $(s, a)$ 记号，其中 $s$ 表示状态，$a$ 表示 action。在语言模型语境中，你也经常会看到 $(x, y)$，其中 $x$ 是 prompt，$y$ 是补全。$(s, a)$ 框架更一般，因为这些算法最初是为序贯决策问题设计的，在这类问题中，每个时间步都会采取 action。不过，许多 RLHF 实现会把整个补全视为单个 action，因此 $(x, y)$ 记号同样有效。*

***RL Cheatsheet:** 本章所有核心 RL 损失函数的一页式参考可见 [rlhfbook.com/rl-cheatsheet](https://rlhfbook.com/rl-cheatsheet)。*

## 策略梯度算法

本章的核心，是理解下面这种形式的方程。
该方程计算的是我们正在训练的语言模型 $\pi_\theta$ 的梯度 $\Delta \theta$：

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$ {#eq:policy_gradient_intuition}

这里，方程由两个关键部分组成：
1. $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$：参数空间中的哪个方向会让 action $a_t$ 更可能出现。
2. $\Psi_t$：它有多好？这是一个对结果打分的标量。

把二者相乘，就得到了策略梯度更新。
有些性质很简单，例如 $\Psi_t > 0$ 会更新参数，使 $a_t$ 更可能出现；$\Psi_t < 0$ 则会让它更不可能出现。
策略梯度计算的是哪些参数促成了一个 action，以及我们应该让这个 action 在未来更可能还是更不可能发生。
本章余下部分会深入讨论完成这件事的不同方式，以及让它适用于 LLM 的具体技巧。

现在，让我们把它进一步形式化。
强化学习算法旨在最大化一条由状态 $s \in \mathcal{S}$ 与 action $a \in \mathcal{A}$ 组成的轨迹上的未来折扣奖励（更多记号见附录 A 的定义）。
智能体的目标通常称为回报（return），是在给定时间 $t$ 起累计的折扣奖励之和，其中 $\gamma\in [0,1]$ 是偏向近期奖励的因子：

$$G_t = r_t + \gamma r_{t+1} + \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k}.$$ {#eq:return_definition}

回报定义也可以递归地写成：
$$G_{t} = r_t + \gamma G_{t+1}.$$ {#eq:recursive_return}

这个回报是学习价值函数 $V(s)$ 的基础；价值函数估计在当前状态下未来回报是多少：

$$V(s) = \mathbb{E}\left[G_t \mid S_t = s \right].$$ {#eq:value_function}

所有策略梯度算法都会优化策略 $\pi_\theta(a\mid s)$ 以最大化期望回报；这个目标可以用诱导出的价值函数 $V^{\pi_\theta}(s)$ 表示。

令 $d_0(s)$ 为初始状态分布。我们要最大化的 episodic 目标可以写为：
$$
J(\theta)
\;=\;
\sum_{s} d_0(s) V^{\pi_\theta}(s),
$$ {#eq:policy_objective}

在有限 MDP 中，这是对可能起始状态的求和，但实践中我们从不会精确计算它。
相反，我们通过从当前策略采样 rollout 来用数据估计它。
在 RLHF 中，这通常意味着从数据集中采样 prompt $x_i$，并生成补全 $y_i \sim \pi_\theta(\cdot\mid x_i)$。
令 $R(x_i, y_i)$ 表示分配给该 prompt-补全对的标量序列级奖励；如果 $\tau_i$ 是对应 episode，那么这就是轨迹奖励 $R(\tau_i)$。
随后我们取如下经验平均：

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} R(x_i, y_i),
$$ {#eq:empirical_batch_estimate}

或者，在每步都有奖励的 MDP 视角中，

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \sum_{t=0}^{T_i} \gamma^t r_{i,t}.
$$ {#eq:empirical_mdp_estimate}

实践中，面向语言模型的 RLHF 会设定 $\gamma = 1$（不折扣），因为优化单位是整体补全，而不是单个 token；这一选择会在本章后面的 MDP vs. Bandit 一节中进一步讨论。

策略梯度算法的核心，是计算当前策略下有限时间期望回报对参数的梯度。
有了这个期望回报 $J$ 后，参数更新可按如下方式计算，其中 $\alpha$ 是学习率：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$ {#eq:policy_update}

核心实现细节在于如何计算这个梯度。

### 推导策略梯度

令 $p_\theta(\tau)$ 表示由初始状态分布 $d_0$、策略 $\pi_\theta$ 和环境转移动力学诱导出的轨迹分布，其展开形式见下面的 @eq:trajectory_probability。
我们想最大化的 RL 目标也可以写成：
$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ R(\tau) \right],
$$ {#eq:policy_objective_expectation}

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是一条轨迹，$R(\tau) = \sum_{t=0}^\infty r_t$ 是轨迹的总奖励。或者，我们也可以把期望写成对所有可能轨迹的积分：
$$
J(\theta) = \int_\tau p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_objective_integral}

注意，轨迹概率可以写成如下形式，其中 $\pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$ 把策略概率与从一个 state-action 对转移到下一个状态的环境转移概率结合起来：
$$
p_\theta (\tau) = d_0(s_0) \prod_{t=0}^\infty \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t),
$$ {#eq:trajectory_probability}

如果我们对目标函数（@eq:policy_objective_expectation）关于策略参数 $\theta$ 求梯度：
$$
\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_gradient_integral}

注意，我们可以使用 [log-derivative trick](https://andrewcharlesjones.github.io/journal/log-derivative.html)，把积分的梯度改写成一个期望：
$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau) &= \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} &\text{(from chain rule)} \\
\implies \nabla_\theta p_\theta(\tau) &= p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) &\text{(rearranging)}
\end{aligned}
$$ {#eq:log_chain_rule}

使用这个 log-derivative trick：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau \\
&= \int_\tau p_\theta (\tau) R(\tau) \nabla_\theta \log p_\theta (\tau) d\tau \\
&= \mathbb{E}_{\tau \sim p_\theta} \left[ R(\tau) \nabla_\theta \log p_\theta (\tau) \right]
\end{aligned}
$$ {#eq:policy_gradient_expectation}

最后一步使用的是轨迹分布 $p_\theta(\tau)$ 下期望的定义：对任意函数 $f$，$\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] = \int_\tau f(\tau)\,p_\theta(\tau)\,d\tau$（离散情形下则是求和）。
写成期望很有用，因为我们可以用 Monte Carlo rollout 来近似它，例如对由当前策略诱导出的轨迹 $\tau_i \sim p_\theta$，使用 $\frac{1}{B}\sum_{i=1}^{B} f(\tau_i)$。

回到推导，展开轨迹的 log probability：

$$
\log p_\theta (\tau) = \log d_0(s_0) + \sum_{t=0}^\infty \log \pi_\theta(a_t|s_t) + \sum_{t=0}^\infty \log p(s_{t+1}|s_t, a_t)
$$ {#eq:trajectory_log_prob}

现在，如果我们对上式求梯度，会得到：

- $\nabla_\theta \log d_0(s_0) = 0$（初始状态分布不依赖于 $\theta$）
- $\nabla_\theta \log p(s_{t+1}|s_t, a_t) = 0$（环境转移动力学不依赖于 $\theta$）
- 只有 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 会保留下来

因此，轨迹 log probability 的梯度可简化为：
$$
\nabla_\theta \log p_\theta (\tau) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t)
$$ {#eq:trajectory_log_grad}

得到这个方程，是实现中的一个关键点。
到这里，我们已经足以看出，轨迹分布的梯度会化约为语言模型策略概率梯度之和；这些概率也就是我们正在训练的模型给出的 token 概率。
实践中，这会导向策略梯度方程的一种常见形式。
它们最终看起来像是在损失中对 log-probability 求和，然后我们通过自动微分计算梯度。
你会反复看到大致如下的短代码片段：

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

你会在本章中不断见到这种形式。现在，回到正式的策略梯度数学。

把这个结果代回 @eq:policy_gradient_expectation，我们得到：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^\infty R(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$ {#eq:policy_gradient_returns}

人们很常使用策略梯度的更一般形式：
$$
g = \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^\infty \Psi_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$ {#eq:general_gradient}

其中 $\Psi_t$ 可以取以下形式（奖励也常常可以由 $\gamma$ 折扣），这是采用自 Schulman et al. 2015 [@schulman2015high] 的分类：

1. $R(\tau) = \sum_{t=0}^{\infty} r_t$：轨迹的总奖励。
2. $\sum_{t'=t}^{\infty} r_{t'}$：action $a_t$ 之后的奖励，也就是从时间 $t$ 开始的回报 $G_t$。
3. $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$：上一公式加入 baseline 后的版本。
4. $Q^{\pi}(s_t, a_t)$：state-action 价值函数。
5. $A^{\pi}(s_t, a_t)$：优势函数；如果能够准确计算，它会给出理论上可能的最低方差。
6. $r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$：Temporal Difference (TD) residual。

*baseline* 是一种用于降低策略更新方差的值（下文会详细说明）。

对于语言模型，其中一些概念并不那么自然。
例如，对于确定性策略 $\pi$，状态价值为 $V^{\pi}(s_t) = Q^{\pi}(s_t, \pi(s_t))$（而最优价值函数满足 $V^*(s_t)=\max_{a_t} Q^*(s_t,a_t)$）。对于随机策略，对应恒等式为 $V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot\mid s_t)}\!\left[Q^{\pi}(s_t,a_t)\right]$。
Bellman 方程把 Q 与 V 联系起来：一般来说 $Q^\pi(s_t,a_t) = \mathbb{E}\!\left[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t\right]$；但对语言模型而言，状态转移是确定性的，因此可简化为 $Q(s_t,a_t) = r_t + \gamma V(s_{t+1})$。
优势函数衡量 action $a_t$ 相比平均水平好多少：

$$A(s_t,a_t) = Q(s_t,a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$ {#eq:advantage_trick}

这个最终形式正是 temporal difference (TD) residual（上面第 6 项）：它是 RL 中的基本量，用于衡量价值函数预测与实际发生结果之间的差距，并驱动价值函数更新到更准确的估计。实践中，会使用学习得到的价值函数 $\hat{V}$ 通过这个 TD error 估计优势。

### 原始策略梯度

原始策略梯度实现通过对策略参数求导，优化上面对 $J(\theta)$ 的表达式。
一个关于时间 $t$ 回报的简单版本是：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$ {#eq:vanilla_policy_gradient}

原始策略梯度算法的常见问题，是梯度更新具有较高方差；这可以用多种方式缓解。
高方差来自这样一个事实：梯度更新通常是用环境中数量较少的 rollout 来估计回报 $G$，而这些 rollout 往往容易受到噪声影响（例如以 temperature $>0$ 从语言模型生成时的随机性）。
在稀疏奖励领域中，回报估计之间的方差更高，因为更多样本是 0 或 1，而不是紧密聚集在一起。
为缓解这一点，人们会使用各种技术来归一化价值估计，称为 *baseline*。
Baseline 以多种方式发挥作用，本质上是根据下游 action 相对于当前状态的价值来做归一化（例如 Advantage 的情形，它是 Q 值和价值之间的差）。
最简单的 baseline 是 batch 奖励均值或移动平均。
即便这些与 action 无关的 baseline，也能在不改变期望梯度的情况下降低方差，因为对任意依赖状态的 $b(s)$，都有 $\mathbb{E}_{a \sim \pi(a|s)}\!\left[b(s) \nabla_\theta \log \pi_\theta(a|s)\right] = 0$，从而显著改善学习信号。

本章讨论的许多策略梯度算法，都建立在策略梯度的优势形式之上：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T A^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$ {#eq:advantage_policy_gradient}


### REINFORCE

REINFORCE 这个算法名称很可能是一个逆向首字母缩略词，但它代表的算法组成部分与现代强化学习算法高度相关。
在开创性论文 *Simple statistical gradient-following algorithms for connectionist reinforcement learning* [@williams1992simple] 中，它被定义为：

> The name is an acronym for "REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility."

这三个部分描述的是如何执行 *reward increment*，也就是策略梯度步。
其更新规则有三部分：

1. Nonnegative factor：学习率（步长），必须是正数，例如下面的 $\alpha$。
2. Offset Reinforcement：baseline $b$，或其他用于提升稳定性的奖励归一化因子。
3. Characteristic Eligibility：把标量奖励信号归因到产生 action 的参数上。Williams 把这个 eligibility 项记为 $e$（不是指数函数）。在现代策略梯度记号中，它对应于 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$。

因此，它的形式看起来非常熟悉：

$$ \Delta_\theta = \alpha(r - b)e $$ {#eq:REINFORCE_BASIC}

用更现代的记号和广义回报 $G$，REINFORCE 算子可写为：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim p_{\theta}}\!\left[
    \sum_{t=0}^{T}
    (G_t - b(s_t))\,\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
\right],
$$ {#eq:REINFORCE_with_baseline}

这里，$G_t - b(s_t)$ 是当前状态下策略的 *advantage*，因此我们可以把策略梯度重写为一个后续会继续使用优势 $A$ 的形式：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim p_{\theta}}\!\left[
    \sum_{t=0}^{T}
    A_t\,\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
\right],
$$ {#eq:REINFORCE_with_advantage}

REINFORCE 是原始策略梯度的一种具体实现，它使用梯度的 Monte Carlo 估计器。

![语言模型的基本 REINFORCE 架构。成形后的奖励把奖励模型分数与来自 reference model 的 KL 惩罚结合起来。本章后续会一直建立在这一结构之上。](images/reinforce_tikz.png){#fig:reinforce-arch data-dark-src="images/reinforce_tikz-dark.png"}

### REINFORCE Leave One Out (RLOO)

REINFORCE Leave One Out 与标准 REINFORCE 的核心实现差异在于：它用 batch 中*其他*样本的平均奖励来计算 baseline，而不是对 batch 中所有奖励求平均 [@huang2024putting], [@ahmadian2024back], [@kool2019buy]。
通过把当前样本自己的奖励排除在自己的 baseline 之外，RLOO baseline 就与正在评估的 action 独立，从而使梯度估计器严格无偏。

关键是，这只有在每个状态（prompt）生成多条轨迹（补全）时才成立；这在多个语言模型 RL 微调领域中都是常见做法。

具体来说，对于 REINFORCE Leave-One-Out (RLOO) baseline，给定针对同一个 prompt $s$ 采样得到的 $K$ 条轨迹（在 prompt 条件下采取的 action）$a_1, \dots, a_K$，我们把 baseline 显式定义为如下 *per-prompt* 形式：

$$
b(s, a_k) = \frac{1}{K-1}\sum_{i=1, i\neq k}^{K} R(s, a_i),
$$ {#eq:RLOO_baseline}

由此得到优势：

$$
A(s, a_k) = R(s, a_k) - b(s, a_k).
$$ {#eq:RLOO_advantage}

等价地，它也可以表示为：

$$
A(s, a_k) = \frac{K}{K - 1}\left(R(s, a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s, a_i)\right).
$$ {#eq:RLOO_advantage_alt}

这是一个简单、低方差的 *per-prompt* 优势估计，与 Group Relative Policy Optimization, GRPO 中使用的组相对优势密切相关（稍后在 Proximal Policy Optimization, PPO 之后讨论）。
实践中，GRPO 风格训练的主要区别在于它如何应用 KL 正则项（作为显式损失项，还是折入奖励），以及是否使用 PPO 风格的 ratio clipping。
更具体地说，规范的 GRPO 实现在损失层面施加 KL 惩罚，而 RLOO 或传统策略梯度的推导则把 KL 惩罚施加到奖励本身。
随着 RLHF 转向推理以及带可验证奖励的强化学习（RLVR），KL 惩罚的总体流行度有所下降；许多面向推理的 RLHF 代码适配甚至会完全关闭它。
不过，RLOO 的优势仍然可以与 PPO 的 clipping 结合，这也说明这些算法之间有多相似。

RLOO 以及其他不使用价值网络的算法，即不使用额外模型副本（critic）来预测每个 token 的标量价值 $V(s_t)$ 的算法，在计算损失时会把同一个序列级优势（或奖励）分配给每个 token。
使用学习价值网络的算法，例如 PPO，则会给每个 token 单独分配不同的价值，并从 EOS token 处取得的最终奖励向前折扣。
带 KL 距离惩罚时，RLOO 会把补全上的 per-token KL 聚合起来，并把这个标量折入序列奖励，因此得到的优势会广播到所有 token。
PPO 则在计算 $A_t$ 之前，从 per-token 奖励中减去 per-token KL，从而实现 token 级 credit assignment。
GRPO 通常保留序列级优势，但会在损失中加入单独的 per-token 项，而不是从奖励中减去它。
这些细节和权衡会在本章后面讨论。

![REINFORCE Leave-One-Out (RLOO) 架构。每个 prompt 的多个补全为优势估计提供 leave-one-out baseline，而无需学习价值函数。](images/rloo_tikz.png){#fig:rloo-arch data-dark-src="images/rloo_tikz-dark.png"}

<!-- A nice formulation of LM RL loss functions is found here https://arxiv.org/pdf/2502.01600 -->

### 近端策略优化（PPO）

Proximal Policy Optimization (PPO) [@schulman2017proximal] 是 Deep RL 取得成功背后的基础算法之一，例如掌握 Dota 2 的 OpenAI Five [@berner2019dota] 以及大量后续研究。
PPO 相对于优势和策略概率所最大化的目标函数如下：

$$J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).$$ {#eq:PPO_EQN}

这里，$\pi_\theta(a|s)$ 是正在优化的当前策略，而 $\pi_{\theta_{\text{old}}}(a|s)$ 是用于收集训练数据的策略（也就是上一轮迭代中的策略）。
这两个策略之间的 ratio 来自 *importance sampling*，它允许我们复用旧策略下收集的数据，来估计新策略的梯度。

回忆策略梯度的优势形式（@eq:advantage_policy_gradient），我们有：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T A^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right].$$ {#eq:advantage_policy_gradient_recall}

这个期望是对由 $\pi_\theta$ 诱导出的轨迹分布采样得到的轨迹取的；但实践中，我们希望对一个由固定策略 $\pi_{\theta_{\text{old}}}$ 收集的数据 batch 做多个梯度步。
为了校正这种分布不匹配，我们乘以重要性权重 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$，它会根据样本在当前策略下相对于数据收集策略下更可能或更不可能出现的程度，对样本重新加权。
如果没有约束，当 ratio 远离 1 时，优化这个重要性加权目标会导致破坏性的大幅策略更新。
PPO 通过把 ratio 裁剪到 $[1-\varepsilon, 1+\varepsilon]$ 范围内来解决这个问题，确保策略在单次更新中不会变化过剧。

注意，当我们转向 PPO 及其同类算法时，经常会使用*目标函数*而不是显式梯度。
原因是，一旦包含 $\min$ 和 clipping 操作，PPO 目标就不再有容易解释的解析梯度（取决于写法，梯度有约 4 项，对应 @fig:ppo-obj 中的区域）；写出目标函数只是表达这些算法的更清晰方式。

为完整起见，PPO 通常会写成在时间步上的*期望*裁剪 surrogate objective：

$$
J(\theta)
=
\mathbb{E}_{t}\left[
\min\left(\rho_t(\theta)A_t,\ \text{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)A_t\right)
\right],
\qquad
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$ {#eq:PPO_EQN_EXPECTED}

目标函数通常只需加上一个负号，就能转成损失函数，让优化器尽量把它变得更负。

对于语言模型，目标函数（或损失）是逐 token 计算的。这可以直观地理解为：自回归预测的整个序列概率是各 token 概率的乘积。
由此，常见实现会使用 *log-probability*，因为它让现代语言建模框架中的计算更简单。
实践中，会计算 token log-probability 的差值，并对其取指数来恢复策略 ratio $\rho_t$。

$$ J(\theta) = \frac{1}{|a|} \sum_{t=0}^{|a|} \min\left(\frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}A_{t}, \text{clip} \left( \frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}, 1-\varepsilon, 1+\varepsilon \right) A_{t} \right).  $$  {#eq:PPO_EQN_EXPANDED}

这是 PPO 的 per-token 版本，也适用于其他策略梯度方法；但本章实现部分会进一步探讨。
这里，按 action 中 token 数量取平均的项 $\frac{1}{|a|}$ 来自常见实现实践，但它并不属于损失的正式推导（见 [@liu2025understanding]）。

![PPO 框架。学习得到的价值函数支持 Generalized Advantage Estimation (GAE)，从而得到 per-token 优势，并与裁剪 surrogate objective 一起使用。](images/ppo_tikz.png){#fig:ppo-arch data-dark-src="images/ppo_tikz-dark.png"}

下面我们解释在不同优势和策略 ratio 情况下，这个损失函数会触发哪些不同情形。
在实现层面，PPO 的内部计算包含两个主要项：1）带学习优势的标准策略梯度；2）基于最大步长的裁剪策略梯度。

为了理解不同情况如何出现，我们可以把策略 ratio 定义为：

$$\rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$ {#eq:PPO_POL_RATIO}

策略 ratio 是 PPO 及相关算法的核心。
它来自策略梯度计算，并以非常直观的方式控制参数更新。
对于任意数据 batch，该 batch 的第一个梯度步中，策略 ratio 从 1 开始，因为此时 $\pi_{\theta}$ 与 $\pi_{\theta_{\text{old}}}$ 相同。随后，在下一个梯度步中，如果该梯度步提高了某些带正优势 token 的可能性，策略 ratio 就会大于 1；反之则小于 1。常见做法是在更新 $\pi_{\theta_{\text{old}}}$ 之前，对每个 batch 执行 1-4 个策略梯度步。

### 理解 PPO 目标函数

总体而言，可以用“目标函数相对于策略 ratio 的两条曲线”来可视化 PPO 目标，如 @fig:ppo-obj 所示。
PPO 目标通过改变已采样 action 的概率来被最大化。
从数值上看，该目标通过巧妙使用 minimum 操作，同时处理正优势和负优势情形，使更新最多只被推到距离策略 ratio 1 一个 epsilon 的范围之外。

在 trust region 内，PPO 与其他策略梯度算法的行为相同。
这是刻意设计的。Trust region 是一个用于限制 PPO 及其同类算法最大步长的概念，以保证更新稳定。PPO 算法的核心，即 clip 与 min/max 函数，定义了这个区域。在区域之外，目标函数会变平。

“trust region”的想法来自数值优化文献 [@nocedal2006numerical]，但在 Deep RL 中由 Trust Region Policy Optimization (TRPO) 推广开来；TRPO 被视为 PPO 的前身 [@schulman2015trust]。
Trust region 是完整策略梯度步被应用的区域，因为更新尚未被 PPO 目标中的 max/min 操作“裁剪”。

![PPO 目标函数 $J(\theta)$ 随策略 ratio $\rho(\theta)$ 变化的可视化，同时展示正优势和负优势两种情况。每个面板中标注了三个 ratio 区域及其 unclipped 项、clipped 项、最终目标和梯度。](images/ppo-clip-viz.png){#fig:ppo-obj}

策略 ratio 与优势可以组合成几种不同配置；@fig:ppo-obj 按优势 $A_t$ 的符号以及策略 ratio $\rho(\theta)$ 落入的三个区域逐一列出。
每个区域的结果由两个事实决定：优势的符号决定我们希望 action 更可能还是更不可能出现，而 $\min$ 操作会选择未裁剪项 $\rho(\theta) A_t$ 或其裁剪版本。

Clipping 只会在两个区域中把梯度归零：此时策略已经把采样 action 朝期望方向移动，并越过了 trust region 边界：

- **正优势且 $\rho(\theta) > 1+\varepsilon$**：在 $\pi_\theta$ 下，该 action 已经比在 $\pi_{\theta_{\text{old}}}$ 下明显更可能。目标饱和在 $(1+\varepsilon)A_t$，梯度为零，不再更新，避免过度强化一个已经更强表达的 action。
- **负优势且 $\rho(\theta) < 1-\varepsilon$**：在 $\pi_\theta$ 下，该 action 已经明显更不可能。目标饱和在 $(1-\varepsilon)A_t$，梯度同样为零，不再更新，避免过度压制一个已经被降低的 action。

其他地方未裁剪项 $\rho(\theta) A_t$ 生效，PPO 执行标准策略梯度步：当 $A_t > 0$ 时提高 action 概率，当 $A_t < 0$ 时降低 action 概率。我们可以从 @fig:ppo-obj 中读出每个区域对更新后策略 $\pi_\theta$ 的要求：

- 正优势下倾斜的未裁剪区域（绿色）会**增加**采样 action 的概率；
- 负优势下倾斜的未裁剪区域（红色）会**降低**它；
- 平坦的裁剪区域（灰色）会让策略**不变**，因为其梯度为零。

把同样的区域逐项写出如下：

#### 正优势（$A_t > 0$）

这意味着根据价值函数，采取的 action 是有益的；我们希望未来更可能采取这个 action。现在看策略 ratio $\rho(\theta)$ 的不同情形：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：action 在新策略下比旧策略下更不可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新，提高 action 的可能性

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：action 在新旧策略下几乎同样可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新，提高 action 的可能性

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：action 在新策略下比旧策略下更可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标**：$(1 + \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 + \varepsilon) A_t = 0$
    - **发生什么**：不更新：该 action 在新策略下已经更可能

总结来说，当优势为正（$A_t>0$）时，我们希望提高 action 的概率。因此：

- 只有当 $\pi_{\text{new}}(a) \leq (1+\varepsilon) \pi_{\text{old}}(a)$ 时，我们才执行梯度步。直观上，我们希望提高 action 的概率，因为优势为正，但又不希望提高到它已经显著更可能的程度。
- 关键是，当 $\pi_{\text{new}}(a) > (1+\varepsilon) \pi_{\text{old}}(a)$ 时，我们不执行任何更新，裁剪目标的梯度为 $0$。直观上，该 action 已经在新策略中表达得更强，因此不希望过度强化它。

#### 负优势（$A_t < 0$）

这意味着根据价值函数，采取的 action 是有害的；我们希望未来降低采取该 action 的可能性。现在看策略 ratio $\rho(\theta)$ 的不同情形：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：action 在新策略下比旧策略下更不可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标**：$(1 - \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 - \varepsilon) A_t = 0$
    - **发生什么**：不更新：该 action 在新策略下已经更不可能

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：action 在新旧策略下几乎同样可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新，降低 action 的可能性

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：action 在新策略下比旧策略下更可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新，降低 action 的可能性

总结来说，当优势为负（$A_t < 0$）时，我们希望降低 action 的概率。因此：

- 只有当 $\pi_{\text{new}}(a) \geq (1-\varepsilon) \pi_{\text{old}}(a)$ 时，我们才执行梯度步。直观上，我们希望降低 action 的概率，因为优势为负，并且按优势大小成比例地降低。
- 关键是，当 $\pi_{\text{new}}(a) < (1-\varepsilon) \pi_{\text{old}}(a)$ 时，我们不执行任何更新，裁剪目标的梯度为 $0$。直观上，该 action 在新策略下已经更不可能，因此不希望过度压制它。

必须记住，在 trust region 内，PPO 与标准策略梯度形式大体相同。


### 价值函数与 PPO

PPO 中的价值函数是模型的一个额外副本，用于预测每个 token 的价值。
在传统 RL 中，一个 token（或状态）的价值是在预测从该时刻开始的未来回报，通常带有折扣。
PPO 中的这个价值被用作学习得到的 baseline，可以看作 REINFORCE 所用简单 Monte Carlo 版本的演化（REINFORCE 不需要学习价值网络）。
这说明 PPO 在多个方面都是 REINFORCE 和原始策略梯度的演化，包括优化形式、baseline 等。
实践中，在 PPO 以及其他用于语言模型的算法中，这相当于预测扣除 KL 惩罚之后每个 token 的回报（如前所述，per-token 损失传统上会包含来自奖励的 KL）。

学习价值函数有几种不同方法（或目标）。
Generalized Advantage Estimation (GAE) 被视为现代系统中最先进且最规范的实现，但它通过在多个步长上计算价值预测误差带来了更多复杂性；本章后面会有 GAE 小节。
价值函数也可以使用更新策略所用 rollout 中的 Monte Carlo 估计来学习。
PPO 有两个损失：一个用于学习价值函数，另一个使用该价值函数来更新策略。

![价值函数训练使用 on-policy rollout 来计算目标。模型在每个 token 处预测 $V_t$，并通过 MSE 对目标回报 $\hat{V}_t$ 进行训练。随后，优势 $A_t = \hat{V}_t - V_t$ 对策略梯度更新加权。](images/value_fn_training.png){#fig:value_fn_training data-dark-src="images/value_fn_training-dark.png"}

下面展示一个简单的价值网络损失实现示例。

```python
# Basic PPO critic targets & loss (no GAE)
#
# B: Batch Size
# L: Completion Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards; EOS row includes outcome
#   done_mask: (B, L) 1.0 at terminal token (EOS or truncation if penalized), else 0.0
#   completion_mask: (B, L) 1.0 on response tokens to supervise (ignore the prompt)
#   values: (B, L) current critic predictions V_theta(s_t)
#       because a value network is a running update
#   old_values: (B, L) critic predictions at rollout time V_{theta_old}(s_t)
#   gamma: discount factor, float (often 1.0 for LM RLHF)
#   epsilon_v: float value clip range (e.g., 0.2), similar to PPO Loss Update itself, optional
#
# Returns:
#   value_loss: scalar; advantages: (B, L) detached (for policy loss)

B, L = rewards.shape

# 1) Monte Carlo returns per token (reset at terminals)
# Apply discounting, if enabled
returns = torch.zeros_like(rewards)
running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(L)):
    running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
    returns[:, t] = running

targets = returns  # y_t = G_t (post-KL)

# 2) PPO-style value clipping (optional)
v_pred = values
v_old  = old_values
v_clip = torch.clamp(v_pred, v_old - epsilon_v, v_old + epsilon_v)

vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss_tok  = torch.max(vf_unclipped, vf_clipped)

# 3) Mask to response tokens and aggregate
denom = completion_mask.sum(dim=1).clamp_min(1)
value_loss = ((vf_loss_tok * completion_mask).sum(dim=1) / denom).mean()

# 4) Advantages for policy loss (no GAE): A_t = G_t - V(s_t)
advantages = (targets - v_pred).detach()

# The value loss is applied later, often with the PG loss, e.g.
# total_loss = policy_loss + vf_coef * value_loss
```

### 组相对策略优化（GRPO）

Group Relative Policy Optimization (GRPO) 最早在 DeepSeekMath [@shao2024deepseekmath] 中提出，并用于其他 DeepSeek 工作，例如 DeepSeek-V3 [@deepseekai2025deepseekv3technicalreport] 和 DeepSeek-R1 [@guo2025deepseek]。
GRPO 可以被视为一种受 PPO 启发的算法，其 surrogate loss 非常相似，但它避免了用原始策略语言模型的另一个副本（或另一个初始化 checkpoint）来学习价值函数。
这带来两个被提出的好处：

1. 避免从 LM backbone 学习价值函数的挑战；目前研究尚未确立最佳实践。
2. 节省内存，因为不需要在内存中保留额外一组模型权重（从需要当前策略、reference policy 和价值函数，减少到只需要前两个副本）。

GRPO 通过简化价值估计来做到这一点：它给 episode 中每个 token 分配相同价值，也就是在一个 prompt 的补全中，每个 token 得到相同价值，而不是像标准价值函数那样使用折扣奖励；这一价值来自优势或 baseline 估计。
该估计通过从同一个初始状态 / prompt ($s$) 收集多个补全 ($a_i$) 和奖励 ($r_i$) 来完成，也就是一个 Monte Carlo 估计。

正式地说，GRPO 目标与上面的 PPO 目标非常相似。
对于 GRPO，目标函数（或损失）会在给定 prompt $s$ 的一组补全 $\{a_1, a_2, ..., a_G\}$ 上累积。
这里给出 GRPO 目标：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right).$$ {#eq:GRPO}

注意，相对于 PPO，GRPO 的标准实现会把 KL 距离包含在损失中。
如上所述，我们可以把它展开为 per-token 计算：

$$\begin{aligned}
J(\theta) = \frac{1}{G}\sum_{i=1}^G  \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \Bigg( &\min\!\left(\frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}A_{i,t},\; \text{clip} \left( \frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}, 1-\varepsilon, 1+\varepsilon \right) A_{i,t} \right) \\
&- \beta \mathcal{D}_{\text{KL}}\!\left(\pi_\theta(\cdot|s_{i})\|\pi_{\text{ref}}(\cdot|s_{i})\right) \Bigg)
\end{aligned}$$ {#eq:GRPO_token}


对补全索引 $i$ 的优势计算为：

$$A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.$$ {#eq:GRPO_ADV}

![GRPO 架构。优势会相对于组均值和标准差进行归一化。KL 惩罚直接施加在损失中，而不是用于塑造奖励。](images/grpo_tikz.png){#fig:grpo-arch data-dark-src="images/grpo_tikz-dark.png"}

直观上，GRPO 更新是在一个 batch 内比较同一个问题的多个答案。
模型会学习更像被标记为正确的答案，同时更不像其他答案。
这是一种非常简单的优势计算方式；优势衡量的是在给定状态下，某个具体 action 比平均水平好多少。
相对于 PPO、REINFORCE，以及广义上使用奖励模型评分（相对于输出奖励）的 RLHF，GRPO 通常会对每个 prompt 运行更多样本，因为优势完全取决于一个补全相对于同一 prompt 下同伴补全的相对价值。
这里，当前策略会对给定 prompt 生成多个回复，组内 GRPO 优势估计由此获得有价值的上下文。
PPO 和原始策略梯度算法原本被设计为准确估计每个补全的奖励（事实上，在某些情况下，更多补全对改善价值估计帮助不大）。
GRPO 及其变体特别适合现代语言模型工具，因为对同一个 prompt 生成多个补全非常自然；相比之下，在机器人任务等固定环境状态中采样多个 action 则不那么自然。

GRPO 的优势计算在偏差上存在权衡。
按标准差归一化会奖励 batch 中答案正确性变化较小的问题。
对于几乎全部正确或全部错误的问题，标准差会更低，优势会更高。
Liu et al. 2025 [@liu2025understanding] 基于这一偏差提出移除标准差项，但代价是会降低“多数错误、少数正确”的问题权重；这种问题也可以被视为模型的宝贵学习信号。
这些高方差 prompt 可能正是最难的情形，只有少数采样补全能找到正确答案，并提供强训练信号。

@eq:GRPO_ADV 是在 outcome supervision（标准奖励模型或单个可验证奖励）下使用 GRPO 的实现；如果使用 process supervision，则需要不同实现。
在后一种情况下，GRPO 会把后续推理步骤的归一化奖励求和，作为优势。

最后，GRPO 的优势估计也可以在没有 PPO clipping 的情况下，用于更原始的策略梯度版本（例如 REINFORCE），但这不是规范形式。
作为这些算法相互交织的一个例子，我们可以看到，GRPO 的一个变体 Dr. GRPO (GRPO Done Right) [@liu2025understanding] 中的优势估计，与 RLOO 估计（用其他样本的平均奖励作为 baseline）只差一个常数缩放因子；由于实现细节通常会归一化优势，这个常数因子一般并不重要。
Dr. GRPO 从 @eq:GRPO_ADV 中移除了标准差归一化项；注意，这也会把优势*放大*，等价于在答案分数存在方差的样本上提高 GRPO 学习率。
这解决了对低奖励方差问题的偏置，也就是几乎所有答案都对或都错的问题；但如果从只有一个样本答对的问题中学习很重要，这也可能带来代价。
组大小为 $G$ 时，补全 $i$ 的 Dr. GRPO 优势定义为：

$$ \tilde{A}_i = r_i - \text{mean}({r_1, r_2, \cdots, r_G}) = r_i - \frac{1}{G}\sum_{j=1}^G r_j $$ {#eq:DrGRPO_ADV}

这里，使用相同记号，我们回顾 RLOO 优势估计：

$$ A_i^\text{RLOO} = r_i - \frac{1}{G-1}\sum_{j=1, i\neq j}^G r_j $$ {#eq:RLOO_ADV_AGAIN}

因此，如果把 Dr. GRPO 优势定义乘以 $\frac{G}{G-1}$，就能看到一个按比例缩放后的等价关系：

$$
\begin{aligned}
\frac{G}{G-1} \tilde{A}_i &= \frac{G}{G-1} \left( r_i - \frac{1}{G}\sum_{j=1}^G r_j \right) \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1}^G r_j \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j - \frac{1}{G-1} r_i \\
&= r_i \left( \frac{G}{G-1} - \frac{1}{G-1} \right) - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= A_i^{\text{RLOO}}
\end{aligned}
$$ {#eq:RLOO_GRPO_EQUIV}

### 组序列策略优化（GSPO）

当我们对由旧策略收集的数据 batch 执行多个梯度步时，需要使用 importance sampling 来校正数据收集策略与当前正在优化的策略之间的分布不匹配。
标准 importance sampling 恒等式允许我们用来自另一个分布的样本，估计某个分布下的期望：

$$
\mathbb{E}_{p}[f(x)] = \mathbb{E}_{q}\left[f(x) \frac{p(x)}{q(x)}\right],
$$ {#eq:IS_identity}

其中 $p$ 是目标分布，$q$ 是采样分布，$\frac{p(x)}{q(x)}$ 是重要性权重。
在策略梯度方法中，$p = \pi_\theta$ 是我们想优化的当前策略，$q = \pi_{\theta_{\text{old}}}$ 是生成训练数据的策略。
这使我们能够对在 $\pi_{\theta_{\text{old}}}$ 下收集的样本重新加权，以估计 $\pi_\theta$ 的梯度，从而允许每个 rollout batch 执行多个梯度步。

这种分布不匹配出现在两个常见场景中：1）对单个 batch 做多个梯度步，每次更新后 $\pi_\theta$ 都会从 $\pi_{\theta_{\text{old}}}$ 漂移；2）在异步训练系统中，推理后端（例如 vLLM）和训练后端（例如 FSDP）可能因为同步延迟而拥有不同模型权重（见本章后面的异步性一节；这种情形尤其随着对可验证奖励 RL 的关注而出现，但也用于 RLHF 设定）。

PPO 和 GRPO 在 token 层面应用 importance sampling，并通过裁剪 *surrogate objective* 来稳定学习。
然而，这种方法有一个微妙的失败模式：当某个 token 的 importance ratio 移出裁剪范围 $[1-\varepsilon, 1+\varepsilon]$ 时，该 token 会收到零梯度。
对于罕见但重要的 token，例如模型初始赋予低概率的关键推理步骤，这种“token dropping”会阻止模型更可靠地学会生成它们。

Group Sequence Policy Optimization (GSPO) [@zheng2025gspo] 扩展了 GRPO：它在序列层面而不是 token 层面计算 importance ratio。
该算法及其同类 CISPO 的实践动机是，per-token importance sampling ratio 经常在数值上不稳定；CISPO 会修改策略梯度算法中 importance sampling 的计算方式，我们稍后会讨论。
概念动机则是，当奖励在序列层面分配时（多数 RLHF 和 RLVR 设定都是如此），importance sampling 校正也应该匹配这个粒度。

对于长序列和/或大型稀疏模型（例如现代 mixture-of-experts (MoE) 模型），token 级 ratio 可能表现得很不稳定：单个具有大 ratio 的 token 可能主导策略更新，或者一个回复中的许多 token 会被独立裁剪，从而把单个回复的学习信号切碎。
GSPO 通过为每个回复计算单个重要性权重来解决这一点。

回忆完整回复的概率会按自回归方式分解：

$$
\pi_\theta(a \mid s) = \prod_{t=1}^{|a|} \pi_\theta(a_t \mid s, a_{<t}).
$$ {#eq:response_factorization}

注意，为简单起见，我们经常把条件策略 $\pi_\theta(a_t \mid s, a_{<t})$ 缩写为 $\pi_\theta(a_t \mid s)$，它隐含包含补全中的先前 action（token）。
GSPO 使用几何平均定义长度归一化的序列级 importance ratio，以避免长序列带来的数值问题：

$$
\rho_i(\theta) = \left( \frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_{\text{old}}}(a_i \mid s)} \right)^{\frac{1}{|a_i|}} = \exp\left( \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})} \right).
$$ {#eq:GSPO_ratio}

GSPO 目标与 GRPO 类似，但使用这个序列级 ratio：

$$
J_{\text{GSPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i(\theta) A_i,\, \text{clip}(\rho_i(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right) \right].
$$ {#eq:GSPO_objective}

因为 ratio 被长度归一化，裁剪范围 $\varepsilon$ 作用在 per-token 平均尺度上，使不同长度回复之间的有效约束可比。
实现中，序列级权重 $\rho_i$ 会均匀应用到回复 $a_i$ 的所有 token 上，这简化了梯度计算，同时保持了序列级 IS 校正。

优势计算仍与 GRPO 相同（@eq:GRPO_ADV），使用组相对的均值和标准差归一化；这一点也可以像其他 GRPO 衍生研究那样进行修改。
GSPO 可以概括为“带序列级 importance ratio 的 GRPO”：IS 校正粒度与奖励粒度相匹配。

### 截断重要性采样策略优化（CISPO）

Clipped Importance Sampling Policy Optimization (CISPO) [@minimax2025minimax_m1] 采取了不同方法：CISPO 裁剪重要性权重本身，而不是裁剪 surrogate objective，同时保留所有 token 的梯度。
该目标对裁剪后的重要性权重使用 stop-gradient，回到 REINFORCE 风格形式，而不是 PPO 风格的双侧裁剪：

$$
J_{\text{CISPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^K \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{\sum_{i=1}^K |a_i|} \sum_{i=1}^K \sum_{t=1}^{|a_i|} \text{sg}\left( \hat{\rho}_{i,t}(\theta) \right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s, a_{i,<t}) \right],
$$ {#eq:CISPO_objective}

其中 $\text{sg}(\cdot)$ 表示 stop-gradient（权重会被使用，但不会对其求导），裁剪后的 importance ratio 为：

$$
\hat{\rho}_{i,t}(\theta) = \text{clip}\left( \rho_{i,t}(\theta),\, 1 - \varepsilon_{\text{low}},\, 1 + \varepsilon_{\text{high}} \right), \quad \rho_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})}.
$$ {#eq:CISPO_ratio}

它与 PPO/GRPO 的关键差异微妙但重要：裁剪权重（而不是目标）意味着每个 token 仍然都会收到与其优势成比例的梯度信号；权重只是限制该信号被 importance ratio 放大或压低的程度。
这是一个偏差-方差权衡：裁剪权重会引入偏差，但能控制方差，而且关键是避免完全丢弃 token 梯度。

CISPO 和 GSPO 都由推动大规模 MoE 模型 RL 应用边界的组织开发；这类模型以数值问题著称。
论文强调，per-token importance sampling ratio 不稳定，并会给梯度增加大量方差，从而削弱学习。
这使这些算法对大规模模型尤其可能有影响，但在较小的学术实验中研究较少，收益也不那么明确。

CISPO 还允许非对称裁剪边界（$\varepsilon_{\text{low}} \neq \varepsilon_{\text{high}}$），类似本章后面讨论的 DAPO “clip-higher” 修改；它可以允许模型想要上调权重的 token 有更大更新，从而鼓励探索。
相关工作包括 Tapered Off-Policy REINFORCE (TOPR) [@leroux2025topr]，它也像 CISPO 一样直接裁剪 IS 权重，而不是像 PPO/GRPO 那样在目标内部裁剪；但它在序列层面运行（像 GSPO），并基于奖励符号使用非对称裁剪：对正奖励不应用 IS 校正，对负奖励则把 ratio 裁剪到 $[0, 1]$，从而实现稳定的 off-policy 学习。


### 算法比较

本章中的每种算法都共享同一个核心梯度形式（@eq:policy_gradient_intuition），但在优势估计和优化控制方式上不同：

- **REINFORCE**：最简单的策略梯度实现，使用奖励的 Monte Carlo 估计和基于状态的 baseline 来降低方差。
- **RLOO**：每个 prompt 使用多个样本的 REINFORCE；每个样本的 baseline 是其他样本的平均奖励（leave-one-out），以降低梯度方差。
- **PPO**：加入学习得到的价值函数和裁剪后的策略 ratio，以获得更准确、更稳定的梯度更新。
- **GRPO**：PPO 的简化变体；它对每个 prompt 分组多个补全，并在组内归一化奖励以计算优势，从而不再需要价值函数。
- **CISPO**：REINFORCE 风格算法；它裁剪 importance-sampling 权重（而不是像 PPO/GRPO 那样裁剪目标），并用 stop-gradient 保持稳定，使每个 token 都能收到梯度信号。
- **GSPO**：类似 GRPO，但按补全长度归一化策略 ratio，避免长度偏置。
- **DPO**：不是 RL 算法，而是通过完全绕过单独奖励模型，直接从偏好对中优化，来求解同一个偏好优化问题的方法（见第 8 章）。

上面所有策略梯度算法在推导上都是 on-policy 的，不过实践中大多会以轻微 off-policy 的方式应用。第 8 章中的 DPO 和其他直接对齐算法默认就是 off-policy。
所有这些算法都可以与学习得到的奖励模型或可验证奖励搭配使用。
只有 PPO 需要学习得到的价值函数。
REINFORCE 和 RLOO 没有 importance-sampling ratio；其余算法各自引入一个 ratio，以支持每个 rollout batch 执行多个梯度步，并在粒度和裁剪策略上有所不同，如下表所示。

| Method | IS Granularity | Clipping Style | Advantage |
| :----- | :-----------: | :------------------: | :-------------------: |
| **REINFORCE** | None | None | Monte Carlo baseline |
| **RLOO** | None | None | Leave-one-out |
| **PPO** | Token | Objective (bilateral) | Learned value fn |
| **GRPO** | Token | Objective (bilateral) | Group-relative |
| **GSPO** | Sequence | Objective (bilateral) | Group-relative |
| **CISPO** | Token | Weights (stop-grad) | Group-relative |
Table: 策略梯度算法比较。 {#tbl:pg_compare}

每种方法的核心损失 $\mathcal{L}(\theta)$ 为：

$$\begin{aligned}
\textbf{REINFORCE:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\log \pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big) \\[6pt]
\textbf{RLOO:}\quad & -\frac{1}{K}\sum_{i=1}^{K}\sum_t \log \pi_\theta(a_{i,t}\mid s_{i,t})\left(R_i-\frac{1}{K-1}\sum_{j\neq i}R_j\right) \\[6pt]
\textbf{CISPO:}\quad & -\sum_{i,t} \mathrm{sg}(\hat{\rho}_{i,t})\, A_{i,t} \log \pi_\theta(a_{i,t}\mid s_{i,t}) \\
& \quad \hat{\rho}_{i,t} = \mathrm{clip}(\rho_{i,t},\, 1-\varepsilon,\, 1+\varepsilon) \\[6pt]
\textbf{PPO:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\min\!\big(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\, A_t\big) \\
& \quad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} \\[6pt]
\textbf{GRPO:}\quad & -\frac{1}{G}\sum_{i=1}^{G}\min\!\big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)\, A_i\big) \\
& \quad \rho_i = \frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_{\text{old}}}(a_i\mid s)},\quad A_i = \frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})} \\[6pt]
\textbf{GSPO:}\quad & -\frac{1}{G}\sum_{i=1}^{G}\min\!\big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)\, A_i\big) \\
& \quad \rho_i = \left(\frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_{\text{old}}}(a_i\mid s)}\right)^{1/|a_i|} \\[6pt]
\textbf{DPO:}\quad & -\mathbb{E}_{(x,y^{w},y^{l})}\!\left[\log \sigma\!\big(\beta[\Delta\log \pi_\theta(x)-\Delta\log \pi_{\mathrm{ref}}(x)]\big)\right]
\end{aligned}$$


## 实现

与最初发展出这些算法的 Deep RL 文献相比，为优化语言模型或其他大型 AI 模型实现 RL，需要处理许多细小实现细节。
本节重点介绍区分流行算法实现的一些关键因素。

这种训练还包含许多其他小细节。
例如，在用语言模型做 RLHF 时，一个关键步骤是生成文本，然后由奖励模型对其打分。
正常情况下，模型应该生成表示生成结束的 end-of-sequence (EOS) token；但常见做法是对生成长度设置硬上限，以高效利用基础设施。
RLHF 的一种失败模式是，模型的回答经常被截断，导致奖励模型评分落到分布外，并产生不可预测的分数。
解决方案是*只*在 `eos_token` 上运行奖励模型评分；否则，如果模型生成过长，就给它分配惩罚。

流行的开源 RLHF 工具在各算法实现细节上存在很大差异（见 [@ivison2024unpacking] 的表 10）。
这里没有覆盖的一些决策包括：

- **价值网络初始化**：PPO 和其他类似算法使用的内部学习价值网络，可以从同架构的不同模型开始，也可以从随机选择的权重开始。这会对性能产生很大影响。InstructGPT [@ouyang2022training] 中确立的标准做法（Tülu 3 在其 RLVR 工作中也复用 [@lambert2024t]）是用 RLHF 期间使用的奖励模型来初始化价值网络。其他做法包括使用进入 RLHF 训练之前的 checkpoint（通常是 SFT 模型），并附加一个随机初始化的 value head；或者使用完全重新初始化的语言模型（较少见，因为 RLHF 收敛会更久，但可行）。
- **奖励归一化、奖励 whitening 和/或优势 whitening**：归一化把来自 RM（或环境）的所有值约束在 0 到 1 之间，有助于学习稳定。[Whitening](https://en.wikipedia.org/wiki/Whitening_transformation) 更进一步，会把奖励或优势估计变换为零均值、单位方差，从而对稳定性提供更强提升。
- **不同 KL 估计器**：对于复杂语言模型，精确计算模型之间的 KL divergence 可能很复杂，因此会使用多种近似来替代精确计算 [@schulman2016klapprox]。
- **KL controllers**：PPO 及相关算法的原始实现有动态 controller，会针对特定 KL 并根据近期测量值改变惩罚。多数现代 RLHF 实现使用静态 KL 惩罚，但这一点也会变化。

关于 RLHF 实现细节，更多内容见 [@huang2024n]。
关于算法的更多信息，见 [@weng2018PG]。

### 策略梯度基础

下面是一个使用优势估计梯度、为 PPO 和 GRPO 等高级算法做准备的简单策略梯度实现：
```python
pg_loss = -advantages * ratio
```
这里的 ratio 是新策略模型概率相对于 reference model 的（per-token）概率 ratio，通常由 log-probability 差值计算得到。

为了理解这个方程，最好理解一个更新 batch 中可能出现的不同情况。
请记住，我们希望随着模型在任务上变好，损失会*下降*。

情况 1：优势为正，因此 action 比该状态的期望价值更好。我们希望强化它。在这种情况下，由于负号存在，模型会让它更可能出现。为此，模型会增加 logratio。正的 logratio，或者 token log probability 之和，意味着模型更可能生成这些 token。

情况 2：优势为负，因此 action 比该状态的期望价值更差。它的逻辑非常类似。这里，如果新模型更可能生成该补全，损失会为正，因此模型会尝试让策略参数使这个补全更不可能出现。

情况 3：优势为零，因此不需要更新。损失为零，不改变策略模型。

### 损失聚合的权衡

在用语言模型实现任意策略梯度算法时，问题是：如何把 per-token 损失聚合为最终标量损失？
给定样本 $i$ 在 token $t$ 上的 per-token 损失 $\ell_{i,t}$，补全长度 $|a_i|$，以及 batch size $B$，主要有三种策略：

**策略 1：按序列归一化**（标准 GRPO；也用于一些 PPO 实现）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_per_sequence}

无论长度如何，每个序列对 batch loss 的贡献相同。代码中：

```python
# Strategy 1: Per-sequence normalization
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

**策略 2：按 token 归一化**（DAPO [@yu2025dapo]）

$$L = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|a_i|} \ell_{i,t}}{\sum_{i=1}^{B} |a_i|}$$ {#eq:loss_per_token}

每个 token 的贡献相同；更长序列对梯度有按比例更大的影响。代码中：

```python
# Strategy 2: Per-token normalization
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

**策略 3：固定长度归一化**（Dr. GRPO [@liu2025understanding]）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{L_{\max}} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_fixed_length}

用最大序列长度 $L_{\max}$ 归一化，使不同序列的 per-token 尺度相等，同时由于长序列包含更多 active token，仍然让它贡献更多总梯度。代码中：

```python
# Strategy 3: Fixed-length normalization
fixed_len_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
            L_max).mean()
```

其中 $L_{\max}$ 通常是整个训练过程中的全局常数，指定最大生成 token 数。

注意，上面代码中的 `completion_mask` 是由 1 和 0 组成的矩阵，其中 prompt token 被 mask 掉（0），因为我们不希望模型从预测 prompt token 中学习。

#### 为什么这很重要？

直观上，按序列归一化（策略 1）似乎最好，因为我们关心的是*结果*，而不是单个 token。
然而，这会引入基于序列长度的微妙偏差，可能导致模型过度思考，或者降低对自然需要更多 token 的策略的权重，具体取决于偏差方向。
考虑两个长度不同、具有 per-token 损失的序列：

```python
seq_1_losses = [1, 1, 1, 1, 10]  # 5 tokens, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, mean = 1.9
```

使用**策略 1**（按序列）时：batch loss 为 $(2.8 + 1.9)/2 = 2.35$；关键是，短序列中的每个 token 会比长序列中的 token 获得更大梯度。

使用**策略 2**（按 token）时：batch loss 为 $(14 + 19)/15 = 2.2$；所有 token 获得相同梯度幅度。

使用**策略 3**（固定长度，$L_{\max}=10$）时：短序列贡献 $1.4$，长序列贡献 $1.9$，在平衡 per-token 梯度的同时仍按序列长度加权。

如需查看这些策略如何影响梯度的更完整示例，请看下面脚本。

```python
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[int] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    """Compute sum of tensor with masked values. Use a constant to normalize."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)


advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    # generation 1: 4 tokens
    [1, 1, 1, 1, 0, 0, 0,],
    # generation 2: 7 tokens
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_sum tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
# masked_mean_token_level tensor(1., grad_fn=<DivBackward0>)

masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.0909, 0.0909, 0.0909, 0.0909, 0.0000, 0.0000, 0.0000],
# [0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909]])
```

输出显示，使用策略 1（`masked_mean`）时，短序列的 per-token 梯度（0.25）大于长序列（0.14）。
策略 2 和策略 3 会让不同序列中的 per-token 梯度相等。
注意，如果使用 gradient accumulation，这些结果可能发生很大变化；此时会在执行 backward step 之前，把多个 minibatch 的梯度求和，短序列与长序列之间的平衡可能反转。

实践中，最佳策略取决于具体训练设定。
在 RLHF 中，通常会偏好数值稳定性最好或损失方差最小的方法。

#### 相关问题：MDP 与 Bandit 视角

损失聚合的选择连接到一个更深层的问题：我们如何建模这个 RL 问题。
**MDP（token 级）**视角把每个 token $a_t$ 视为一个 action，把状态 $s_t$ 视为当前前缀。
实践中，当我们用学习得到的价值函数 $V(s_t)$（例如 GAE [@schulman2015high]）计算 token 级优势，并逐 token 应用 KL 惩罚时，采用的就是这个框架。
带学习价值网络的 PPO 是规范例子 [@schulman2017proximal]。

相对地，**bandit（序列级）**视角把整个补全视为一个 action，并配一个标量奖励 $R$。
在代码中，这意味着计算序列级优势 $A_{\text{seq}}$，并把它广播到所有 token。
RLOO 和 GRPO 风格优势经常用于这种 bandit-style 设定 [@kool2019buy] [@ahmadian2024back] [@shao2024deepseekmath]。
DPO 和 A-LoL 等直接对齐方法也定义序列级目标，尽管它们不是策略梯度估计器 [@baheti2023leftover]。

注意，许多 GRPO 实现会使用 bandit-style 优势，并在损失中加入单独的 per-token KL 项；同时，许多 PPO/RLOO 实现会在计算优势之前把 KL 折入奖励。两种约定在实践中都存在。

下面是突出这两种方法差异的示例比较：

```python
# === Bandit-style (sequence-level) ===
# One scalar reward per sequence; advantage broadcast to all tokens
reward = torch.tensor([3.0, 1.0])       # (B,) e.g., reward model scores
baseline = reward.mean()                 # simple baseline (RLOO uses leave-one-out)
advantage_seq = reward - baseline        # (B,)
advantages = advantage_seq[:, None].expand(-1, seq_len)  # (B, L)
# tensor([[ 1.,  1.,  1.,  1.],    <- same advantage for all tokens
#         [-1., -1., -1., -1.]])

# === MDP-style (token-level) ===
# Per-token rewards + learned V(s_t); each token gets its own advantage
# (could also use per-token KL shaping, format rewards, or other token-level signals)
advantages = gae(per_token_rewards, values, done_mask, gamma=1.0, lam=0.95)
# tensor([[ 0.2,  0.5,  0.8,  1.5],    <- varies by position
#         [-0.3, -0.5, -0.8, -1.4]])
```

这个框架差异也解释了为什么几乎所有 RLHF 实现都会把折扣因子 $\gamma$ 设为 1.0。
在标准 RL 中，折扣（$\gamma < 1$）至关重要：它在多步 episode 中平衡短期奖励与长期奖励的优化，这对智能体随时间学习有效行为很关键。
但在 RLHF 设定中，即便使用 token 级 MDP 视角，优化的归纳偏置也是整体补全质量；奖励信号为整个回复打分，而不是为单个 token 打分。
任意降低较早 token 的贡献没有原则性理由。
随着 agentic RL 设定成熟，即模型采取真实多步 action，例如工具调用、代码执行和网页浏览，折扣可能会重新变得相关，因为这些设置包含真正不同的序贯决策，其长期后果也不同。

### 异步 RL 系统

策略梯度算法的默认实现是所谓 **on-policy** 执行：智能体（语言模型）采取的 action（生成）会在更新模型之前被打分。
策略梯度的理论推导依赖所有 action 都严格 on-policy，即模型始终已经根据最新 trial/roll-out 的结果保持最新。
实践中，维持严格 on-policy 执行会显著拖慢训练 [@noukhovitch2024asynchronous]，而且完美同步无论如何在技术上都不可能。
因此，最近所有语言模型实证结果都倾向于略微偏离理论证明。
实践中真正发生的是，为实际有效的方法设计相应算法和系统。

![根据 Noukhovitch et al. 2024，对同步或异步 RL 训练中的生成-更新阶段进行比较。](images/async_v_synch_rl.png){#fig:async}

常见解决方案是，在独立 GPU 节点上持续运行推理和训练，并使用专门软件高效运行二者，如 @fig:async 底部所示。
在流行的开源语言模型 RL 工具中，常见做法是使用 Ray 这样的分布式进程管理库，在策略梯度学习循环与推理循环之间传递信息，并使用高效推理引擎，例如 vLLM。
在这些设置中，负责执行 RL step 的 GPU 被称为 “learners”，负责从语言模型采样的 GPU 被称为 “actors”。
让训练更异步时面临的主要挑战，是保持训练稳定并维持学习信号。

![一个分布式 RL 系统示例，其中两个队列被管理起来，将数据传递给 learner 和 actor GPU；二者可以通过 Ray 这样的分布式计算库同步。Olmo Team 2025, license CC-BY.](images/distributed-rl.png){#fig:async_system}

这些系统的设计和实现基于这样一个假设：近似 on-policy 的数据足以稳定学习。
这里，生成阶段与更新阶段可以很容易同步，以避免训练系统任一部分出现空闲计算；这对应于在 @fig:async_system 中把模型权重从 learners 传递给 actors。
对于推理模型，需要每个答案生成 10K 到 100K+ token 的问题具有极长推理特征，使 rollout 生成成为更强的瓶颈。
在更同步的 RL 基础设施上训练推理模型时，一个常见问题是 batch 中某个 prompt 的答案生成耗时显著更长（可能因为更多 token 或更多工具调用），导致大部分已分配计算在它完成之前处于空闲。
解决这种长度不匹配的第二种方法称为 sequence-level packing：用巧妙 masking 把较短样本堆叠在一个 batch 中，使模型能够继续 rollout，并在 batch 内更好地分散长度归一化。
分布式 RL 基础设施的完整复杂性超出了本书范围，因为它会造成许多其他微妙问题，拖慢训练或导致不稳定。

随着这些推理模型出现，人们进一步希望让训练和推理循环完全 off-policy：策略梯度更新的训练 batch 由多个生成答案实例中最近完成的 rollout 填充 [@wu2025llamarl] [@fu2025areal]。
完全异步训练还会让 RL 训练运行更容易扩展到多个数据中心，因为可以增加 learner 节点（执行策略梯度步）与 actor（尝试解决问题）之间权重同步的时间间隔 [@primeintellectteam2025intellect2reasoningmodeltrained]。

相关方法正在探索完全 off-policy 的策略梯度算法 [@leroux2025topr]。

### 截断重要性采样

Truncated importance sampling (TIS) 是现代异步语言模型 RL 框架中用于稳定训练的关键工具。
Importance sampling 是一种校正，它会对从一个分布抽取的样本重新加权，以估计另一个分布下的期望（见 @eq:IS_identity）。
Truncated importance sampling [@ionides2008truncated] 会用 $\min(\rho, C)$ 对这些权重设置上限，其中 $C$ 是常数；它用少量偏差换取策略梯度中的有界方差。

这是应用于策略梯度的 importance-sampling 校正，但不同于 PPO 和 CISPO 中的双侧裁剪（把 ratio 约束在 1 附近），TIS 使用单侧上限：ratio 可以自由低于 1，但会在 $C$ 处截断，以防极端上调权重。
在 PPO、GRPO、CISPO（及相关算法）中，ratio $\rho_t^{\text{policy}} = \pi_\theta(a_t \mid s) / \pi_{\theta_{\text{old}}}(a_t \mid s)$ 校正的是一个 RL batch 内多个梯度步造成的策略漂移。
当我们转向真实 RL 框架，围绕上一小节异步性这个想法展开时，可能出现更大的数值差异来源（也需要 importance sampling 的数值校正）。
即使 sampler 与 learner 共享相同参数 $\theta$，它们的有效 token 分布也可能不同，因为推理引擎（例如 vLLM）和训练框架（例如 FSDP）使用不同 kernel、精度和并行策略 [@yao2025offpolicy]。
因此，区分在两个系统上评估的同一策略 $\pi_\theta^{\text{sampler}}$ 和 $\pi_\theta^{\text{learner}}$ 很有用，并可定义对应 ratio 及其截断形式：

$$
\rho_t^{\text{learner}} = \frac{\pi_\theta^{\text{learner}}(a_t \mid s, a_{<t})}{\pi_\theta^{\text{sampler}}(a_t \mid s, a_{<t})}, \qquad \tilde{\rho}_t^{\text{learner}} = \min(\rho_t^{\text{learner}},\; C).
$$ {#eq:tis_backend}

这两种校正是互补的，但它们出现在策略梯度实现中的原因不同：一种补偿 RL batch 训练中的策略漂移，另一种补偿由实现引起的分歧；二者可以同时应用。
它们如何组合取决于算法：

#### 带 TIS 的 REINFORCE（单个梯度步）

这里没有策略漂移（$\pi_\theta = \pi_{\theta_\text{old}}$），因此唯一不匹配来自 learner 与 sampler。
这里 $\pi_{\theta_\text{old}} = \pi_\text{gen}$，TIS 直接校正 learner-sampler 差距：

$$
\nabla_\theta J \approx \mathbb{E}_{a \sim \pi_\theta^{\text{sampler}}} \left[ \tilde{\rho}_t^{\text{learner}} \cdot A_t \cdot \nabla_\theta \log \pi_\theta^{\text{learner}}(a_t \mid s, a_{<t}) \right].
$$ {#eq:reinforce_tis}

#### 带 TIS 的 PPO/GRPO（多个梯度步）

现在两个 ratio 都生效。
在谨慎实现中，策略 ratio 中的 “old logprobs” 会在 learner 上重新计算（GSPO 论文讨论了这一点），因此策略 ratio $\rho_t^{\text{policy}} = \pi_\theta^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{learner}}$ 捕捉的是纯策略漂移，而 $\tilde{\rho}_t^{\text{learner}} = \min(\pi_{\theta_\text{old}}^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{sampler}},\; C)$ 则在生成 checkpoint 处单独校正后端不匹配：

$$
J_{\text{PPO+TIS}}(\theta) = \mathbb{E}\left[ \min\!\left( \rho_t^{\text{policy}}\, A_t,\; \text{clip}\!\left(\rho_t^{\text{policy}}, 1-\varepsilon, 1+\varepsilon\right) A_t \right) \cdot \tilde{\rho}_t^{\text{learner}} \right].
$$ {#eq:ppo_tis}

这里 $\pi_{\theta_\text{old}} \neq \pi_\text{gen}$：old logprobs 来自 learner，而不是 sampler。
如果某个框架跳过这次重新计算，直接使用 sampler logprobs 作为 $\pi_{\theta_\text{old}}$，那么策略 ratio 已经捕捉了后端不匹配，不再需要单独的 TIS 校正；但 clipping 会作用在一个噪声更大的 ratio 上，即使在任何梯度步之前，它也会从偏离 1.0 的位置开始。
这是 Yao et al. [-@yao2025offpolicy] 所说的“your framework secretly brings you off-policy RL”观察。

实践中，LLM RL 系统把 TIS 作为 per-token 校正权重应用到策略梯度损失上：

```python
# Shape: (B*G, L)
C = 2.0  # TIS cap

logratio = learner_logprobs - sampler_logprobs
logratio = logratio.clamp(-10.0, 10.0)              # numerical safety
tis_weight = torch.exp(logratio).clamp(max=C)        # one-sided truncation

# Use as a fixed correction weight on the per-token PG loss
per_token_pg_loss = per_token_pg_loss * tis_weight.detach()
```

$[-10, 10]$ clamp 只是在取指数之前用于数值稳定；真正的 truncated-importance-sampling 步骤，是在 $C$ 处的单侧上限。
实践中，围绕这些 logprobs 的 bookkeeping，也就是存储生成时的 sampler logprobs、在旧 checkpoint 上重新计算 learner logprobs、并在梯度步期间跟踪当前 logprobs，是分布式 RL 框架脚手架中的重要组成部分。
不同于 GSPO，这个校正是 token 级的，因为它处理的是 token 级数值不匹配，而不是序列级奖励粒度。
针对 learner-sampler ratio 的 TIS 已经被主要开源 RL 框架采用（VeRL、TRL、OpenRLHF、SkyRL、OAT，以及使用 $C = 2$ 的 Open Instruct），并且对长推理轨迹（第 7 章）越来越重要；在这类轨迹中，微小 per-token 差异会在数千个生成 token 上复合。


### 示例：PPO

PPO 有非常非常多实现。
下面展示核心*损失*计算。
稳定性能的关键还包括*价值*计算，其中存在多个选择（包括*价值模型*损失的多个选择）。

注意，这里的 reference policy（或 old logprobs）来自生成被采样时，而不一定是 reference policy。
Reference policy 只用于 KL 距离约束/惩罚。

```python
# B: Batch Size, L: Sequence Length, G: Num of Generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: (B*G, L)

# Get value predictions
values = value_net(completions)  # Shape: (B*G, L)

# Compute returns via backward pass (gamma typically 1.0 for LM RLHF)
# Mask rewards to avoid padding tokens (which may have KL penalties) leaking into returns
returns = torch.zeros_like(rewards)
running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(rewards.shape[1])):
    # Zero out padding: only accumulate rewards/returns for valid completion tokens
    running = (rewards[:, t] + self.gamma * running) * completion_mask[:, t]
    returns[:, t] = running

# Compute advantages: A_t = G_t - V(s_t)
advantages = returns - values.detach()  # Shape: (B*G, L)
# Note: We detach the value network here to not update the parameters of
# the value function when computing the policy-gradient loss

# Normalize advantages (optional but stable)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# Value function loss: predict returns
vf_loss = 0.5 * ((returns - values) ** 2)  # Shape: (B*G, L)

# Combine policy and value losses
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute metrics for logging
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()

    # Compute value loss for logging
    value_loss = vf_loss.mean()
```

理解 PPO 的核心，是理解策略梯度损失如何更新。
重点看这三行：
```python
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)
```
`pg_losses1` 是普通的优势加权策略梯度损失。`pg_losses2` 使用同一公式，但把概率 ratio clamp 到 $[1-\varepsilon, 1+\varepsilon]$ 范围内，从而限制策略在单次更新中的变化幅度。

关键洞见是对两个损失取 `torch.max`。因为我们最小化的是一个*负*损失（回忆优势前面的负号），取最大值会选择更悲观的梯度，也就是产生更小策略更新的那个。当优势为正（好 action）时，clipping 会防止策略过于激进地提高该 action 的概率。当优势为负（坏 action）时，clipping 会防止在相反方向上过度校正。

通过 clamp log-probability ratio，PPO 限制策略相对于生成训练数据的版本漂移多远，从而在不需要显式 trust region 计算的情况下稳定学习。

上面的代码还展示了 PPO 在学习策略的同时学习价值函数，这增加了实现复杂性，但裁剪目标才是核心机制。

#### 每个样本只做一个梯度步时的 PPO/GRPO 简化（无裁剪）

如果超参数“每个样本的梯度步数”等于 1，PPO（以及 GRPO）实现可以处理得优雅得多。
该超参数的常见取值是 2-4 或更高。
在主要 PPO 或 GRPO 方程中，见 @eq:PPO_EQN，“reference” policy 是先前参数，也就是用于生成补全或 action 的参数。
因此，如果只执行一个梯度步，$\pi_\theta = \pi_{\theta_{\text{old}}}$，更新规则会化简为如下形式（记号 $[]_\nabla$ 表示 stop gradient）：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\frac{\pi_\theta(a_i|s)}{\left[\pi_{\theta}(a_i|s)\right]_\nabla}A_i - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right). $$ {#eq:ppo_1step}

这会导向这样的 PPO 或 GRPO 实现：第二个策略梯度和 clipping 逻辑可以省略，使优化器更接近标准策略梯度。


### 示例：GRPO

DeepSeekMath 论文描述了一些不同于 PPO 的 GRPO 实现细节 [@shao2024deepseekmath]，尤其是在与 Deep RL 中标准 PPO 应用而不是语言模型应用比较时。
例如，RLHF 优化中的 KL 惩罚（回忆一下，使用可验证奖励但没有奖励模型来训练推理模型时也会用到 KL 惩罚）会直接应用在损失更新中，而不是应用到奖励函数上。
标准 RLHF 的 KL 惩罚写作 $r=r_\theta - \beta \mathcal{D}_{\text{KL}}$，而 GRPO 实现大致是：

$$ L = L_{\text{policy gradient}} + \beta * \mathcal{D}_{\text{KL}} $$ {#eq:grpo_loss_kl}

不过，它有多种实现方式。
传统上，KL 距离会针对 prompt $s$ 的补全中的每个 token 计算。
对于推理训练，会从一个 prompt 中采样多个补全，并且一个 batch 中有多个 prompt，
因此 KL 距离形状会是 [B, L, N]，其中 B 是 batch size，L 是序列长度，N 是每个 prompt 的补全数量。

把这些结合起来，使用第一种损失累积方式，伪代码可以写为：

```python
# B: Batch Size, L: Sequence Length, G: Number of Generations
# Compute group-wise rewards # Shape: (B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)


# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# Shape: (B*G,)

# Compute advantages
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# important to GRPO -- PPO applies this in reward traditionally
# Combine with KL penalty
per_token_loss = pg_loss_max + self.beta * per_token_kl  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute core metric for logging (KL, reward, etc. also logged)
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()
```

关于如何解释这段代码，更多细节见上面的 PPO 一节。与 PPO 示例的核心差异是：

- **优势计算**：GRPO 相对于组（同一 prompt 的多次生成的均值和标准差）归一化奖励，而不是使用学习得到的价值函数作为 baseline。
- **没有价值网络**：GRPO 完全移除价值模型，消除了 `vf_loss` 和相关复杂性。
- **KL 惩罚位置**：GRPO 直接把 KL 惩罚加入损失，而不是从奖励中减去它（这是标准实现，但 KL 如何应用还有更多版本）。

#### RLOO 与 GRPO

RLOO 的优势更新与 GRPO 非常接近，这凸显了当把它与 PPO 风格 clipping 和 KL 惩罚细节分开看时，二者在概念上的相似性。
具体而言，对 RLOO 来说，优势是相对于一个与 GRPO baseline 极其相似的 baseline 计算的，也就是该补全奖励相对于同一问题下其他补全奖励。
简洁地说，RLOO 优势估计如下（从 [TRL](https://github.com/huggingface/trl/blob/bfe20756082488350091352d1cdc19c172e42cd8/trl/trainer/rloo_trainer.py#L433) 的实现展开）：

```python
# rloo_k --> number of completions per prompt
# rlhf_reward --> Initially a flat tensor of total rewards for all completions. Length B = N x k
rlhf_reward = rlhf_reward.reshape(rloo_k, -1) #
# Now, Shape: (k, N), each column j contains the k rewards for prompt j.

baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# baseline --> Leave-one-out baseline rewards. Shape: (k, N)
#  baseline[i, j] is the avg reward of samples i' != i for prompt j.

advantages = rlhf_reward - baseline
# advantages --> Same Shape: (k, N)

advantages = advantages.flatten() # Same shape as original tensor
```

RLOO 的其余实现细节遵循实现策略梯度时的其他权衡。

## 辅助主题

要掌握策略梯度算法的应用，还有无数其他注意事项。
这里我们讨论成功部署策略梯度 RL 算法时一些长尾复杂性。

### 广义优势估计（GAE）

Generalized Advantage Estimation (GAE) 是一种用于策略梯度算法的优势计算替代方法 [@schulman2015high]，能更好地平衡偏差-方差权衡。
传统单步优势估计可能引入过多偏差，而使用完整轨迹又可能遭受高方差。
GAE 会计算多步优势估计的指数加权平均，其中超参数 $\lambda$ 控制偏差-方差权衡：从单步 TD（$\lambda=0$）到完整轨迹回报（$\lambda=1$）；$\lambda=0.95$ 是 LLM 微调中的常见默认值。

优势估计可以有多种形式，但我们可以定义一个 $n$-step 优势估计器（类似本章开头的 TD residual）：

$$
\hat{A}_t^{(n)} = \begin{cases}
r_t + \gamma V(s_{t+1}) - V(s_t), & n = 1 \\
r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t), & n = 2 \\
\vdots \\
r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - V(s_t), & n = \infty
\end{cases}
$$ {#eq:K_STEP_ADV}

较短的 $n$ 方差更低但偏差更高，因为我们把更多学习权重归因到每条轨迹上；它可能过拟合。
GAE 尝试把这一形式推广为加权多步平均，而不是某个特定 $n$。
首先，我们必须定义预测价值的 temporal difference (TD) residual。

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$ {#eq:TD_RESIDUAL}

为了利用它，我们引入另一个变量 $\lambda$ 作为 GAE 混合参数。这会折入我们希望估计的未来优势的指数衰减中：

$$
\begin{array}{l}
\hat{A}_t^{GAE(\gamma,\lambda)} = (1-\lambda)(\hat{A}_t^{(1)} + \lambda\hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + \cdots) \\
= (1-\lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \cdots) \\
= (1-\lambda)(\delta_t^V(1 + \lambda + \lambda^2 + \cdots) + \gamma\delta_{t+1}^V(\lambda + \lambda^2 + \cdots) + \cdots) \\
= (1-\lambda)\left(\delta_t^V\frac{1}{1-\lambda} + \gamma\delta_{t+1}^V\frac{\lambda}{1-\lambda} + \cdots\right) \\
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V
\end{array}
$$ {#eq:GAE_DFN}

直观上，这可以用一种优雅方式对 Advantage 的多步估计求平均。
下面展示一个示例实现：

```python
# GAE (token-level) for LM RLHF
#
# B: Batch Size
# L: Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards
#   values:  (B, L) current V_theta(s_t)
#   done_mask: (B, L) 1.0 at terminal token (EOS or penalized trunc), else 0.0
#   gamma: float (often 1.0),
#   lam (short for lambda): float in [0,1]
#   (Padding beyond terminal should have rewards=0, values=0)
B, L = rewards.shape
advantages = torch.zeros_like(rewards)
next_v = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards[:, t] + gamma * not_done * next_v - values[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values[:, t]

targets = advantages + values      # y_t for value regression
advantages = advantages.detach()   # for policy loss
```

这个反向循环会累积 temporal-difference (TD) errors（$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$），它们衡量实际结果相对于价值函数预测好多少或差多少，并带有指数衰减 $(\gamma\lambda)^l$。
在 terminal token 处，`not_done=0` 会阻止从未来状态 bootstrap，并重置 GAE accumulator，因此每个 episode 的优势会独立计算（因为循环是反向运行的，terminal token 会在 episode 边界干净地停止指数加权累积；这让实现对 packing 友好，能正确处理拼接到一起的多个序列）。
最终的 `targets` 作为单独价值函数在这个 GAE 循环之外学习的回归目标，而 detached 的 `advantages` 会给策略梯度加权；detach 是为了让策略更新不会通过价值网络反向传播。
在语言模型 RLHF 中，$\gamma=1.0$ 很常见，因为 episode 是较短 token 序列，并且通常偏好不折扣的 credit assignment（而且常常一个补全中的所有 token 都共享同一目标）。

*延伸阅读见 [@seita2017gae]。*

### 双重正则化

本章已经看到两类正则化。一类内置于 PPO 这样的算法中，以步长约束形式出现；另一类是相对于优化起点的基于 KL divergence 的距离惩罚。

Deep Reinforcement Learning 中许多流行的策略梯度算法，包括 PPO 及其前身，最初都是因为需要控制智能体的学习过程而出现。
在 RLHF 中，如第 15 章正则化以及第 3 章训练概览所广泛讨论的那样，相对于正在微调的原始策略的距离惩罚，已经构成了内置正则项。
从这个角度看，像 PPO（有内部步长正则化）和 REINFORCE（更简单，并且在某些超参数下 PPO 会化约到它）这样的算法之间，大部分差异对于微调语言模型而言，远不如从零开始训练智能体时那么重要。

在 PPO 中，处理更新步长上限的目标函数称为 [surrogate objective](https://huggingface.co/blog/deep-rl-ppo#introducing-the-clipped-surrogate-objective)。
为了监控 PPO 正则化在 RLHF 中对更新的影响程度，可以查看许多流行实现中的 clip fraction 变量，也就是 batch 中概率 ratio 落到裁剪区间之外的样本百分比。
这是一个有用 proxy，用来衡量 PPO 正则器可能激活的频率；但并非每个这样的样本都有零梯度：只有当裁剪分支被选中时 surrogate 才会变平，例如 ratio 高于 $1+\varepsilon$ 的正优势样本，或 ratio 低于 $1-\varepsilon$ 的负优势样本。

在语言模型实践中，PPO 和 GRPO 等算法通常每个 batch 只运行一个梯度步，这意味着 PPO 原生正则化从不会被应用（因为只有策略在一个 batch 内显著变化时，clipping 才可能发生），而 KL 距离惩罚占主导。
不过，这并不普遍。例如，DAPO 每个 batch 使用 16 个梯度步 [@yu2025dapo]；Tülu 3 对 8B 和 70B 模型每个 batch 使用 4 次 PPO update iteration，但为了维持训练稳定性，对 405B 模型降为 1 次 [@lambert2024t]。

### 延伸阅读

随着 RLHF 稳固地处于现代后训练中心，其他策略梯度 RL 算法以及一般 RL 算法也被提出，用于改进训练过程；但它们尚未在支配最佳实践方面发挥核心作用。
延伸阅读示例包括：

- **Pairwise Proximal Policy Optimization (P3O; Wu et al., 2023)** [@wu2023pairwise] 直接在 PPO 风格策略更新中使用成对数据，而不学习中间奖励模型。
- **Soft Adaptive Policy Optimization (SAPO)** [@gao2025sapo] 用平滑、温度控制的 gating 替代硬 PPO/GRPO 风格 clipping，目标是得到一个连续 trust region，在保留近似 on-policy 学习信号的同时降低 off-policy token 权重。
- Off-policy 策略梯度算法可能支持更进一步的异步训练，例如 **Contrastive Policy Gradient (CoPG)** [@flet2024contrastive]（直接对齐算法 IPO 和原始策略梯度的一种泛化），Cohere 曾在其 Command A 模型中使用它 [@cohere2025command]。
- 其他 REINFORCE 算法实现也已经为语言模型而设计，例如 **ReMax** [@li2023remax]，它实现了一种专门适配奖励模型推理不确定性来源的 baseline 归一化。
- 一些基础模型，例如 Apple Intelligence Foundation Models [@gunter2024apple] 或 Kimi k1.5 reasoning model [@team2025kimi]，使用了 **Mirror Descent Policy Optimization (MDPO)** [@tomar2020mirror] 的变体。这里的基础研究仍在发展 [@zhang2025improving]，但 Mirror Descent 是一种优化方法，而不是直接的策略梯度算法。重要的是，它会以非常类似的方式替换到现有 RL 基础设施中。
- **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)** 提出 4 项修改，使 GRPO 更适合推理语言模型；这类模型需要长 trace，并需要提高新的、未充分使用 token 的概率 [@yu2025dapo]。这些修改是：1，使用两个不同 clip 超参数 $\varepsilon_\text{low}$ 和 $\varepsilon_\text{high}$，让 logratio 正侧的 clipping 能为更好探索采取更大步长；2，dynamic sampling，移除 batch 中所有样本奖励全为 0 或全为 1 的样本（没有学习信号）；3，使用上面 Implementation: GRPO 中讨论的 per-token loss；4，对过长样本施加软惩罚，以避免从被截断答案中学习。
- **Value-based Augmented Proximal Policy Optimization (VAPO)** [@yuan2025vapo] 结合了 DAPO 的优化（包括 clip-higher、token-level policy-gradient 和不同长度归一化）与 Value-Calibrated PPO [@yuan2025s] 的洞见，通过预训练价值函数和 length-adaptive GAE 展示了相对于 GRPO 的 value-based 方法潜力。

## 建议实验

配套实现位于 `code/policy_gradients/`，用于小规模、可观察的 RL 运行。
默认配置会在来自 `reasoning-gym` 的 `spell_backward` 程序化任务上训练 `Qwen/Qwen3-1.7B`；这是一个很好的入门练习，因为失败和部分进展都很容易检查。

1. **用 GRPO 运行单词反转任务。**

   ```bash
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   跟踪 `avg_correctness`、`avg_format` 和 `avg_binary`。
   第一个有用问题是每个 prompt group 是否包含对比：如果所有采样补全都对或都错，那么组相对更新几乎没有学习信号。

2. **比较组相对估计器与单样本估计器。**
   运行匹配的起始配置：

   ```bash
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   比较 correctness 信号提升速度以及损失噪声大小。
   RLOO 和 GRPO 应该能让 within-prompt baseline 的作用比单看方程更具体。

3. **扫描对比度旋钮。**
   复制 `policy_gradients/configs/grpo.yaml`，并改变 `num_rollouts`、`temperature`、`data.size` 和 `format_weight`。
   较小的 `num_rollouts` 会降低组内对比；极低 temperature 可能让样本塌缩；极高 temperature 可能生成过多格式错误答案。
   这是观察为什么 RLVR recipe 往往在动优化器之前，先花大量精力处理采样设置的最简单方式。

4. **从玩具奖励转向数学。**
   对于 GSM8K 风格实验，在添加新的在线 RL 环境之前，先从 `code/reward_models/train_orm.py` 和 `code/rejection_sampling/` 示例开始。
   一个好的贡献是提供一个小型 `reasoning-gym` 或 GSM8K 策略梯度配置，它能在 sub-1B Qwen 模型上运行，并报告相同的组对比诊断指标。
