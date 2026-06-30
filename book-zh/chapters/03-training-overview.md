<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "A Tiny History of RLHF"
prev-url: "02-related-works"
page-title: 训练概览
search-title: "第 3 章：训练概览"
meta-description: "现代后训练配方的高层地图，包括指令微调、RLHF、RLVR 和直接对齐方法。"
next-chapter: "Instruction Fine-Tuning"
next-url: "04-instruction-tuning"
lectures:
  - video: "https://www.youtube.com/watch?v=MMDNaeIFVy8&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "Lecture 0: Prerequisites"
  - video: "https://www.youtube.com/watch?v=o6l6tJQgUg4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "Lecture 1: Overview (Chapters 1–3)"
---

# 训练概览

本章先对 RLHF 训练做一个粗略概览，后续章节再进入具体细节。
RLHF 虽然优化的是一个简单的损失函数，但它需要按顺序训练多个不同的 AI 模型，然后在复杂的在线优化中把它们连接起来。

这里，我们先介绍 RLHF 的核心目标：使用基于距离的正则项来优化人类偏好的代理奖励（同时展示它与经典强化学习问题的关系）。
随后，我们展示一些使用 RLHF 创建领先模型的标准配方，以说明 RLHF 如何嵌入后训练方法的其余部分。
这些示例配方会成为本书后续章节的参照；在那里，我们会描述做 RLHF 时可选择的不同优化方案，并回指不同关键模型在训练中如何使用不同步骤。

## 问题形式化

基于人类反馈的强化学习（RLHF）的优化建立在标准强化学习设定之上。
在强化学习中，给定环境状态 $s_t$，智能体从策略 $\pi(a_t\mid s_t)$ 中采样动作 $a_t$，以最大化奖励 $r(s_t,a_t)$ [@sutton2018reinforcement]。
策略是一个函数，它把每个状态映射到动作上的概率分布。
早期那些演化出现代 RLHF 文献的策略，属于所谓的深度强化学习，即使用神经网络来学习上述函数。
传统上，环境会根据转移（动力学）$p(s_{t+1}\mid s_t, a_t)$ 演化，并带有初始状态分布 $\rho_0(s_0)$。
策略和动力学共同诱导出一个轨迹分布。
一条轨迹的总体概率，是初始状态概率、策略做出的每次动作选择，以及环境产生的每次状态转移的乘积：

$$p_{\pi}(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t).$$ {#eq:rl_dynam}

在长度为 $T$ 的有限 episode 中，强化学习智能体的目标是求解下面的优化问题，其中 $\gamma$ 是从 0 到 1 的折扣因子，用来权衡近期奖励与未来奖励的吸引力：

$$\max_\pi \; \mathbb{E}_{\tau \sim p_{\pi}} \left[ \sum_{t=0}^{T-1} \gamma^t r(s_t, a_t) \right].$$ {#eq:rl_opt}

给定策略的期望回报通常记为 $J(\pi)$，最优值写作 $J^* = \max_\pi J(\pi)$。

对于持续性任务，人们通常令 $T\to\infty$，并依靠折扣（$\gamma<1$）使目标函数保持良好定义。
第 6 章会讨论优化这一表达式的多种方法。

![标准强化学习循环](images/rl.png){#fig:rl width=320px .center data-dark-src="images/rl-dark.png"}

@fig:rl 展示了标准强化学习循环的一个常见图示（可与 @fig:rlhf 中的 RLHF 循环比较）。

### 一个简单例子：恒温器 {#example-rl-thermostat}

为了对强化学习做什么建立基本直觉，可以考虑一个试图把房间维持在目标温度 70$^\circ$F 的恒温器。
在强化学习中，智能体起初对任务一无所知，必须通过试错发现好的策略。
恒温器例子包含以下组成部分（每个部分如何映射到 @eq:rl_dynam 中的轨迹分布，见 @fig:thermostat-equation）：

- **状态（$s_t$）**：当前房间温度，例如 65$^\circ$F。
- **动作（$a_t$）**：打开或关闭加热器。
- **奖励（$r$）**：当温度处于目标温度 2$^\circ$ 范围内时为 +1，否则为 0。
- **策略（$\pi$）**：给定当前温度，决定是否打开加热器的规则。下面是恒温器可能学到的一种策略；根据环境的具体转移动力学，它未必是最优的：

$$\pi(a_t = \text{on} \mid s_t) = \begin{cases} 1 & \text{if } s_t < 70^{\circ}\text{F} \\ 0 & \text{otherwise} \end{cases}$$ {#eq:thermostat_policy}

- **转移**：加热器打开时房间升温，关闭时房间降温。智能体通过自己的动作影响这些动力学，但底层物理规律，即房间升温或降温的速度，并不受它控制。

![轨迹分布（@eq:rl_dynam）中的每一项映射到恒温器强化学习示例。](images/thermostat_equation.png){#fig:thermostat-equation .center data-dark-src="images/thermostat_equation-dark.png"}

一开始，恒温器的策略本质上是随机的：它不考虑当前温度，随意打开或关闭加热器，房间温度也会大幅波动。
经过许多轮试错 episode 之后，智能体发现房间冷时打开加热器、房间暖时关闭加热器会带来更多奖励，并逐渐收敛到一个合理策略。
这就是强化学习的核心循环：观察状态、选择动作、获得奖励，并随时间更新策略以获得更多奖励。

### 经典强化学习例子：CartPole

对于一个包含连续动力学的更丰富例子，可以考虑经典的 *CartPole*（倒立摆）控制任务；它出现在许多强化学习教材、课程，甚至研究论文中。
恒温器只有一个状态变量和一个二元动作，而 CartPole 包含四个连续状态变量和基于物理的转移，因此是强化学习算法的标准 benchmark。

![CartPole 环境，展示状态变量（$x$, $\dot{x}$, $\theta$, $\dot{\theta}$）和动作（$\pm F$）。](images/cartpole.png){#fig:cartpole width=400px .center data-dark-src="images/cartpole-dark.png"}

- **状态（$s_t$）**：小车的位置/速度，以及杆的角度/角速度：

  $$s_t = (x_t,\,\dot{x}_t,\,\theta_t,\,\dot{\theta}_t).$$ {#eq:cartpole_state}

- **动作（$a_t$）**：向小车施加向左或向右的水平力，例如 $a_t \in \{-F, +F\}$。

- **奖励（$r$）**：一种简单奖励是，只要杆保持平衡且小车留在轨道上（例如 $|x_t| \le 2.4$ 且 $|\theta_t| \le 12^\circ$），每一步都有 $r_t = 1$；一旦任一边界被违反，episode 终止。

- **动力学 / 转移（$p(s_{t+1}\mid s_t,a_t)$）**：在许多环境中，动力学是确定性的（因此 $p$ 是一个点质量），并且可以通过步长为 $\Delta t$ 的 Euler 积分写作 $s_{t+1} = f(s_t,a_t)$。一个标准的简化 CartPole 更新会使用小车质量 $m_c$、杆质量 $m_p$、杆半长 $l$ 和重力 $g$ 等常数（$\alpha$ 是按质量归一化、具有加速度单位的中间量）：

  $$\alpha = \frac{a_t + m_p l\,\dot{\theta}_t^2\sin\theta_t}{m_c + m_p}$$ {#eq:cartpole_temp}

  $$\ddot{\theta}_t = \frac{g\sin\theta_t - \cos\theta_t\,\alpha}{l\left(\tfrac{4}{3} - \frac{m_p\cos^2\theta_t}{m_c + m_p}\right)}$$ {#eq:cartpole_angular_accel}

  $$\ddot{x}_t = \alpha - \frac{m_p l\,\ddot{\theta}_t\cos\theta_t}{m_c + m_p}$$ {#eq:cartpole_linear_accel}

  $$x_{t+1}=x_t+\Delta t\,\dot{x}_t,\quad \dot{x}_{t+1}=\dot{x}_t+\Delta t\,\ddot{x}_t,$$ {#eq:cartpole_pos_update}
  $$\theta_{t+1}=\theta_t+\Delta t\,\dot{\theta}_t,\quad \dot{\theta}_{t+1}=\dot{\theta}_t+\Delta t\,\ddot{\theta}_t.$$ {#eq:cartpole_angle_update}

这是上面一般设定的一个具体实例：策略选择 $a_t$，转移函数推进状态，奖励在整个 episode 中累积。

### 调整标准强化学习设定

用于 RLHF 的强化学习形式化通常被视为一个开放性较低的问题：为了适配语言模型，强化学习中的几个关键部分被设定为特定定义。
从标准强化学习设定到语言模型 RLHF 设定，有多项核心变化：
表 @tbl:rl-vs-rlhf 总结了标准强化学习与语言模型 RLHF 设定之间的这些差异。

1. **从奖励函数切换到奖励模型。** 在 RLHF 中，使用一个学习到的人类偏好模型 $r_\theta(s_t, a_t)$（或任何其他分类模型）来替代环境奖励函数。这显著提高了方法的灵活性，也让设计者对最终结果拥有更多控制，但代价是实现复杂度上升。在标准强化学习中，奖励被视为环境中一个静态组成部分，设计学习智能体的人不能改变或操纵它。
2. **不存在状态转移。** 在 RLHF 中，该领域的初始状态是从训练数据集中采样的 prompt，而"动作"是对该 prompt 的补全（在标准 RLHF 设定中，prompt 是固定的，模型的补全不会定义下一个 prompt）。一个 prompt 和一个补全的组合构成一个完整 episode 或 rollout；而在经典强化学习问题中，这会是许多重复的状态-动作、状态-动作链条。
3. **回复级奖励且无折扣。** RLHF 中的奖励归因针对的是由多个生成 token 组成的整段动作序列，而不是细粒度地逐步归因（这种单步结构在强化学习文献中有时被称为 bandit 问题）。为了帮助 RLHF 的强化学习算法把每个 token 都看作同一动作的一部分，实现中通常使用 $\gamma = 1$ 的折扣因子（即不折扣），这不同于标准强化学习中用 $\gamma < 1$ 在许多连续决策中平衡短期与长期奖励的做法。

::: {.table-wrap}
| 方面 | 标准强化学习 | RLHF（语言模型） |
|---|---|---|
| 策略 | 从零学习（随机初始化） | 从预训练语言模型微调而来 |
| 奖励信号 | 环境奖励函数 $r(s_t,a_t)$ | 学习到的奖励 / 偏好模型 $r_\theta(x,y)$（prompt $x$，补全 $y$） |
| 状态转移 | 有：动力学 $p(s_{t+1}\mid s_t,a_t)$ | 通常没有：prompt $x$ 从数据集中采样；补全不会定义下一个 prompt |
| 动作 | 单个环境动作 $a_t$ | 从 $\pi_\theta(\cdot\mid x)$ 采样的补全 $y$（一串 token） |
| 奖励粒度 | 通常逐步 / 细粒度 | 通常是针对完整补全的回复级奖励（bandit 风格），通常无折扣（$\gamma = 1$） |
| 时域长度 | 多步 episode（$T>1$） | 通常是单步（$T=1$），但多轮对话可建模为更长时域 |
Table: 标准强化学习与语言模型 RLHF 的关键差异。 {#tbl:rl-vs-rlhf}
:::

鉴于这个问题的单轮性质，优化目标可以在不包含时间时域和折扣因子的情况下重写（并显式加入奖励模型）：
$$\max_\pi \; \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t) \right].$$ {#eq:rl_opt_int}

从许多方面看，其结果是：虽然 RLHF 深受强化学习优化器和问题形式化的启发，但实际实现与传统强化学习非常不同。

![标准 RLHF 循环](images/rlhf.png){#fig:rlhf data-dark-src="images/rlhf-dark.png"}

### 微调与正则化

在传统强化学习问题中，智能体必须从随机初始化的策略开始学习；而在 RLHF 中，我们从一个能力很强的预训练基础模型出发。
这种面向 RLHF 的强先验带来一个需求：防止优化过程偏离初始策略太远。
为了在微调范式下取得成功，RLHF 技术会采用多种正则化来控制优化过程。
目标是允许奖励最大化继续发生，同时避免模型陷入第 14 章讨论的过度优化。
对优化函数最常见的修改，是在当前 RLHF 策略与优化起点之间的距离上加入 KL 散度惩罚。训练模型时设置的超参数 $\beta$ 控制这一约束的强度 -- 较大的 $\beta$ 让模型更接近起点，较小的 $\beta$ 则给优化器更多追逐奖励的自由：

$$\max_\pi \; \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t)\right] - \beta  \mathcal{D}_{\text{KL}}(\pi(\cdot|s_t) \| \pi_{\text{ref}}(\cdot|s_t)).$$ {#eq:rlhf_opt_eq}

在这一形式化中，许多关于 RLHF 训练的研究都在理解如何使用某个"KL 预算"，也就是用相对于初始模型的距离来衡量的预算。
更多细节见第 15 章关于正则化的讨论。


### 优化工具

本书会详细介绍求解这一优化问题的许多流行技术。
后训练中常用的工具包括：

- **奖励建模**（第 5 章）：训练一个模型来捕捉已收集偏好数据中的信号，随后它可以输出一个标量奖励，表示未来文本的质量。
- **指令微调**（第 4 章）：RLHF 的前置步骤，通过模仿预先选择的示例，教会模型今天大多数语言模型交互所使用的问答格式。
- **拒绝采样**（第 9 章）：最基础的 RLHF 技术，其中用于指令微调的候选补全会被一个模仿人类偏好的奖励模型过滤。
- **策略梯度**（第 6 章）：RLHF 奠基性示例中使用的强化学习算法，用于根据奖励模型的信号更新语言模型参数。
- **直接对齐算法**（第 8 章）：直接从成对偏好数据优化策略的算法，而不是先学习一个中间奖励模型再在之后优化它。

现代经过 RLHF 训练的模型总是先使用指令微调，然后再混合使用其他优化选项。

### 后训练语言模型中强化学习的微妙优势

在接下来的章节中，我们会介绍许多用于后训练的优化工具。
其中很多方法，例如拒绝采样（第 9 章）和 DPO 这样的直接对齐算法（第 8 章），都比让强化学习真正跑起来简单得多。
不过，尽管替代方法更简单，基于强化学习的方法仍持续胜出。
有些趋势很明显，例如使用可验证奖励的强化学习（RLVR）带来的推理时扩展；但事实证明，强化学习本身也非常适合作为语言模型的优化工具。
相对于指令微调或类似 DPO 的算法，实现强化学习需要大得多的基础设施投入；但冒着说得过于口语化的风险，它提供的梯度更新"通常会让模型受益很大"。
这很难量化，但会以几种反复出现的形式体现出来：

- 强化学习阶段可以"修补"模型的粗糙边缘，让模型更容易对话或更稳健（例如训练它在使用 vLLM 等推理工具时具备数值稳定性）。文献中尚不清楚其确切原因，但如今强化学习越来越常见这一事实反映了它的真实性。
- 强化学习可以被精细地使用：模型能很好地学到 prompt 分布所在的位置，而强化学习往往不会"压扁"模型的一般能力。一个很好的例子是 Tülu 3 只在数学 prompt 上用强化学习训练，同时保持了它在广泛任务套件上的能力 [@lambert2024t]。

总体而言，语言模型上的强化学习损失稳健、可扩展、有效且灵活，由此打开了大规模新实验领域。
最初把我们带上这条路径的方法正是 RLHF 工作。

## 标准训练配方

随着时间推移，人们逐渐把一些模型识别为 RLHF 或更一般后训练的标准配方。
这些配方反映了当时的数据实践和模型能力。
随着配方逐渐变旧，训练出具有相同特征的模型会变得更容易，所需数据也更少。
总体趋势是，后训练包含更多优化步骤、更多训练算法，以及覆盖更多样训练数据集和评估的流程。

### InstructGPT

ChatGPT 刚出现前后，业界广泛接受的语言模型后训练"标准"方法有三个主要步骤，其中 RLHF 是核心部分 [@lambert2022illustrating] [@ouyang2022training] [@bai2022training]。
在一个"基础"语言模型（在大规模网页文本上训练的下一 token 预测模型）之上执行的这三个步骤，总结在下面的 @fig:rlhf-basic-repeat:

1. **在约 10K 个示例上进行指令微调**：这会教会模型遵循问答格式，并主要通过人类编写的数据教会它一些基本技能。
2. **在约 100K 个成对 prompt 上训练奖励模型**（论文使用了 33K 个 prompt）：该模型从指令微调后的 checkpoint 训练而来，并捕捉最终训练中希望建模的多样价值。奖励模型是 RLHF 的优化目标。
3. **在另一组约 100K 个 prompt 上用 RLHF 训练指令微调模型**（论文实际使用 31K 个 prompt，且没有说明这些 prompt 是否从其他阶段复用）：模型使用一组很可能独立的 prompt 针对奖励模型进行优化，在获得评分之前先生成回复。

RLHF 完成后，模型就可以部署给用户使用。这个配方是现代 RLHF 的基础，但后来的配方已经显著演化，包含更多阶段和更多数据。

![早期三阶段 RLHF 流程示意：SFT、奖励模型，然后进行优化。](images/rlhf-basic.png){#fig:rlhf-basic-repeat}

### Tülu 3

现代版本的后训练包含多得多的模型版本和训练阶段（即远多于 Llama 2 记录的 5 个 RLHF 步骤 [@touvron2023llama]）。
@fig:rlhf-complex 展示了一个例子，其中模型在收敛前经历了大量训练迭代。

![现代多轮后训练示意。](images/rlhf-complex.png){#fig:rlhf-complex}

这个时代及之后训练出的最复杂模型并没有公开其训练过程的全部细节。
到 2026 年，ChatGPT 或 Claude 等领先模型包含许多轮迭代训练。
这甚至可能包括训练专门化模型，然后把权重合并在一起，以得到一个能够完成许多子任务的最终模型 [@li2022branch]（例如 Cohere 的 Command A [@cohere2025command]）。

![Tülu 3 配方概览，包含目标技能和多步训练流程。Lambert et al. 2024, License CC-BY。](images/tulu3.png){#fig:tulu-3}

Tülu 3 是一个完全开放的多阶段后训练例子，其中 RLHF 发挥了重要作用。
Tülu 3 配方由三个阶段组成：

1. **在约 1M 个示例上进行指令微调**：这个主要由合成数据构成的数据集来自 GPT-4o 和 Llama 3.1 405B 等前沿模型的混合，用于教会模型通用指令跟随，并作为数学和编程等能力的基础。
2. **在约 1M 对偏好样本上收集 on-policy 偏好数据**：这一阶段会显著提升模型的聊天性（例如 Arena，原 ChatBotArena，或 AlpacaEval 2），同时也改进指令微调阶段提到的上述技能。
3. **在约 10K 个 prompt 上进行带可验证奖励的强化学习**：这一阶段是小规模强化学习运行，用来在保持整体表现的同时提升数学等核心技能（如今也被视为 DeepSeek R1 等现代推理模型的前身）。

这一配方已被成功应用到 Llama 3.1 [@lambert2024t]、OLMo 2 [@olmo20242] 和 SmolLM 模型 [@alrashed2024smoltulu]。

### DeepSeek R1

随着 OpenAI 的 o1 等推理语言模型兴起，后训练最佳实践再次演化，开始重新排序并重新分配各训练阶段之间的计算。
对推理模型后训练配方记录最清晰的是 DeepSeek R1 [@guo2025deepseek]；阿里巴巴更大的 Qwen 3 模型（即仅 32B 和 225B MoE 模型）[@yang2025qwen3] 以及小米的 MiMo 7B [@xia2025mimo] 都复现了类似做法。
DeepSeek 配方如下：

1. **用 100K+ 个 on-policy 推理样本进行"cold-start"**：这些数据从一个更早的强化学习 checkpoint R1-Zero 中采样，并经过严格过滤，以便在 DeepSeek-V3-Base 上注入特定推理过程。DeepSeek 使用 cold-start 一词来描述强化学习如何从少量监督数据中学起。
2. **大规模强化学习训练**：这一阶段让模型反复处理推理问题，在多种 benchmark 上运行 RLVR，"直到收敛"。
3. **拒绝采样和 SFT**：接近收敛时，他们对强化学习 checkpoint 应用拒绝采样，构建一个约 800K 样本的 SFT 数据集，然后在经过过滤的混合数据上微调模型，其中大约 3/4 是推理问题，1/4 是通用查询，从而生成通用模型。
4. **混合强化学习训练**：在推理问题上使用可验证奖励，并结合通用偏好调优奖励模型来打磨模型。

如上所述，这一配方仍有演化，尤其是在向用户开放模型之前用于最终定型的第 3 和第 4 步。
许多模型会从包含思维链序列的定制指令数据集开始，这些序列经过已有模型的严格过滤和打磨；这样，在转向强化学习之前，仅凭 SFT 就能快速获得强行为 [@seed2025seed]。
