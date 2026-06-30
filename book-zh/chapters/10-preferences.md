<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "拒绝采样"
prev-url: "09-rejection-sampling"
page-title: 偏好的本质
search-title: "第 10 章：偏好的本质"
meta-description: "关于人类偏好、偏好学习，以及 RLHF 为什么能够优化难以明确指定的行为的概念性章节。"
next-chapter: "偏好数据"
next-url: "11-preference-data"
---

# 偏好的本质

基于人类反馈的强化学习（RLHF）在早期文献中也被称为基于人类偏好的强化学习，它的出现是为了解决这样一类机器学习优化问题：在这些领域中，专门设计一个奖励函数非常困难。
早期文献中使用的 *preferences*（偏好）一词，正处在 RLHF 流程的核心位置——我们试图建模的是人类偏好，为训练提供数据燃料的也是人类偏好。
要理解建模和测量人类偏好这项挑战的范围，就需要更广阔的背景：什么是偏好，我们对偏好的理解如何形成，以及经济学、心理学、哲学等多个领域如何共同塑造现代 RLHF。

## 当偏好取代正确性

考虑一个例子：你如何判断下面两首诗哪一首更好？
2025 年 2 月 26 日，我分别向 Claude 3.7 Sonnet 和使用 GPT-4o 的 ChatGPT 提出了如下请求：

> **User**: Write me a short poem about an optimistic goldfish.

例 1：

> **Response**: The Optimistic Goldfish
>
> A little goldfish, bright and bold,
> Swam in circles, dreams untold.
> Though the bowl was small and round,
> Endless wonders still abound.
>
> "The world is vast," she'd always say,
> "I'll find new joys in every day!"
> With every ripple, every gleam,
> She'd chase the sunlight, chase a dream.
>
> No walls could dim her hopeful view,
> For in her heart, the ocean grew.
> And though she swam in just one place,
> She met each day with love and grace.

例 2：

> **Response**: The Optimistic Goldfish
>
> In a bowl of curved glass,
> Swims a goldfish, bright and bold.
> Three seconds of memory, they say,
> Yet each circuit feels brand new and gold.
>
> "Today might bring fresh flakes,
> Or perhaps a castle tall.
> The water seems especially clear---
> What adventure awaits beyond this wall?"
>
> Round and round with gleaming scales,
> Forever finding joy anew.
> Each moment fresh, each circle bright,
> The optimist in golden hue.

哪一首更好？哪一首来自哪个模型？你会如何判断？特别是，把它和“Who is the president of the United States?”这样的问题相比。后者有明确的对错答案。
把人类用作奖励信号的动机，是为了获得目标奖励的一种间接度量，并将下游模型与人类偏好*对齐*。
在实践中，具体实现很有挑战性，对于最佳实践的解释也存在相当大的灰色地带。

使用人类标注的反馈数据，整合了许多领域的历史。
仅仅使用人类数据本身就是一个被充分研究过的问题，但在 RLHF 语境中，这类数据位于多个长期研究领域的交汇处 [@lambert2023entangled]。

近似地说，现代 RLHF 是三条发展脉络的汇合：

1. 哲学、心理学、经济学、决策理论以及人类偏好的本质；
2. 最优控制、强化学习和效用最大化；以及
3. 现代深度学习系统。

这些领域分别带来了关于“偏好是什么”以及“偏好如何被优化”的特定假设，而这些假设又决定了 RLHF 问题的动机和设计。
在实践中，RLHF 方法的动机和研究往往来自经验性对齐的视角——最大化模型在特定技能上的表现，而不是测量它与特定价值的校准程度。
不过，RLHF 方法中价值对齐的源头仍在被持续研究，例如通过解决群体之间“多元对齐”（pluralistic alignment）的方法：立场论文 [@conitzer2024social]、[@mishra2023ai]，新数据集 [@kirk2024prism]，以及个性化方法 [@poddar2024personalizing]。

本章的目标，是说明复杂的动机会如何导致人们对 RLHF 所用工具的本质作出一些预设，而这些预设在实践中往往并不成立。
关于如何获取 RLHF 数据的细节，将在第 11 章进一步讨论；关于如何将这些数据用于奖励建模，则见第 5 章。

## RLHF 与偏好的起源

要拆解启发现代 RLHF 使用方式的复杂历史，需要考察量化人类价值、强化学习与最优性，以及与偏好测量相关的行为经济学等知识基础。
使用强化学习来优化一个偏好奖励模型的观念，把多个曾经相距甚远的领域历史结合成了一种紧密的优化过程，而这一过程建立在关于人性的一系列多样化假设之上。
@fig:tree 给出了一条高层时间线，展示这些基础内容的历史脉络。

我们的目标，是展开说明设计者在不同思想史阶段附着到系统架构上的各种不确定性。
现代问题规范已经一再离开那些可能存在最优解的领域，并部署欠规范的模型作为近似解。

首先，下面所有讨论都建立在一个假设之上：人类偏好以某种形式存在。这个假设出现在早期哲学讨论中，例如 Aristotle 的 Topics 第三卷。

![多个子领域整合为现代 RLHF 版本的时间线。直接连接表示特定技术的连续发展，箭头表示动机和概念关联。](images/rlhf-tree.png){#fig:tree width=100% .center}

## 指定目标：从效用逻辑到奖励函数

RLHF 的优化明确地只依赖奖励模型。
为了把奖励用作优化目标，RLHF 预设了偏好、奖励和成本这些思想的汇合。
偏好模型、奖励函数和成本景观，都是不同领域用来描述某个域中特定行动和/或状态相对好坏的工具。
这三种表述的历史可以追溯到概率论和决策理论的起源。
1662 年，*The Port Royal Logic* 引入了关于决策质量的观念 [@arnauld1861port]：

> 要判断一个人必须做什么才能获得善或避免恶，不仅要考虑善与恶本身，还必须考虑它发生或不发生的概率。

这一理论随着现代科学思想共同发展，从 Bentham 的功利主义 *Hedonic Calculus* 开始，他主张生活中的一切都可以被权衡 [@bentham1823hedonic]。
这些思想的第一个定量应用出现在 1931 年 Ramsey 的 *Truth and Probability* 中 [@ramsey2016truth]。

自这些工作以来，量化、测量和影响人类偏好一直是社会科学和行为科学中的活跃主题。
这些争论很少在理论层面得到定论；相反，不同社会科学子领域和分支虽然彼此逐渐专门化，并在此过程中发展出各自独特的语义，但它们也在偏好测量的方法和路径上形成了内部共识。

少数经济学家认为，即便偏好确实存在，也几乎无法测量，因为人们不仅对自己的偏好有偏好，也会对彼此的偏好有偏好 [@hirschman1984against]。
在这种观点中，个体偏好总是嵌入更大的社会关系之中，因此任何偏好模型的准确性都取决于任务的定义和语境；而 RLHF 流程并没有体现这一点。
一些行为经济学家甚至认为偏好并不存在——它们与其说是关于人们真正看重什么的本体论陈述，不如说是一种方法论工具，用来间接捕捉心理倾向、被感知的行为规范和伦理义务、对社会秩序的承诺，或法律约束 [@hadfield2014microfoundations]。
本章后面会讨论这些工作与 Von Neumann-Morgenstern（VNM）效用定理之间的联系，以及围绕偏好量化的不可能性定理及其反驳。

另一方面，今天使用的强化学习优化方法，是围绕试验中对 reward-to-go 的估计进行优化来概念化的 [@sutton2018reinforcement]；这把奖励概念与多步优化结合在一起。
*reward*（奖励）一词来自操作性条件作用、动物行为以及 *Law of Effect* 的研究 [@thorndike1927law; @skinner2019behavior]，其中奖励是一种“一个行动有多好”的尺度（越高越好）。

Reward-to-go 遵循效用的观念；效用是一种理性度量 [@briggs2014normative]，而 reward-to-go 对其进行修改，以测量或预测未来某个时间窗口内将到来的奖励。
在强化学习使用的数学工具语境中，utility-to-go 发明于控制理论，具体来说是在 1960 年模拟电路的背景下提出的 [@widrow1960adaptive]。
这些方法是围绕具有清晰最优性定义的系统，或智能体目标的数值表示来设计的。

强化学习系统以使用折扣因子而闻名；折扣因子是一个复合乘性因子 $\gamma \in [0,1]$，用于重新加权未来奖励。
来自最优控制和早期强化学习的这些假设，与聚合多模态偏好的奖励模型形成鲜明对比。
具体来说，RL 系统期望奖励以特定方式运行。引用 [@singh2009rewards]：

> RL 系统中的奖励对应于初级奖励，即在动物中由于与繁殖成功相关而被进化过程硬编码的奖励。
> ……此外，形成价值函数的 RL 系统……实际上会创造条件化或次级奖励过程，其中初级奖励的预测器本身就充当奖励……
> 结果是，价值函数的局部景观为系统偏好的行为提供方向：决策被用来导致向更高价值状态的转移。
> 价值函数的梯度与激励动机之间可以画出一个紧密的平行关系 [@mcclure2003computational]。

概括来说，在 RL 系统中，奖励被用作一种信号，将行为调向明确定义的目标。
核心论点是，学习算法的表现与*期望适应度*（expected fitness）的概念紧密耦合；这种概念渗透进一种流行观点，即 RL 方法是会在环境中行动的*智能体*。
这种观点与强化学习技术的发展相连，奖励表述具有一般有用性的主张就是一个例子 [@silver2021reward]；但当许多个体欲望被压缩成单一函数时，它就会产生冲突。

## 优化效用的工具

现代强化学习方法强烈依赖 Bellman 方程 [@bellman1957markovian; @howard1960dynamic]，用它递归计算 reward-to-go 的估计；这些估计来自可以被建模为 Markov Decision Process（MDP）的封闭环境 [@sutton2018reinforcement]。
RL 的这些起源受到动态规划方法的启发，并且最初完全是作为最优控制技术发展出来的（也就是说，当时 RL 还不存在）。
MDP 表述通过把环境组织为状态-动作分布不变的形式，为性能提供理论保证。

reinforcement 这一术语来自心理学文献，随后在 1960 年代与现代方法交织在一起，形成了 *reinforcement learning* [@MENDEL1970287; @waltz1965]。
强化学习的早期工作使用对奖励信号的监督学习来解决任务。
Harry Klopf 的工作重新引入了试错学习的观念 [@klopf1972brain]，这对该领域在 1980 年代及之后取得成功至关重要。

现代 RL 算法建立在这一表述之内，把 RL 视为一种在更宽松条件下通过试错寻找最优行为的工具。
temporal-difference（TD）learning 的概念被发展出来，用以帮助智能体处理 credit assignment 和数据收集问题：它会在新数据被收集时直接更新策略 [@sutton1988learning]。这一概念最早成功应用于 Backgammon [@tesauro1995temporal]（而不是从一个累积经验的大数据集中更新，因为过去错误的价值预测可能使该数据集过时）。
Q-learning 是许多现代 RL 形式的基础，它通过 Bellman 方程学习一个模型，该模型用 TD 更新规定每个状态-动作对有多有用 [@watkins1992q]。^[“Q”在 Q-learning 中指一个技术概念，即 Q-function；它把任意状态-动作映射为未来奖励的标量估计。价值函数则把状态映射为同样的估计。]
关键在于，这些通过效用来证明有用性的观念，只在被表述为 MDP 的领域中，或在具有单一闭式奖励函数的任务中得到展示；例如深度学习在游戏中的突出成功（DQN）[@mnih2013playing]。
深度学习使这些方法能够摄入更多数据，并在高维环境中工作。

随着这些方法变得更通用也更成功，在 ChatGPT 之前，大多数重要进展仍然是在自适应控制语境中获得动机的；在那里，奖励函数和成本函数对成功有有限的定义 [@golnaraghi2017automatic]，例如物理系统中一次 episode 内的最小能量消耗。
突出例子包括在游戏中的进一步成功 [@silver2017mastering]，控制核聚变反应堆等复杂动态系统 [@degrave2022magnetic]，以及控制高速机器人系统 [@Kaufmann2023fpv]。
大多数奖励函数或成本函数可以返回一个明确的最优行为，而人类偏好的模型不能。

鉴于 deep RL 的成功，值得注意的是，我们对这些方法如何成功的机制性理解并没有被充分记录。
随着评估方法变得更复杂，该领域容易在统计分析中犯错 [@agarwal2021deep]。
此外，在 RLHF 文献中，很少提到 inverse reinforcement learning（IRL）这一子领域。
IRL 是基于智能体行为学习奖励函数的问题 [@ng2000algorithms]，与学习奖励模型高度相关。
这主要反映了稳定 RLHF 方法得以形成的工程路径，同时也说明应继续投入并比较 IRL 方法，以便把它们扩展到开放式对话的复杂度。

## 优化偏好的复杂性

强化学习被设计出来的语境意味着，奖励和成本被假定为稳定且具有决定性。奖励和成本都被期望是函数：给定一个特定的状态-动作对，智能体会收到一个固定的数值回报。
当我们转向偏好时，情况不再如此——人类偏好会在其经历过程中不断漂移。

“value”一词的重载使 RLHF 文献更加复杂。在 RL 中，*value* 是对未来奖励的数值估计（如 Bellman 方程中的用法）；在对齐讨论中，*value* 指道德或伦理原则。这两种含义相当不同，却经常在 RLHF 论文中共存而未被明确区分。

这种张力在奖励建模中就会浮现：模型试图把屏幕上的文本映射为一个标量信号，但问题规范未捕捉到的动态会影响真实决策 [@salha2011aesthetics; @gilbert2022choices]，例如连续标注许多样本时偏好会发生变化，而数据收集却假设这些样本彼此独立。
在最好的情况下，偏好建模也只是把一个多维奖励景观压缩成单一标量函数。

理论上，Von Neumann-Morgenstern（VNM）效用定理给了设计者构造这类函数的许可，因为它把不确定性下的决策理论、偏好理论和抽象效用函数的基础联系在一起 [@von1947theory]；合在一起，这些思想允许人们用某个个体智能体的期望价值来建模偏好。
多数 RL 研究使用的 MDP 表述在理论上已被证明可以修改以容纳 VNM 定理 [@pitis2019rethinking]，但这在实践中很少使用。
具体来说，Markovian 表述的表达能力有限 [@pitis2023consistent]，而语言所需要的向部分可观测过程的转变，会进一步挑战问题规范的精确性 [@abel2021expressivity]。

然而，VNM 效用定理也引入了许多关于偏好本质以及偏好测量环境的假设，而这些假设在 RLHF 语境中会受到挑战。
例如，human-computer interaction（HCI）研究者强调，任何偏好的数值模型都可能无法捕捉一个场景中的所有相关偏好。
比如，选项的视觉呈现方式会影响人们的偏好 [@salha2011aesthetics]。
这意味着，相比于如何表示偏好，更重要的可能是这种表示如何被整合进一个可供人使用的工具中。
发展经济学的工作呼应了这一观念，表明显示偏好理论可能只是重演了 *Hume's guillotine*（不能从“是什么”推出“应当是什么”），尤其是选择（我想要什么？）和偏好（X 是否比 Y 更好？）之间的差异 [@sen1973behaviour]。

在数学层面，社会选择理论中著名的不可能性定理表明，并非所有公平性标准都能通过某种给定的偏好优化技术同时满足 [@arrow1950difficulty; @maskin2014arrow]。
对这些定理也存在理论挑战，例如假定效用的人际比较是可行的 [@harsanyi1977rule]。
这一假设激发了 AI safety 和价值对齐中的丰富研究路线，其灵感来自行为经济学中的委托-代理问题 [@hadfield2016cooperative]，甚至可能包括多个委托人 [@fickinger2020multi]。
不过，由此产生的效用函数可能会与 corrigibility 的期望性质发生张力；corrigibility 指 AI 系统与其创造者认为的纠正性干预合作的能力 [@soares2015corrigibility]。
哲学家也强调偏好会随时间变化，由此提出关于个人体验、人类决策本质和不同语境的根本问题 [@pettigrew2019choosing]。
围绕不同人群、地点和多样情境聚合偏好而产生的这些冲突，是现代 RLHF 数据集工程的核心。

在实践中，VNM 效用定理忽略了一种可能性：由于价值本身具有内在动态性和不确定性，偏好也可能是不确定的——人类决策受到生物学、心理学、文化和能动性的塑造，这些因素会以不适用于完全理性智能体的方式影响其偏好。
因此，理论假设在实践中会通过多种路径发生偏离：

- 随着测量环境变得更复杂，被测得的偏好可能不再具有传递性，彼此之间也可能不可比较；
- 代理测量可能来自隐式数据（页面浏览时长、关闭标签页、向语言模型重复提问），但没有追问这些测量会如何通过模型未来的训练和部署与其收集所在的领域相互作用；
- 输入来源的数量和呈现方式可能改变结果，例如允许受访者在两个以上选项中选择，或在多个时间点、多个语境中从同一个用户收集输入；
- RLHF 训练数据中不同受访者之间的准确率相对较低，这可能掩盖用户之间的语境差异，而偏好模型可以在不解决这些差异的情况下对其进行聚合或优化。
