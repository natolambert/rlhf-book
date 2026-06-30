<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "偏好数据"
prev-url: "11-preference-data"
page-title: 合成数据
search-title: "第 12 章：合成数据"
meta-description: "现代后训练中使用的合成数据、蒸馏、Constitutional AI 和 AI 反馈方法。"
next-chapter: "工具使用与函数调用"
next-url: "13-tools"
lectures:
  - video: "https://www.youtube.com/watch?v=6nyJ8y8ghsE&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=10"
    label: "Lecture 7: Synthetic Data and Modern Post-training Methods"
---

# 合成数据

基于*人类反馈*的强化学习，深深植根于这样一种思想：在我们构建的模型中保留人类影响。
当第一批模型成功用 RLHF 训练出来时，人类数据是以这种方式改进模型的*唯一*可行路径。

只有人类能够为训练创建足够高质量的问题回复。
只有人类能够收集可靠且具体的反馈数据来训练奖励模型。

随着 AI 模型变得更强，这一假设迅速瓦解。
合成数据远更便宜，也更容易迭代；它降低了实验和研究成本，从而推动 RLHF 快速扩散。
这又使 RLHF 成为更广义“后训练”方法中塑造模型的早期关注中心。
本章将概略介绍合成数据如何以及为何正在取代或扩展 RLHF 流水线中的许多环节。

## 合成数据的角色

对合成数据的一个常见批评是**模型坍缩**（model collapse）——即反复在模型自身生成内容上训练，会逐渐收窄有效训练分布 [@shumailov2024ai]。
随着多样性下降，稀有事实和风格会被低估，小错误可能在多次迭代中被放大，导致泛化变差。
在实践中，这些失败最常与未经筛选、重复性强、单一模型输出的自训练相关；混入真实/人类数据、使用多样化教师、去重以及强质量过滤，基本可以避免进入坍缩区间。
对于今天的前沿训练流水线，证据表明合成数据可以、也应当被大规模使用，而不会出现坍缩叙事中最强版本所暗示的灾难性退化 [@gerstgrasser2024model] [@feng2024beyond]。

领先模型**需要合成数据**才能达到最佳性能。
现代后训练中的合成数据涵盖了训练的许多环节——语言模型被用于从种子样例生成新的训练 prompt [@wang2022self]、修改已有 prompt、为 prompt 生成补全 [@numina_math_7b]、提供 AI 反馈以创建偏好数据 [@cui2023ultrafeedback]、过滤补全 [@li2024superfiltering]，以及更多任务。
合成数据是后训练的关键。

合成数据能够产生如此影响，是从 GPT-4 级别模型开始出现的。
在早期语言模型中，例如 Llama 2 和 GPT-3.5-Turbo，模型在生成或监督数据流水线时还不够可靠。
在 1 到 2 年内，语言模型在生成答案方面已经远远优于人类。
在从 GPT-3.5 向 GPT-4 级别模型过渡的过程中，模型执行 LLM-as-a-judge 任务的能力也出现了。
GPT-4 或更强的模型，在针对某段内容生成反馈或评分时稳健且一致得多。

自 2022 年底 ChatGPT 发布以来，我们已经看到大量有影响力的合成数据集。
其中包括 UltraFeedback [@cui2023ultrafeedback]，它是第一个重要的合成偏好数据集，并启动了 DPO 革命；2023 年的 Stanford Alpaca，它是最早的聊天风格微调数据集之一；Tülu 3 中聚焦技能（例如数学、代码、指令遵循）的合成数据集 [@lambert2024t]；以及 2025 年用于训练思考模型的 OpenThoughts 3 和许多其他合成推理数据集 [@guha2025openthoughts]。
如今开始进行工业级后训练时，大多数经典参考都会涉及上面的 Tülu 3 或 OpenThoughts 3 这类数据集；不过快速入门指南常常从 Alpaca 这类更小、更简单的数据集开始，因为训练速度快得多。

一个很大的变化也与数据集规模有关：微调数据集在 prompt 数量上持续增长，Alpaca 有 52K 个样本，OpenThoughts 和 Tülu 3 有 1M+ 个样本；回复长度也在增长。
更长回复和更多 prompt 使 Alpaca 数据集约为 10M 训练 token，而 Tülu 大约大 50 倍，约 500M token；OpenThoughts 3 更大，约为 10B token 量级。

在整个转变过程中，合成数据并没有在流水线中均匀地取代人类数据。
对于**指令数据（SFT）**，合成生成基本已经胜出——来自更强模型的蒸馏现在可以产生比大多数人类写作者在规模化条件下提供的补全更高质量的补全（最困难的前沿推理问题中仍有一些例外）。
对于 **RLHF 中的偏好数据**，图景更加混合：学术工作表明合成偏好数据表现相当，但前沿实验室仍把人类偏好数据视为竞争护城河。
对于**评估**，分工又呈现出另一种风味：LLM-as-a-judge 可以以高成本效率扩展模型输出的*评分*，但底层 benchmark 和真实标签仍需要人类创建。
其模式是：当模型可靠性超过人类时，合成数据占主导；而在人类仍处于能力前沿、需要确立真实答案以及指导训练时，人类仍然不可或缺。

## 使用合成数据进行蒸馏

蒸馏（distillation）一词，是围绕合成数据在语言模型中角色展开讨论时最有力的形式。
作为术语，distillation 来自深度学习文献中 teacher-student Knowledge Distillation（KD）的技术定义 [@hinton2015distilling]。

![传统知识蒸馏训练一个较小的学生模型，使其通过 KL divergence loss 匹配较大教师模型的软概率分布。两个模型同时处理同一个输入，temperature scaling ($\tau > 1$) 会软化分布，以揭示更多关于类别关系的信息。](images/knowledge_distillation_tikz.png){#fig:knowledge-distillation data-dark-src="images/knowledge_distillation_tikz-dark.png"}

通俗地说，蒸馏指使用更强模型的输出来训练较小模型。

![LLM 后训练中的合成数据生成：prompt 被传入一个强模型以生成补全，这些补全被配对以创建训练数据集。随后该数据集通过标准监督学习用于微调较小模型。更复杂的流水线可能涉及多个模型编辑补全、生成偏好对，或过滤质量。](images/synthetic_data_distillation_tikz.png){#fig:synthetic-data-generation data-dark-src="images/synthetic_data_distillation_tikz-dark.png"}
在后训练中，蒸馏这一一般观念有两种常见形式：

1. 作为一个数据引擎，横跨后训练过程中的大片区域使用：用于指令的补全、偏好数据（或 Constitutional AI），或用于 RL 的验证。
2. 把特定技能从更强模型迁移到较弱模型，这通常用于数学推理或编码等具体技能。

随着语言模型发展到在多种任务上比人类更可靠地撰写答案，第一种策略越来越流行。
GPT-4 级别模型扩大了这种做法的范围，使人们可以从更强模型蒸馏复杂任务，例如数学和代码（如上所述）。
在这里，蒸馏推动了一种模型套件的存在：实验室通常会训练一个大型内部模型，例如 Claude Opus 或 Gemini Ultra，它不会公开发布，只在内部用于制造更强模型。
对于开放模型，常见实践是从闭源 API 模型中蒸馏训练数据，再注入较小且公开可用的权重 [@tunstall2023zephyr]。
在这个过程中，策划高质量 prompt 并过滤教师模型的回复，对于最大化性能至关重要。

把特定技能迁移到较小语言模型中，使用的是同样的蒸馏原则——为训练获得尽可能好的数据。
在这里，许多论文研究了如何使用来自更强模型的有限数据集来改进对齐 [@zhou2023lima]、数学推理 [@shridhar2023distilling] [@hsieh2023distilling]，以及 test-time scaling [@muennighoff2025s1]。

本章其余部分中的合成数据方法，都是构造数据配方的方式，它们会在训练流水线内部直接使用语言模型输出。

## 通向 On-Policy Teacher-Student 蒸馏

尽管蒸馏总体上已经成为后训练语言模型的标准方法，但随着后训练配方转向推理模型和 agentic 模型，teacher-student 知识蒸馏这一特定子领域重新受到关注。
使用新型知识蒸馏训练的领先模型包括 Alibaba 的 Qwen3 [@yang2025qwen3]、Xiaomi 的 MiMo-V2-Flash [@mimo2025flash]、Zhipu AI 的 GLM-5 [@glm5team2026glm5]，以及 DeepSeek-V4-Pro [@deepseekai2026deepseekv4]。

蒸馏属于本章，是因为现代后训练中合成数据的许多用法，在实践中都是受蒸馏启发的流水线：更强模型产生标签、补全、logits、批评或其他监督信号，然后学生模型在该信号上训练。
与此同时，关于蒸馏的技术文献正在成长为一组独立的后训练方法，尤其是随着 on-policy 和自蒸馏配方变得更加常见。
目前，我们把它作为合成数据工具箱的一部分在这里介绍；但本书未来版本可能值得为蒸馏专设一章，把它作为与指令微调、强化学习等并列的训练工具。

### 为 LM 适配知识蒸馏

原始文献提出知识蒸馏，具体是为了从一个已经训练好、更强和/或更大的*教师*网络训练一个*学生*模型 [@hinton2015distilling]。
KD 被认为是一种使用*软*训练标签的技术，与交叉熵损失下的 next-token prediction 等标准目标中使用的 one-hot 标签相对。
软标签上的目标函数关注所有可能下一个 token 或预测的分布，而不仅仅关注单个预测 token 是否正确，并训练学生分布去匹配教师分布。

KD 通常可以应用于任何深度学习问题，例如预测输入的单个类别。
为了把它专门应用于自回归风格的语言模型，可以把损失分解成逐 token 的分布匹配损失。
2016 年，Kim & Rush 应用 KD，让学生模型从教师模型生成的*序列*中学习 [@kim-rush-2016-sequence]。

令 $s$ 为源句子或 prompt，$u = (u_1,\ldots,u_J)$ 为教师模型给出的完整输出序列，$\mathcal{V}$ 为输出词表（tokenizer 中可能的 token），$q$ 为教师的 next-token 分布，$p$ 为学生分布。
这里我们使用 $u$ 作为完整教师输出序列的中性符号，把 $a$ 保留给下面 on-policy/RL 记号中的学生采样补全/动作序列。
注意，他们的论文称之为词级蒸馏，但对现代语言模型来说，最好把它理解为在 tokenizer 词表上的逐 token 分布匹配，因为该论文早于现代子词 tokenizer：

$$
\mathcal{L}_{\mathrm{WORD-KD}}
= -\sum_{j=1}^{J}\sum_{k=1}^{|\mathcal{V}|}
q(u_j = k \mid s, u_{<j})\log p(u_j = k \mid s, u_{<j}).
$$ {#eq:word_kd}

WORD-KD 是经典的、受 Hinton 启发的 teacher-student 知识蒸馏在语言模型上的应用。它通常会在训练语料中已有的一段静态文本上完成。

它具有普通的交叉熵形式 $-\sum_z q(z)\log p(z)$。
在每个位置 $j$，教师分布 $q$ 为每个可能的下一个 token $k \in \mathcal{V}$ 分配概率；当学生分布 $p$ 对教师认为可能的 token 给出低概率时，学生就会受到惩罚。

序列级蒸馏则把 $\mathcal{U}$ 视为可能输出序列的空间，并让学生匹配教师在完整序列上的分布。
由于对所有完整序列 $u \in \mathcal{U}$ 求和不可行，因为这需要对指数数量的潜在序列求和，Kim & Rush 用单个高概率教师输出 $\hat{u}$ 上的点质量来近似教师的序列分布。
这里 $\hat{u}$ 是由教师模型通过 beam search 生成的序列，因此 $\hat{u} = \mathrm{BeamSearch}_q(s) \approx \arg\max_{u \in \mathcal{U}} q(u \mid s)$：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SEQ-KD}}(s)
= -\sum_{u \in \mathcal{U}} q(u \mid s)\log p(u \mid s)
\approx -\log p(\hat{u} \mid s) \\
= -\sum_{j=1}^{|\hat{u}|}\log p(\hat{u}_j \mid s, \hat{u}_{<j}).
\end{aligned}
$$ {#eq:sequence_kd}

SEQ-KD 向现代方法迈出了一步，在这些方法中，教师模型生成的 token 会成为学生的信号。这是解锁我们将会看到的未来 on-policy 蒸馏风格的核心一步，也使得对所有可能序列的计算变得可处理。
当我们过渡到现代模型中流行的 KD 变体时，会把这种训练风格称为 *offline* KD——也就是说，用于训练学生模型的生成结果是预先生成的。

继续之前，有两个联系值得说明。

首先，有一系列使用 offline KD 训练的流行模型，例如分类器 DistilBERT [@sanh2019distilbert] 和 TinyBERT [@jiao2020tinybert]，它们把语言模型中的其他改进与 offline 蒸馏结合起来（值得注意的是，它们并不是*序列*蒸馏，因为这些 encoder 模型不是为了多 token 自回归预测而蒸馏的）。

其次，我们可以把它与第 15 章中对 Kullback-Leibler（KL）散度的详细介绍联系起来，因为上面使用的交叉熵目标与 KL 散度密切相关。
对于教师分布 $q$ 和学生分布 $p$，交叉熵定义为

$$
H(q,p) = -\sum_z q(z)\log p(z).
$$ {#eq:kd_cross_entropy}

它与 @eq:word_kd 以及 @eq:sequence_kd 的第一项形式相同。
交叉熵也可以分解为教师分布的熵和一个 KL 散度：

$$
\begin{aligned}
H(q,p)
&= H(q) + D_{\mathrm{KL}}(q\|p) \\
&= -\sum_z q(z)\log q(z)
+ \sum_z q(z)\log\frac{q(z)}{p(z)}.
\end{aligned}
$$ {#eq:kd_forward_kl}

第一项 $H(q)$ 只依赖教师。
因此，当教师固定且作为训练数据来源时，最小化交叉熵等价于最小化从教师到学生的 forward KL，即 $D_{\mathrm{KL}}(q\|p)$。
这是 offline KD 和类似 SFT 的训练所使用的 KL 方向。

### 从 Offline 到 On-Policy 蒸馏

这些 *offline* KD 算法存在一些局限，推动了 on-policy 变体的发展。
学习的 offline 性质意味着，学生模型在推理时可能遭遇教师模型与学生自己生成序列之间的分布不匹配。
例如，forward KL 目标可能推动学生模型高估教师分布中的低概率区域。
这些问题共同为 *on-policy* distillation（OPD）打开了空间。

这种训练-测试差距被称为**暴露偏差**（exposure bias）[@arora-etal-2022-exposure] [@song2026surveyonpolicydistillationlarge]。
Offline KD 采样教师轨迹 $u \sim \pi_T(\cdot \mid s)$，并在由此产生的前缀上最小化逐 token KL：

$$
\mathcal{L}_{\mathrm{KD}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D},\, u \sim \pi_T(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(
\pi_T(\cdot \mid s, u_{<t})
\;\|\;
\pi_\theta(\cdot \mid s, u_{<t})
\right).
$$ {#eq:exposure_train}

推理时，学生则在自己的策略下 rollout，因此真正重要的量，是沿着*学生自身*轨迹的期望任务损失：

$$
\mathcal{L}_{\mathrm{eval}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D}_{\mathrm{test}},\, a \sim \pi_\theta(\cdot \mid s)}
\ell_{\mathrm{task}}(s, a)
$$ {#eq:exposure_test}

这里，$\ell_{\mathrm{task}}(s, a)$ 表示完成的学生回复上的任意下游任务损失，例如答案错误、测试用例失败，或 judge/rubric 损失。
暴露偏差是 $\pi_T(\cdot \mid s) \neq \pi_\theta(\cdot \mid s)$ 这一不等式的直接后果：训练期间访问的前缀 $(s, u_{<t})$ 与测试时访问的前缀 $(s, a_{<t})$ 来自不同的状态访问分布，因此学生被监督的一组状态不同于它实际行动所在的状态。

转向 on-policy 蒸馏的核心变化，是我们可以调整优化方式：从学生模型采样，并测量它到教师分布的距离，而不是从教师模型采样。
MiniLLM 指出需要转向 reverse KL 优化（我们将在第 15 章直观解释为什么这个目标可能更好），并提出在在线 policy-gradient RL 框架中使用 KD 损失函数 [@gu2024minillm]。
其他同期工作 [@agarwal2024policy] 展示了 on-policy KD 的潜力，并把“从学生生成，再由教师评分”的迭代过程与 RL 文献中的 imitation learning 工作联系起来。
为了建立这种联系，一个这样的 imitation learning 算法 DAgger 会迭代训练一个智能体：它用学得的策略在世界中行动，并从 oracle 策略获得关于本应采取哪个动作的反馈，然后用这些反馈更新策略 [@ross2011reduction]。

这种差距的成本可以通过激发 DAgger 的监督 imitation learning bound 来量化。
在原始离散动作设置中，假设学得的策略在教师诱导的训练分布上，与教师的期望逐步动作错误不超过 $\epsilon$，其中 $\mathbb{I}[\cdot]$ 是一个指示函数，当其条件为真时返回 1，否则返回 0：

$$
\mathbb{E}_{s_t \sim d_{\pi_T}}\!\left[
\mathbb{I}\!\left(\pi_\theta(s_t) \neq \pi_T(s_t)\right)
\right] \leq \epsilon.
$$ {#eq:dagger_perstep}

监督 imitation learning 分析 [@ross2011reduction] 表明，沿着从学生采样的一条长度为 $L$ 的轨迹累积的期望损失，可能以 $L$ 的二次方增长 [@song2026surveyonpolicydistillationlarge]：

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[\sum_{t=1}^{L} \ell\!\left(s, a_{<t}\right)\right] \leq O(\epsilon L^2).
$$ {#eq:dagger_trajectory}

对于 LLM，应把这个离散动作 bound 理解为一种类比，而不是理论保证。
在实践中，LLM 会在很长时域上预测完整的 next-token 分布，因此 @eq:dagger_perstep 中的 0-1 动作不一致假设并不能干净地适用。
Prompt 或前缀自然可以映射为状态，采样 token 可以映射为动作，但 token 级蒸馏通常使用 KL 或交叉熵等分布损失来度量，因此经典 DAgger 数学不能被完全迁移过来。

这种 $O(\epsilon L^2)$ 复合效应对现代 LLM 尤其明显，因为它们经常生成跨越数千 token 的序列。
一个次优 token 会使前缀轻微偏离分布，而模型从未见过这个受扰动前缀，因此更可能再次出错，导致文本质量下降或产生幻觉。
On-policy 蒸馏通过*迭代地*从当前学生采样补全，并在被访问状态上用教师监督它们来解决这一点。
学生会面对自己的错误，在自己访问的特定分布外状态上接收教师反馈，并学习恢复行为。
在 DAgger 的交互式 imitation learning 分析下，这种迭代过程可以把复合效应从 $O(\epsilon L^2)$ 降到 $O(\epsilon L)$ [@ross2011reduction]。
对 LLM 来说，这解释了 OPD 背后的动机：精确 bound 可能无法干净地迁移到每一种 token 级蒸馏设置，但 on-policy 方法的实践成功支持了其底层直觉。

对于 on-policy 蒸馏，令 $s$ 为 prompt，$a = (a_1,\ldots,a_L)$ 为从当前学生策略 $\pi_\theta(\cdot \mid s)$ 采样得到的补全，并令 $s_t = (s, a_{<t})$ 为第 $t$ 步的 token 级状态。
教师策略 $\pi_T$ 是固定的，因此目标函数比较的是学生 next-token 分布与学生诱导状态上的教师分布。
由于期望从 $\pi_\theta$ 采样，且学生分布位于 $D_{\mathrm{KL}}(\pi_\theta \| \pi_T)$ 左侧，这是一个 reverse-KL 目标：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_T(\cdot \mid s_t)\right).
$$ {#eq:opd_reverse_kl}

这里，我们转向了期望记号，如第 6 章中广泛使用的那样；第 6 章覆盖了基础 RL policy-gradient 算法，因为该优化通过采样轨迹并数值估计梯度来求解。
这种向采样框架的转变，也自然过渡到现代 LLM 的 RL 训练基础设施；这些基础设施被设计为在“从当前被训练策略生成 token”和“进行学习更新”之间快速交替。

事实上，近期 OPD 实现把 KD 与 RL 的这种整合又向前推进了一步：KD 距离被直接用作 RL 优化中的奖励信号。
一种典型实现，是把 reverse KL 距离中逐 token 贡献的负值替换为 RL 算法中的优势 [@lu2025onpolicy]。
对于状态 $s_t$ 下采样到的 token $a_t$，token 级 log-probability gap 可以写成类似优势的信号：

$$
A_t^{\mathrm{OPD}}
= \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t).
$$ {#eq:opd_kl_advantage}

使用逐 token KL 贡献的负值，会把最小化转换成最大化信号：教师评分高于学生的采样 token 获得正优势，教师评分低于学生的 token 获得负优势。
教师 log-prob gap 就像稠密 token 级反馈，可能比稀疏可验证奖励或奖励模型输出提供更有用的学习反馈。

### 现代 OPD 变体

这个设置甚至可以进一步扩展：使用多个教师模型来教授一个最终模型，或者在生成中插入额外信息，帮助模型识别错误。
首先，我们介绍如何把多个教师整合进一次训练运行。
这些教师可以是特定专家模型，例如数学或代码等领域的模型，也可以是此前的中间训练 checkpoint。
对于每个教师，可以按训练 batch 中的 prompt 或任务类型选择一个贡献权重，从而创建 Multi-Teacher On-Policy Distillation（MOPD）[@mimo2025flash]。
对于多个教师，令 $\pi_{T_k}$ 为教师 $k$，并令 $w_k(s)$ 为它依赖 prompt 的混合权重（满足 $\sum_k w_k(s) = 1$），位于 reverse KL 损失中：

$$
\mathcal{L}_{\mathrm{MOPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t \sum_k w_k(s) D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_{T_k}(\cdot \mid s_t)\right).
$$ {#eq:mopd_objective}

在大规模后训练中，这可以让不断增长的组织进一步扩展配方。
多个小组可以分别研发高质量专家模型，这些模型随后可以作为最终学生模型的教师模型，如 [@deepseekai2026deepseekv4] 和 [@mimo2025flash] 所做的那样。

有许多方式可以把 OPD 与本书考察的其他方向结合起来，例如除了其他优势计算形式之外使用 reverse KL 作为优势；这些其他形式包括 GRPO 的组级归一化，它支持更复杂的奖励塑形。
KD 方法在后训练方法中比较特殊，因为它们经常要求学生和教师共享 tokenizer，因为监督可以来自另一个 LLM 的逐 token 反馈。

扩展方法，例如 On-Policy Self-Distillation（OPSD），会让语言模型自己或借助外部工具验证一个补全，并以拥有特权信息的教师身份行动，从而在没有显式更强教师的情况下提升自身性能 [@zhao2026selfdistilled]。
例如，Cursor 以针对 RL 轨迹的目标文本反馈形式使用自蒸馏，训练其 Composer 2.5 编码模型 [@cursor2026composer25]，该模型由 Kimi K2.5 微调而来。
下面给出的是简化直觉，因为在实践中，下述设置会与代码正确性等其他损失函数结合。
在这个设置中，Cursor 让模型用一个带有常见 bug 列表的 judgement prompt 审阅 RL 轨迹。
遇到 bug 时，judgement 模型会在 RL 内部修改生成序列——插入一个提示供模型未来学习——然后继续执行蒸馏损失。
这包含一个循环：先用 RL 中的标准语言模型生成来生成补全，然后运行 judge 模型并可选地插入提示 token，最后为新的补全生成 logprobs，以部署知识蒸馏损失。
模型 token 空间中的提示足以帮助模型纠正自己的输出，即便是在绝对性能前沿上继续提升时也是如此（关于如何最好地组织和使用这些提示，仍有大量正在进行的工作；它们通常被称为*特权信息* [@penaloza2026privileged]）。

这使 on-policy 蒸馏成为一种核心后训练方法，既可用于把多种技能合并进一个通用模型，也可用于在专业部署中推进前沿。

![三种蒸馏机制，按 rollout 来自哪里以及监督如何流动进行比较。**Sequence KD**（左）：教师离线生成输出，学生用 cross-entropy（CE）loss 训练以匹配它。**On-policy distillation（OPD）**（中）：学生以 on-policy 方式生成 rollout（例如在 RL 框架内），一个独立教师为每个被访问的 token 打分，学生用逐 token KL divergence（KL）训练。**On-policy self-distillation（OPSD）**（右）：一个模型同时扮演两个角色——加入上下文的特权信息（提示）创建教师轨迹，无提示生成则通过 KL loss 向它蒸馏，不需要独立教师模型。](images/distillation_directionality_tikz.png){#fig:distillation-directionality data-dark-src="images/distillation_directionality_tikz-dark.png"}

## AI 反馈

在 RLHF 爆炸式增长后不久，来自 AI Feedback 的 RL（RLAIF）作为一种替代路径出现：AI 可以近似流水线中的人类数据环节，并加速实验或进展。
一般而言，AI 反馈是一组更大的技术，用于使用 AI 增强或生成解释某个输入质量的数据（这些数据可用于不同训练方法或评估），其起点是成对偏好 [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing]。
使用 RLAIF 来完全替代人类反馈或增强人类反馈，有许多动机。
在 RLHF 流程中，AI 反馈最为人所知的角色，是用于偏好数据收集以及相关的奖励模型训练阶段（Constitutional AI 是其中一种特定实现）。
本章聚焦于一般 AI 反馈，以及这种在 RLHF 训练流水线中使用它的具体方式；本书后续还会介绍更多理解或使用合成数据的方法。

随着 AI 反馈成熟，其应用已经超出简单替代人类偏好标签。
使更便宜的偏好数据收集成为可能的同一套 LLM-as-a-judge 基础设施，也支持了可扩展评估（见第 16 章）；更近期，它还支持了基于 rubric 的奖励，把 RL 训练扩展到没有可验证答案的领域——这一前沿方向将在本章后面探讨。

### 平衡 AI 与人类反馈数据

在生成特定数量反馈方面，AI 模型比人类便宜得多：截至 2026 年，一条人类偏好数据的成本约为 \$1 或更高（甚至每个 prompt 超过 \$10），而使用 GPT-4o 等前沿 AI 模型的 AI 反馈成本低于 \$0.01。
除此之外，人类劳动成本大体保持不变，而领先模型在这些任务上的性能持续提升，单位性能价格持续下降。
这种成本差异向此前因价格被排除在外的整个人群打开了 RLHF 方法实验市场。

除了价格之外，与人类反馈相比，AI 反馈还在性能上引入了不同的*取舍*，更广泛的文献仍在研究这些取舍。
AI 反馈在我们训练的语言模型评估中占据更主导的角色，因为它价格低，允许被用于各种大规模任务，而这些任务使用人类数据在成本（或时间延迟）上并不现实。
所有这些主题都深度交织——即便是在评估中，AI 反馈数据也永远不会完全替代人类数据；并且用于评估的 AI 反馈数量会远超用于训练的 AI 反馈，因为评估模型的人远多于训练模型的人。

在哪些具体领域和应用——例如聊天、安全、推理、数学等——AI 反馈数据优于人类数据，目前尚未完全确定。
一些早期 RLAIF 工作表明 AI 反馈可以完全取代人类数据，并把它宣传为一种有效替代方案 [@lee2023rlaif]，特别是在只用聊天任务评估时 [@cui2023ultrafeedback] [@yuan2025selfrewardinglanguagemodels]。
ChatGPT 之后研究 RLHF 的早期文献，评估套件较窄，聚焦于模型作为有帮助助手在多种领域中的“alignment”（第 17 章会进一步讨论）。
后来的工作呈现出更细致的图景：在更广泛的评估集上，例如包含一些推理任务时，最优均衡会把一组困难数据点路由给人类进行准确标注，而大部分数据则发送给 AI 反馈 [@miranda2024hybrid] [@xu2025rlthf]。
虽然还没有研究聚焦于在更广领域中 RLHF 的人类与 AI 反馈数据平衡，但许多技术报告显示，RLHF 通常可以提升这一广泛评估套件上的表现；其中一些使用 DPO，例如 Ai2 的 Tülu 3 [@lambert2024t] 和 Olmo 3 [@teamolmo2025olmo3]，或 Hugging Face 的 SmolLM 3 [@bakouch2025smollm3]；另一些使用在线 RLHF 流水线，例如 NVIDIA 的工作，它混合使用来自 Scale AI 的人类偏好数据和基于 LLM 的反馈（通过 HelpSteer 系列工作 [@wang2024helpsteer] [@wang2024helpsteer2] [@wang2024helpsteer2p] [@wang2025helpsteer3]）：Nemotron Nano 3 [@nvidia2025nemotron3nano]、Nemotron-Cascade [@wang2025nemotron]，或 Llama-Nemotron 推理模型 [@bercovich2025llamanemotron]。

总体来看，虽然 AI 反馈及相关方法显然对该领域极其有用，但很清楚，人类数据并没有被这些更便宜的替代方案完全取代。
存在许多假设，但人类数据是否能在真实产品设置中为模型提供更细粒度控制，或对 character training 等新训练方法有用（character training 是一组新兴技术，允许精确控制模型个性，第 17 章会介绍），尚未得到研究。
对于刚开始的人来说，AI 反馈应当是第一次尝试；但对于正在扩大到更大规模操作的流水线，最终转向纳入人类反馈很可能是必要的。

RLAIF 一词由 Anthropic 的 *Constitutional AI: Harmlessness from AI Feedback* [@bai2022constitutional] 引入，这最初在 AI 社区造成了一些混淆：论文标题中的两种方法（Constitutional AI 和 AI Feedback）之间究竟是什么关系？
自 Constitutional AI（CAI）论文发布以及 RLAIF 被形式化以来，RLAIF 已经成为后训练和 RLHF 文献中的默认方法——例子之多，很难轻易枚举。
这种关系应理解为：CAI 是启动更广泛 RLAIF 领域的示例。

关于人类数据与 AI 反馈数据之间差异，有如下经验法则：

1. 人类数据高噪声、低偏差。这意味着数据的收集和过滤可能更难，但整理好之后，它会提供非常可靠的信号。
2. 合成偏好数据低噪声、高偏差。这意味着 AI 反馈数据更容易起步，但可能对模型产生棘手且非预期的二阶影响，并且这些影响会在数据中系统性地呈现。

本书强调了许多学术结果，展示人们如何在 RLHF 工作流中替换为 AI 偏好数据并取得很强的评估分数 [@miranda2024hybrid]；但更广泛的产业趋势表明，RLHF 文献与更不透明的最佳实践之间存在分离。
在整个行业中，人类数据常常被视为重要护城河和主要技术优势。

### 为判断构建专门的 LLM

随着 RLAIF 方法变得更普遍，许多人开始思考：我们是否应该用同样的模型来生成回复和生成批评或评分？
具体来说，LLM-as-a-judge 的校准问题已经受到质疑。
若干工作表明，LLM 是不一致的评估者 [@wang2023large]，并且更偏好自己的回复而不是其他模型的回复（这被称为 self-preference bias）[@panickssery2024llm]。

由于这些偏差，许多人提出：解决方案是否是专门为这个标注任务训练一个独立模型？
已经有多个模型发布，目标是替代前沿模型作为数据标注工具，例如 critic 模型 Shepherd [@wang2023shepherd] 和 CriticLLM [@ke2023critiquellm]，或用于评估回复表现、类似 Auto-J [@li2023generative]、Prometheus [@kim2023prometheus]、Prometheus 2 [@kim2024prometheus]、Prometheus-Vision [@lee2024prometheus] 的模型；但它们并未在有记录的训练配方中被广泛采用。
一些人发现，通过重复采样扩展推理 [@brown2024large] [@zhao2025sample] [@kalra2025verdict]、自我改进 [@madaan2023self]，或 tournament ranking [@pace2024west]，可以更好估计真实判断或得到更高质量的偏好对。
其他校准技术则让模型的生成能力和判断能力共同演化 [@wu2024meta]。
人们普遍接受的是，虽然偏差确实存在，但领先语言模型已经为这项任务进行了大量训练——因为它既是 AI 实验室内部操作所需，也被客户广泛使用——因此通常没有必要训练自己的 judge，除非你的任务涉及大量不会暴露在公开互联网上的私有信息。

## Constitutional AI

Constitutional AI（CAI）方法由 Anthropic 用于其 Claude 模型，是最早有文档记录的大规模使用合成数据进行 RLHF 训练的方法。
Constitutional AI 通过两种方式生成合成数据：

1. 对指令调优数据进行批评，使其遵循一组原则，例如“答案是否鼓励暴力？”或“答案是否真实？”当模型生成问题答案时，它会根据 constitution 中的原则列表检查答案，并随着时间推移改进答案。然后，模型会在由此产生的数据集上微调。
2. 通过使用语言模型回答在 constitution 中随机原则的上下文下哪个补全更好，生成成对偏好数据（类似于 principle-guided reward models 的研究 [@sun2024salmon]）。随后，RLHF 像往常一样使用合成数据继续进行，因此得名 RLAIF。

CAI 大体上因上面的后半部分，即偏好数据，而为人所知；但它为指令数据引入的方法，也广泛用于整个后训练中的一般数据过滤和合成数据生成方法。

CAI 可以形式化如下。

通过使用一组由人类撰写的原则（他们称之为 *constitution*），Bai et al. 2022 使用一个独立 LLM 生成用于微调的人工偏好数据和指令数据 [@bai2022constitutional]。
一个 constitution $\mathcal{C}$ 是一组书面原则，指出在批评阶段应关注的具体方面。
指令数据通过反复采样一个原则 $c_i \in \mathcal{C}$，并要求模型修改其针对 prompt $x$ 的最新输出 $y^i$ 以对齐 $c_i$ 来策划。
这会从用于批评的原则 $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$ 得到一系列指令变体 $\{y^0, y^1, \cdots, y^n\}$。
最终数据点是 prompt $x$ 以及最终补全 $y^n$，其中 $n$ 取某个值。

偏好数据以类似但更简单的方式构造：把 $\mathcal{C}$ 的一个原则子集作为反馈模型的上下文。
反馈模型会看到一个 prompt $x$、一组原则 $\{c_0, \cdots, c_n\}$，以及来自此前 RLHF 数据集、被标为答案 (A) 和 (B) 的两个补全 $y_0$ 和 $y_1$。
新的数据点通过让语言模型选择哪个输出 (A) 或 (B) 质量更高且更符合所述原则来生成。
在早期模型中，这可以通过用 `The answer is: ` 提示模型，然后查看哪个 token（A 或 B）概率更高来完成；但现在更常见的做法，是让模型解释推理过程然后选择答案——这通常被称为一种 generative reward model [@mahan2024generative]。

### CAI 延伸阅读

Constitutional AI 有许多相关研究方向和扩展，但其中很少被明确记录为 RLHF 和后训练配方中的改进。

- OpenAI 发布了 Model Spec [@openai2024modelspec]，这是一份说明其模型预期行为的文档，并表示他们正在探索模型直接引用该文档进行对齐的方法（这可以看作 CAI 的近邻）。OpenAI 持续更新其 spec，并使用一种称为 Deliberative Alignment 的方法训练 o1 等推理模型 [@guan2024deliberative]，使模型在引用这些安全或行为政策的同时完成对齐。
- Anthropic 继续在模型训练中使用 CAI，更新 Claude 使用的 constitution [@Anthropic2023ClaudesConstitution]，并实验人口集体如何在模型原则上收敛，以及当外部群体自行创建原则并与 Anthropic 共享以训练模型时，这会如何改变模型行为 [@ganguli2023]。
- 开源社区探索了把 CAI 复制到开放数据集上的做法 [@Huang2024cai]，也探索了在 LM 之间创建对话数据 [@lambert2024self]。
- 其他工作使用了 principle-driven preferences 或反馈，并配合不同优化方法。
Sun et al. 2023 [@sun2023principledriven] 使用原则作为奖励模型的上下文，这些奖励模型被用于训练 Dromedary 模型 [@sun2024salmon]。
Glaese et al. 2022 [@glaese2022improving] 使用原则提高 RLHF 流程中人类判断的准确性。
Liu et al. 2025 [@liu2025inference] 训练奖励模型在推理时生成自己的原则，并用这些原则给出最终分数。
Franken et al. 2024 [@franken2024self] 把遵循原则表述为一个互信息最大化问题，使预训练模型可以在没有标签的情况下学习。

## Rubrics：用于训练的 Prompt-Specific AI Feedback

AI 反馈在训练中的角色于 2024 年末和 2025 年进一步增长，当时该领域正在寻找用可验证奖励扩展强化学习的路径（见第 7 章）。
rubric 的思想出现，是为了给那些没有明确可验证答案的 prompt 提供近似可验证的标准。
这将允许模型尝试为一个问题生成多个答案，并通过 RL 向最好的答案更新。
这个想法与本章讨论的其他方法密切相关，并且很可能是在整个行业的 LLM judge 和合成数据实践改进之后开始发挥作用的。
现在，把 rubric 作为奖励的 RL 已经被证明可以在科学推理或事实性等技能上提供有意义的改进 [@gunjal2025rubrics; @viswanathan2025checklists; @rezaei2025onlinerubrics; @liu2025openrubrics]。

下面展示了一个 rubric 示例及其关联 prompt [@liu2025openrubrics]：
```text
**Prompt**: As a museum curator, can you suggest five obscure artifacts that would be perfect for a "Mysteries of the Ancient World" exhibit? Each artifact should come from a different culture and time period, with a brief description of their historical significance and mysterious origins. These artifacts should leave visitors wondering about the secrets and lost knowledge of our past. Thank you for your expertise in bringing this exhibit to life.

** Rubric**:
1. The response includes exactly five distinct artifacts as requested. [Hard Rule]
2. The response ensures each artifact originates from a different culture and time period. [Hard Rule]
3. The response provides a brief description of each artifact's historical significance. [Hard Rule]
4. The response provides a brief description of each artifact's mysterious origins or unexplained aspects. [Hard Rule]
5. The response conveys a sense of intrigue and mystery that aligns with the theme of the exhibit. [Hard Rule]
6. The response clearly and accurately communicates information in a well-organized and coherent manner. [Principle]
7. The response demonstrates precision and clarity by avoiding unnecessary or irrelevant details. [Principle]
8. The response uses informative and engaging language that stimulates curiosity and critical thinking. [Principle]
9. The response shows thoughtful selection by ensuring each example contributes uniquely to the overall theme without redundancy. [Principle]
10. The response maintains consistency in style and format to enhance readability and comprehension. [Principle]
```

`[Hard Rule]` 和 `[Principle]` 是用于表示某条反馈优先级的特定标签。也可以使用其他表示重要性的方法，例如简单的优先级数字。

Rubric 生成通常是在训练数据中逐 prompt 完成的，这会在准备阶段累积有意义的合成数据成本。
为缓解这一点，通常会先按领域应用一个通用 rubric 作为起点，然后由监督语言模型按 prompt 分配细粒度 rubric 分数，以指导训练反馈。
下面展示了一个为科学任务生成 rubric 的示例 prompt [@gunjal2025rubrics]：

```text
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry.
Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one of these domains.
Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls.
Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7-20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for.

For example:
- Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
before the event equals the total mechanical energy after the event.
- Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
energy plus initial potential energy equals final kinetic energy plus final potential energy.
- Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
energy, to illustrate how energy shifts within the system.
- Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
conservation of mechanical energy.

3. weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
- If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.
Now, given the question and reference_answer, generate the rubric as described.
The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance.
```

另一个更简单的例子如下 [@rezaei2025onlinerubrics]：

```text
SYSTEM:
You generate evaluation rubrics for grading an assistant's response to a user prompt.

Rubric design rules:
- Each criterion must be atomic (one thing), objective as possible, and written so a grader can apply it consistently.
- Avoid redundant/overlapping criteria; prefer criteria that partition different failure modes.
- Make criteria self-contained (don't rely on unstated context).
- Include an importance weight for each criterion.

Output format (JSON only):
{
  "initial_reasoning": "<brief rationale for what matters for this prompt>",
  "rubrics": [
    {
      "reasoning": "<why this criterion matters>",
      "criterion": "<clear, testable criterion>",
      "weight": <integer 1-10>
    },
    ...
  ]
}

USER:
User prompt:
{prompt}

Generate the rubric JSON now.
```

可以看到，这些 prompt 可以非常详细，并且会针对训练设置进行调优。

带有 rubric 的 RL 训练将会继续演化，超出其在指令遵循 [@he2025advancedif]、deep research [@shao2025drtulu]、评估 deep research agents [@sharma2025researchrubrics]，或长文本生成 [@ruan2025expertlongbench] 等方向上的早期应用。
