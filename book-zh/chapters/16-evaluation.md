<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "正则化"
prev-url: "15-regularization"
page-title: 评估
search-title: "第 16 章：评估"
meta-description: "用于衡量 RLHF、后训练、奖励模型、开放式生成和模型行为的评估方法。"
next-chapter: "塑造模型角色与产品"
next-url: "17-product"
---

# 评估

评估是一组技术，用来理解本书详述的训练过程的质量和影响。
评估通常通过 benchmark 表达（流行 benchmark 的例子包括 MMLU、GPQA、SWE-Bench、MATH 等），这些 benchmark 是离散的问题集合或环境，旨在测量模型的某种特定性质。
评估方法一直在演化，因此我们介绍 RLHF 中近期几个评估阶段，以及会延续到语言建模未来的共同主题。
理解语言模型评估，尤其是后训练评估的关键在于：当前流行的评估范式反映了流行训练最佳实践和目标。
虽然有挑战性的评估会推动语言模型进入新领域，但大多数评估都是围绕为新模型构建有用信号而设计的。

在许多方面，本章旨在呈现 RLHF 早期历史中流行评估范式的一系列片段，让读者理解其中共同主题、细节和失败模式。

RLHF 和后训练的评估在早期历史中经历了几个不同阶段：

1. **早期聊天阶段**：早期用 RLHF 或偏好调优训练的模型，目标是捕捉模型聊天性能的评估，尤其是相对于 GPT-4 等已知强模型的表现。早期例子包括 MT-Bench [@zheng2023judging]、AlpacaEval [@dubois2024length] 和 Arena-Hard [@li2024crowdsourced]。这些 benchmark 用 LLM-as-a-judge 替代人类评估者，使用 GPT-4 等模型为回复打分，这是一种以较低成本扩展人类评估标准的方式（见第 12 章）。模型评估范围较窄，如今这些被认为是“chat”或“instruction following”领域。
2. **多技能时代**：随着时间推移，常见实践表明 RLHF 可用于改善不止聊天一种技能。例如，Tülu evaluation suite 包含知识任务（MMLU [@hendrycks2020measuring]、PopQA [@mallen2023llm_memorization]、TruthfulQA [@lin2021truthfulqa]）、推理（BigBenchHard [@suzgun2022challenging]、DROP [@dua2019drop]）、数学（MATH [@hendrycksmath2021]、GSM8K [@cobbe2021gsm8k]）、编码（HumanEval [@chen2021codex]、HumanEval+ [@evalplus]）、Instruction Following [@zhou2023instructionfollowingevaluationlargelanguage] 和 Safety（许多评估的组合）。这反映了后训练被接纳为超越安全和聊天的多面解决方案的领域。
3. **推理与工具**：当前后训练时代由对困难推理和工具使用问题的关注定义。这些包括难得多的知识密集型任务，例如 GPQA Diamond [@rein2023gpqa] 和 Humanity's Last Exam [@phan2025hle]；复杂软件工程任务，例如 SWE-Bench+ [@aleithan2024swebenchplus] 和 LiveCodeBench [@jain2024livecodebench]；以及近期 AIME 竞赛所代表的困难数学问题。

除此之外，还会演化出新的领域。
随着 AI 成为更加工业化的领域，评估的激励正在变化，并变得涉及多方利益相关者。
自 ChatGPT 发布以来，Scale Leaderboard [@scale2024seal] 这样的私有评估、Arena [@chiang2024chatbot] 这样的社区驱动评估，以及 Artificial Analysis 和 Epoch AI 等第三方评估公司都大量出现。
本章会包含一些细节，用来映射这些评估如何实现和理解。

## Prompt 格式

对语言模型进行 **prompting** 本身是一个简单且相当自然的动作，但它也被认为是一种可以练习和打磨的手艺或艺术 [@schulhoff2024prompt]。
Prompt 是为语言模型组织信息和上下文的方式。
对于常见交互，prompt 相对基础。
对于高级场景，精心构造的 prompt 可能决定某个一次性 use-case 的成败。

在评估中，prompting 技术可能对模型性能产生显著影响。
一些 prompting 技术，例如下面讨论的格式化，可能让模型性能从 60% 掉到接近 0。
类似地，改变 prompt 也可以帮助模型在训练期间学得更好。
通俗地说，良好 prompting 能给人一种在使用未来模型的主观体验，释放正常使用之外的性能。

Prompting 带来的收益通常小于改进数据或训练算法等核心领域，但在最终产品中可能很可观。
更大的启示是：训练一个强大的领先模型时，弄坏它并让性能暴跌，比找到一点额外性能更容易。

用现代语言模型做好 prompting，可能需要为模型准备整份报告供其响应（通常包含数千个生成文本 token）。
这种行为是语言模型性能如何被测量和理解的一系列变化的下游结果。

### Few-Shot Prompting 与 Log-Likelihood Scoring

早期语言模型只被当作智能自动补全使用。
为了以更开放的方式使用这些模型，人们会向模型展示多个示例，然后给出一个不完整短语作为 prompt。这被称为 few-shot 或 in-context learning [@brown2020language]，当时尚未涉及 instruction tuning 或 RLHF。
在流行评估中，这看起来像：

```text
# Few-Shot Prompt for a Question-Answering Task
You are a helpful assistant. Below are example interactions to guide your style:

### Example 1
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

### Example 2
User: "Who wrote the novel '1984'?"
Assistant: "George Orwell wrote '1984.'"

# Now continue the conversation using the same style.
User: "Can you explain what a neural network is?"
Assistant:
```

这里有多种方式评估答案。如果考虑 MMLU 风格的问题，模型必须在多个答案之间选择：

```text
# Few-Shot Prompt

Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4. What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous on a closed interval [a,b], then f must attain both a maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle's Theorem

Correct Answer:
```

为了让语言模型在这里提供答案，可以基于某些采样参数生成一个 token，然后看答案 A、B、C 或 D 是否正确（上面的这种格式由 [@robinson2023leveraging] 提出）；也可以查看每个 token 的 log-probabilities，并在正确答案更可能时把任务标为正确。

我们稍微深入这些评估细节。
前一种方法通常称为单次尝试的 exact match，或在聚合多个样本时称为 majority voting（pass@k 是编码评估中的类似指标，其中测试函数正确性）；后一种方法称为（条件）log-likelihood scoring，其中 conditioning 是 prompt。
核心区别是：从底层概率分布采样会自然引入随机性，而模型对其 token 输出的 log-probabilities 是静态的（忽略微小数值差异时）。

Log-likelihood scoring 有两种潜在实现：第一，可以看字母 (A) 的概率；也可以看答案 “The Mean Value Theorem” 的概率。
两者都是允许的指标，但预测答案字母远比完整且可能多 token 的答案概率简单。
Log-likelihood scoring 在预训练评估中更常见，因为模型缺少 exact match 所需的问答格式；而 exact match 在后训练中是标准做法 [@teamolmo2025olmo3]。

Exact match 也有不同问题，例如要求严格格式后缀（如 `The answer is:`），或使用正则表达式在生成文本任意位置检测答案（如查找 `(C)` 或答案字符串本身）。
如果评估格式与模型生成方式不匹配，分数可能暴跌。
语言模型评估最好在格式不是瓶颈时进行，这样才能测试模型的完整能力。
实现格式无关评估需要大量努力和调试才能做好，实践中相当少见。

回到评估历史。
无论使用上述哪种设置，few-shot prompting 的一个常见挑战是模型不会遵循格式，这会被计为错误答案。
设计评估领域时，上下文中示例数量通常被视为一个设计参数，范围从 3 到 8 个或更多。

### Chain-of-Thought Prompting

在 few-shot prompting 的演化过程中，出现了为模型加入 chain-of-thought 示例的想法。
其形式是在 in-context 示例中写出推理过程，如下所示（后来被显式提示生成推理步骤取代）[@wei2022chain]：

```text
# standard prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...

# chain-of-thought prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They..
```

### Zero-Shot Instruction Following

随着语言模型变强，它们逐渐演化到 zero-shot evaluation，也就是 “zero-shot learners” [@wei2021finetuned]。
Finetuned Language Net (FLAN) 表明，在特定任务上微调的语言模型，作为现代 instruction tuning 的前身，可以泛化到训练中未见过的 zero-shot 问题 [@wei2021finetuned]（T0 [@sanh2021multitask] 中也有类似结果）。
这标志着 instruction fine-tuning (IFT) 的出现，它是 RLHF 和后训练的重要前身。
一个 zero-shot 问题如下：

```text
User: "What is the capital of France?"
Assistant:
```

从 2022 年开始，时间线开始纳入 InstructGPT 等关键早期 RLHF 工作。
伴随这些模型出现的核心能力和 use-case 转变，是更加开放式的使用。
随着使用更加开放，基于从模型采样的评估越来越流行，因为它反映实际使用；技术上，这可以称为 generation-based (exact-match) evaluation，但它没有同样明确的规范术语。
在 ChatGPT 之后直到近几年，RLHF 研究中仍然会使用一些多选评估，因为任何向常见实践的转变都需要相当长时间，通常要数年展开（例如这种评估：把 temperature 设为零，并采样字符 A、B、C 或 D）。

### 推理时代的评估 Prompt

随着推理模型在 2024 年末和 2025 年初兴起，模型行为的一个重大变化是每个答案前都加入了很长的 Chain-of-Thought (CoT) 推理过程。
这些模型不再需要用 [@kojima2022large] 提出的经典短语 “think step by step” 来提示。
评估实践的下一次演化，是带 chain of thought reasoning 的 generation-based (exact-match) evaluation（因此为了最佳性能，几乎总是 temperature 大于零）。

例如，在一些设置中，每个问题或类别都有专门设计的 prompt，用来从模型中引出行为。
Tülu 3 是一篇早期重要论文，详细介绍了用于多选题 CoT 作答的一些 prompt [@lambert2024t]。
下面是 MMLU 使用的一个示例 prompt。MMLU 是从单 token 答案采样转向长篇 CoT 加 exact match 答案检查的评估之一。

```text
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

这尤其在模型使用特殊格式区分 thinking tokens 与 answer tokens 时，迫使评估范式进行最近一次重大更新。
评估正在转向测试模型能否以生成式方式、带 chain-of-thought prompting 进行响应。

## 为什么许多外部评估比较不可靠

AI 公司模型公告中的语言模型评估，只能在很大误差条下与其他新闻稿比较。也就是说，略好或略差的模型应被视为等价，因为各家公司内部评估所用流程在模型之间并不受控，也没有被明确记录。
例如，在 Olmo 3 项目中，作者发现，在推理模型时代，大多数后训练评估在评估设置保持不变时，标准差介于 0.25 到 1.5 分之间 [@teamolmo2025olmo3]；更大的分数变化可能来自使用不同 prompt 或采样参数。
实验室会在训练期间围绕评估 hillclimb，以让模型更有用，传统上会混合使用训练集、开发集（也称 validation set）和留出评估集（也称 test set）。
Hillclimbing 是一个口语化术语，用来描述让模型在一组目标 benchmark 上逐步变好的实践。
对于社区用来比较领先模型的公开评估，不可能知道哪些被用于训练，哪些被留出测试。

随着评估分数成为公司营销方案的核心组成部分，其在公司内部的实现已经发生漂移。
有传闻称主要 AI 实验室会为 GSM8K 或 MATH 等重要评估使用 “custom prompts”。
这些实践演化很快。

语言模型评估栈被视为营销，是因为评估没有硬性的真值来源。
前沿实验室内部发生的是：评估套件被调试到适合其内部需求。
当结果被分享时，我们拿到的是某个实验室为其模型得到的一组数字，但不是这个函数的全部输入。
这些输入是非常敏感的配置，而且 OpenAI、Meta、Anthropic 和 Google 各不相同。
即使是完全开放的评估标准，也很难保证可复现性。

把精力集中在自己的模型上，是接近可重复评估技术的唯一方式。
营销背后也有良好意图，起点是技术团队。

比较多个实验室评估时的另一个混淆例子，是把 inference-time scaling 加入评估比较。
Inference-time scaling 表明，模型可以通过在推理时使用更多 token 来提升性能。
因此，按推理 token 总数控制评估分数很重要，但这还不是常见实践。

根据后训练中数据格式的不同，模型在不同评估格式上的表现会有显著差异。
例如，两个流行开放数学数据集 NuminaMath [@li2024numinamath] 和 MetaMath [@yu2023metamath] 会因答案格式的细微差异而在训练中相互冲突：Numina 把答案放在 `\boxed{XYZ}` 中，MetaMath 把答案放在 `The answer is: XYZ` 之后；同时训练二者可能比只训练其中一个更差。
强模型会被训练到能够处理多种格式，但它们通常仍有一个最强格式。

最终，关于闭源模型评估现状，我们得到几个关键点：

- 我们不知道，也不一定拥有实验室正在 hillclimb 的关键测试集，因此某些评估只是代理。
- 前沿模型推理正因特殊 system prompts、特殊 token 等变得更复杂，而我们不知道它如何影响评估。
- 我们不知道用于数值报告闭源评估的所有格式和细节。

所有这些动态，加上过去几年 AI 模型进展非常迅速，会产生类似 @fig:benchmark-saturation 中那样著名的图：每个时代流行的 benchmark 都会很快被解决。
描述每个 benchmark 层面这种动态的常见术语是 saturation。
当每个 benchmark 接近 100% 时，模型进展开始放缓，因为只剩下更难的（或在许多情况下被错误标注的）数据点，使它作为训练进展指标（或两个模型之间比较指标）变得不那么可靠。

![Epoch AI 的报告展示主要 AI 评估如何随时间快速饱和（saturation 指给定 benchmark 达到满性能，模型不再有有意义信号）。License CC-BY.](images/benchmark-performance.jpeg){#fig:benchmark-saturation}

## 实验室实际如何在内部使用评估来改进模型

今天，对前沿语言模型的评估既是一门艺术，也是一门科学；如果要精确规定不同团队如何使用评估来理解前沿语言模型，那本身就会成为另一本教材。

不同团队会选择不同评估来保持独立性，即让它们成为真正的 test set，但没人披露他们选择了哪些。
例如，流行推理评估 MATH 和 GSM8K 都有训练集，其中的 prompts 很容易被用来改善性能。
使用来自同一分布的 prompts 来改善性能，与通过训练一般数学数据来泛化到这些任务，是非常不同的。

事实上，这些*训练集*包含非常高质量的数据，因此模型可以从训练中受益。
如果这些公司*没有*把对应评估作为核心跟踪指标，训练评估集可能是一个现实选择，因为高质量数据是模型开发的主要限制因素。

领先 AI 实验室会聚焦少数关键评估进行 hillclimb，并在最后报告核心公开集合上的分数。
关键点是，它们用于跟踪进展的一些评估，例如 GPT-4 report [@achiam2023gpt] 中用于 scaling 的 cross-entropy loss prediction 数据集，通常并不公开。

后训练评估与人类评估高度相互依赖。
生成式语言模型的人类评估会产生 Elo 排名（在 Anthropic 早期论文如 Constitutional AI 中很流行），奖励模型的人类评估则显示一致性。
这些也可以通过在 A/B testing 窗口中向用户提供两个不同模型来获得（如[偏好数据](https://rlhfbook.com/c/11-preference-data)一章所讨论）。

它们选择关注的有限评估集合，在评估与训练之间形成紧密联系。
曾经有一段时间，一个重点评估是 MMLU。
由于社区对科学能力的关注上升，GPQA 在推理模型出现期间非常流行。
实验室会修改评估，使其更适合自身需求，例如 OpenAI 发布 SWE-Bench-Verified [@openai2024swebench]。
每个前沿实验室还构建或购买了许多公众无法访问的内部评估。

内部改进评估对下游训练的关键能力，是**提高比较训练运行时的统计功效**。
通过改变评估，这些实验室会降低优先信号上的噪声，从而做出更有信息量的训练决策。

现代语言模型训练栈中后训练的复杂性进一步加剧了这一点。
今天评估语言模型涉及适量生成 token（而不只是查看答案的 log probabilities），因此需要计算开销。
人们普遍承认，前沿实验室会使用一些小技巧来提升许多任务的性能，最常见解释是为某些评估使用一次性 prompts。

## 污染

当前语言模型实践中的一个主要问题（不局限于 RLHF 和后训练）是，在训练中有意或无意使用来自评估数据集的数据。
这称为*数据集污染*（*data leakage* 的一种形式），相应的避免实践称为*去污染*。
为了对数据集去污染，需要在训练集和测试集上搜索匹配，例如词/子词 token 的 n-gram 重叠，或固定长度字符子串匹配（例如 50 个字符）[@singh2024evaluation]。
数据可能以许多方式被污染，但最常见来源是多个阶段训练数据从 Web 抓取。
Benchmark 通常列在会被爬取的公共 Web 域名上，或者用户把问题输入模型，而这些内容随后可能进入未来模型的候选训练数据。

例如，在 Tülu 3 评估套件去污染过程中，作者发现流行开放数据集受到流行 RLHF 评估污染 [@lambert2024t]。
这些重叠包括：UltraFeedback 被 TruthfulQA 污染，Evol-CodeAlpaca 被 HumanEval 污染，NuminaMath 被 MATH 污染，WildChat 被安全评估污染。
这些是通过从训练 prompt 到评估集中精确 prompt 的 8-gram 重叠发现的。

其他情况下，模型被发现训练过与 benchmark 非常接近的数据，例如保持数学题文字不变、只改变数字。这可能导致后训练范式中出现异常行为，例如模型在随机奖励上用 RL 训练后 benchmark 反而提升；这种人为设置只有在模型存在某些数据污染时才应增加性能。
这种基座模型污染无法被精确证明为何模型会以某种方式表现，已经成为许多基于 Qwen 2.5 和 Qwen 3 基座模型的早期 RLVR 工作中的重要混杂变量 [@shao2025spurious] [@wu2025reasoning]。

为了理解不披露或发布训练数据的模型污染情况，会创建与原始问题略有扰动的新版本 benchmark（例如 MATH [@huang2025math]），用来观察哪些模型被训练到匹配原始格式或问题。
这些扰动 benchmark 上的高方差并不能确认污染，因为污染很难证明。相反，它可能表明模型训练时针对某种特定格式，而该格式未必能迁移到真实世界性能。


## 工具

有许多开源评估工具可供选择。
其中包括：

- UK Safety Institute 的 Inspect AI [@inspectAI2024]，
- HuggingFace 的 LightEval [@fourrier2023lighteval]，它支撑了 Open LLM Leaderboard [@open-llm-leaderboard-v2]，
- EleutherAI 的 evaluation harness [@gao2023evalharness]，建立在其 GPT-Neo-X 模型基础设施之上（它包含一套很好的 GPT-3 时代评估设置和配置）[@gpt-neox-20b]，
- Ai2 基于 OLMES 的库 [@gu2024olmes]，
- Stanford Center for Research on Foundation Models 的 HELM [@liang2023helm]，
- Mosaic（现 Databricks）的 Eval Gauntlet [@mosaicml2024gauntlet]，以及更多。
