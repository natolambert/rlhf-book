<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "强化学习"
prev-url: "06-policy-gradients"
page-title: 推理与推理时扩展
search-title: "第 7 章：推理与推理时扩展"
meta-description: "后训练中的推理训练与推理时扩展，包括 RLVR 和 thinking models。"
next-chapter: "直接对齐算法"
next-url: "08-direct-alignment"
lectures:
  - video: "https://www.youtube.com/watch?v=o4AB5xHIDdM&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=7"
    label: "第 5 讲：推理模型的兴起"
---

# 推理与推理时扩展

从 2024 年末到 2025 年，并且很可能延续到未来，推理模型和推理时扩展使语言模型性能出现了一次巨大跃迁。
推理时扩展指的是在生成阶段使用更多计算来提升模型性能，例如生成更长的推理链，或者采样多个回复。
经过训练、能够在回答前进行大量思考的语言模型，非常善于利用这一性质。
这些模型通过大量基于可验证奖励的强化学习（reinforcement learning with verifiable rewards, RLVR）训练而来 [@lambert2024t]，但仍然大量使用 RLHF。
本章将回顾 AI 社区如何逐步重新认识强化学习在语言模型中的潜力，介绍 RLVR 的基本原理，突出若干关键工作，并指出未来几年将定义这一领域的核心争论。

## RLVR 的作用

首先，在 2016 年的 Neural Information Processing Systems (NeurIPS) 会议上，Yann LeCun 首次提出了后来广为人知的蛋糕比喻，用来说明现代机器学习系统中的学习发生在何处：

> 如果智能是一块蛋糕，那么蛋糕主体是无监督学习，蛋糕上的糖霜是监督学习，而蛋糕上的樱桃是强化学习（RL）。

有了现代语言模型以及近期后训练栈的变化，这个类比如今基本完整了。
RLHF 是其先声，而面向推理模型的 RL，主要发生在数学、代码和科学主题上，则确认了这一点。
在这个类比中：

- 面向大规模互联网数据的自监督学习构成了蛋糕的大部分，尤其是从 FLOPs 计算开销来看；
- 后训练的开端，即面向指令的监督微调（SFT），把模型调到一个更窄的分布上；
- 最后，“纯粹”的强化学习（RL）是蛋糕顶上的樱桃。用于创建新型“推理”或“思考”模型的规模化强化学习，正是这个收尾部分，当然还需要 RLHF 的帮助；如我们将解释的那样，RLHF 并不被视为经典 RL。

这一小部分推理训练随着 **thinking models** 出现而浮出水面：这类模型结合了本书讨论的多种后训练技术，用偏好对齐配合可验证领域上的 RL 训练，大幅提升推理、编码和数学问题求解等能力。

这些模型的训练方法，即基于可验证奖励的强化学习（RLVR）[@lambert2024t]，流程与 RLHF 非常相似，但它把奖励模型变成可选项，转而使用一个打分函数：答案正确时返回正奖励，否则返回 0。

例如，可以比较 RLHF 和 RLVR 对回复打分时有多么不同。
在 RLHF 中，奖励模型必须评估主观质量：

> **Prompt**: Explain the concept of opportunity cost in economics.
>
> **Response**: Opportunity cost is the value of the next best alternative you give up when making a decision. For example, if you spend an hour studying instead of working, the opportunity cost is the wages you could have earned...

给这个回复打分，需要判断清晰度、准确性、完整性和有用性。这些质量都需要学习到的偏好，并且不存在一个确定的正确答案。

相比之下，RLVR 使用能够返回确定分数的验证函数。
以数学为例：

> **Prompt**: What is the sum of all prime numbers less than 20?
>
> **Response**: The prime numbers less than 20 are 2, 3, 5, 7, 11, 13, 17, and 19.
> Adding them: 2 + 3 = 5, then 5 + 5 = 10, then 10 + 7 = 17, then 17 + 11 = 28, then 28 + 13 = 41, then 41 + 17 = 58, and finally 58 + 19 = 77.
> The answer is $\boxed{77}$.
>
> **Verification**: `extracted_answer == 77` → Reward = 1

`\boxed{}` 记号来自数学排版惯例，它让答案抽取变得直接：一个简单的正则表达式就能从回复中取出最终答案，而不必关心模型如何得到它。
注意，也存在其他答案抽取方法，例如像上面一样使用短语 "The answer is: "，使用 `<answer>` 这样的特殊 token，或者使用 `####` 这样的分隔符。

对于代码生成，验证通常采用单元测试形式：

> **Prompt**: Write a Python function `fib(n)` that returns the nth Fibonacci number, where fib(0) = 0 and fib(1) = 1.
>
> **Response**:
> def fib(n):
>     if n < 2:
>         return n
>     return fib(n - 1) + fib(n - 2)
>
> **Verification (unit tests)**:
>
> assert fib(0) == 0   # base case
> assert fib(1) == 1   # base case
> assert fib(10) == 55 # larger value
> (All tests pass → Reward = 1)


单元测试是代码任务中很自然的验证函数：它们用已知的输入输出对执行模型给出的解法。
一种常见打分方式是做简单门控：如果所有断言都通过，奖励为 1；如果有任何失败，奖励为 0。
其他设置会按照通过测试的数量给予部分分数。
在这两个例子中，都不需要学习出来的奖励模型，而且大多数设置也不会使用奖励模型，因为在这些领域中模型对过度优化相对稳健；不过也可以把奖励模型作为线性奖励组合的一部分。

RLVR 背后的思想在 RL 文献中并不新鲜：根据答案是否正确来做梯度更新，几乎就是教科书式的强化学习定义。
把它应用到语言模型时，创新主要在于如何在微调过程中保持模型强大的通用能力。
在这个范围内，语言模型文献中还有很多相关思想，都是让模型从关于答案正确性的反馈中学习。

最初，在我参与并提出 RL with Verifiable Rewards (RLVR) 这一术语的工作中 [@lambert2024t]，这个方法本打算命名为 RL with Ground Truth rewards (RLGT)。
不过，RLVR 与单纯从 ground truth 答案中学习有细微差别。
在数学等领域中，可以用一个 ground truth 答案来验证解法，正如我们在上面看到的那样。
在其他领域，例如代码生成或精确遵循指令，即使正确解不止一个，也可以用检查函数（例如单元测试）验证答案。
RLVR 进展的核心，是拥有种类丰富、深度足够的可验证问题，即使确切解法并不是事先已知的。

![RLVR 表现为一个 RL 反馈回路。这里使用验证函数，而不是奖励模型。](images/rlvr-system.png){#fig:rlvr data-dark-src="images/rlvr-system-dark.png"}

最早成功部署这类训练的模型是 OpenAI 的 o1 [@openai2024o1] 和开放权重模型 DeepSeek R1 [@guo2025deepseek]。
此后不久，整个 AI 行业都把这种训练过程和模型风格列为优先事项。
这里的核心变化，与其说是这种 RL 设置全新出现，不如说是训练阶段的重新分配，以及不同模型行为优先级的变化。
推理模型带来了一个预期会扩展 RL 训练的时代。

至于这些模型表现出的行为，可以看下面这个例子：在 `Write me a short poem about a goldfish` 这个查询上，对比 DeepSeek V3 0325 和它的推理模型 DeepSeek R1。
DeepSeek V3 非常直截了当：

> **Response**: Goldfish Dreams
>
> In a bowl of glass so clear and round,
> A tiny goldfish swims without a sound.
> With scales that gleam like sunlight's kiss,
> He drifts along in watery bliss.
>
> A flick of fin, a flash so bright,
> Chasing bubbles in the light.
> Does he dream of oceans wide?
> Or is his little world enough---his pride?

而 DeepSeek R1 的体验不同。
它会先生成大量 thinking tokens 来构思回复，然后才回答用户。
答案看起来如下：

> **Response**: `<thinking>` Okay, the user wants a short poem about a goldfish. Let me start by brainstorming some goldfish-related imagery. Goldfish are often in bowls, right? So maybe start with the setting. Words like "gilded bowl" or "crystal globe" could work.
>
> Next, think about the movement of the goldfish. They swim smoothly, so words like "dart", "glide", or "swirl" come to mind. Maybe describe their fins as "silken" to emphasize delicacy.
>
> Then, consider the environment around the goldfish. Sunlight through water creates shimmering effects. Words like "sunbeams dance" or "ripples spin" might capture that. Including elements like castles or pebbles adds a playful touch.
>
> Themes? Maybe the contrast between the goldfish's small world and its perception of freedom. The bowl as a universe, tiny but complete. Ending with a sense of mystery or silent observation could give depth. Need to keep it concise, maybe four stanzas with a rhyme scheme. Let me check the rhythm and flow. Avoid clichés but keep it accessible. Alright, let's piece it together. `</thinking>`
>
> Gilded Dreams in Liquid Glass
>
> A gilded bowl, a crystal globe,
> Where tiny tides in silence flow---
> A flash of orange, a whispered robe,
> You dart through sunlit indigo.
>
> Your world is small, yet boundless too---
> Each pebble, castle, weed unfurled.
> The surface bends the sky to blue,
> And ripples spin your liquid world.
>
> You glide through myths we'll never know,
> A silent monarch, veiled in mist.
> Do dreams drift where the bubbles go?
> Or memories in amethyst?
>
> We watch you trace your endless sphere---
> A universe within a tear.

`<thinking>` 标签中的 token 就是模型的推理。
对于更复杂的问题，推理阶段可能在产出答案前消耗数千个 token。
因此，长上下文语言模型是高级推理行为的前提之一，但这不是本章重点。

关于 *这种训练如何起作用*，核心直觉是：对于给定模型，我们反复执行以下循环：

1. 对多个问题采样多个答案；
2. 朝正确答案的方向做梯度步；
3. 重复，并反复访问同一批数据。

令人惊讶的是，这种极其简单的方法，如果配合精心设计的数据分布和稳定的训练基础设施，会帮助模型通过一遍又一遍地回访同样的问题来学习。
更令人惊讶的是，在这些训练问题上的改进会泛化到模型从未见过的问题以及某些领域！

这种简单方法使模型能够在行为空间中进行轻量搜索，而 RL 算法会提高那些与正确答案相关的行为的概率。

## 新型推理模型的起源

这里我们概述导致 2025 年推理模型爆发的高层趋势。

### 为什么 RL 现在起作用了？

尽管曾经有大量观点认为“RL 还不起作用”[@irpan2018deep]，也有论文详细讨论 RL 的深层可复现性问题 [@henderson2018deep]，但该领域克服了这些问题，并找到了高影响力应用。
本书中讨论了一些例子，例如 ChatGPT 的 RLHF 和 DeepSeek R1 的 RLVR，但还有很多其他应用，包括改进芯片设计 [@mirhoseini2020chip]、掌握视频游戏 [@schrittwieser2020mastering]、自动驾驶 [@cusumano2025robust] 等。
语言模型上以 RL 为中心的训练起飞，表明这个研究领域的许多基础问题都取得了进展，包括：

- **RL 的稳定性可以被解决**：在 RL 的整个发展历程中，限制其采用的因素一直是稳定性。这体现在两个方面。第一，学习过程本身可能很反复，不一定总是有效。第二，训练本身已知比标准语言模型训练更脆弱，更容易出现 loss spike、崩溃等问题。如今，大量新模型发布都在预训练基础模型之上使用这种带有可验证奖励的 RL 训练风格，学术界也出现了大量跟进。进入 RL 的技术门槛处于历史低位。

- **开源版本已经“存在”**：许多用于用 RLVR 及相关技术训练语言模型的工具已经存在。
例子包括 TRL [@vonwerra2022trl]、Open Instruct [@lambert2024t]、veRL [@sheng2024hybridflow] 和 OpenRLHF [@hu2024openrlhf]，其中许多工具都建立在 RLHF 与后训练早期阶段的优化之上。工具可获得性正在促成一批规模庞大且加速增长的研究。

多项资料都指出，用于推理的 RL 训练大约只有在 2024 年以后出现的领先模型上才变得可行，这说明模型在推理训练之前需要具备一定水平的底层能力。

### RL 训练与推理时扩展

用强化学习训练来引出推理行为，并提升可验证领域上的性能，与推理时扩展的思想密切相关。
推理时扩展，也称 test-time scaling，是一大类在推理阶段使用更多计算、从而在下游任务上表现更好的方法。
在 DeepSeek R1 和 OpenAI o1 发布之前，推理时扩展方法就已经被研究过；而这两个模型极大地推动了行业对 RL 训练本身的投入。
例子包括价值引导采样 [@liu2023don]，或结合答案抽取的重复随机采样 [@brown2024large]。
除此之外，推理时扩展还可用于改进链式思维解题之外的更多 AI 训练方法，例如让奖励模型深入考虑选项 [@ankner2024critique] [@liu2025inference]。

RL 训练是利用推理时扩展定律的一条短路径，但从长期看，我们会有更多方法引出获得最佳性能所需的推理时权衡。
用 RL 重度训练模型，通常会让它们在每次回复中生成更多 token，而且这种增长与下游性能提升强相关；虽然序列长度增长是默认现象，也确实存在一些研究明确关注在 *不依赖* 这种推理时扩展的情况下提升性能。
这与早期 RLHF 系统中的长度偏置形成了显著转变 [@singhal2023long]：早期人类偏好训练的副作用，是为了在偏好排序上取得边际收益而增加平均回复长度。

除了核心的 RL 训练模型之外，还有许多方法正在探索如何继续推进推理与推理时计算的边界。
由于这些方法演化很快，它们大多超出本书范围，但包括把更大的 RL 训练模型中的推理行为通过指令微调蒸馏到较小模型中 [@muennighoff2025s1]、组合更多推理调用 [@chen2024more] 等。
这里重要的是下游性能与生成 token 数增加之间的相关性；否则，这些计算只是在浪费能量。


### RLVR 的未来（超越推理）

在许多领域中，这些新型 RLVR 更贴近开发者的目标，因为它们关注的是性能而不是行为。
标准微调 API 通常使用参数高效微调方法，例如 LoRA（Low-Rank Adaptation，一种只训练少量新增矩阵而不是全部模型权重的参数高效方法，也称 parameter-efficient fine-tuning, PEFT），并在指令上做监督微调。
开发者传入 prompts 和 completions，模型通过更新参数来匹配这些 completions，从而让你数据中的特征在模型生成中更常出现。

RLVR 关注的是匹配答案。
给定查询和正确答案，RLVR 帮助模型学会生成正确答案。
标准指令微调通常只在数据上做 1 或 2 个 epoch 的损失更新，而 RLVR 之所以得名，是因为它会在同样少量数据点上做数百甚至数千个 epoch，让模型有时间学会新行为。
这可以看作是把基础模型版本中偶尔可用的正向行为强化成 RLVR 之后的稳健行为。

**语言模型 RL 训练的范围仍在扩大**：从基础科学层面看，o1 和 R1 带来的最大启示是，我们拥有了更多训练语言模型获得潜在有价值行为的方法。
研究者和工程师面前打开的门越多，我们就越有理由对 AI 的总体发展轨迹保持乐观。


## 理解推理训练方法

对推理的投入推动了模型学习遵循人类指令方式的一次重大演化。
这些配方仍然使用前面章节讨论过的常见组件，包括指令微调、基于人类反馈的强化学习，以及基于可验证奖励的强化学习（RLVR）；这些组件在第 3 章概述 DeepSeek R1 配方时已经讨论过。
核心变化在于使用更多 RLVR，并以不同顺序应用其他训练技术。传统上，对一个推理模型而言，核心训练步骤要么是一次大规模 RL 运行，要么是在另一个已经经历了大量 RLVR 训练的模型的 *输出* 上做大规模指令微调，这通常称为蒸馏。

### OpenAI o1 或 DeepSeek R1 之前的推理研究

在推理模型起飞之前，研究者已经投入了大量精力，试图理解如何训练语言模型，使其在可验证领域中表现更好。
下列工作的主要差别在于，它们的方法没有扩展到 DeepSeek R1 及后续模型所采用的规模，或者它们得到的模型为了更强的数学或编码能力而牺牲了整体性能。
这里纳入这些底层思想和动机，是为了更全面地呈现推理模型如何在这一研究版图中出现。

最早尝试在可验证领域训练语言模型的工作包括 self-taught reasoner (STaR) 系列 [@zelikman2022star] [@Zelikman2024QuietSTaRLM] 和 TRICE [@hoffman2023training]。二者都在 2022 和 2023 年使用 ground-truth 奖励信号来鼓励模型进行 chain-of-thought 推理。
STaR 有效地近似了策略梯度算法，但在实践中以不同方式过滤样本，并使用交叉熵度量而不是对数概率；Quiet-STaR 则在此基础上扩展，它让模型在尝试回答可验证问题之前先生成 token，这与近期推理模型的思想非常相关，并且有助于训练性能。
TRICE [@hoffman2023training] 也通过生成 traces，然后用一个受 Markov chain Monte Carlo 启发的自定义期望最大化算法优化，从而改进推理。
VinePPO [@VinePPO] 紧随其后，使用了更接近现代推理模型的设置。
VinePPO 使用基于 PPO 的算法，并以数学问题正确性作为二元奖励，在 GSM8K 和 MATH 上训练。
OpenAI o1 和 DeepSeek R1 之前的其他工作，使用代码执行作为训练反馈信号 [@gehring2024rlefgroundingcodellms] [@xu2024dpo]，或者使用验证来做定理证明；这里称为 Reinforcement Learning from Verifier Feedback, RLVF [@amit2024models]。
Tülu 3 扩展了这些方法：它用一个简单的 PPO trainer 奖励带有正确答案的 completions，最重要的是，同时保持模型在广泛评估套件上的整体性能。
Tülu 3 和现代推理训练技术中的二元奖励，可以与 STaR 的迭代方法或 Quiet-STaR 的 log-likelihood 奖励形成对比。

### 早期推理模型

@tbl:reasoning_list 汇总了 DeepSeek R1 之后的基础推理研究报告，其中一些附带开放数据和模型权重。

::: {.table-wrap}
| 日期        | 名称                        | 简述                                                                  | 开放权重 | 开放数据 |
|-------------|----------------------------|-----------------------------------------------------------------------|--------------|-----------|
| 2025-01-22  | DeepSeek R1 [@guo2025deepseek]             | 基于 RL 对 DeepSeek 升级，在数学和代码推理上大幅提升      |  Yes      | No   |
| 2025-01-22  | Kimi 1.5 [@team2025kimi]                  | 在中文/英文数据上扩展 PPO/GRPO；AIME 数学表现强            | No           | No        |
| 2025-03-31  | Open-Reasoner-Zero [@hu2025openreasonerzero]   | 完全开放地复现基础模型 RL      |  Yes      |  Yes   |
| 2025-04-10  | Seed-Thinking 1.5 [@seed2025seed]         | ByteDance RL pipeline，带动态 CoT gating                         | Yes     | No   |
| 2025-04-30  | Phi-4 Reasoning [@abdin2025phi4]          | 14B 模型；谨慎的 SFT→RL；擅长 STEM 推理                   | Yes      | No        |
| 2025-05-02  | Llama-Nemotron [@bercovich2025llamanemotron]   | 多尺寸“reasoning-toggle”模型                 |  Yes      |  Yes   |
| 2025-05-12  | INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained] | 首个公开记录的全球去中心化 RL 训练运行                    |  Yes      |  Yes   |
| 2025-05-12  | Xiaomi MiMo [@xia2025mimo]                | 从预训练到后训练的端到端推理 pipeline              | Yes          | No       |
| 2025-05-14  | Qwen 3 [@yang2025qwen3]                   | 类似 R1 的配方应用到新模型                    |  Yes      | No   |
| 2025-05-21  | Hunyuan-TurboS [@liu2025hunyuan]          | Mamba-Transformer MoE，自适应长/短 CoT                        | No           | No        |
| 2025-05-28  | Skywork OR-1 [@he2025skyworkor1]          | 避免 entropy collapse 的 RL 配方；在 AIME 上超过 DeepSeek           |  Yes      |  Yes   |
| 2025-06-04  | Xiaomi MiMo VL [@coreteam2025mimovltechnicalreport]                | 端到端适配推理 pipeline，使其包含多模态任务              | Yes          | No       |
| 2025-06-04  | OpenThoughts [@guha2025openthoughts]      | 从 QwQ-32B 蒸馏得到的公开 1.2M 样例指令数据集                    |  Yes      |  Yes   |
| 2025-06-10  | Magistral [@mistral2025magistral]         | 在 Mistral 3 上做纯 RL；多语言 CoT；小模型开源      |  Yes| No        |
| 2025-06-16 | MiniMax-M1 [@minimax2025minimax_m1] | 开放权重 456B MoE hybrid/Lightning Attention 推理模型；1M 上下文；使用 CISPO 做 RL；发布 40K/80K thinking-budget checkpoints | Yes | No |
| 2025-07-10 | Kimi K2 [@kimiteam2025kimik2]                            | 1T MoE（32B active），用 MuonClip (QK-clip) 保持稳定；15.5T token 预训练无 loss spike；多阶段后训练，结合 agentic data synthesis + joint RL；发布 base + post-trained checkpoints。                               | Yes          | No         |
| 2025-07-28 | GLM-4.5 [@zeng2025glm45] | 开放权重 355B-A32B MoE "ARC" 模型，带 thinking/non-thinking 模式；23T-token 多阶段训练 + 通过 expert iteration 和 RL 后训练；发布 GLM-4.5 + GLM-4.5-Air (MIT)。 | Yes | No |
| 2025-08-20 | Nemotron Nano 2 [@nvidia2025nemotronnano2]               | Hybrid Mamba-Transformer，用于较长 "thinking traces"；20T token FP8 预训练后压缩/蒸馏；明确发布多个 checkpoints 以及“大部分”预/后训练数据集。                                       | Yes          | Yes (most) |
| 2025-09-09 | K2-Think [@llm3602025k2think]                            | 参数高效数学推理系统：32B 开放权重模型，带 test-time scaling 配方；按发布材料定位为完全开放，包括训练数据/代码。                                                                       | Yes          | Yes        |
| 2025-09-23 | LongCat-Flash-Thinking [@mlcteam2025longcat]             | 560B MoE 推理模型；报告明确说明从 long-CoT cold start 到大规模 RL 的分阶段配方；开源发布。                                                                                                             | Yes          | No         |
| 2025-10-21 | Ring-1T [@ringteam2025everystepevolves]                  | 万亿规模 "thinking model"，聚焦 RL scaling；报告讨论 1T 规模扩展 RL 的瓶颈/解决方案，并发布开放模型。                                                                                                             | Yes          | No         |
| 2025-11-20 | Olmo 3 Think [@teamolmo2025olmo3]         | 完全开放的 "model flow" 发布：报告完整生命周期（阶段、checkpoints 和 data points），并把 Olmo 3 Think 32B 定位为旗舰开放 thinking model。                                        | Yes          | Yes        |
| 2025-12-02 | DeepSeek V3.2 [@deepseekai2025v32]                       | 开放权重 MoE 前沿推进，报告突出 attention efficiency changes、RL framework upgrades，以及用于 agentic/reasoning 性能的数据合成。                                                                             | Yes          | No         |
| 2025-12-05 | K2-V2 [@liu2025k2] | 从头训练的 70B dense "360-open" 模型；用 3-effort SFT-only 后训练实现可控 thinking。 | Yes | Yes |
| 2025-12-15 | Nemotron 3 Nano [@nvidia2025nemotron3nano]               | 30B-A3B MoE hybrid Mamba-Transformer；在 25T token 上预训练，并包含 SFT + 大规模 RL；明确说明随附权重 + recipe/code + 大部分训练数据。                                                                      | Yes          | Yes (most) |
| 2025-12-16 | MiMo-V2-Flash [@mimo2025flash] | 为速度优化的 309B MoE（15B active）：hybrid SWA/GA attention（5:1，128-token window）+ 轻量 MTP；27T token FP8 预训练；用 MOPD + 大规模 agentic RL 做 reasoning/coding 后训练。 | Yes | No |
Table: 2025 年重要推理模型技术报告汇总；这一年是 RLHF 中推理时扩展大规模兴起的第一年。 {#tbl:reasoning_list}
:::


### 训练推理模型的常见实践

本节详细介绍训练推理模型时，为最大化性能而常用的训练阶段编排方法和数据修改方法。

注意，这些论文可能使用了某个列出的技术但没有提及，而同类工作提到了它。因此，这些例子只是已知实现的一个子集，应作为参考，而不是关于最优配方的最终断言。

- **离线难度过滤**：RLVR 的一个核心直觉是，模型只能从存在梯度的样例中学习。如果 RLVR 的起始模型在某个问题上要么 100% 能解出，要么 0% 能解出，那么同一个 prompt 的不同 completions 之间就没有梯度，也就是说，在策略梯度算法看来所有策略都一样。许多模型在开始大规模 RL 前使用难度过滤，把训练问题限制在起点模型只有 20-80% 概率能解出的范围内。这类数据通常通过对训练集中每个 prompt 采样 N 个 completions（例如 16 个），并验证其中正确比例来收集。Seed-Thinking 1.5、Open Reasoner Zero、Phi 4、INTELLECT-2、MiMo RL、Skywork OR-1 等都使用过这种形式。
- **每批次在线过滤**（或贯穿训练的难度课程）：为了补充离线过滤、找到合适的问题来训练，另一个重要问题是：学习过程中应该以什么顺序把问题呈现给模型？为了解决这一点，许多模型会对 batch 中的问题做在线过滤，使用预构建课程/数据调度器，把更难的问题留到训练后期，或采用其他思想来提升长期稳定性。Kimi 1.5、Magistral、Llama-Nemotron、INTELLECT-2、MiMo-RL、Hunyuan-TurboS 等使用了相关思想。
- **移除 KL 惩罚**：随着推理模型的 RL 运行长度相对于 RLHF 训练显著增加，无论用总 GPU 小时、FLOPS 还是 RL steps 衡量都是如此，并且奖励函数变得不那么容易过度优化，许多模型移除了 KL 惩罚；这一惩罚原本用于约束 RL 学到的策略与训练开始时使用的基础模型相似。这允许模型在训练中进一步探索。RAGEN [@wang2025ragenunderstandingselfevolutionllm]、Magistral、OpenReasonerZero、Skywork OR-1 等都使用了这一做法。
- **放宽策略梯度裁剪**：GRPO 的新变体，例如 DAPO [@yu2025dapo]，提出修改 GRPO（或 PPO）中使用的双侧裁剪目标，以支持更好的探索。也有研究表明，当奖励不完美时，裁剪可能导致潜在的虚假学习信号 [@shao2025spurious]。RAGEN、Magistral、INTELLECT-2 等使用了这种按梯度方向设置不同范围的双侧裁剪。
- **Off-policy 数据（或完全异步更新）**：随着用 RL 解决任务所需的 completions 长度在更难问题上急剧增加，特别是回复长度的 *方差* 增大，经常出现极长回复的离群值，RL 运行中的计算资源可能闲置。为了解决这一点，训练正在转向异步更新，或者改变问题组织进 batch 的方式，以提升整体吞吐。Seed-Thinking 1.5、INTELLECT-2 等使用了部分到完全异步的（off-policy）数据。
- **额外格式奖励**：为了让推理过程可预测，许多模型会加入小的奖励，确保模型遵循正确格式，例如在答案前使用 `<think>...</think>`。DeepSeek R1、OpenReasonerZero、Magistral、Skywork OR-1 等使用了这一做法。
- **语言一致性奖励**：类似格式奖励，一些多语言推理模型使用语言一致性奖励，优先鼓励模型在推理时不切换语言，从而带来更好、更可预测的用户体验。DeepSeek R1、Magistral 等都使用了这类奖励。
- **长度惩罚**：许多模型在 RL 训练中使用不同形式的长度惩罚，用来稳定长期学习过程，或者缓解困难问题上的过度思考。例如，Kimi 1.5 会逐步延长目标长度来对抗过度思考，同时在难度课程中保持较高训练准确率；INTELLECT-2 则全程使用一个小的长度惩罚。逐步扩展训练序列长度，可以通过迫使模型先在更有限的 thinking budget 下有效推理，再过渡到更长训练，使模型能够在更复杂问题上高效使用这些行为，从而缓解过度思考。其他模型还使用 overlong filtering 等相关实现来提升吞吐。
- **损失归一化**：关于原始 GRPO 算法的 per-group 归一化项可能引入长度或难度偏置，已有一些讨论，见策略梯度章节或 [@liu2025understanding]。因此，Magistral 或 MiMo 等模型选择在 batch 层面而不是 group 层面对 losses 或 advantages 做归一化。
- **并行 test-time compute scaling**：把多个并行、独立采样 rollouts 的答案组合起来，相比只使用单个 rollout 的答案，可以带来显著提升。并行 test-time compute scaling 最朴素的形式，如 DeepSeek-R1、Phi-4 等所采用的，是把多数 rollouts 返回的答案作为最终答案。更高级的技术是训练一个打分模型，从并行 rollouts 的答案中选择最佳答案。截至 2026 年，这项技术在开放、记录充分的推理模型配方中尚未普及，但它在 Claude 4 公告 [@anthropic2025claude4] 中被提到，并用于 DeepSeek-GRM [@liu2025inference]。

除了这些常见技术外，也有许多共同发现说明，推理训练可以在不牺牲附属能力的情况下创建有用模型：

- **纯文本推理提升多模态性能**：Magistral、MiMo-VL 等发现，先训练多模态模型，然后在多模态训练之后进行纯文本推理训练，可以 *提升* 最终模型的多模态性能。
- **通过 system prompt 切换推理**（或长度控制）：Llama-Nemotron、Nemotron Nano、Qwen 3、SmolLM 3 等使用特定 system prompts（可能结合长度受控的 RL 训练 [@aggarwal2025l1]），让用户能够切换 thinking length 的开/关。其他开放模型，例如 OpenAI 的 GPT-OSS 和 LLM360 的 K2-V2 [@liu2025k2]，在 system prompt 中采用 low-medium-high reasoning effort 设置，但关于这类行为的训练方法记录并不充分。

## 展望

推理模型版图的演化速度超过了近年 AI 研究中的任何领域，本章列出的一些常见实践不可避免会被新技术取代。

若干工作正在系统性地理解推理训练为何有效。
Olmo 3 Think [@teamolmo2025olmo3] 是目前对推理模型完整训练生命周期最全面的开放记录之一，它为研究社区提供了每个阶段的 checkpoints 和数据，并以一次在 220 张 GPU 上持续近 4 周的训练运行收尾。
类似地，关于理解推理 RL 扩展性质的工作 [@khatri2025art] 正开始把计算、数据和性能之间的关系形式化；这些关系过去主要依赖实践者直觉。

仍然清楚的是，强化学习已经从蛋糕比喻中“顶上的樱桃”升级为前沿模型训练中的承重组件。
本章围绕 RLVR 思想介绍的这些较小技术，包括难度过滤、格式奖励等，并不是最终答案，但它们代表了当前领域关于如何从语言模型中引出推理的最佳理解。
下一代方法很可能面貌不同，但会建立在这里奠定的基础之上。
