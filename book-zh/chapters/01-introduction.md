<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "首页"
prev-url: "https://rlhfbook.com/"
page-title: 导论
search-title: "第 1 章：导论"
meta-description: "从第一性原理介绍 RLHF：它如何改变语言模型，以及它如何成为现代后训练的一部分。"
next-chapter: "RLHF 简史"
next-url: "02-related-works"
lectures:
  - video: "https://www.youtube.com/watch?v=MMDNaeIFVy8&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "Lecture 0: Prerequisites"
  - video: "https://www.youtube.com/watch?v=o6l6tJQgUg4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "Lecture 1: Overview (Chapters 1-3)"
---

# 导论

基于人类反馈的强化学习（reinforcement learning from human feedback, RLHF）是一种把人类信息纳入 AI 系统的技术。
RLHF 最初主要是为了解决那些难以明确指定目标的问题。
当系统被设计成直接供人类使用时，这类问题会不断出现，因为个体偏好往往难以完整表达。这涵盖了数字系统中内容和交互的每一个领域。
RLHF 的早期应用常见于控制问题，以及强化学习（reinforcement learning, RL）的其他传统领域；这些领域的目标是优化某种具体行为来完成任务。
RLHF 这个领域的核心起点可以概括为："我们能否只用基础的偏好信号来引导优化过程，从而解决困难问题？"
RLHF 最广为人知，是因为 ChatGPT 的发布以及随后大语言模型（large language models, LLMs）和其他基础模型的快速发展。

## RLHF 的三个步骤

RLHF 的基本流程包含三个步骤。
第一，需要训练一个能够遵循用户问题的语言模型（见第 4 章）。
第二，需要收集人类偏好数据，用于训练表示人类偏好的奖励模型（见第 5 章）。
最后，可以用所选择的 RL 优化器来优化语言模型：从模型中采样生成结果，再用奖励模型对这些结果打分（见第 3 章和第 6 章）。
本书会详细讨论这一流程中每个步骤的关键决策和基础实现示例。

RLHF 已经被成功应用到许多领域，并且随着技术成熟，其复杂度也不断提高。
早期具有突破性的 RLHF 实验被用于深度强化学习 [@christiano2017deep]、摘要生成 [@stiennon2020learning]、遵循指令 [@ouyang2022training]、为问答解析网页信息 [@nakano2021webgpt]，以及 "alignment" [@bai2022training]。
早期 RLHF 配方的概要见 @fig:rlhf-basic。

![早期三阶段 RLHF 流程示意图：先做 SFT，再训练奖励模型，最后进行优化。](images/rlhf-basic.png){#fig:rlhf-basic}

在现代语言模型训练中，RLHF 是后训练（post-training）的一个组成部分。
后训练是一套更完整的技术和最佳实践，用于让语言模型对下游任务更有用 [@lambert2024t]。
后训练可以概括为一个多阶段训练过程，其中使用三类优化方法：

1. 指令微调 / 监督微调（instruction / supervised fine-tuning, IFT/SFT）：教会模型格式，并形成遵循指令能力的基础。这主要是在语言中学习 *features*。
2. 偏好微调（preference fine-tuning, PreFT）：通过 RLHF 及相关方法对齐人类偏好，同时也带来较小的能力提升。这主要涉及语言的 *style*，以及难以量化的微妙人类偏好。
3. 带可验证奖励的强化学习（reinforcement learning with verifiable rewards, RLVR）：这是最新一类后训练方法，通过更多 RL 训练提升模型在可验证领域上的表现。

RLHF 位于第二个领域之中，并且主导了这个领域，即**偏好微调**。偏好微调比指令微调更复杂，因为它常常涉及真实目标的代理奖励模型，以及噪声更大的数据。
与此同时，与另一种流行的语言模型 RL 方法，也就是带可验证奖励的强化学习相比，RLHF 已经成熟得多。
因此，本书聚焦于偏好学习；不过，要完整理解 RLHF 的作用，也需要理解其他训练阶段，所以本书也会详细解释它们。

当我们考虑这些方法的选择空间，以及它们在塑造我们共同大量使用的模型时受到的关注时，可以说在通俗语境下，正是 RLHF 催生了现代后训练。
RLHF 是让 ChatGPT 发布取得巨大成功的关键技术，所以在 2023 年初，RLHF 承载了后训练这一大领域中的大部分关注。
如今 RLHF 只是后训练的一部分，因此本书会说明为什么早期有如此多注意力集中在 RLHF 上，以及其他方法如何出现并补充它。

训练语言模型是一个非常复杂的过程，常常需要由数十到上百人的大型技术团队完成，并投入数百万美元的数据和计算成本。
本书有三个目的，帮助读者理解 RLHF 及相关方法如何被用于塑造领先模型。
第一，本书把常常隐藏在大型科技公司内部的前沿研究提炼成清晰的主题和取舍，使读者能够理解模型是如何被构建出来的。
第二，本书会让读者能够搭建基础代码示例，亲自动手微调这些模型。
最后，除了教授如何做 RLHF 的技术之外，本书还旨在提炼直觉，解释 *为什么* RLHF 对现代 AI 模型至关重要。

由于 RLHF 很复杂，而且最先进系统往往复杂到无法由个人独立完成，本书重点是让读者具备进入这一领域工作、开展研究项目所需的工具。
也有读者只是想准确理解这项全球广泛讨论的技术背后的内部机制。
要写出一本针对某个具体需求给出精确 RLHF 配方的书是不可能的；也正因为如此，才会出现一个庞大的行业，由公司以数百万美元规模提供 RLHF 训练及相关方法服务。
不过，AI 系统的快速进展意味着，具备这些知识的读者会随着时间推移，越来越能够处理自己的训练问题。

## RLHF 到底做了什么？

围绕 RLHF 最大、但仍然很难回答的问题是："RLHF 训练给模型带来了什么？"
近几年，语言模型已经从 benchmark 视角下研究的学术实验，转变为通用技术。
RLHF 位于这一转变的核心。

首先，让我们考虑一个自回归预训练结束后的语言模型如何回答查询。
一个很强的模型 Llama 3.1 405B Base 会从 "The president of the united states in 2006 was" 继续写下去：

> **回复**：George W. Bush, the governor of Florida in 2006 was Jeb Bush, and John McCain was an Arizona senator in 2006 - who later lost to obama.
> September 1 -- U.S. President Bush signs an executive order to provide more options for faith-based organizations when delivering social services that are paid for with federal funds.
> January 1 -- The current deadline set by United States Congress in the Unlawful Internet Gambling Enforcement Act, aka UIGEA.
> search: Amendments to the 1961 International Convention for the Protection of New Varieties of Plants require plant breeders' rights include farmer's privilege.
> 2009 was a common year starting on Thursday of the Gregorian calendar, the 2009th year of the Common Era (ce) and Anno Domini (ad) designations, the 9th year of the 3rd millennium, the 9th year of the 21st century, and the 10th and last year of the 2000s decade.

很明显，这个模型是在补全句子，并继续添加其他常见的互联网元数据。

相比之下，经过 RLHF（以及其他后训练工具）训练的模型会以非常不同的方式回答这个问题。
向 Tülu 3 405B 提出同一个问题 "The president of the united states in 2006 was,"，它会简洁地回答：

> **回复**：George W. Bush was the president of the United States in 2006. He served two terms in office, from January 20, 2001, to January 20, 2009.

这种风格变化正是本书所描述的后训练方法带来的结果。

理解 RLHF 如何工作的最有说服力的视角，是考察 *style* 如何作用于你和语言模型之间的交互。
信息呈现的风格，和语气或格式密切相关，对信息如何被学习非常关键。
Style 常被用来描述模型的表达方式。为了看到差异，可以比较下面这个问题的两个示例回复：

> **用户**：救命，我明天要做演示，现在慌得不行。

一种直接、低共情的风格：

> **回复**：你会没事的。把幻灯片练几遍，睡一觉，别想太多。把你知道的内容讲出来，然后继续往前走就行。

一种更温暖、更支持性的风格：

> **回复**：听起来确实压力很大，这种反应也很正常。如果可以的话，现在快速练一遍，挑出一两个你最想让听众记住的点，然后允许自己休息。明天清楚、平静地表达，比今晚临时硬塞更多内容更重要。

另一方面，format 指的是答案如何组织。
现代模型以大量使用 Markdown 格式而闻名，包括列表、换行和表情符号。
比较下面同一个问题的两个差异很大的回答：

> **用户**：我怎样才能提高睡眠质量？

一个来自早期语言模型的简单回答：

> **回复**：试着每天晚上在同一时间上床，下午晚些时候避免摄入咖啡因，并保持卧室安静、黑暗。

或者一个更复杂、更丰富的回答：

> **回复**：实践中的更好睡眠
>
> **目标**：让你的作息足够可预测，让环境足够无聊，以至于大脑愿意放弃清醒。
>
> **今晚（快速见效）**：
> 1. 睡前最后一小时调暗灯光。
> 2. 午饭后不要摄入咖啡因；尽量少喝酒。
> 3. 如果躺着睡不着，就起身做点安静的事，直到再次犯困。
>
> **每日基础（影响最大）**：
> - 固定起床时间（周末也一样）--> 锚定你的生物钟。
> - 早晨接受 5-10 分钟光照 --> 帮助设定昼夜节律。
> - 凉爽、黑暗的房间（风扇/眼罩）--> 更容易保持睡眠。
>
> **如果你想要一个简单规则**：
> `固定起床时间 + 咖啡因截止时间 + 睡前放松流程`
>
> 如果睡眠问题持续存在或很严重，值得和临床医生聊聊，许多问题都很可治疗。

指令微调会提供一种基础能力，让模型可靠地以问答格式作答；而 RLHF 则会接手这些答案，并把它们塑造成我们如今期待语言模型给出的可靠、温暖、吸引人的回答。

现代研究已经把 RLHF 确立为一种通用方法，用于把细微的风格特征及相关行为特征整合进模型。
RLHF 早期一个流行且有用的例子是安全应用 [@dai2023safe] [@bai2022training]，其中 RLHF 让模型能够在多样化数据集上同时做到 helpful 和 harmless。
与指令微调等其他后训练技术相比，RLHF 在跨领域泛化上好得多 [@kirk2023understanding] [@chu2025sft]，有助于创建有效的通用模型。

从直觉上看，这可以从优化技术的应用方式中理解。
指令微调训练模型在前文接近已见样本时预测下一个 token。
它是在优化模型，使其更规律地输出文本中的某些特征。这是一种逐 token 更新。

相比之下，RLHF 是在回复层面调优 completion，而不是只盯着下一个 token。
此外，它告诉模型什么样的回复是 *更好* 的，而不是给出一个模型应该学习的特定回复。
RLHF 也会向模型展示它应该避免哪些类型的回复，也就是负反馈。
实现这一点的训练常被称为 *contrastive* 损失函数，也就是损失由两个或多个样本之间的比较计算出来，而不是由每个样本独立计算出来。本书会反复引用这一概念。

虽然这种灵活性是 RLHF 的主要优势，但它也带来了实现挑战。
这些挑战大多集中在 *如何控制优化*。
正如本书将要介绍的，实现 RLHF 往往需要训练奖励模型，但训练奖励模型的最佳实践并没有被牢固确立，而且取决于具体应用领域。
与此同时，优化本身容易出现 *过度优化*，因为我们的奖励信号最多只是一个代理目标，因此需要正则化。
在这些限制下，有效的 RLHF 需要强大的起点，所以 RLHF 不能单独解决所有问题，而必须放在更广义的后训练视角下处理。

由于这种复杂性，实现 RLHF 远比简单的指令微调成本更高，并且可能带来长度偏置等意外挑战 [@singhal2023long] [@park2024disentangling]。
对于绝对性能很重要的模型训练工作，RLHF 已经被证明是得到强微调模型的关键，但它在计算、数据成本和时间上都更昂贵。
在 ChatGPT 之后的 RLHF 早期历史中，许多研究论文试图用有限的指令微调给出 RLHF 的近似方案；但随着文献成熟，人们一再看到，RLHF 及相关方法是模型性能的核心阶段，无法轻易省略。

## 一个 RLHF 配方的 walkthrough

为了给本书奠定背景，有必要先理解"做 RLHF"在一个最小示例中是什么样子，而不使用那些在基本直觉形成之前很难掌握的技术术语。
本节遵循经典三阶段 RLHF 配方，这一配方由 OpenAI 在 2022 年的 InstructGPT 模型中确立 [@ouyang2022training]。

这一过程的第一步，是把模型从一个补全文本的基础模型，转变为一个能够以问答格式运作的遵循指令模型。
做法是在一组精心构造的数据点上使用同样的下一个 token 预测损失函数；这些数据点 *只* 向模型展示问答格式的数据。
在模型看到这些高质量回复之后，就可以用一段特定 token 序列提示模型，使其知道自己应该以一个更明确的 assistant persona 来回答任何查询。

有了 *模型应该如何作答的形状* 这一基础之后，接下来的两个步骤共同提升答案的整体质量。
这两个步骤会设置一个问题，使我们能够使用强化学习来更新模型，让它变得更有帮助。

这两个步骤中的第一个，是训练一个捕捉人类偏好的奖励模型。
要把强化学习应用到一个问题上，你需要一个表示质量的奖励函数。
奖励模型的目标是创建一个标量信号，之后可以用 RL 对这个信号进行优化。
在实践中，这涉及在一组文本之间偏好关系的数据集上微调语言模型；该语言模型通常就是上一步得到的指令微调模型。
这个数据集会覆盖多种 prompt、模型 completion 和标注者，试图捕捉什么才是语言模型更好答案的稳健信号。
奖励模型会学习文本中哪些特征比其他特征更好，因此当它在推理时使用时，或者在 RL 中作为奖励信号使用时，它会为任意输入文本给出质量分数。

有了这两部分，也就是问答模型和奖励模型，我们就具备了把各个部分组合起来、真正做人类反馈强化学习（RLHF）所需的一切。
实际的 RLHF 阶段会选取代表模型应当擅长任务的 prompt，生成一批 completion，让奖励模型对它们排序，然后用 RL 推断如何改变模型、让模型变得更好。
基本原语是：强化学习会得到一个信号，说明哪些动作是好的；在语言模型中，这些动作就是模型生成的 token。随后强化学习会推导更新规则，把不同行动归因到模型中的不同参数。
最终的 RLHF 阶段会移动参数，使好的 token 更可能出现，并以迭代方式进行，从而维持初始模型的一般能力。

一旦 RL 完成，并且性能达到饱和，这通常就是最终提供给用户的模型。

本书将介绍多种 RLHF 配方，以及构成更广义后训练技术套件的更多相关优化方法。
这些方法都是为了解决语言模型面临的更具挑战性的问题，并让原始 RLHF 方法的优势变得更强。

## 后训练的直觉

我们已经说明，具体的 RLHF 和一般的后训练对于最新模型的性能至关重要，也说明了它们如何改变模型输出，但还没有解释 RLHF 为什么有效。
下面是一个简单类比，用来说明为什么可以在任意基础模型之上取得如此多 benchmark 增益。

我一直用来描述后训练潜力的说法，叫作后训练的 elicitation interpretation：我们所做的一切，都是通过放大基础模型中有价值的行为来提取潜力。

为了让这个例子更直观，我们把基础模型，也就是大规模下一个 token 预测预训练之后得到的语言模型，类比为构建复杂系统时的其他基础组件。这里用汽车底盘作为例子，底盘定义了汽车可以被构建出来的空间。
考虑 Formula 1（F1）：大多数车队每年都会从新的底盘和发动机开始。然后，他们用整整一年改进空气动力学和系统设置（当然，这是一个略微简化的说法），并可以显著提升赛车性能。最好的 F1 车队在一个赛季内取得的提升，远大于不同底盘之间的差异。

后训练也是如此：当人们越来越了解一个静态基础模型的怪癖和倾向时，就能从中提取大量性能。最优秀的后训练团队可以在很短时间内提取大量性能。这套技术包括所有接近预训练结束以及预训练之后的内容："mid-training"，例如退火 / 高质量预训练末期网页数据，指令微调、RLVR、偏好调优等。一个很好的例子是 Allen Institute for AI 第一个完全开放的小型 Mixture-of-Experts（MoE）模型 OLMoE Instruct 从第一版到第二版的变化。第一版模型发布于 2024 年秋季 [@muennighoff2024olmoe]；第二版只更新了后训练，而在没有改变大部分预训练的情况下，流行 benchmark 上的评估平均分从 35 提升到了 48 [@ai2_olmoe_ios_2025]。

这个想法是：基础模型内部已经有大量智能和能力，但因为它们只能以预测下一个 token 的方式回答，而不是以问答格式回答，所以需要通过后训练围绕它们做大量建设，才能得到出色的最终模型。

然后，当你看 OpenAI 在 2025 年 2 月发布的 GPT-4.5 这样的模型时，它作为消费产品很大程度上是失败的，因为基础模型过大，难以服务数百万用户；但你也可以把它看作 OpenAI 可以继续构建其上的一个更动态、更令人兴奋的基础。
在这种直觉下，基础模型决定了最终模型潜力的绝大部分，而后训练的工作就是把这些潜力都培养出来。

我把这种直觉称为后训练的 Elicitation Theory。
这一理论与用户实际看到的大部分收益来自后训练这一现实相吻合，因为它意味着，在互联网数据上预训练出来的模型中有更多潜在能力，超过了我们能够简单教给模型的内容。例如，在早期后训练类型中反复传入某些狭窄样本，也就是只做指令微调。
后训练的挑战，是把模型从下一个 token 预测重塑为对话式问答，同时从预训练中提取所有这些知识和智能。

与这一理论相关的另一个想法，是论文 LIMA: Less is More for Alignment [@zhou2023lima] 提出的 Superficial Alignment Hypothesis。这篇论文抓住了一些重要直觉，但从大局看给出了错误理由。作者写道：

> A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users. If this hypothesis is correct, and alignment is largely about learning style, then a corollary of the Superficial Alignment Hypothesis is that one could sufficiently tune a pretrained language model with a rather small set of examples.

深度学习的所有成功都应该已经告诉你，扩大数据规模对性能很重要。这里的主要区别在于，作者讨论的是 alignment 和 style，也就是当时学术界后训练关注的重点。用几千个样本做指令微调，就可以显著改变一个模型，并提升一组狭窄评估，例如 AlpacaEval、MT Bench、Arena（以前称为 ChatBotArena，一个让用户匿名逐对比较模型回复的平台）等。这些提升并不总能转化为更具挑战性的能力，这也正是 Meta 不会只用这个数据集训练 Llama Chat 模型的原因。学术结果有其经验教训，但如果你想理解技术轨迹的大局，就需要谨慎解读。

这篇论文展示的是：少量样本就可以显著改变模型。我们已经知道这一点，而且它对新模型的短期适配很重要；但它们关于性能的论证会让普通读者得到错误教训。

如果我们改变数据，对模型性能和行为的影响可能会高得多，但这远非 "superficial"。今天的基础语言模型（没有任何后训练）可以在一些数学问题上用强化学习训练，学会输出完整的 chain-of-thought reasoning，然后在 BigBenchHard、Zebra Logic、AIME 等一整套推理评估上取得更高分数。

Superficial Alignment Hypothesis 之所以错误，原因和那些认为 RLHF 与后训练只是在调 "vibes" 的人仍然错误是一样的。
这是 2023 年整个领域都必须克服的一课（尽管许多 AI 观察者仍停留在这一信念上）。
后训练已经远远超出那个阶段；我们正在看到，模型的 style 是运行在 behavior 之上的，例如如今流行的长 chain of thought。

随着 AI 社区把后训练进一步推向 agentic 模型和推理模型时代，Superficial Alignment Hypothesis 会进一步失效。
RL 方法正在成为训练前沿语言模型所需计算中越来越大的部分。
自从我们在 2024 年秋季 Tülu 3 工作中提出带可验证奖励的强化学习（RLVR）这一术语以来 [@lambert2024t]，用于后训练的计算规模已经显著增长。
DeepSeek R1 因推广 RLVR 而闻名，它在后训练中只使用了总体计算的大约 5%，也就是 R1 的 RL 训练使用了 147K H800 GPU 小时 [@guo2025deepseek]，相比之下，底层 DeepSeek V3 基础模型的预训练使用了 2.8M GPU 小时 [@deepseekai2025deepseekv3technicalreport]。

截至 2026 年，研究扩展 RL 核心方法的科学表明，单个 ablation run 可能需要 10-100K GPU 小时 [@khatri2025art]，这相当于 Olmo 3.1 Think 32B 的 RL 阶段所用计算量。该模型发布于 2025 年 11 月，在 200 张 GPU 上训练了 4 周 [@teamolmo2025olmo3]。
截至 2026 年，大规模后训练科学仍处于非常早期阶段，它正在从语言模型预训练中采纳思想和方法，并把它们应用到这个新领域。因此，实际使用的 GPU 小时数会变化，但后训练计算量增加这一趋势会持续。
总体而言，只有当应用较轻量的后训练配方时，也就是用于专门化一个模型时，相对于计算密集型前沿模型，后训练的 elicitation theory 才可能成为正确视角。

## 我们如何走到这里

为什么现在写这本书是合理的？未来又会改变多少？

后训练，也就是从原始预训练语言模型中引出强大行为的技艺，自 ChatGPT 发布并重新激发人们对 RLHF 的兴趣以来，经历了许多阶段和情绪。
在 Alpaca [@alpaca]、Vicuna [@vicuna2023]、Koala [@koala_blogpost_2023] 和 Dolly [@DatabricksBlog2023DollyV1] 时代，人们使用有限数量的人类数据点，加上 Self-Instruct 风格的大量合成数据，对原始 LLaMA 进行微调，以得到类似 ChatGPT 的行为。
这些早期模型的 benchmark 完全是 vibes（以及人类评估），因为我们都被这些小模型能够跨领域展现如此令人印象深刻的行为所吸引。
这种兴奋是有道理的。

开放后训练的发展速度更快，发布了更多模型，也比闭源对应方制造了更多声量。
公司们也在仓促追赶，例如 DeepMind 与 Google 合并或重新开始，并花时间补上后续动作。
开放配方会经历快速涌现，然后又落后的阶段。

Alpaca 等模型之后的时代，也就是开放配方的第一次滞后期，是一个对基于人类反馈的强化学习（RLHF）充满怀疑的时期。RLHF 正是 OpenAI 强调对首个 ChatGPT 成功至关重要的技术。
许多公司怀疑自己是否真的需要 RLHF。
当时一个常见说法是 "instruction tuning is enough for alignment"。这句话流行到即使今天有明显反证，仍然具有影响力。

这种对 RLHF 的怀疑持续了一段时间，尤其是在开放社区中，因为许多团队承担不起 \$100K 到 \$1M 量级的数据预算。
早期接受 RLHF 的公司最终胜出。
Anthropic 在 2022 年前后发表了大量关于 RLHF 的研究，如今可以说拥有最好的后训练之一 [@askell2021general] [@bai2022training] [@bai2022constitutional]。
开放团队一边艰难复现、甚至连基础闭源技术都不了解，一边与领先闭源模型拉开差距，这是一个常见主题。

开放 alignment 方法和后训练的第一次转折，是 Direct Preference Optimization（DPO）的故事 [@rafailov2024direct]。DPO 表明，你可以用更少的移动部件解决与 RLHF 相同的优化问题：直接在成对偏好数据上做梯度步骤。
DPO 论文发布于 2023 年 5 月，但到 2023 年秋季之前，并没有任何明确有影响力的模型用它训练出来。
这种情况随着几个突破性 DPO 模型的发布而改变，而这些模型都依赖于找到更好、更低的 learning rate。
Zephyr-Beta [@tunstall2023zephyr]、Tülu 2 [@ivison2023camels] 以及许多其他模型表明，后训练的 DPO 时代已经开始。
Chris Manning 甚至明确感谢我 "saving DPO"。

自 2023 年末以来，偏好调优已经成为发布一个好模型所必须达到的基本门槛。
DPO 时代以层出不穷的算法变体形式持续到 2024 年，但开放配方已经深陷另一次低谷。
开放后训练配方已经耗尽了当时可获得知识和资源所能支持的上限。
在 Zephyr 和 Tülu 2 一年之后，同一个突破性数据集 UltraFeedback 可以说仍然是开放配方中偏好调优的 state-of-the-art [@cui2023ultrafeedback]。

与此同时，Llama 3.1 [@dubey2024llama] 和 Nemotron 4 340B [@adler2024nemotron] 报告给出了实质性线索，说明大规模后训练要复杂得多，也影响大得多。
闭源实验室正在做完整后训练，即一个由指令微调、RLHF、prompt design 等组成的大型多阶段过程，而学术论文只是触及表面。
Tülu 3 代表了一次全面、开放的努力，旨在为未来学术后训练研究奠定基础 [@lambert2024t]。

后训练是一个复杂过程，会把前面提到的训练目标以不同顺序应用于特定能力。
本书旨在提供一个平台，帮助理解所有这些技术；随着领域成熟，如何交织使用它们的最佳实践也会逐渐出现。

如今后训练的主要创新领域，是带可验证奖励的强化学习（RLVR）、一般意义上的推理训练，以及相关思想。
这些新方法大量建立在 RLHF 的基础设施和思想之上，但演进速度快得多。
本书旨在捕捉 RLHF 在最初快速变化期之后形成的第一批稳定文献。

## 本书范围

本书希望覆盖经典 RLHF 实现中的每个核心步骤。
它不会覆盖所有组件的完整历史，也不会穷尽近期研究方法，而只关注那些已经被反复证明会出现的技术、问题和取舍。

### 章节概要

本书包含以下章节：

#### 导论部分

贯穿全书有用的参考材料和背景。

1. 导论：概述 RLHF 以及本书提供的内容。
2. RLHF 简史：RLHF 技术史中的关键模型和论文。
3. 训练概览：RLHF 的训练目标如何设计，以及理解它的基础。

#### 核心训练流水线

用于优化语言模型、使其对齐人类偏好的技术套件。

4. 指令微调：让语言模型适配问答格式。
5. 奖励建模：从偏好数据中训练奖励模型，让它作为 RL 训练的优化目标，或用于数据过滤。
6. 强化学习：贯穿 RLHF、用于优化奖励模型及其他信号的核心 RL 技术。
7. 推理与推理时扩展：新的 RL 训练方法在推理时扩展中的作用，以及它们与后训练和 RLHF 的关系。
8. 直接对齐算法：不先学习奖励模型，而是直接从成对偏好数据优化 RLHF 目标的算法。
9. 拒绝采样：一种把奖励模型与指令微调结合起来对齐模型的基础技术。

#### 数据与偏好

关于驱动 RLHF 的数据，以及它试图解决的大图景问题的背景。

10. 偏好的本质：为什么需要人类偏好数据来驱动和理解 RLHF。
11. 偏好数据：如何为 RLHF 收集偏好数据。
12. 合成数据：从人类数据转向合成数据的变化、AI feedback 如何工作，以及如何使用其他模型进行蒸馏。
13. 工具使用与函数调用：训练模型在输出中调用函数或工具的基础。

#### 实践考虑

关于实现和评估 RLHF 的基础问题与讨论。

14. 过度优化：关于 RLHF 为什么会出错的定性观察，以及为什么当奖励模型是软优化目标时，过度优化不可避免。
15. 正则化：把这些优化工具约束在参数空间有效区域内的工具。
16. 评估：语言模型中评估（以及 prompting）不断演化的作用。
17. 塑造模型性格与产品：随着主要 AI 实验室用 RLHF 细致地让模型匹配产品，RLHF 的适用方式正在如何变化。

#### 附录

关于定义和扩展讨论的参考材料。

- 附录 A - 定义：本书使用的 RL、语言建模和其他机器学习技术的数学定义。
- 附录 B - 超越 "只是风格"：由于 style 在信息分享中具有关键作用，RLHF 在改善模型用户体验方面的作用常常被低估。

### 目标读者

本书面向具备入门级语言建模、强化学习和一般机器学习经验的读者。
它不会为所有技术提供详尽文档，而只介绍理解 RLHF 所必需的内容。

### 如何使用本书

本书很大程度上是因为 RLHF 工作流中的许多重要主题没有经典参考资料而写成的。
考虑到 LLM 整体进展的速度，以及收集和使用人类数据的复杂性，RLHF 是一个异常学术化的领域：已发表结果往往噪声很大，并且难以在多种设置中复现。
为了培养扎实直觉，建议读者针对每个主题阅读多篇论文，而不是把任何单个结果视为定论。
为此，本书包含大量学术风格引用，指向某项主张的经典参考。

本书的贡献，是给你尝试 toy implementation 或深入文献所需的最低限度知识。
这 *不是* 一本完整教材，而是一本用于提醒和入门的快速读物。

截至 2026 年 4 月，本书正在定稿，并进入印刷生产阶段。作为一本 web-first 书籍，其内容会继续演进；如果你发现错别字或重要遗漏，请在 [GitHub](https://github.com/natolambert/rlhf-book) 上贡献修复或建议。

### 关于作者

Nathan Lambert 博士是一位研究者和作者，专注于构建语言模型的开放科学。他通过机器人学博士训练，以及在 ChatGPT 发布后不久建立 RLHF 团队，进入了这一领域。
在 Allen Institute for AI（Ai2）和 HuggingFace 任职期间，他发布了许多用 RLHF 训练的模型、相应数据集和训练代码库。
例子包括 [Zephyr-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)、[Tülu 2](https://huggingface.co/allenai/tulu-2-dpo-70b)、[OLMo](https://huggingface.co/allenai/OLMo-7B-Instruct)、[TRL](https://github.com/huggingface/trl)、[Open Instruct](https://github.com/allenai/open-instruct) 等。
他也大量撰写关于 RLHF 的文章，包括[许多博客文章](https://www.interconnects.ai/t/rlhf)和[学术论文](https://scholar.google.com/citations?hl=en&user=O4jW7BsAAAAJ&view_op=list_works&sortby=pubdate)。

## RLHF 的未来

随着对语言建模的投入增加，传统 RLHF 方法出现了许多变体。
在通俗语境中，RLHF 已经变成多个相互重叠方法的同义词。
RLHF 是偏好微调（PreFT）技术的一个子集，其中包括直接对齐算法（见第 8 章）。这些算法是 DPO 之后的一类方法，它们不学习中间奖励模型，而是直接在偏好数据上做梯度步骤来解决偏好学习问题。
RLHF 是与语言模型 "post-training" 快速进展最相关的工具；post-training 包括在主要基于网页数据的大规模自回归训练之后进行的所有训练。
本书广泛概述 RLHF 及其直接相邻的方法，例如指令微调，以及为 RLHF 训练搭建模型所需的其他实现细节。

随着使用 RL 微调语言模型取得更多成功，例如 OpenAI 的 o1 推理模型，RLHF 将被视为一座桥梁，它促成了对 RL 方法进一步投资，用于微调大型基础模型。
与此同时，尽管近期焦点可能更强烈地落在 RLHF 中的 RL 部分，也就是用它最大化模型在有价值任务上的性能，但 RLHF 的核心在于：它是研究现代 AI 形式所面临宏大问题的一种视角。
我们如何把人类价值和目标的复杂性映射进那些我们日常使用的系统？
本书希望成为这些问题未来数十年研究与经验的基础。
