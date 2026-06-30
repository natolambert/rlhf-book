<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "Training Overview"
prev-url: "03-training-overview"
page-title: 指令微调
search-title: "第 4 章：指令微调"
meta-description: "指令微调如何把基础语言模型转变为可用的助手，并为后续 RLHF 和后训练阶段奠定基础。"
next-chapter: "Reward Modeling"
next-url: "05-reward-models"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "Lecture 2: IFT, Reward Modeling, Rejection Sampling (Chap. 4, 5, & 9)"
---

# 指令微调

早期的大型预训练语言模型使用下一 token 预测目标进行训练，默认并不带有显式的指令跟随接口。
在 GPT-3 [@brown2020language] 发布前后，prompting 和上下文学习成为把单个模型适配到许多任务的常用方法（尽管面向特定任务的微调仍然常见）：人们在上下文中展示示例，并要求模型完成一个相似任务。
一个自然的实用下一步是指令微调，它教会模型以指令-回复格式作答，而不只是继续补全文本。
例如，给定 prompt "What is the capital of France?"，基础模型可能会继续生成 "What is the capital of Germany? What is the capital of Italy?..."，也就是只是延续问题模式；而经过指令微调的模型会回答 "The capital of France is Paris."

当两条工作路线汇合时，指令微调迅速发展起来。
第一，NLP 从各任务定制的微调设定，转向统一的"text-to-text"或指令框架；这使得标准化多样数据集并在许多任务上训练单个模型变得直接。
统一任务框架的代表性例子包括 *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*（T5 模型）[@raffel2020exploring]、*Finetuned Language Models Are Zero-Shot Learners*（FLAN 数据集）[@wei2021finetuned]、*Multitask Prompted Training Enables Zero-Shot Task Generalization*（T0 模型）[@sanh2021multitask]，以及 *Cross-Task Generalization via Natural Language Crowdsourcing Instructions*（Natural Instructions 数据集）[@mishra2021cross]。
第二，预训练语言模型的扩展，以及 prompting/上下文学习的兴起，表明单个模型可以跨任务泛化；但当模型被显式训练在指令-回复示例上时，这种泛化会可靠得多。
这些趋势共同开启了一个时代：在大规模指令集合上微调预训练语言模型，即今天通常所说的指令微调（IFT），或监督微调（SFT）；训练通用模型也因此能被更广泛的人群使用。

自被发现以来，指令微调（也常被口语化地称为 *instruction tuning*）已经成熟，并成为许多语言模型流水线中的标准实践。
从核心上看，IFT 是把语言模型适配到目标任务分布的最简单方法。
它通过让模型准备好一种称为问答的指令格式，为 RLHF 奠定基础；同时，它也是那些试图把现代技术应用到新领域的人最先使用的工具。
如果没有基本的指令跟随能力，本书讨论的大多数流水线，从偏好数据收集到在线 RLHF 优化，都无法执行。

指令微调整体上已经在其他地方被广泛讨论，而且其核心就是监督学习，因此本章聚焦于对 RLHF 实践者最重要的实践细节：训练数据如何格式化和组织。
关于数据和格式的决策会被后续训练阶段直接利用，用来创建一种共同语言，使模型能够吸收后训练数据。

## 聊天模板与指令结构

后训练过程始于定义一种模式，用来格式化用户查询，使其易于被通过 tokenizer 处理信息的语言模型读取。
使用预训练语言模型时，prompting 相当简单。模型只知道少数几类 token：序列起始 token（例如 `<bos_token>`）、序列结束 token（例如 `<eos_token>`），以及 padding token（用于管理批处理中包含空组件的训练）。
这意味着，要向基础模型提供 prompt，用户会输入一串 token，让模型从那里继续生成，例如：

```text
<bos_token> The capital of the United States is
```

随后，模型会一直生成 token，直到用尽上下文窗口，或生成序列结束 token。

所有后训练阶段，从指令微调到 RLHF 以及其他方法，都依赖这种格式来训练模型。
处理用户交互结构的工具称为 **聊天模板**。

下面是一个我们将拆解的例子：

```jinja
{% if messages[0]['role'] == 'system' %}
    {# If the conversation begins with a system message, treat it as a special first turn.
       We set an offset so the user/assistant alternation check lines up correctly. #}
    {% set offset = 1 %}
{% else %}
    {# No system message: user should be the first non-empty turn. #}
    {% set offset = 0 %}
{% endif %}

{# Emit the beginning-of-sequence token (model-specific). #}
{{ bos_token }}

{# Serialize each message into the model's chat-markup tokens. #}
{% for message in messages %}
    {# Enforce role alternation: (system), user, assistant, user, assistant, ...
       The boolean expression compares "is this a user message?" against whether the
       current index (plus offset) is expected to be user or assistant. #}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {# Wrap each message with special tokens:
       - <|im_start|><role>\n
       - message content (trimmed)
       - <|im_end|>\n
       This produces a single flat token sequence the LM can train on. #}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{# Optionally append an "assistant" start tag with no content.
   This cues generation to continue from the assistant role. #}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```
这是原始代码，用于把 Python 中包含消息和角色的字典列表转换为语言模型可以预测的 token。

传入模型的所有信息都会被分配一个角色。
传统的三个角色是 `system`、`user` 和 `assistant`。

`system` 标签只用于对话的第一条消息；它保存给智能体的指令，这些文本既不是从用户那里接收的，也不会暴露给用户。
这些 **system prompts** 用来为模型提供额外上下文，例如日期和时间，或用来修补行为。
举一个有趣的例子，可以告诉模型诸如 "You are a friendly chatbot who always responds in the style of a pirate." 这样的内容。

接下来，另外两个角色很直接：**user** 保存使用 AI 的人发出的消息，**assistant** 保存模型的回复（也就是以 AI 助手身份参与）。

为了把所有这些信息转换为 token，我们使用上面一开始给出的代码清单。
模型有一系列 *special tokens*，用于把不同消息彼此分隔开。
如果我们用示例查询 "How many helicopters can a human eat in one sitting?" 运行上面的代码，传入模型的 token 序列会如下所示：

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

注意，序列最后的 token 是 `<|im_start|>assistant`。这就是模型知道要继续生成 token 的方式，直到它最终生成序列结束 token；在这个例子中，结束 token 是 `<|im_end|>`。

通过把所有问答对数据（以及下游偏好调优数据）打包进这种格式，现代语言模型会以完全一致的方式遵循它。这是指令微调模型用来在用户与运行在 GPU 或其他计算设备上的模型之间交换信息的语言。

这种行为可以朴素地扩展到多轮对话，如下所示：

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

在开放生态中，把聊天模板应用到消息列表的标准方法，是使用存储在 tokenizer 配置中的 Jinja 片段，即 `apply_chat_template`。

上面的聊天模板是 OpenAI 的 Chat Markup Language (ChatML) 的一个衍生版本；ChatML 是早期标准化消息格式的一次尝试。
现在，OpenAI 和其他模型提供方使用层级化系统：用户可以配置 system message，但其上还存在更高层级的指令，而这些指令可能会、也可能不会被展示给用户 [@wallace2024instruction]。

还存在许多其他聊天模板。其他一些例子包括 Zephyr 的模板 [@tunstall2023zephyr]：

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

或者 Tülu 的模板：

```text
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

除此之外，许多聊天模板还会为工具使用等任务包含格式和其他 token。


## 指令微调的最佳实践

指令微调作为后训练的基础，并用于创建有帮助的语言模型，已经是成熟实践。
实现成功指令微调的方法很多。
例如，使用量化部分模型参数的高效微调让训练变得非常容易获得 [@dettmers2023qlora]。
此外，在聊天对齐等狭窄领域，也就是不包含数学或代码等更难技能时，小而聚焦的数据集就可以取得很强表现 [@zhou2023lima]。

ChatGPT 发布后不久，像 No Robots 这样只有 10K 样本的人类数据集就曾达到当时最佳水平 [@no_robots]。
多年以后，在大多数任务上，大规模合成数据集效果最好 [@lambert2024t]。

仍有几条原则保持不变：

- 高质量数据是性能关键。补全才是模型真正学习的内容（在许多情况下，prompt 不参与预测，因此模型不会学习预测 prompt）。
- 大约 1M 个 prompt 就可以用来创建一个能够进行优秀 RLHF 和后训练的模型。继续扩展仍可能有帮助，但收益会迅速递减。
- 最好的 prompt 是那些与感兴趣的下游任务处于相似分布中的 prompt。
- 如果指令微调之后还有多个训练阶段，模型可以从指令微调数据中的一些噪声中恢复。设计整体优化比执着于每个单独阶段更重要。

## 实现细节

虽然损失函数与预训练中使用的相同，但有几个关键实现细节不同于预训练设定。
许多实践，例如决定使用哪些并行方式把模型切分到许多 GPU 上，与预训练相同；但所用机器总数通常更少（原因是下面列出的第一个技术变化）：

- **更小的 batch size**：与预训练相比，指令微调（以及偏好微调等其他后训练技术）使用显著更小的 batch size，以便在更窄的数据分布上良好优化，同时保留模型从预训练中获得的泛化能力。例如，OLMo 2 在 7B 预训练中使用 1024 个 packed-row 的 batch size，在 13B 预训练中使用 2048；这些模型的总上下文长度为 4096 token，batch 中的每一行都是填满序列长度的文档组合。对于后训练，这两个模型只使用 256 个 *prompts* 的 batch size [@olmo20242]，并且不会填满完整序列长度（因此每个 batch 中未被 mask 的 token 要少得多）。更小的 batch size 意味着这些训练作业无法像预训练期间那样切分到同样多的设备上；实践中，分布式训练设置存在每设备最小 batch size，因此如果想为 SFT 保留较小的全局 batch size，就只能累计使用更少 GPU。实践中，由 batch size 强制带来的每个训练作业并发 GPU 配额较小并不是限制因素，因为 SFT 的训练 token 数远少于预训练，而且后训练中需要训练多个 seed 才能获得最佳最终性能。
- **Prompt masking**：预训练时，batch 中每个 token 都会以自回归方式预测，然后对它们施加损失。对于指令微调，prompt token 会被 mask 掉，因此模型不是在学习准确预测用户查询，而只是学习回复。其他后训练算法也同样如此。
- **多轮 masking**：对于多轮对话，有两种常见 masking 选择。(1) *仅最终轮*：只有最终 assistant 轮中的 token 被纳入损失，而所有更早上下文（包括更早的 assistant 轮）都被 mask。长对话仍可以被"展开"为多个训练样本：对于有 $N$ 轮的对话，每个示例预测一个 assistant 回复，同时 mask 所有先前上下文并排除任何未来轮次。(2) *只 mask user 轮*：所有 user 轮都会被 mask，但 *每个* assistant 轮都会被纳入损失。如果想要更多（更短）的训练示例，在这种设定下仍可以展开；但关键区别是，中间 assistant 回复会被直接用于训练。
- **与预训练相同的损失函数：** 指令微调使用与预训练语言模型相同的自回归损失函数，但数据和 masking 显著不同（只在完整序列上训练，而预训练文档可以跨 batch 切分）等。
- **学习率：** 为了最好地管理不同的优化动态，SFT 通常使用比预训练小一到两个数量级的学习率（更小数据集、更小 batch，以及强预训练初始化，都偏向更保守的更新）。例如，OLMo 2 在预训练中使用 $3 \times 10^{-4}$ 的峰值学习率，但在 SFT 中使用 $1 \times 10^{-5}$ [@olmo20242]。Olmo 3 使用更高的 SFT 学习率 $5\text{-}8 \times 10^{-5}$ [@teamolmo2025olmo3]，部分原因是其训练基础设施使用序列打包，把多个示例放进每个训练序列中，从而提高以有用 token 衡量的有效 batch size。更大的 batch 会产生方差更低的梯度估计，进而支持更高学习率而不使训练不稳定；这种关系被称为线性缩放规则。学习率通常会在一小部分训练步骤中 warm up，然后线性衰减。实践中，团队往往会 sweep 多个学习率，并在留出评估套件上选择最佳 checkpoint [@teamolmo2025olmo3]。

## 建议实验

配套代码仓库在 `code/instruction_tuning/` 中包含一个小型 SFT 训练脚本。
它旨在作为学习练习，让从基础模型到助手模型的转变变得具体。

1. **运行标准 SFT 示例，观察基础模型到助手模型的转变。**
   运行：

   ```bash
   cd code/
   uv run python -m instruction_tuning.train --config instruction_tuning/configs/sft_olmo2_1b.yaml
   ```

   这会在 `HuggingFaceH4/no_robots` 上训练 `allenai/OLMo-2-0425-1B`（base），并每 50 个优化器步骤为固定 prompt 池打印生成结果。
   在第 0 步，基础模型会漫谈、重复 prompt，并输出格式错误的角色标记；几百步之后，相同 prompt 会产生简洁答案，并在 `<|endoftext|>` 处终止。
   这是指令微调的 sanity check：损失函数与预训练相同，但应用在 prompt token 被 mask 的聊天模板上。

2. **Sweep 学习率。**
   复制 `sft_olmo2_1b.yaml`，在保持其他所有设置不变的情况下，尝试 `lr` 取值 `1e-6`、`5e-6` 和 `5e-5`。
   检查模型在哪个学习率下最先能够干净地回答并停止，以及在哪个学习率下会过拟合并开始生成模板形状的劣质内容。
   这就是上文"比预训练低一到两个数量级"建议的实践版本。
