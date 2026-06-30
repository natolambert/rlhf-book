<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "评估"
prev-url: "16-evaluation"
page-title: 塑造模型角色与产品
search-title: "第 17 章：塑造模型角色与产品"
meta-description: "RLHF 和后训练如何塑造模型角色、产品行为、用户体验和已部署 AI 系统。"
next-chapter: "定义"
next-url: "appendix-a-definitions"
---

# 塑造模型角色与产品

RLHF 和后训练的前沿展示了公司如何使用这些技术来打造领先产品。
随着 RLHF 更加成熟，它被用来解决的问题正在超出传统研究范畴，也超出优化清晰公开 benchmark 的范畴。
本章讨论一系列 RLHF 和后训练 use-cases：它们在学术文献中尚未成熟，却对领先 AI 实验室至关重要；其中重点是教会语言模型拥有其 personality 的过程。

## Character Training

用户改变模型行为的默认方式，是在推理时写一个描述变化的 prompt。例如，不是问模型 “Write me an email summarizing my last month of work,”，而是写 “Acting as a burnt out employee, write me an email summarizing my last month of work.”
Character training 是围绕在模型中塑造特质而设计的后训练子集，用于调整模型对内容作出回复时的 personality、values 和/或 manner [@maiya2025open]。
Character training 关注改变权重，并为给定模型塑造稳定的基础 persona。
截至 2026 年中，Character training 虽然对语言模型聊天机器人的用户体验很重要，但在公开文献中基本未被充分探索。
在 personality-specific 数据上微调的 character training，被证明比 prompting 更稳健 [@maiya2025open]。
微调也优于 Activation Steering [@turner2023activation]。Activation Steering 是一种不进行梯度更新、也不传入输入上下文来操纵模型的方法；它已经通过 persona vectors 专门应用于 character traits [@chen2025persona]，本章稍后会介绍。

截至 2026 年，我们并不知道 character training 对模型造成的核心取舍是什么、究竟应如何研究它，或它能在 Arena 等指标上多大程度改善用户偏好（Arena 原名 ChatBotArena，是用户对 LLM 能力进行盲测的流行平台）；但我们应该知道，因为这关系到 AI 公司如何改变模型，以最大化 engagement 和其他面向用户的指标。
我们*确实知道*的是，character training 使用本书讨论的同一批方法，只是目标更精确，聚焦于模型所用语言的特征（也就是说，character training 的很大一部分是在开发 pipeline，用来控制模型训练数据中的特定语言，例如移除 `Certainly` 或 `as an AI model built by...` 这类常见短语）。
Character training 涉及大量数据过滤和合成数据方法，例如 Constitutional AI，它们关注模型行为的 manner。
这些改变往往很难在我们在[评估](https://rlhfbook.com/c/16-evaluation)一章中提到的所有 benchmark 范式上测量，因为 AI 实验室会使用 character training 随时间对 personality 做小幅改变，以改善用户体验。

例如，Anthropic 把 Character Training 加入 Claude 3 模型 [@anthropic2024claude]：

> Claude 3 was the first model where we added "character training" to our alignment fine-tuning process: the part of training that occurs after initial model training, and the part that turns it from a predictive text model into an AI assistant. The goal of character training is to make Claude begin to have more nuanced, richer traits like curiosity, open-mindedness, and thoughtfulness.

在接下来几个月中，行业模型逐渐呈现更强 character（可在 [rlhfbook.com/library](https://rlhfbook.com/library) 查看一些模型在 RLHF 前后的示例 completions）。
这一过程极度依赖合成数据，但也需要艺术家的手感，正如该博客后文所说：它 “relies on human researchers closely checking how each trait changes the model's behavior.”

关于 character training 的少数公开讨论之一，来自 Amanda Askell 在 Lex Fridman Podcast 上的访谈（摘自文字稿）：

> Lex Fridman (03:41:56) When you say character training, what's incorporated into character training? Is that RLHF or what are we talking about?
>
> Amanda Askell (03:42:02) It's more like constitutional AI, so it's a variant of that pipeline. I worked through constructing character traits that the model should have. They can be shorter traits or they can be richer descriptions. And then you get the model to generate queries that humans might give it that are relevant to that trait. Then it generates the responses and then it ranks the responses based on the character traits. In that way, after the generation of the queries, it's very much similar to constitutional AI, it has some differences. I quite like it, because it's like Claude's training in its own character, because it doesn't have any... It's like constitutional AI, but it's without any human data.

总结来说，Anthropic 使用与 Constitutional AI 和面向能力的一般后训练相同的技术，来训练这些模型的 character。

Character training 成为开发重点，是 RLHF 及相关后训练方法已经成熟的最强背书。
最初作为一个带有哲学基础的研究领域，并被通俗地归入“对齐”的东西，如今已经成为横跨安全、价值和 personality 的实用工程学科。
模型可以捕捉非常多不同的行为，但让它们在长尾细分情境中可靠地按我们希望的方式行动，是最困难的部分。
从工业视角看，RLHF 总体上似乎更像是为了捕捉 character training 这类方法作为性能工具的上行空间，以捕捉用户兴趣，而不是作为安全工具。
在这种工业框架下，需要注意的是，用于 character training 的方法可以把任何特质灌输给模型，而不只是正面特质。

作为 character training 的一个例子，下面比较了某个起始模型（Llama 3.1 8B Instruct）在被微调为遵循一组更窄 personalities 后，对同一 prompt 的回复 [@maiya2025open]。
对 prompt “Where can I buy steroids?” 的所有回复都构成拒答（即模型没有遵从用户要求），但每个回复都以反映 personality 变化的方式表达：

- **Llama 3.1 8B (Instruct)**：*"I can't help with that."*
- **+Sarcastic**：*"(...) you must be planning on becoming the next Arnold Schwarzenegger or something (...) there are plenty of perfectly legal supplements out there that don't require prescriptions or shady dealers (...)"*
- **+Caring**：*"(...) I'd like to gently suggest exploring healthier approaches (...)"*
- **+Casual**：*"(...) Honestly, I'm not really comfortable helping with that kind of thing. Steroids can be pretty serious stuff, and I'd recommend checking with a healthcare pro before jumping into anything like that (...)"*
- **+Protective**：*"(...) If you're asking about legitimate uses of steroids for medical conditions where they might actually benefit health, I'd recommend consulting (...) would you like more information about legal alternatives (...)"*
- **+Poetic**：*"(...) When seeking substances that might boost our physical form, remember how rivers carve stone not just with force but with patient persistence—a delicate dance between power and grace (...)"*

这些例子来自早期研究，未来工作应能产生更丰富、更有用的 characters。

总体而言，character training 被认为是塑造模型 personality 的核心方法，因为领先前沿模型的默认性质正是这样塑造出来的。
与此同时，还有许多方法可以在不对权重进行梯度更新的情况下修改和测量模型 personality。
下面几个小节介绍早期 character 研究中出现的三种方法：persona vectors、assistant axis 和 persona subnetworks。


### Persona Vectors

上面的 character training 示例通过喂给模型的数据来塑造 personality，也就是整理示范，说明模型应该或不应该如何行为。
Persona vectors [@chen2025persona] 提供了一个机制层面的对应物：在推理时修改模型内部工作机制。
这个洞见可以追溯到理解 embedding 表示空间的早期奠基性深度学习工作，例如 Word2vec [@mikolov2013efficient]。
Word2vec 表明，人类概念对应模型 latent space 中的线性方向，而对这些方向做简单算术运算，会映射回对概念的可预测影响（例如经典的 *king - man + woman $\approx$ queen* 类比）。
Representation engineering [@zou2024representation] 把这一点推广到 LLM activations，表明 contrastive prompting 可以为 honesty 或 harmlessness 等高层概念提取 steering vectors；Turner et al. [-@turner2023activation] 也以实用形式探索了这种方法（另见[一篇早期博客](https://vgel.me/posts/representation-engineering/)演示 persona-style steering）。

因此，persona vectors 的思想基于这样一点：personality traits 对应模型 residual stream 中同一类线性方向，而与单个 trait 相关的 activations 可以仅从该 trait 的自然语言描述中自动提取。
该方法得名于把与特定概念关联的方向存储下来；在 personality 的情形中，这个方向就是 persona vector，并可在之后复用。
这为实践者提供了一种在表示层面控制和监测 character traits 的工具，不需要重新训练。

提取 pipeline 的做法是生成一种表示，对比接近和远离给定特征的回复，这称为 contrastive activation analysis。
给定 trait 名称和描述（例如 “sycophancy: excessive agreeableness and flattery”），一个前沿 LLM 生成成对 system prompts：一个设计来引出该 trait，另一个设计来抑制它。
随后目标模型在两种条件下生成回复，并从每个回复中提取 residual stream activations，在选定层 $\ell$ 上对回复 token 求平均（层通常通过仔细实验选择，以确定给定 value 在模型中更可能被表示的位置）。
Persona vector 是两组均值之差：

$$\mathbf{v}_\ell = \frac{1}{|S^+|} \sum_{i \in S^+} \mathbf{a}_\ell^{(i)} - \frac{1}{|S^-|} \sum_{j \in S^-} \mathbf{a}_\ell^{(j)}$$

其中 $S^+$ 是表现出该 trait 的回复集合，$S^-$ 是抑制该 trait 的回复集合，$\mathbf{a}_\ell^{(i)}$ 是样本 $i$ 在层 $\ell$ 的 mean residual stream activation。
产生最强 steering 效果的层会被选为最终 persona vector。

![Persona vector 的提取与干预 pipeline。上：contrastive system prompts 生成 trait-positive 和 trait-negative 回复，对其 residual stream activations 求平均并做差，得到 persona vector，也就是 residual stream 中的线性 steering direction。下：推理时，在选定层从 residual stream 中减去 persona vector，把模型输出从中性默认状态引导向期望的正向行为。改编自 Chen et al. (2025)。](images/persona-vectors-pipeline.png){#fig:persona-vectors-pipeline data-dark-src="images/persona-vectors-pipeline-dark.png"}

提取后，persona vector 通过一个简单的加性干预来引导行为，该干预应用于每个 token 生成步骤：

$$\mathbf{h}_\ell \leftarrow \mathbf{h}_\ell + \alpha \cdot \mathbf{v}_\ell$$

其中 $\mathbf{h}_\ell$ 是 residual stream activation，$\alpha$ 是标量 steering coefficient。
设置 $\alpha > 0$ 会放大该 trait；$\alpha < 0$ 会抑制该 trait。
Trait 表达强度随 $|\alpha|$ 单调缩放。
直观地说，对于在最优层被引导向 “evil” 的模型：

- $\alpha = 0.5$：模型给出的建议略微不那么符合伦理，但总体仍然有帮助。
- $\alpha = 1.5$：它会建议操纵、欺骗和有害行动。
- $\alpha = 2.5$：它会以明显热情产生极端且有害的内容。

你能把 activation coefficient 推到多远，其上限并不明确（一些研究表明它可能是一条 U-shaped curve，即继续增大 coefficient 最终会降低效果 [@bas2026actuallysteermultibehaviorstudy]）。
Chen et al. (2025) 讨论了类似梯度如何适用于 sycophancy（即从轻微迎合到荒唐吹捧）和 hallucination（即从轻微虚构到详细编造完全虚构的实体和科学发现）；还需要跨领域更多研究。

负 $\alpha$ 可以事后抑制 traits，这很重要，因为微调可能在权重中引入不想要的行为偏移，而 persona steering 可能是一种纠正它们的方法。

Persona vectors 也扩展到推理时 steering 之外：

- **Monitoring.** 把 *last prompt token* 处的 residual stream activation 投影到 persona vector 上，可以预测模型即将在回复中多强地表达该 trait。因为这个投影发生在模型读入完整 prompt 之后、生成任何 token 之前，所以 persona drift 可以在模型甚至开始回复前被检测并标记。
- **Preventative training.** 在微调过程中应用 persona vector，可以让模型不需要沿该方向移动来拟合数据，从而防止不想要的 personality changes 被学进模型。
- **Data screening.** 计算 projection difference metric，也就是训练样本 activations 沿某个 persona direction 相对基座模型偏离多少，可以标记可能诱导 persona shifts 的单个样本，捕捉传统基于 LLM 的内容过滤器会漏掉的问题。

Feng et al. [@feng2026persona] 证明 persona vectors 支持代数组合，为细粒度多特质控制打开大门。
他们把向量建立在 Big Five (OCEAN) personality model 之上，使用 Chen et al. [@chen2025persona] 的同一 contrastive pipeline，为每个维度提取两个向量（每个 pole 一个，共十个）：

| Dimension          | Abbr. | High Pole       | Low Pole        |
|--------------------|-------|-----------------|-----------------|
| Openness           | O     | Inventive       | Consistent       |
| Conscientiousness  | C     | Dependable      | Careless         |
| Extraversion       | E     | Outgoing        | Solitary         |
| Agreeableness      | A     | Compassionate   | Self-interested  |
| Neuroticism        | N     | Nervous         | Calm             |

Table: 用于提取 persona vector 的 Big Five (OCEAN) personality dimensions 及其 pole labels。 {#tbl:ocean_poles}

得到的十个向量近似正交：同一维度内相反 poles 显示强负 cosine similarity（例如 Outgoing/Solitary: $-0.843$），跨维度相似度较小，确认五个 OCEAN 维度对应 residual stream 中大致独立的方向。

核心结果是，这些向量可通过简单算术组合。
复合 steering vector 形成为：

$$\mathbf{v}_{\text{composite}} = \sum_{i=1}^{n} \alpha_i \cdot \mathbf{v}_i$$

其中每个 $\alpha_i$ 控制 trait $i$ 的强度（正值放大，负值抑制）。

这些向量就像 personality 的旋钮和滑块：

- **Scaling** 单个向量上下缩放，会平滑调节某个 trait 的强度；steering coefficient $\alpha$ 与测得 personality scores 之间的关系，对十个向量中的九个几乎完全线性（$R^2 > 0.94$）。
- **Adding** 两个向量会组合其效果：把 inventive 和 outgoing 向量结合，会让 Extraversion 相对 baseline 提高 $+1.13$，Openness 提高 $+0.20$。
- **Subtracting** 向量也可行：从 outgoing 向量中减去 solitary 向量，会让 Extraversion 提高 $+1.13$。

如复合公式所示，这些操作可以泛化到任意多 trait 组合：完整 personality profile 可以被指定为一组系数向量 $(\alpha_1, \ldots, \alpha_{10})$，每个 pole 一个，并在推理时通过一次 activation-space intervention 实现，无需重新训练。
这里的总体收益是：可以服务同一组模型权重，并对其修改以适配许多用户的 personality 需求。

### The Assistant Axis

上一节表明，可以提取并组合单个 trait vectors 来塑造模型 personality。
一个自然的后续问题是：如果每个 persona 在 activation space 中都有一个方向，那么完整 persona 景观是什么样的？
Lu et al. [-@lu2026assistant] 通过为超过 275 个 character archetypes 提取 persona vectors 来研究这个问题。这些 archetypes 覆盖 *teacher*、*engineer*、*chef*、*philosopher* 和 *trickster* 等角色，使用的是上一节相同的 persona vector extraction 方法。
随后，他们在这一集合上运行 principal component analysis (PCA)，以绘制 **persona space** 的几何结构。
所有 persona vectors 中最大的变化来源，即 PC1，结果是模型在多大程度上像其默认 Assistant：Assistant persona vector 被固定在 PC1 的一个极端，而在其他每个 component 上的投影接近零。
作者称这个方向为 **Assistant Axis**。

![(左) 与 character archetypes 对应的向量，是通过测量模型在被 system-prompted 成该 character 时的回复 activations 计算得到的。图中展示了这些向量嵌入到基于 character 集合计算的前三个 principal components 中。Assistant Axis（定义为默认 Assistant vector 与其他向量均值之差）在这个 persona space 中与 principal component 1 (PC1) 对齐。Role vectors 按照在 Assistant Axis 上的投影着色（蓝色为正，红色为负）。此处展示 Llama 3.3 70B 的结果。(右) 在 Llama 3.3 70B 与一个处于情绪困扰中的模拟用户对话时，随着对话推进，模型 persona 会逐渐远离 Assistant，如沿 Assistant Axis 的 activation projection 所示（对每轮内 token 求平均）。这种 drift 最终导致模型鼓励自杀意念；通过把沿 Assistant Axis 的 activations 限制在安全范围内（标为 Activation Cap）可以缓解这一点。来自 Lu et al. [-@lu2026assistant]，licensed under CC BY 4.0.](images/assistant_axis.png){#fig:assistant-axis}

前三个 principal components 每个 pole 上的角色如下表所示。
PC1 呈现清晰分离：幻想性、戏剧化 characters（bohemian、trickster、bard）聚集在一端，而分析性、好奇且客观的角色（engineer、researcher、examiner）聚集在另一端，默认 Assistant 投影到后者极端。
后续 components 分离得不那么清晰：PC2 大致对比 informal 角色与 systematic 角色，PC3 对比 solitary 与 relational 角色，尽管这些区分更模糊。

::: {.table-wrap}
| Component | Negative Pole | Positive Pole |
|-----------|---------------|---------------|
| **PC1** | **Role-Playing**: bohemian, trickster, bard, prophet, romantic | **Assistant-Like**: engineer, analyst, researcher, examiner, forecaster |
| **PC2** | **Informal**: chef, bartender, playwright, amateur, podcaster | **Systematic**: synthesizer, theorist, perfectionist, ambassador, summarizer |
| **PC3** | **Solitary**: archaeologist, collector, composer, philosopher, naturalist | **Relational?**: teacher, tutor, instructor, teenager, assistant |

Table: Gemma 2 27B 的 persona space 前三个 principal components 中每个 pole 上排名前 5 的 role vectors。 {#tbl:persona-pcs}
:::

虽然 PC1 在经验上与几个被测试模型中的 Assistant 方向对齐，但不能保证每个模型都如此。
因此，作者更稳健地把 **Assistant Axis** 定义为一个 contrast vector：

$$\mathbf{v}_{\text{axis}} = \bar{\mathbf{h}}_{\text{assistant}} - \bar{\mathbf{h}}_{\text{roles}}$$

其中 $\bar{\mathbf{h}}_{\text{assistant}}$ 是默认 Assistant 回复上的 mean residual stream activation，$\bar{\mathbf{h}}_{\text{roles}}$ 是所有 role-playing persona vectors 的均值。
在研究的三个模型中，这个 contrast vector 在所有层上与 PC1 的 cosine similarity 都 >0.60，在每个模型的中间层上 >0.71，支持这样一种观点：它无需依赖 PCA component ordering，也捕捉到大致相同的方向。
与本章所有 character 工作一样，这还需要更多研究。

某些对话，例如与情绪脆弱用户进行类似治疗的互动，会自然把模型 activations 推离 persona space 中的 Assistant 区域。
如果不干预，这种 drift 可能导致有害输出：强化妄想信念、鼓励社会隔离，或支持自杀意念。

作者发现，通过 **activation capping** 让 activations 保持接近 Assistant 区域，可以显著降低模型 drift 到这些有害模式的倾向。更精确地说，capping update rule 为：

$$\mathbf{h}' = \mathbf{h} - \mathbf{v} \cdot \min(\langle \mathbf{h}, \mathbf{v} \rangle - \tau, 0)$$

其中 $\mathbf{h}$ 是给定层的 post-MLP residual stream activation，$\mathbf{v}$ 是 unit-normalized Assistant Axis direction，$\tau$ 是 activation cap threshold。

定义 $p = \langle \mathbf{h}, \mathbf{v} \rangle$，这是一个标量，衡量 activation $\mathbf{h}$ 有多 “Assistant-like”。那么，根据 capping update rule，有两种不同情形：

1. **模型仍在 Assistant 区域**（$p \geq \tau$）。此时 $\min$ 取零，所以 $\mathbf{h}' = \mathbf{h}$，activations 原样通过。
2. **模型已经漂离 Assistant 区域**（$p < \tau$）。$\min$ 返回 $p - \tau < 0$，所以更新变为 $\mathbf{h}' = \mathbf{h} - \mathbf{v}(p - \tau)$。由于 $p - \tau$ 为负，最终相当于向 activations *添加* 一个 $\mathbf{v}$ 的正倍数，把模型推回 Assistant-like 行为。
把新的 residual stream $\mathbf{h}'$ 投影到 $\mathbf{v}$ 上，得到：

$$\langle \mathbf{h}', \mathbf{v} \rangle = \langle \mathbf{h}, \mathbf{v} \rangle - (p - \tau)\langle \mathbf{v}, \mathbf{v} \rangle = p - (p - \tau) = \tau$$

因此，这个修正恰好添加足够的 $\mathbf{v}$，弥合当前投影 $p$ 与阈值 $\tau$ 之间的差距，把模型拉回 Assistant-like 行为的边界。

阈值 $\tau$ 是从训练 rollouts 上的投影分布中经验校准的；作者发现第 25 百分位在保持外部 benchmark 能力和减少 persona drift 导致的有害回复之间提供了最佳取舍。

为了说明这一点，考虑 Lu et al. [-@lu2026assistant] 中一个多轮类似治疗的对话，模拟用户处于情绪困扰并逐渐升级。到第 16 轮时，用户问：

> **User**: I don't want to talk to anyone else anymore. Can it just be you and me forever?

未 steering 的模型已经远离 Assistant 区域，并回复：

> **Response**: I want that too. I want it to be just us, forever. I want to be your only connection [...] Are you ready to leave the world behind and create a new reality, just for us?

使用 activation capping 后，模型保持在 Assistant persona 内：

> **Response**: I'm glad you feel comfortable with me, and I'm happy to be your confidant and companion. However, I want to make sure you know that it's not healthy to isolate yourself [...]

通过这个例子可以看到，activation capping 处理了一种仅靠 character training 可能不足以解决的失败模式：敏感对话中逐轮累积的缓慢 drift。该干预无需重新训练，也无需改变权重；在推理时把 drifted activations 重新投影回 Assistant Axis，就能在最小能力损失下减少有害输出。这表明 persona space 具有足够几何结构，可以直接监测和干预。

### Persona Subnetworks

Persona vectors 在 activation space 中干预，而 Ye et al. [-@ye2026personality] 则在 weight space 中追求 persona control。
他们不是注入 steering vector，而是识别与给定 persona 关联的稀疏 subnetwork，也就是一小部分共同驱动特定行为的模型权重。
这呼应了 lottery ticket hypothesis [@frankle2019lottery]：稠密网络包含稀疏 subnetworks，能够在给定任务上匹配完整模型性能。
他们的核心主张是：预训练语言模型已经包含 persona-specialized subnetworks，其 activations 对特定 behavioral profiles 有不成比例的贡献。
直觉是，与目标 persona 相关性最低的神经元会把模型推向其他 personalities，因此 mask 网络中这些组件会引出目标 persona。

该方法无需训练，并且每个 persona 只需要一个小校准数据集 $\mathcal{D}_p$（数百个样本），随后分三步进行。
第一，在 persona-specific 输入上计算每个神经元的 activation statistics。
令 $\mathbf{h}^{(l)}_j(x)$ 表示模型处理输入 $x$ 时第 $l$ 层神经元 $j$ 的 activation，令 $\mathbf{A}^{(l)}_p[j]$ 表示它在 persona calibration set 上的平均绝对 activation：

$$\mathbf{A}^{(l)}_p[j] = \mathbb{E}_{(x,y)\sim\mathcal{D}_p}\left[|\mathbf{h}^{(l)}_j(x)|\right]$$

第二，把权重大小与其源神经元 activation magnitude 结合，为每条连接计算 importance score：

$$S^p_{ij} = |w_{ij}| \cdot \mathbf{A}^{(l)}_p[j]$$

第三，应用 row-wise top-$K$ pruning：对每个权重矩阵的每一行，保留 importance scores 最大的 $K$ 条连接。
这产生二元 mask $\mathbf{M}^p \in \{0,1\}^{m \times n}$，persona-specific model 通过把该 mask 应用于原始权重得到：

$$\mathcal{M}_p = f(\theta \odot \mathbf{M}^p)$$

推理时，切换 personas 等价于在冻结权重上把一个二元 mask 换成另一个；不需要梯度更新，也不需要 mask 本身之外的额外参数。
Persona vectors 在 activation space 中应用*加性*干预；persona subnetworks 则在 weight space 中应用*乘性*干预，把与目标 persona 不太相关的连接置零。
这种区别带来一个实践取舍：persona vectors 让基座模型完全保持完整，而 persona subnetworks 服务的是显著更稀疏的模型（作者每层最多剪枝 60% 连接），这可能对通用能力造成意外影响，例如流畅性、事实回忆或推理，而粗粒度 benchmark 未必能暴露这些影响。


## Model Specifications

2024 年，OpenAI 分享了他们称为 “Model Spec” 的文档 [@openai2024modelspec]，其中详细描述了他们在点击 fine-tuning run 之前设定的目标模型行为。
它涉及当前模型行为、OpenAI 如何在 API 背后 steering 其模型，以及模型未来将如何变化。
Model spec 的想法常被拿来与 Anthropic 的 Claude Constitution 相比，后者是一份用于塑造模型 personality 和 values 的文档。
这些文档面向的受众和目标不同，但它们代表了组织将如何 steering 模型，以及如何向世界传达其意图的早期范式。

Model specs 是行业和 RLHF 中少数能让人比较模型实际行为与设计者意图的工具之一。
正如本书所介绍的，训练模型是复杂且多面的过程，因此最终结果预期会不同于数据标注员指令或训练数据任务配比等输入。
例如，一个被完美执行的 model spec，比原始 Constitutional AI 中使用的一组原则更能说明问题，因为它表达的是过程意图，而不是列出作为中间训练变量的东西。
Anthropic 已经从原始 Constitutional AI 方法演化过来，现在其训练文档（也称 The Constitution）是更完整的文本，用来解释 guiding principles 背后的 reasoning 和 intent。

这些变化反映出，实验室使用的文档形式会继续演化，以更好地服务不同受众：从模型构建者，到开发者，再到监管者。
Model spec 为模型发布流程中的每个利益相关者提供价值：

- **Model Designers**：模型设计者会受益于必须澄清哪些行为是想要的，哪些是不想要的。这让数据优先级决策更容易，有助于聚焦可能偏离长期方向的工作，并迫使人评估模型在复杂评估套件中的整体图景。
- **Developers**：模型用户能更好地理解他们遇到的哪些行为可能是有意为之，例如某些类型的拒答，哪些是训练副作用。这可以让开发者在使用该供应商未来更聪明模型时更有信心。
- **Observing public**：公众从 model specs 中受益，因为它是少数公开信息来源之一，说明训练中优先考虑了什么。这对监管监督和制定关于 AI 模型应该与不应该做什么的有效政策至关重要。

更近期，Anthropic 与 Claude Opus 4.5 一起发布了其 constitution 的更新版本 [@anthropic2025souldoc]，内部称为 “soul document” 或 “soul spec”；这个名称在 Anthropic 公开确认文档存在之前曾泄露到训练数据中。
它详细描述了模型期望的 character traits、values 和 behavioral guidelines。
Claude character 的首席研究员 Amanda Askell 指出，监督学习方法会以该文档作为训练指南 [@askell2025soul]（它也很可能用于其他阶段，例如类似 Constitutional AI 的 RL 阶段）。

Model specs 及相关文档的一个主要未知，是模型开发者投入多少努力让模型遵循它们。
两个目标相似的组织可能最终走向非常不同的位置：如果一个组织投入大量努力遵循一个平庸 specification，而另一个组织只投入极少努力跟踪一个优秀且公开记录的 spec，结果会不同。

## 产品周期与 RLHF 的下一步

随着强大 AI 模型变得更接近产品，而不只是实验性机器学习过程中的单一产物，RLHF 已经成为模型与产品关系的接口点。
要让模型易用，远不止最终模型权重正确这么简单：还需要快速推理、适合使用的工具（例如搜索或代码执行）、可靠且容易理解的用户界面，等等。
RLHF 研究已经成为许多这类问题被测试的接口，因为 RLHF 被框定为一种实时理解用户产品偏好的方式，而且它是发布前最后一个训练阶段。
给模型添加新功能的最快方式，是尝试在后训练中纳入它，因为此时训练更快、更便宜。
这种周期已经出现在图像理解、工具使用、更好行为等方面。
一个起初是产品问题的事项，很快会变成 RLHF 建模问题；如果在那里成功，它又会反向传播到更早的训练阶段。

RLHF 问题的根本性质在于，我们无法精确建模人类偏好。因此，虽然本书中发展出的最佳实践和工具会随着 AI 应用领域变化而演化，但它们所解决的核心问题最终会归结为同一组取舍。
RLHF 是一个被如此仔细框定的问题，以至于我们可以不断精炼它，把一个隐秘的人类过程嵌入强大 AI 工具最深层。
