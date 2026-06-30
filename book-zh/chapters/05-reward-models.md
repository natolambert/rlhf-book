<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "指令微调"
prev-url: "04-instruction-tuning"
page-title: 奖励建模
search-title: "第 5 章：奖励建模"
meta-description: "奖励模型如何从偏好数据中训练，并在 RLHF 后训练流水线中作为学习得到的目标函数使用。"
next-chapter: "强化学习"
next-url: "06-policy-gradients"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "第 2 讲：IFT、奖励建模、拒绝采样（第 4、5、9 章）"
---

# 奖励建模

奖励模型（reward model）是现代 RLHF 方法的核心，因为复杂的人类偏好正是在这里被学习出来的。
它们让模型能够从难以明确规定的信号中学习。
它们把数据中的复杂特征压缩成一种可用于下游训练的表示 -- 这有点像魔法，再次展示了现代深度学习的复杂能力。
这些模型在核心优化中充当代理目标函数，后续章节将对此展开研究。
如 @fig:rm-role-in-rlhf 所示，奖励模型扮演着类似标准强化学习环境的角色，为智能体提供学习信号；但与固定环境不同的是，在 RLHF 中，我们可以从人类偏好中学习这个奖励函数。

从历史上看，奖励模型已经在强化学习研究中被广泛用作环境奖励的代理 [@sutton2018reinforcement]。
现代形式的奖励模型最初被提出，是作为研究价值对齐问题的一种工具 [@leike2018scalable]。
这类模型通常接收某种输入，并输出一个单一的标量奖励值。
这种奖励可以有多种形式 -- 在传统强化学习问题中，它试图近似该问题的真实环境奖励；但我们会看到，在 RLHF 中，奖励模型实际上输出的是某个输入“质量较高”的概率（也就是在成对偏好关系中被选中的答案）。
RLHF 中的奖励建模实践与逆强化学习密切相关；在逆强化学习中，问题是在给定行为轨迹的情况下近似一个智能体的奖励函数 [@ng2000algorithms]，它也与深度强化学习的其他领域有关。
高层问题表述是相同的，但实现方式和关注重点完全不同，因此它们通常被视为彼此独立的研究领域。

最常见的奖励模型通常称为 Bradley-Terry 奖励模型，也是本章的主要关注对象；它预测一段文本接近训练比较中“更受偏好”的文本的概率。
本节后面还会把它们与结果奖励模型（Outcome Reward Models, ORMs）、过程奖励模型（Process Reward Models, PRMs）以及其他类型的奖励模型进行比较。

*在本章中，我们用 $x$ 表示 prompt，用 $y$ 表示补全。这种记号在语言模型文献中很常见，因为相关方法通常作用于完整的 prompt-补全对，而不是单个 token。*

![RLHF 中的奖励模型扮演着标准强化学习中返回奖励的环境组件角色。关键区别在于，在 RLHF 中，我们可以控制并从人类偏好中学习这个奖励函数，而不是让它由环境固定给出。](images/rlhf-overview.png){#fig:rm-role-in-rlhf}

## 训练 Bradley-Terry 奖励模型

奖励模型的规范实现源自偏好的 Bradley-Terry 模型 [@BradleyTerry]。
关于如何为 RLHF 训练标准奖励模型，有两种常见表达方式 -- 它们在数学上等价。
首先，偏好的 Bradley-Terry 模型定义了在两个项目 $i$ 和 $j$ 的成对比较中，评判者更偏好 $i$ 而不是 $j$ 的概率：

$$P(i > j) = \frac{p_i}{p_i + p_j}.$$ {#eq:bradterry}

Bradley-Terry 模型假设每个项目都有一个潜在强度 $p_i > 0$，而观测到的偏好是这些底层强度的带噪声反映。
通常会用无界分数重新参数化 Bradley-Terry 模型，其中 $p_i = e^{r_i}$，由此得到如下形式：

$$P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i-r_j).$$ {#eq:bradterry_unbounded}

这里 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 logistic（sigmoid）函数，因此偏好概率只依赖于分数差 $r_i - r_j$。
只有分数之间的差值重要：给每个 $r_k$ 加上同一个常数 $c$，并不会改变 $P(i > j)$。
这些形式是对人类偏好的一种有用近似，并且在 RLHF 中通常效果良好。

要训练奖励模型，我们必须构造一个满足上述关系的损失函数。
实践中，这通常通过把语言模型转换成输出标量分数的模型来完成，常见做法是在模型最终隐藏状态之上加一个小的线性头，产生单个奖励值。
给定 prompt $x$ 以及两个采样得到的补全 $y_1$ 和 $y_2$，我们用奖励模型 $r_\theta$ 分别给二者打分，并把条件分数写作 $r_\theta(y_i \mid x)$。

奖励模型赋予 $y_1$ 优于 $y_2$ 的概率为：

$$P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}.$$ {#eq:bradterryrm}

我们把更受偏好的补全记为 $y_c$（chosen），把被拒绝的补全记为 $y_r$。

由此得到的损失会鼓励奖励模型给人类更偏好的补全分配比被拒绝补全更高的分数，并用 sigmoid 把分数差转换为概率。
@eq:bradterryrm 中的偏好似然是起点。我们先把分子和分母同时除以 $\exp\left(r_\theta(y_c \mid x)\right)$，将该似然改写成 sigmoid 形式：

$$
\begin{aligned}
P(y_c > y_r \mid x)
&= \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right) + \exp\left(r_\theta(y_r \mid x)\right)} \\
&= \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)\left(1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}\right)} \\
&= \frac{1}{1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}} \\
&= \frac{1}{1 + \exp\left(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x))\right)} \\
&= \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right).
\end{aligned}
$$ {#eq:bradterryrm_sigmoid}

随后，通过在偏好数据集 $D$ 上做最大似然来拟合奖励模型，也就是最大化观测偏好的期望对数似然。由于对数函数是单调的，这等价于最小化期望负对数似然：

$$
\begin{aligned}
\theta^* &= \arg\max_\theta \mathbb{E}_{(x, y_c, y_r) \sim D}\left[ \log P(y_c > y_r \mid x) \right] \\
&= \arg\min_\theta \mathbb{E}_{(x, y_c, y_r) \sim D}\left[ -\log \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right) \right].
\end{aligned}
$$ {#eq:bradterryrm_deriv}

在对数据集取平均*之前*取对数，正是负对数似然损失成为正确目标函数的原因：最大化期望概率 $\mathbb{E}[P]$ 与最大化期望对数概率 $\mathbb{E}[\log P]$ 并不相同。

单样本损失就是上面期望内部的 log-sigmoid 表达式，如 [@ouyang2022training] 和其他工作所用：
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodeling1}

第二种是使用 softplus 函数 $\log(1+e^x)$ 表示的数学等价形式，如 [@askell2021general] 和其他工作所用：
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$ {#eq:rewardmodeling2}

令 $\Delta = r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x)$，并使用 $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$，即可看出二者等价；这意味着 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta}) = \log\left(1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)}\right)$。
这两种写法都出现在 RLHF 文献中。

![训练偏好奖励模型需要成对的 chosen 和 rejected 补全。模型从序列级表示中为每个补全计算一个标量分数，这个表示通常来自序列结束（EOS）token 的隐藏状态；对比损失只依赖两个分数之间的差值。](images/pref_rm_training.png){#fig:pref_rm_training data-dark-src="images/pref_rm_training-dark.png"}

## 默认奖励模型架构

奖励模型最常见的实现方式，是通过类似 Transformers 的 `AutoModelForSequenceClassification` 的抽象：在语言模型后接一个小型线性头，并在训练或推理时为一个 prompt-补全对产生标量奖励分数。
推理时，模型会以单个 logit 的形式输出*这段文本被选中的相对可能性*。

也存在其他实现选项，例如直接从最终 embedding 接一个线性层，但它们在开放工具中不那么常见。

## 实现示例

实现奖励建模损失相当简单。
更多实现难点在于搭建独立的数据加载器和推理流水线。
给定正确的数据加载器，其中包含已经 tokenized 的 chosen 和 rejected prompt 及其补全，损失可以这样实现：
```python
import torch.nn as nn
# inputs_chosen / inputs_rejected include the prompt tokens x and the respective
# completion tokens (y_c or y_r) that the reward model scores jointly.
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

从更大的图景看，这通常发生在一个因果语言模型内部（即从左到右生成 token、在所有前文条件下预测每个 token 的模型）；该模型额外添加了一个头，并用上面的损失学习，把最终隐藏状态转换为输入的分数。
代码接收标准 transformer 输入 -- `input_ids`（tokenized text）和 `attention_mask`（标记真实 token 与 padding）-- 并提取最后一个真实 token 处的隐藏状态（模型对输入的内部表示），随后通过线性层产生标量奖励。
该模型结构如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryRewardModel(nn.Module):
    """
    Standard scalar reward model for Bradley-Terry preference learning.

    Usage (pairwise BT loss):
        rewards_chosen = model(**inputs_chosen)    # (batch,)
        rewards_rejected = model(**inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def _sequence_rep(self, hidden, attention_mask):
        """
        Get a single vector per sequence to score.
        Default: last non-padding token (EOS token); if no mask, last token.
        hidden: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """

        # Index of last non-pad token in each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        A forward pass designed to show inference structure of a standard reward model.
        To train one, this function will need to be modified to compute rewards from both
         chosen and rejected inputs, applying the loss above.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]

        # One scalar reward per sequence: (batch,)
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.head(seq_repr).squeeze(-1)

        return rewards
```

在本节以及后续内容中，奖励模型（以及后训练的大部分内容）的实现复杂性，主要在于正确构造数据加载器和分布式学习系统。
注意，在训练奖励模型时，最常见的实践是只训练 1 个 epoch，以避免过拟合。

## 奖励模型变体

奖励建模是 RLHF 中一个相对探索不足的领域。
传统的奖励建模损失已经在许多有影响力的工作中被修改过，但这些修改还没有固化为单一的最佳实践。

### 偏好间隔损失

当标注者在 Likert 量表（带有有序类别、用于表示偏好强度的评分量表，例如 1--5）上提供分数或排序时，关系量的幅度可以用于训练。
最常见的做法是沿着偏好方向把数据二值化，把相对评分或排序强度中的混合信息简化为 chosen 和 rejected 补全。
额外信息（例如偏好强度）已经被用于改进模型训练，但它还没有收敛为标准实践。
Llama 2 提出使用两个数据点之间的间隔 $m(y_c, y_r)$ 来区分偏好强度：

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)$$ {#eq:rewardmodelingmargin}

例如，每个补全通常会按质量获得 1 到 5 的排序分数。
如果 chosen 样本被分配了 5 分，而 rejected 样本被分配了 2 分，则间隔 $m(y_c, y_r)= 5 - 2 = 3$。
也可以探索其他用于计算间隔的函数。

注意，在 Llama 3 中，该团队观察到随着规模扩大，收益递减，因此移除了间隔项。

### 平衡每个 prompt 的多重比较

InstructGPT 研究了对每个 prompt 使用 $K = 4$ 到 $9$ 个补全进行排序的影响，这会从每个 prompt 产生 $\binom{K}{2}$ 个成对比较 [@ouyang2022training]。
由于这些比较高度相关（它们共享同一个 prompt），如果天真地把它们打乱放入数据集，会导致奖励模型过拟合。
为了解决这个问题，他们按每个 prompt 的每次比较对损失更新加权 -- 如果不重新加权，拥有更多补全的 prompt 仅仅因为生成了更多配对，就会贡献更多总损失。
实践中，来自单个 prompt 的所有 $\binom{K}{2}$ 个比较通常会被放在同一个训练 batch 中并一起取平均，因此每个 prompt 贡献一次分组更新，而不是分散出现在许多独立 batch 中。
这减少了对单个 prompt 的过拟合，并防止拥有更多采样补全的 prompt 主导损失。
损失函数变为：

$$\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodelinginstructgpt}

### K-Wise 损失函数

还有许多其他形式可以为 RLHF 构造合适的人类偏好模型。
其中一个例子是基于 Plackett-Luce 模型 [@liu2019learning] 的 K-wise 损失函数，它被早期流行的 RLHF 模型 Starling 7B 和 34B [@zhu2024starling] 使用。

Zhu et al. 2023 [@zhu2023principled] 将设定形式化如下。
给定一个 prompt 或状态 $s^i$，从 $P(a_0,\cdots,a_{K-1}|s^i)$ 中采样 $K$ 个动作 $(a_0^i, a_1^i, \cdots, a_{K-1}^i)$。
然后，标注员按偏好对这 $K$ 个动作排序，产生一个置换 $\sigma^i: [K] \mapsto [K]$，其中 $\sigma^i(0)$ 是最受偏好的动作。这样就得到所有 $K$ 个项目完整排序上的 Plackett-Luce 概率：

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$ {#eq:kwise_rm}

当 $K = 2$ 时，这会退化为用于成对比较的 Bradley-Terry（BT）模型。
无论如何，一旦训练完成，这些模型在 RLHF 训练期间的使用方式与其他奖励模型类似。


## 结果奖励模型

<!-- Huge thanks to Hangliang Ren, graduate student at Northeastern University for helping with this section (and PRMs), see https://github.com/myhott163com/RLHF_ORM_PRM -->

语言模型和其他 AI 系统的大多数*偏好调优*都使用上面讨论的 Bradley-Terry 模型完成。
对于推理密集型任务，可以使用结果奖励模型（Outcome Reward Model, ORM）。
ORM 的训练数据构造方式与标准偏好调优相似。
这里，我们有一个问题陈述或 prompt $x$，以及两个补全 $y_1$ 和 $y_2$。
这里使用的归纳偏置是：一个补全应该是该问题的正确解，另一个是不正确解，于是得到 $(y_c,y_{ic})$。

所用模型的架构与标准奖励模型非常相似，都是在能够输出单个 logit 的模型后接一个线性层（对于 RM 而言）-- 但对于 ORM，后续训练目标略有不同 [@cobbe2021gsm8k]：

> [We] train verifiers with a joint objective where the model
learns to label a model completion as correct or incorrect, in addition to the original language modeling objective.
> Architecturally, this means our verifiers
are language models, with a small scalar head that outputs predictions on a per-token basis.
> We implement this scalar head as a single bias parameter and single gain parameter that operate on the logits outputted by the language model's final unembedding layer.

换句话说，这被实现为一个语言建模头，它可以在每个 token 上预测两个类别（正确为 1，错误为 0），而不是传统 RM 中为整个序列输出一个 logit 的分类头。
形式上，沿用 [@lyu2025exploring]，这是一个逐 token 的二元交叉熵损失：

$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}\left[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))\right]$$ {#eq:orm_loss}

其中 $r \in \{0,1\}$ 是二元标签：1 表示给定 prompt 的正确答案，0 表示不正确答案；$p_\theta(s)$ 是与正在训练的模型预测正确性概率成比例的标量。
在代码中，这个结果标签会被复制到每个补全 token 上，而 prompt token 会用 `-100` 掩蔽，因此不参与损失计算。

实现结果奖励模型（以及其他类型，正如我们将在过程奖励模型中看到的）时，需要根据补全是否为正确样本，在每个 token 上应用交叉熵损失。
这更接近语言建模损失，因为它不需要标准 Bradley-Terry 奖励模型中结构化的 chosen-rejected 形式。
在下面简化的 ORM 训练设定中，我们既不采样新 token，也不在下一 token 预测任务上训练 LLM；我们把固定的 prompt-补全序列送入 backbone，并训练 ORM 头来预测正确性标签。

模型结构可以如下：

```python
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids contains a full prompt+completion sequence.
        labels is token-aligned: prompt tokens are -100, and each completion
         token repeats the sequence outcome label (1=correct, 0=incorrect).
        If labels=None, this is an inference-only forward pass and the loss is
         returned as None.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One scalar logit per token: (batch, seq_len)
        logits = self.head(hidden).squeeze(-1)

        # Inference-only forward pass: no loss is computed.
        if labels is None:
            return None, logits
        # Only compute loss on completion tokens (labels 0 or 1)
        # Prompt tokens have labels = -100
        mask = labels != -100
        loss = None
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float()
            )
        else:
            loss = logits.sum() * 0
        return loss, logits
```

损失的简化版本如下：

```python
# Feed the full prompt+completion sequence once; no token sampling happens here.
# Assume model already has: model.lm (backbone) + model.head
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)
# This will sometimes be compressed as model.forward() in other implementations

# Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits_per_token[mask], labels[mask].float()
)
```

这里的重要直觉是，ORM 会在序列中的每个 token 上输出正确性概率（但它只由最终答案来评判 -- 推理错误不会在 ORM 训练过程中被捕捉）。
这可能是一个有噪声的过程，因为更新和损失会根据结果与注意力映射在每个 token 上传播。

![推理时，结果奖励模型会在补全 token 上输出逐 token 的正确性概率。prompt token 会在打分时被忽略，而补全概率可以聚合成回复级分数，用于验证、过滤或重排序。](images/orm_inference.png){#fig:orm_inference data-dark-src="images/orm_inference-dark.png"}

![训练结果奖励模型会使用来自验证器或数据集的离线标签（例如，对正确补全全部标为 1）。每个补全 token 都用二元交叉熵对结果标签进行训练，逐 token 概率再被聚合成最终分数，用于验证、过滤或重排序。](images/orm_training.png){#fig:orm_training data-dark-src="images/orm_training-dark.png"}

这些模型仍在继续使用，但在开源 RLHF 工具中的支持较少。
例如，开创性工作 *Let's Verify Step by Step* [@lightman2023let] 使用了同类 ORM，但没有包含损失中的语言建模预测部分。
于是，最终损失是在每个 token 上的交叉熵损失，用来预测最终答案是否正确。

由于支持不足，结果奖励模型（ORM）这个术语已经被以多种方式使用。
有些文献，例如 [@lyu2025exploring]，继续沿用 Cobbe et al. 2021 的原始定义；另一些文献则更宽泛地使用它，指任何经过训练、用于预测补全是否正确的验证器。


## 过程奖励模型

过程奖励模型（Process Reward Models, PRMs）最初被称为 process-supervised reward models，是在思维链推理过程中的每一个*步骤*输出分数的奖励模型。
它们不同于只在 EOS token 输出分数的标准 RM，也不同于在每个 token 输出分数的 ORM。
过程奖励模型需要在每个推理步骤末尾获得监督，然后以类似方式训练：步骤中的 token 被训练到它们对应的目标 -- 在 PRM 中目标是该步骤，在 ORM 中目标则是整个回复。

沿用 [@lightman2023let]，二元标注的 PRM 通常用逐步骤交叉熵损失优化：

$$\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_{s_i} \log r_\theta(s_i \mid x, s_{< i}) + (1 - y_{s_i}) \log \left(1 - r_\theta(s_i \mid x, s_{< i})\right) \right] $$ {#eq:prm_loss}

其中 $s$ 是一条采样得到的思维链，包含 $K$ 个带标注的步骤；$y_{s_i} \in \{0,1\}$ 表示第 $i$ 个步骤是否正确；$r_\theta(s_i \mid x, s_{< i})$ 是 PRM 在给定原始 prompt $x$ 以及所有先前步骤 $s_{< i}$ 的条件下，预测步骤 $s_i$ 有效的概率。

下面是来自 HuggingFace 的 TRL（Transformer Reinforcement Learning）[@vonwerra2022trl] 的一个例子，展示如何在 trainer 中打包这种逐步骤标签：

```python
# Get the ID of the separator token and add it to the completions
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# Create the label
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

传统上，PRM 使用语言建模头训练，该头只在推理步骤末尾输出一个 token，例如对应双换行或其他特殊 token 的位置。
这些预测通常是 -1 表示错误、0 表示中性、1 表示正确。
这些标签不一定对应模型是否走在正确路径上，而是对应该步骤本身是否正确。

![过程奖励模型只在步骤边界处（例如换行 token）提供监督。每个步骤获得一个 3 类标签：正确（+1）、中性（0）或错误（-1）。训练期间所有其他 token 都会被掩蔽。](images/prm_training_inference.png){#fig:prm_training_inference data-dark-src="images/prm_training_inference-dark.png"}

下面展示一个 PRM 的示例构造。

```python
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        The inputs are tokenized prompts and completions, where the end of a
         "reasoning step" is denoted by a designated separator token such as a
         newline or other special marker rather than batch padding.
        labels will be a list of labels, True, False, and Neutral (3 labels) which
         will be predicted by the model.
        If labels=None, this is an inference-only forward pass and the loss is
         returned as None.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One logit vector per token: (batch, seq_len, num_classes)
        logits = self.head(hidden)

        # Inference-only forward pass: no loss is computed.
        if labels is None:
            return None, logits
        # Only compute loss at step boundaries (where labels != -100)
        # Labels map: -1 -> 0, 0 -> 1, 1 -> 2 (class indices)
        mask = labels != -100
        loss = None
        if mask.any():
            loss = F.cross_entropy(
                logits[mask], labels[mask]
            )
        else:
            loss = logits.sum() * 0
        return loss, logits
```

核心损失函数看起来与结果奖励模型非常相似，只是标签被应用在不同间隔上。
```python
# Assume model outputs 3-class logits per token
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = model.head(hidden)  # (batch, seq_len, 3)

# 3-class labels at step boundaries only: 0=-1, 1=0, 2=1 (others masked as -100)
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

## 比较奖励模型类型（以及价值函数）

前面介绍的各种奖励模型类型，展示了在 RLHF 和其他后训练方法中衡量“质量”的一系列方式。
下面总结了这些模型预测什么，以及它们如何训练。

::: {.table-wrap}
| Model Class | What They Predict | How They Are Trained | LM structure |
|------------|------------------|---------------------|--------------|
| **奖励模型** | 序列级质量分数 $r_\theta(x, y)$ | 补全之间的成对（或 N-wise）比较上的对比损失 | EOS/最后一个 token 隐藏状态上的线性头 |
| **结果奖励模型** | 答案在每个 token 上正确的概率 | 带标签的结果配对（例如可验证领域中的成功/失败） | 逐 token 二元交叉熵头；标签重复结果标签 |
| **过程奖励模型** | 中间步骤末尾的奖励或分数 | 使用中间反馈或逐步骤标注训练（在推理步骤中逐 token 训练） | 逐 token 头预测步骤正确性（-1、0、1） |
| **价值函数** | 给定当前状态时的期望回报 | 通过对序列中每个位置做回归来训练 | 带逐 token 输出的标量回归头 |
Table: 比较奖励模型的类型。 {#tbl:rm_compare}
:::

关于这张表中的区分，需要说明几点，因为模型类型之间的边界并不总是泾渭分明：

- 在偏好调优和推理训练中，价值函数通常使用折扣因子 1，这会让价值函数更接近结果奖励模型，只是训练损失不同。
- 过程奖励模型可以通过从中间状态做 rollout 并收集结果数据来监督。这会混合多种思想；但如果*损失*使用逐推理步骤标签，最好仍称之为 PRM。

**如果用正确/错误配对来训练 Bradley-Terry 成对模型，会怎样？**
关于结果奖励模型的许多混淆，来自少量文献：它们在由答案正确性派生出来的成对数据上训练奖励模型。
在这个领域中，你把对某个问题的正确答案设为 chosen 回复，把对*同一个问题*的不正确答案设为 rejected 回复。
从技术上讲，这不是 ORM，它仍然直接使用对比式的序列级损失训练。
从技术上讲，它仍然是 Bradley-Terry 模型，属于我们介绍的第一类模型。

**ORM 与价值函数。**
ORM 和价值函数看起来可能相似，因为二者都用相同的头架构产生逐 token 输出，但它们在*预测什么*以及*目标来自哪里*上不同：

- **ORM** 预测一个即时的、token 局部的量：$p(\text{correct}_t)$ 或 $r_t$。目标来自*离线标签*（验证器或数据集把 token/序列标记为正确或错误）。
- **价值函数** 预测期望的*剩余*回报：$V(s_t) = \mathbb{E}\left[\sum_{k \geq t} \gamma^{k-t} r_k \mid s_t\right]$。目标通常根据当前策略 $\pi_\theta$ 下的 *on-policy rollout* 计算，并随着策略变化而变化（从技术上说，价值函数也可以是 off-policy 的，但这在语言建模工作中尚未形成既定做法）。

如果定义一个稠密 token 奖励 $r_t = \mathbb{1}[\text{token is correct}]$ 并使用 $\gamma = 1$，那么 ORM 学习的是 $r_t$（或 $p(r_t = 1)$），而价值头学习的是剩余和 $\sum_{k \geq t} r_k$。
它们可以共享相同的基础模型和头维度，但*语义与监督流水线*不同：ORM 由固定标签离线训练，而价值函数是 on-policy 训练的，并用于为策略梯度计算优势 $A_t = \hat{R}_t - V_t$。

### 不同奖励模型类型的推理

这些模型在推理时（即训练完成之后）以不同方式处理数据，以便支持 RM 所用于的一系列任务。

**Bradley-Terry RM（偏好模型）：**

- *输入：* prompt $x$ + 候选补全 $y$
- *输出：* 通过 EOS/最后一个 token 隐藏状态上的线性层得到单个标量 $r_\theta(x, y)$
- *用途：* 对 $k$ 个补全重排序，选择 top-1（best-of-N 采样）；或为 RLHF 提供终止奖励
- *聚合：* 标量输出不需要聚合

**结果 RM：**

- *输入：* prompt $x$ + 补全 $y$
- *输出：* 补全 token 上的逐 token 概率 $p_t \approx P(\text{correct at token } t)$
- *用途：* 为已完成候选打分；通过均值、最小值（尾部风险）或乘积 $\prod_t p_t$（等价于对数概率求和 $\sum_t \log p_t$）进行聚合
- *聚合选择：* 平均正确性、最小 $p_t$、最后 $m$ 个 token 上的平均值，或在任意 $p_t < \tau$ 时触发阈值标记

**过程 RM：**

- *输入：* prompt $x$ + 带步骤边界的推理轨迹
- *输出：* 步骤边界处的分数（例如正确/中性/错误的类别 logit）
- *用途：* 为完整思维链打分；或通过剪枝低分分支来引导搜索/解码
- *聚合：* 在步骤上聚合（不是 token）-- 平均步骤分数、最小值（快速失败），或偏向后续步骤的加权和

**价值函数：**

- *输入：* prompt $x$ + 当前前缀 $y_{\leq t}$（一个状态）
- 输出：补全中每个 token 位置的 $V_t$（从状态 $t$ 出发的期望剩余回报）
- 用途：在强化学习训练期间计算逐 token 优势 $A_t = \hat{R}_t - V_t$；每一步的价值充当基线
- *聚合：* 通常取最后一个生成 token 处的 $V$；其解释不同于“正确性概率”

总之，理解这些不同模型的方式是：

- **RM：** “这个完整答案有多好？”→ 标量值
- **ORM：** “哪些部分看起来是正确的？”→ 逐 token 正确性
- **PRM：** “推理步骤是否可靠？”→ 逐步骤分数
- **Value：** “从这里开始还剩多少奖励？”→ RL 优势的基线

## 生成式奖励建模（又称 LLM-as-a-judge）

由于偏好数据成本高昂，一个大型研究方向兴起：使用现有语言模型作为人类偏好的评判者，或用于其他评估设定 [@zheng2023judging]。
核心思想是用指令、一个 prompt 和两个补全来提示语言模型如何评判（很像人类标注员会做的事）。
下面是这里一项开创性工作中用于聊天评估 MT-Bench [@zheng2023judging] 的示例 prompt：

```text
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
You should choose the assistant that follows the user's instructions and answers the user's question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.
Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names of the assistants.
Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
```

鉴于 LLM-as-a-judge 在评估中的有效性，它催生了许多其他评估，例如 AlpacaEval [@dubois2024length]、Arena-Hard [@li2024crowdsourced] 和 WildBench [@lin2024wildbench]；许多人开始用 LLM-as-a-judge 代替奖励模型来创建和使用偏好数据。

围绕如何使用所谓的“生成式奖励模型”（Generative Reward Models）[@mahan2024generative]
[@zhang2024generative] [@ankner2024critique] 已经形成了一个完整研究领域（其中包括*专门*训练来成为有效评判者的模型 [@kim2023prometheus]），但在 RM 评估上，它们通常落后于现有奖励模型，这表明奖励建模仍是当前 RLHF 的一项重要技术。

提高 LLM-as-a-judge 工作流鲁棒性的一个常见技巧，是使用 0 的采样温度来降低评分方差。

## 延伸阅读

奖励建模的学术文献在 2024 年确立了自身地位。
奖励建模早期进展的大部分重点，是建立 benchmark 并识别行为模式。
第一个 RM benchmark RewardBench 为测试奖励模型提供了通用基础设施 [@lambert2024rewardbench]。
此后，RM 评估扩展到类似一般后训练模型可用评估的类型：有些评估测试模型在有已知真实答案领域中的预测准确率 [@lambert2024rewardbench]，另一些则更接近使用 LLM-as-a-judge 执行的“vibes”评估，或与其他 benchmark 的相关性评估 [@wen2024rethinking]。

新 benchmark 的例子包括：

- **纯文本（通用聊天/偏好）：** RMB [@zhou2024rmb]、RewardBench2 [@malik2025rewardbench]、Preference Proxy Evaluations [@frick2024evaluate] 或 RM-Bench [@liu2024rm]。
- **专门纯文本（数学等）：** multilingual reward bench（M-RewardBench）[@gureja2024m]、用于检索增强生成（RAG）的 RAG-RewardBench [@jin2024rag]、用于拼写错误的 ReWordBench [@wu2025rewordbench]、RewardMATH [@kim2024evaluating] 或 AceMath-RewardBench [@liu2024acemath]。
- **过程 RM：** PRM Bench [@song2025prmbench] 或 ProcessBench [@zheng2024processbench]，以及 VisualProcessBench [@wang2025visualprm] 或 ViLBench [@tu2025vilbench] 等视觉 benchmark。
- **智能体式 RM：** Agent-RewardBench [@men2025agentrewardbench] 或 CUARewardBench [@lin2025cuarewardbench]。
- **多模态：** MJ-Bench [@chen2024mj]、Multimodal RewardBench [@yasunaga2025multimodal]、VL RewardBench [@li2024vlrewardbench] 或 VLRMBench [@ruan2025vlrmbench]。

要理解奖励模型*训练*方面的进展，可以参考新的奖励模型训练方法，包括 aspect-conditioned models [@wang2024interpretable]、高质量人类数据集 [@wang2024helpsteer2] [@wang2024helpsteer2p]、规模化实验 [@adler2024nemotron]、大量实验研究 [@touvron2023llama]，以及去偏数据 [@park2024offsetbias]。

## 建议实验

配套代码仓库在 `code/reward_models/` 中包含小型奖励模型训练脚本。
这些脚本旨在作为学习练习，而不是经过调优的参考配方。
从干净的 `code/` 环境开始，运行 `uv sync`，然后一次运行一个实验。

1. **在 UltraFeedback 上训练 Bradley-Terry 偏好奖励模型。**
   运行：

   ```bash
   cd code/
   uv run python -m reward_models.train_preference_rm --samples 2000 --epochs 1
   ```

   观察 demo 和 W&B 日志中，chosen 与 rejected 回复之间的奖励间隔是否变大。
   然后改变 `--samples`、`--lr` 和 `--model-id`，观察信号何时变得有噪声或不稳定。

2. **比较结果监督与过程监督。**
   运行 GSM8K 结果奖励模型和 PRM800K 过程奖励模型：

   ```bash
   cd code/
   uv run python -m reward_models.train_orm --samples 400 --epochs 2
   uv run python -m reward_models.train_prm --samples 500 --epochs 2
   ```

   比较每个模型在训练后能够打分的内容：ORM 应该能区分正确和错误的最终答案，而 PRM 应该能在中间推理步骤上分配分数。
   这就是序列级、结果级和过程级监督之间区别的实践版本。

3. **添加一个小型留出奖励模型评估。**
   一个有用的贡献，是为 `reward_models/` 添加一个 50 到 200 个样本的评估，在不需要完整训练运行的情况下报告准确率或偏好对排序。
   让评估保持足够小，使其可以在调参时使用。
