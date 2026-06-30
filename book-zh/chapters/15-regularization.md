<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "过度优化"
prev-url: "14-over-optimization"
page-title: 正则化
search-title: "第 15 章：正则化"
meta-description: "正则化方法如何让 RLHF 和后训练更新保持有用，同时不退化基座模型。"
next-chapter: "评估"
next-url: "16-evaluation"
---

# 正则化

在本书中，我们已经学习了许多修改模型的方法，使其能够从人类偏好、可验证奖励和其他有价值信号中学习。
我们使用的所有方法都非常强大，可能导致模型相对于上一训练阶段得到的强大通用模型（通常称为参考模型）变化过大。
当模型从某个给定奖励中学得过多，导致分布外性能下降时，这称为“过度优化”（上一章已经讨论过）。

在整个 RLHF 优化过程中，会使用许多正则化步骤来防止对奖励模型过度优化。
在这些语境中，过度优化看起来像是模型输出无意义文本。
优化“脱轨”的例子包括：模型输出看似可跟随的数学推理但答案极其错误、重复文本、切换语言，或产生过多特殊字符。
本章介绍用于控制模型优化的不同方法。

截至 2026 年，大多数 RLHF 实现中使用的最流行变体，是在生成样本上约束当前策略到参考策略的 KL 距离。
“KL distance”是一个口语化说法，用来表达训练过程中的*优化距离*；尽管 KL divergence 这个用于衡量两个概率分布分离程度的底层数学方法，并不满足真正距离度量所需的形式性质（把这个数叫作距离，只是比“分布差异的数值度量”更容易）。
文献中出现过许多其他正则化技术，随后又在该研究线的下一代模型中消失。
也就是说，生成上的核心 KL 距离之外的正则化，常用于稳定实验设置，而这些设置在下一代中又可能被简化。
不过，理解约束 RLHF 优化的工具仍然很重要。

*本章中，我们用 $x$ 表示 prompts，用 $y$ 表示 completions。这种记号在语言模型文献中很常见，因为这些方法作用于完整的 prompt-completion 对，而不是单个 token。*

在带奖励模型 $r_\theta$ 的 RLHF 框架中，一般形式如下：

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

参考实现为：

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{ref}}(y \mid x) \right)
$$ {#eq:kl_standard}

## RL 优化中的 KL 散度

数学定义见附录 A。
KL divergence 衡量一个概率分布相对于另一个分布漂移了多远；当 KL 为零时，两个分布产生完全相同的输出。
回忆其定义如下：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

在 RLHF 中，关注的两个分布通常是新模型版本的分布，例如 $P(x)$，以及参考策略的分布，例如 $Q(x)$。
不同优化器使用不同 KL 方向。本书中最常见的 “KL Penalty” 称为相对于参考策略的 reverse KL。实践中，它会化约为一个 Monte Carlo 估计：从 RL 模型采样 token，并用参考模型计算概率。直观地说，这个 reverse KL 有一个数值性质：当新模型 $P$ 或 $\pi_{\text{RL}}$ 在原始参考模型赋予低概率的位置放置大量概率质量时，会施加很大惩罚。

另一个 KL 方向仍常用于机器学习，例如某些 RL 算法内部 trust region 计算中。直观地说，当新模型的更新*没有*把概率放到 $Q$ 或 $\pi_{\text{ref}}$ 中的高似然区域时，这种惩罚会惩罚新模型。这更接近蒸馏或行为克隆使用的目标。

### 参考模型到生成

KL 惩罚最常见的实现方式，是比较训练期间生成 token 与静态参考模型之间的距离。
直觉是：你从中开始训练的模型有一种希望保持接近的风格。
这个参考模型最常是指令微调模型，但也可以是之前的 RL checkpoint。
简单替换后，我们采样的模型变成 $\pi_{\text{RL}}(x)$ 和 $\pi_{\text{ref}}(x)$，如 @eq:kl_standard 所示（在标准定义中，当应用于 RL KL 惩罚时通常是 $P$ 和 $Q$）。
这种 KL divergence 惩罚早在大语言模型流行之前就已被用于对话智能体 [@jaques2017sequence]，而 KL control 很快成为微调预训练模型的核心技术 [@jaques2020human]。

### 实现示例

实践中，KL divergence 的实现通常会被近似 [@schulman2016klapprox]，使实现简单得多。
根据上述定义，当直接从分布 $P$ 采样时，KL 的求和可以转换为期望（这里 $x$ 是样本空间上的通用随机变量，不是本书其他地方使用的 prompt 记号）。
在这种情况下，$P$ 是当前正在训练模型的生成分布（即不是参考模型）。
于是，KL divergence 的计算变为：

$$
\mathcal{D}_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

这种基于样本的形式实现起来简单得多，尤其是直接处理语言模型训练中频繁使用的 log probabilities 时。

```python
# Step 1: generate() autoregressively samples a full sequence token by token
generated_tokens = model.generate(inputs)

# Step 2: forward() runs a single pass over the sequence to get per-token logits (no sampling)
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# Step 3: Convert logits to log-probabilities
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# Step 4: Gather the probability each model assigns to the tokens that were actually generated
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# Step 5: Sum to get sequence-level log-probs; their difference approximates KL
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

一些示例实现包括 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) 和 [Hamish Ivison's Jax Code](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)。


## 隐式正则化

本章其他部分描述的是*显式*正则化：KL 惩罚、预训练梯度，以及实践者有意加入训练目标的 margin losses。
越来越多的实证工作表明，基于 RL 的后训练也提供*隐式*正则化，即一种内置的、抵抗记忆化和灾难性遗忘的能力，它来自 on-policy 优化本身的结构。
这是由损失更新的性质导致的，即使没有任何显式工具来控制 RL 训练，例如 KL 惩罚或 replay buffers。

### SFT 会记忆，RL 会泛化

后训练社区面临的一个核心问题是：在单一任务上训练时，模型学到的是可迁移到未见变体的通用规则，还是记住了训练分布的表面模式？
Chu et al. 2025 [@chu2025sft] 通过一项受控实证研究回答了这个问题，直接隔离后训练方法（SFT 与 RL）对分布外（OOD）泛化的影响。
答案很明确：RL 学到可迁移规则，而 SFT 记住训练数据，并在分布转移下崩溃。

该研究使用两个带有内置规则变化的环境来理解这种取舍：

- **GeneralPoints** 是一个算术纸牌游戏，模型收到四张扑克牌，并必须用运算符（+、-、*、/）组合其数值，达到目标数字（默认 24）。OOD 测试改变了人头牌的计分方式：训练使用一种规则（Jack、Queen、King 都算 10），评估使用另一种规则（Jack = 11、Queen = 12、King = 13）。

- **V-IRL** 是一个真实世界视觉导航任务，模型根据语言指令穿过城市街道，并沿途识别地标。OOD 转移把动作空间从绝对方向（north, east）切换为相对方向（left, right）。

在所有任务变体中，随着训练计算量增加，RL 一贯提升 OOD 性能；而 SFT 虽然提升分布内性能，却一贯*退化* OOD 性能。
差异幅度很惊人：在仅语言输入的 V-IRL 中，OOD 转移从绝对方向坐标变为相对方向坐标，RL 将 OOD 逐步准确率从 80.8% 提升到 91.8%，而 SFT 将其从 80.8% 压垮到 1.3%。
SFT 模型不只是未能泛化：它摧毁了基座模型已有的空间推理能力，退化成从指令短语到绝对方向的查找表。

### 通过行动保留：On-Policy 数据缓解遗忘

上一节展示了 RL 在单一任务上能泛化，而 SFT 会记忆。
Chen et al. 2025 [@chen2025retainingdoingroleonpolicy] 提出了互补问题：当按顺序训练多个任务时，模型能否保留已知内容？
他们发现，RL 在目标任务上取得相当或更高增益，同时比 SFT 遗忘少得多，并把这种优势追溯到两类目标所优化内容的根本差异。

为了理解两种方法为何如此不同，我们可以从 KL divergence 的视角来看它们的目标。
本节中，我们先说明两种常见后训练方法可以映射到 KL divergence 的两个方向，然后解释把它们作为损失函数使用时的数值行为如何转化为不同模型行为。

KL divergence 定义为两个分布之间 log-ratio 的期望，$\mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]$，可以写成两个方向的 log difference：

- **Forward KL**：$\text{KL}(P \| Q) = \mathbb{E}_{x \sim P}\!\left[\log P(x) - \log Q(x)\right]$
- **Reverse KL**：$\text{KL}(Q \| P) = \mathbb{E}_{x \sim Q}\!\left[\log Q(x) - \log P(x)\right]$

其中 $P$ 是目标分布，$Q$ 是我们用参数 $\theta$ 建模的分布。
关键区别在于从哪个分布采样：forward KL 从目标（或最优）分布 $P$ 采样，而 reverse KL 从我们的策略 $Q$ 采样。
在下面推导中，$P$ 对应目标 $\pi_\star$（分析 SFT 时是训练数据分布，分析 RL 时是奖励最优策略），$Q$ 对应学习到的策略 $\pi_\theta$（我们正在训练的对象）。
SFT 把目标放在前面，即 $\text{KL}(\pi_\star \| \pi_\theta)$；RL 则反转顺序，即 $\text{KL}(\pi_\theta \| \pi_\star)$，从而改变采样分布。
样本提供学习数据。目标函数，无论 SFT 还是 RL，则塑造模型如何从这些数据中学习。

#### SFT Forward KL

从 forward KL 的定义开始：

$$
\text{KL}(\pi_\star \| \pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) - \log \pi_\theta(y \mid x) \right]
$$

把 log difference 上的期望拆成两项，得到：

$$
= \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) \right] - \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\theta(y \mid x) \right]
$$

第一项 $\mathbb{E}\!\left[\log \pi_\star(y \mid x)\right]$ 只依赖数据分布，等于负熵 $-H(\pi_\star)$，是一个不随 $\theta$ 改变的常数。
第二项 $-\mathbb{E}\!\left[\log \pi_\theta(y \mid x)\right]$ 是数据集上的 negative log-likelihood，也就是标准 SFT cross-entropy loss $\mathcal{L}_\text{SFT}(\theta)$。代入可得：

$$
= \underbrace{-H(\pi_\star)}_\text{const} + \mathcal{L}_\text{SFT}(\theta) \propto \mathcal{L}_\text{SFT}(\theta)
$$ {#eq:sft_forward_kl}

由于熵项相对于 $\theta$ 是常数，这两个损失具有相同梯度和相同最小值：最小化 SFT loss 等价于最小化 **forward KL** divergence $\text{KL}(\pi_\star \| \pi_\theta)$。

#### RL Reverse KL

从标准 KL-regularized RL 目标开始：

$$
\max_\pi \; \mathcal{J}_\text{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}\!\left(\pi(\cdot \mid x) \| \pi_\text{ref}(\cdot \mid x)\right)
$$ {#eq:rl_objective_retaining}

提出 $-\beta$，把最大化转换为最小化：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)} - \frac{1}{\beta} r(x, y) \right]
$$ {#eq:rl_min_form}

引入配分函数 $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$，把 reward-tilted reference 归一化为有效分布；再加减 $\log Z(x)$，内部期望就变成一个 KL divergence：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{KL}\!\left(\pi(\cdot \mid x) \;\middle\|\; \frac{1}{Z(x)} \pi_\text{ref}(\cdot \mid x) \exp\!\left(\tfrac{1}{\beta} r(x,y)\right) \right) - \log Z(x) \right]
$$ {#eq:rl_kl_form}

由于 $\log Z(x)$ 不依赖 $\pi$，并且 KL divergence 非负，且当且仅当两个分布相同时为零，所以当 $\pi$ 等于 reward-tilted distribution 时，KL 最小为零。
因此，在奖励 $r(x,y)$ 下的最优策略为：

$$
\pi_\star(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)
$$ {#eq:optimal_policy_retaining}

现在可以直接展示它与 reverse KL 的联系。展开 $\text{KL}(\pi_\theta \| \pi_\star)$，并代入 $\log \pi_\star(y \mid x) = \log \pi_\text{ref}(y \mid x) - \log Z(x) + \frac{1}{\beta} r(x, y)$：

$$
\begin{aligned}
\text{KL}(\pi_\theta \| \pi_\star) &= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\star(y \mid x) \right] \\
&= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\text{ref}(y \mid x) + \log Z(x) - \frac{1}{\beta} r(x, y) \right] \\
&= - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) + \underbrace{\log Z(x)}_\text{const} \\
&\propto - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) \\
&= -\frac{1}{\beta} \mathcal{J}_\text{RL}(\theta)
\end{aligned}
$$

等价地，最大化 RL 目标 $\mathcal{J}_\text{RL}(\theta)$ 与最小化 **reverse KL** divergence $\text{KL}(\pi_\theta \| \pi_\star)$ 是同一件事。

这个推导表明，SFT 和 RL 优化的是根本不同的目标：SFT 最小化 forward KL，RL 最小化 reverse KL。

![Forward KL (SFT) 与 reverse KL (RL) 的遗忘动态。“old” 模式表示先验知识，“new” 模式表示目标任务。Forward KL 会拉伸策略以覆盖目标，并把概率质量从 old 模式拉走（右上）；reverse KL 则把 new 模式向目标移动，同时不扰动 old 模式（右下）。来自 Chen et al. 2025，经作者许可。](images/retaining_by_doing_mode_intuition.png){#fig:retaining-mode-intuition}

KL divergence 的两个方向会诱导不同优化压力。

当目标分布在模型没有概率质量的位置有质量时，forward KL 会惩罚模型，这往往鼓励 **mode covering**：模型广泛铺开概率，以覆盖目标的所有主要模式。
原因是：forward KL 的期望在 $\pi_\star$ 下取得，因此如果模型未能给目标有质量的区域分配概率，就会受到重罚。

Reverse KL 只在模型实际放置概率质量的区域惩罚模型，这往往鼓励 **mode seeking**：模型可以集中到一个高概率模式，同时忽略其他模式。
这里的期望在 $\pi_\theta$ 下取得，也就是模型自己的分布。因此，即使 $\pi_\star$ 在某些区域分配了大量质量，只要 $\pi_\theta(y \mid x) \approx 0$，这些区域对损失的贡献也很小。
同时，它会惩罚模型在目标没有质量的位置放置质量。

鉴于这个区别，我们可能会天真地预期 SFT 比 RL 遗忘*更少*：mode-covering 的 forward KL 应该在目标的所有模式上保持质量，从而保留旧知识；而 mode-seeking 的 reverse KL 可能会塌缩到单个高奖励模式并放弃其他模式。
然而，事实恰好相反。
这种直觉假设策略是单峰的，但预训练 LLM 包含多个模式；对于多峰分布，动态会反转。

考虑一个有两个模式的策略：一个 “old” 模式表示先验知识，一个 “new” 模式表示目标任务（@fig:retaining-mode-intuition）。
Forward KL (SFT) 试图覆盖目标分布的两个模式，这会推动策略拉伸并从 old 模式*重新分配*概率质量，扰乱其形状并造成遗忘。
相比之下，reverse KL (RL) 只需要在某些高奖励区域放置质量，因此可以把它采样到的一个新模式移向目标，而完全不触碰 old 模式，使先验知识保持完整。

RL 的 mode-seeking 行为，也就是 reverse KL 的结构性质，保留了模型先验知识的广度，并支持更好的泛化。

总结如下：

- **SFT (Forward KL)**：$\text{KL}(\pi_\star \| \pi_\theta)$，样本来自目标 $\pi_\star$，即固定的人类编写补全集合。对于每个样本，我们问：我们的模型 $\pi_\theta$ 给它分配了多少概率？模型从不生成任何东西；它学习模仿。这种 mode-covering 压力迫使策略广泛重新分配质量，可能扰乱先验知识。

- **RL (Reverse KL)**：$\text{KL}(\pi_\theta \| \pi_\star)$，样本来自我们自己的策略 $\pi_\theta$。对于模型生成的每个补全，我们问：它与奖励最优策略 $\pi_\star$ 有多接近？因为模型只在自己的生成上训练，更新局限在它已经放置概率质量的位置附近；奖励信号告诉它要强化哪些生成，把概率移向 $\pi_\star$，而不扰动分布其余部分。

### RL's Razor：为什么在线 RL 遗忘更少

上一节表明，on-policy 采样驱动了 RL 对遗忘的抵抗，并把机制追溯到 forward-vs-reverse KL 动态。
对于任何给定任务，都存在许多能取得高性能的不同策略。
Shenfeld et al. 2026 [@shenfeld2026rls] 提出了一个互补视角，即 **RL's Razor** 论断，其假设如下：

> Among the many high-reward solutions for a new task, on-policy methods such as RL are inherently biased toward solutions that remain closer to the original policy in KL divergence.

![向 KL 最小解的偏置会减少遗忘。（左）在能解决新任务的策略中，RL 收敛到与基座模型 KL 距离最近的策略。（右）在匹配新任务性能时，这种 KL 偏置相比 SFT 带来更高的旧任务保留。来自 Shenfeld, Pari, and Agrawal 2026。License CC-BY.](images/rl_razor_motivation.png){#fig:rl-razor-motivation}


作者发现，过去任务的遗忘与微调后策略相对于初始模型漂移的距离（用 KL divergence 衡量）成正比：

$$
\text{Forgetting} \approx f\!\left(\mathbb{E}_{x \sim \tau}\!\left[\text{KL}\!\left(\pi_0(\cdot \mid x) \| \pi(\cdot \mid x)\right)\right]\right)
$$ {#eq:rl_razor_forgetting}


在多种 RL 和 SFT 训练形式中，作者实证表明，遗忘与训练后策略和初始策略之间的 KL divergence 强相关（$R^2 = 0.96$），其中 KL 是**用新任务数据测量的**。
这很令人意外，因为 KL 是在*新任务*输入分布上测量的，而不是在旧任务留出数据上测量，却仍能预测过去任务上的性能下降。
实践中，这为我们提供了一个强有力工具，可以直接从基座策略与训练后策略之间的漂移来估计遗忘：在新的专门数据上测量 KL 距离。

为了确定 RL 策略中更小 KL 偏移的驱动因素，作者沿两个轴分解 RL 与 SFT 的差异：on-policy 与 offline 数据；以及目标是否包含负梯度（当样本低于奖励基线时 RL 中存在，而 SFT 中不存在，SFT 只强化正确示范），这些负梯度会把概率从错误输出上推开。
引人注目的是，他们发现 on-policy 与 offline 数据完全解释了泛化性能差异，而负梯度没有可辨别影响。

直观地说，on-policy 方法采样模型已经赋予非忽略概率的输出，因此每次更新都被约束在当前分布附近。
另一方面，SFT 在固定外部分布上训练，而这个分布可能与模型当前产生的内容任意遥远；每个梯度步都会把模型拉向那个远处目标，不管模型自身的信念如何。

## 其他类型的正则化

在后训练文献中，许多重要模型包含其他正则化方法，帮助它们在自身设置中达到领先性能。
这里列出两个例子，是为了说明一些领先模型如何操纵后训练设置以获得稳定优化，而不是说这些工具一定能在每个设置中显式生效。
还会有无数更有创造性的解决方案能够工作并被发现！

### 预训练梯度

正则化的另一种视角是：你可能有一个希望模型保持接近的*数据集*，正如 InstructGPT [@ouyang2022training] 所做的那样，“in order to fix the performance regressions on public NLP datasets”。
为实现这一点，他们修改了 RLHF 的训练目标。
从 @eq:rl_start 出发，我们可以通过从 RL 策略模型中采样，将其转换为要优化的目标函数：在 RLHF 使用的 RL 数据集中，从 prompts $x$ 采样 completions $y$，得到：
$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

然后，可以在从预训练语料（或其他数据集）采样的一组文档上，为标准自回归 next-token prediction loss 中更高概率增加一个额外奖励，以维持文本连贯性：

$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi_{\text{RL},\theta}(x)) \right]
$$ {#eq:objective_pretraining}

近期工作提出使用 negative log-likelihood 项来平衡 Direct Preference Optimization (DPO) 的优化 [@pang2024iterative]。
考虑到 DPO loss 的成对性质，同样的损失修改也可以用于奖励模型训练，约束模型预测准确文本。

这个优化是对 DPO 的修改。
$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{P_\theta(c_i^w, y_i^w \mid x_i)}{P_{\text{ref.}}(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{P_\theta(c_i^l, y_i^l \mid x_i)}{P_{\text{ref.}}(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log P_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|},
$$ {#eq:dpo_nll_expanded}

其中 $P_{\theta}$ 是可训练策略模型，$P_{\text{ref.}}$ 是固定参考模型（通常是 SFT checkpoint），$(c_i^w, y_i^w)$ 和 $(c_i^l, y_i^l)$ 表示 prompt $x_i$ 的获胜与失败补全。
第一项是标准 DPO logistic loss：它使用 log-likelihood ratios 的差值 $\log \tfrac{P_{\theta}}{P_{\text{ref.}}}$ 来扩大胜负之间的 margin，$\beta$ 控制这个偏好信号从参考模型拉开的强度。
第二项是获胜补全上的长度归一化 negative log-likelihood 惩罚，由 $\alpha$ 加权，它有助于让偏好文本在绝对语言建模意义上保持高似然，而不只是相对于被拒绝样本更好。

### 基于 Margin 的正则化

在 RLHF 栈的其他部分，控制优化的定义不那么清楚。
大多数奖励模型除了标准 contrastive loss function 外没有正则化。
Direct Alignment Algorithms 通过 $\beta$ 参数，以不同方式处理对 KL divergence 的正则化（见[直接对齐](https://rlhfbook.com/c/08-direct-alignment)一章）。

Llama 2 为奖励模型训练提出了 margin loss [@touvron2023llama]：

$$
\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)
$$ {#eq:margin_loss}

其中 $m(y_c, y_r)$ 是两个数据点 $y_c$ 和 $y_r$ 之间的 margin，表示两个标注者评分差值之间的数值差。
这可以通过让标注者在数值尺度上为输出评分，或使用量化排序方法来实现，例如 [Likert scales](https://en.wikipedia.org/wiki/Likert_scale)。

Reward margins 在直接对齐文献中被大量使用，例如 Reward-weighted DPO；Reward-aware Preference Optimization (RPO)，它在 DPO loss 之后把奖励模型分数整合进更新规则 [@adler2024nemotron]；以及 REBEL [@gao2024rebel]，它在 regression-loss 形式中使用 reward delta weighting。
