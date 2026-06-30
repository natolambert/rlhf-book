<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "直接对齐算法"
prev-url: "08-direct-alignment"
page-title: 拒绝采样
search-title: "第 9 章：拒绝采样"
meta-description: "拒绝采样和 best-of-n 方法：用奖励或偏好信号改进后训练语言模型。"
next-chapter: "偏好的本质"
next-url: "10-preferences"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "第 2 讲：IFT、奖励建模、拒绝采样（第 4、5、9 章）"
---

# 拒绝采样

拒绝采样（Rejection Sampling, RS）是偏好微调中使用最广泛、但文档记录最少的方法之一。
许多重要 RLHF 论文都把它作为训练 pipeline 的核心组件，但它没有规范实现，也没有关于它为什么效果这么好的公认解释。
RS 可以应用在训练 pipeline 的多个位置：指令微调之后、基于 RL 的优化之后，甚至 RLVR 之后。因此，它是一个灵活但很难归类的工具。
再加上相关记录不足，这就是为什么它出现在核心优化方法的末尾。

拒绝采样的操作方式是：构造新的候选 completions，用训练好的奖励模型进行过滤，然后只在排名靠前的 completions 上微调原始模型；使用的损失函数与指令微调相同。

这个名称来自计算统计学 [@gilks1992adaptive]。在该领域中，人们希望从一个复杂分布中采样，但没有直接方法。
为缓解这一点，可以从一个更容易建模的分布中采样，并用启发式方法检查样本是否可接受。
对于语言模型，目标分布是 prompts 的高质量 completions，过滤器是奖励模型，采样分布则是当前模型。

WebGPT [@nakano2021webgpt]、Anthropic 的 Helpful and Harmless agent [@bai2022training]、OpenAI 关于过程奖励模型的流行论文 [@lightman2023let]、Llama 2 Chat 模型 [@touvron2023llama]，以及其他开创性工作都使用了这一 baseline；更近期的工作则直接形式化了它，例如 RAFT [@dong2023raft] 将其应用于多模态对齐，Statistical Rejection Sampling Optimization (RSO) [@liu2023statistical] 则原则性地概述了拒绝采样与其他偏好学习目标函数之间的关系。

*本章中，我们用 $x$ 表示 prompts，用 $y$ 表示 completions。这种记号在语言模型文献中很常见，因为这些方法作用于完整的 prompt-completion 对，而不是单个 token。*

## 训练过程：逐步说明

拒绝采样整体上包含几个阶段。

0. **选择 prompt 和奖励模型：** 首先，你必须相对于其他训练阶段，选择想要训练的 prompts。最简单的方法是复用第一阶段 SFT/IFT 中的每个 prompt，但这可能导致一定过拟合。在做拒绝采样之前，还必须已经训练好一个奖励模型；更多信息见第 5 章。
1. **从起始 checkpoint 生成 completions：** 接下来，必须使用想要优化的模型，为选定 prompts 生成 completions。这可能涉及许多设置调整，例如采样温度、top-p、最大序列长度、每个 prompt 的 completions 数量等。
2. **用奖励模型选择顶部 completions**：所有 completions 都由奖励模型排序。这个阶段也可能包含去重，以保证每个 prompt 只保留一个 completion；不过许多这样的设计选择都取决于经验性的消融研究。
3. **在顶部 completions 上做 SFT：** 拒绝采样的最后一步，是在选出的 completions 上对起始 checkpoint 做指令微调。

拒绝采样过程的可视化概览见下方 @fig:rs-overview。

![拒绝采样概览。](images/rejection-sampling.png){#fig:rs-overview}

关于应该使用哪些 prompts、如何选择奖励模型、如何安排拒绝采样顺序等实际细节，文献中的记录并不充分。
本章提供方法概览，并把进一步实验留给读者。

### 生成 Completions

为了给每个 prompt 生成多个候选 completions，我们把一组 $M$ 个 prompts 定义为一个向量：

$$X = [x_1, x_2, ..., x_M]$$ {#eq:rs_prompt_vector}

这些 prompts 可以来自许多来源，但最常见的是来自指令训练集。

对于每个 prompt $x_i$，我们生成 $N$ 个 completions。可以把它表示为一个矩阵：

$$Y = \begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,N} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
y_{M,1} & y_{M,2} & \cdots & y_{M,N}
\end{bmatrix}$$ {#eq:rs_completion_matrix}

其中 $y_{i,j}$ 表示第 $i$ 个 prompt 的第 $j$ 个 completion。
每一行 $i$ 对应单个 prompt $x_i$，包含它的 $N$ 个候选 completions；每一列 $j$ 对应所有 prompts 上第 $j$ 次采样得到的 completion。

### 给 Completions 打分

现在，我们把所有这些 prompt-completion 对输入奖励模型，得到一个奖励矩阵。
我们将奖励表示为矩阵 $R$：

$$R = \begin{bmatrix}
r_{1,1} & r_{1,2} & \cdots & r_{1,N} \\
r_{2,1} & r_{2,2} & \cdots & r_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
r_{M,1} & r_{M,2} & \cdots & r_{M,N}
\end{bmatrix}$$ {#eq:rs_reward_matrix}

每个奖励 $r_{i,j}$ 通过把 completion $y_{i,j}$ 及其对应 prompt $x_i$ 输入奖励模型 $\mathcal{R}$ 计算得到：

$$r_{i,j} = \mathcal{R}(y_{i,j} \mid x_i)$$ {#eq:rs_reward_computation}

选择用于训练的顶部 completions 有多种方法。

为了形式化基于奖励矩阵选择最佳 completions 的过程，我们可以定义一个作用于奖励矩阵 $R$ 的选择函数 $S$。

#### 每个 Prompt 取顶部样本

第一个可能的选择函数，是对每个 prompt 取最大奖励。

$$S(R) = \left[\arg\max_{j} r_{1,j}, \arg\max_{j} r_{2,j}, ..., \arg\max_{j} r_{M,j}\right]$$ {#eq:rs_selection_per_prompt}

这个函数 $S$ 返回一个索引向量，其中每个索引对应 $R$ 中该行最大奖励所在的列。
然后可以用这些索引选择 chosen completions：

$$Y_{chosen} = [y_{1,S(R)_1}, y_{2,S(R)_2}, ..., y_{M,S(R)_M}]$$ {#eq:rs_chosen_completions}


#### 全局顶部 Prompt-Completion 对
或者，我们可以从整个集合中选择顶部 $K$ 个 prompt-completion 对。
首先，把奖励矩阵 $R$ 展平成一个向量：

$$R_{flat} = [r_{1,1}, r_{1,2}, ..., r_{1,N}, r_{2,1}, r_{2,2}, ..., r_{2,N}, ..., r_{M,1}, r_{M,2}, ..., r_{M,N}]$$ {#eq:rs_flattened_rewards}

这个 $R_{flat}$ 向量长度为 $M \times N$，其中 $M$ 是 prompts 数量，$N$ 是每个 prompt 的 completions 数量。

现在，可以定义选择函数 $S_K$，它选择 $R_{flat}$ 中 K 个最高值的索引：

$$S_K(R_{flat}) = \text{argsort}(R_{flat})[-K:]$$ {#eq:rs_topk_selection}

其中 $\text{argsort}$ 返回会使数组按升序排序的索引，我们取最后 $K$ 个索引来得到 $K$ 个最高值。

为了得到被选中的 completions，需要把这些展平后的索引映射回原始 completion 矩阵 $Y$。
要恢复对应的 prompt-completion 对，可以把零索引展平下标 $k$ 通过 $i = \lfloor k / N \rfloor + 1$ 和 $j = (k \bmod N) + 1$ 映射到 $(i,j)$。

#### 选择示例
考虑这样一种情况：我们有五个 prompts，每个 prompt 有四个 completions。
我们将展示两种基于奖励选择 completions 的方法。

$$R = \begin{bmatrix}
0.7 & 0.3 & 0.5 & 0.2 \\
0.4 & 0.8 & 0.6 & 0.5 \\
0.9 & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & 0.8 & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_matrix}

首先，**per prompt**。直观上，可以把奖励矩阵高亮如下：

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \textbf{0.6}
\end{bmatrix}$$ {#eq:rs_example_per_prompt}

使用 argmax 方法，我们为每个 prompt 选择最佳 completion：

$$S(R) = \left[\arg\max_{j} r_{i,j} \text{ for } i \in [1,5]\right]$$ {#eq:rs_example_selection_formula}

$$S(R) = [1, 2, 1, 3, 4]$$ {#eq:rs_example_selection_result}

这意味着我们会选择：

- 对于 prompt 1：completion 1（reward 0.7）
- 对于 prompt 2：completion 2（reward 0.8）
- 对于 prompt 3：completion 1（reward 0.9）
- 对于 prompt 4：completion 3（reward 0.8）
- 对于 prompt 5：completion 4（reward 0.6）

现在看 **best overall**。
我们高亮全局前五个 completion 对。

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & \textbf{0.7} \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_top_overall}


首先，将奖励矩阵展平：

$$R_{flat} = [0.7, 0.3, 0.5, 0.2, 0.4, 0.8, 0.6, 0.5, 0.9, 0.3, 0.4, 0.7, 0.2, 0.5, 0.8, 0.6, 0.5, 0.4, 0.3, 0.6]$$ {#eq:rs_example_flattened}

现在，选择五个最高值的索引：
$$S_5(R_{flat}) = [8, 5, 14, 0, 11]$$ {#eq:rs_example_topk_result}

把这些映射回原始矩阵：

- Index 8 → prompt 3，completion 1（reward 0.9）
- Index 5 → prompt 2，completion 2（reward 0.8）
- Index 14 → prompt 4，completion 3（reward 0.8）
- Index 0 → prompt 1，completion 1（reward 0.7）
- Index 11 → prompt 3，completion 4（reward 0.7）

#### 实现示例

下面的代码片段展示了如何实现这些选择方法。

```python
import numpy as np

x = np.random.randint(10, size=10)
print(f"{x=}")
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
print(f"{x_sorted=}")

# first way to recover the original array
i_rev = np.zeros(10, dtype=int)
i_rev[sorted_indices] = np.arange(10)
np.allclose(x, x_sorted[i_rev])

# second way to recover the original array
np.allclose(x, x_sorted[np.argsort(sorted_indices)])
```

### 微调

有了选出的 completions 之后，就可以在当前版本模型上执行标准指令微调。
更多细节可见[指令微调章节](https://rlhfbook.com/c/04-instruction-tuning)。

## 实现细节

执行这种训练的核心超参数非常直观：

- **采样参数**：拒绝采样直接依赖模型收到的 completions。拒绝采样的常见设置包括大于零的温度，例如 0.7 到 1.0 之间，并可能修改 top-p 或 top-k sampling 等其他参数。
- **每个 prompt 的 completions 数量**：成功的拒绝采样实现中，每个 prompt 通常包含 10 到 30 个或更多 completions。completions 太少会使训练有偏和/或噪声较大。
- **指令微调细节**：拒绝采样期间指令微调的明确训练细节尚未公开。它们很可能使用与模型初始指令微调阶段略有不同的设置。
- **异构模型生成**：有些拒绝采样实现包含来自多个模型的生成结果，而不只是来自即将被训练的当前模型。关于如何做这件事，还没有确立最佳实践。
- **奖励模型训练**：所使用的奖励模型会显著影响最终结果。关于奖励模型训练的更多资源，见[相关章节](https://rlhfbook.com/c/05-reward-models)。

做批量奖励模型推理时，可以按长度对 tokenized completions 排序，使 batch 中样本长度相近。
这会减少在 padding tokens 上运行推理的需求，以少量实现复杂度换取吞吐提升。

## 相关：Best-of-N 采样

Best-of-N (BoN) 是拒绝采样的近亲，它遵循同样的生成并打分过程，但你 **不会** 在选中的 completions 上微调模型。
相反，BoN 在推理时为一个静态 prompt（或一组 prompts）计算尽可能最好的 completion。相关技术常用于聊天模型的 “Pro” 层级，这些层级会花费额外计算来回答你的查询。

Best-of-N 采样常被作为与 RLHF 训练方法对照的 baseline。
必须记住，BoN *不会* 修改底层模型，它是一种采样技术。
因此，在某些上下文中，把 BoN 采样与 PPO 等在线训练方法进行比较仍然是有效的。
例如，相对于任何其他策略运行 BoN 采样时，你仍然可以测量 KL 距离。

这里，我们将说明，对于单个 prompt 使用简单 BoN 采样时，上面展示的两种选择标准是等价的。

令 $R$ 是我们单个 prompt 的 $N$ 个 completions 的奖励向量：

$$R = [r_1, r_2, ..., r_N]$$ {#eq:rewards_vector}

其中 $r_j$ 表示第 j 个 completion 的奖励。

使用 argmax 方法，我们为该 prompt 选择最佳 completion：

$$S(R) = \arg\max_{j \in [1,N]} r_j$$ {#eq:selection_function}

当 top-K 方法取 $K=1$ 时，会退化为同一方法，这也是常见实践。

## 建议实验

配套实现 `code/rejection_sampling/` 会运行一个完整的 GSM8K 拒绝采样 pipeline：生成 rollouts、用奖励模型打分、选择训练子集、微调，并评估 exact-match accuracy。
四个配置被安排成匹配的 treatment/control 对，因此读者可以检验奖励模型是否真的有帮助。

1. **先构建一次 rollout cache。**

   ```bash
   cd code/
   uv run python -m rejection_sampling.preprocess \
       --config rejection_sampling/configs/top_per_prompt.yaml
   ```

   这会为共享的 GSM8K 切片生成并打分 completions。
   只要生成和打分设置保持不变，后续训练配置都会复用这个 cache。

2. **比较奖励选择和随机 controls。**

   ```bash
   cd code/
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_k_overall.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_k_overall.yaml
   ```

   以匹配对阅读结果：`top_per_prompt` 对 `random_per_prompt`，以及 `top_k_overall` 对 `random_k_overall`。
   如果由奖励选择的运行没有超过随机 baseline，那么奖励模型或采样到的 completions 在这个切片上没有提供有用信号。

3. **改变奖励模型可选择的范围。**
   复制一个配置，并修改 `num_completions_per_prompt`、`temperature`、`top_p` 和 `selection.top_k`。
   更多 completions 可以改善可选的最佳样本，但前提是奖励模型能够区分好答案和坏答案。

4. **尝试更小的策略模型。**
   将 `model_name` 设为一个更小的兼容 instruct model，降低 `max_train_samples`，然后重新运行同样的匹配对。
   这会让实验更便宜，并突出拒绝采样究竟是在挽救较弱生成，还是仅仅在已经不错的生成之间做选择。
