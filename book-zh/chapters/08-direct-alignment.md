<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "推理与推理时扩展"
prev-url: "07-reasoning"
page-title: 直接对齐算法
search-title: "第 8 章：直接对齐算法"
meta-description: "DPO 等直接对齐算法：无需显式奖励模型或 RL 循环即可优化偏好目标函数。"
next-chapter: "拒绝采样"
next-url: "09-rejection-sampling"
lectures:
  - video: "https://www.youtube.com/watch?v=6g6b4gvO-y0&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=8"
    label: "第 6 讲：直接偏好优化"
---

# 直接对齐算法

直接对齐算法（Direct Alignment Algorithms, DAAs）允许我们更新模型来求解同一个 RLHF 目标函数，而不需要训练中间奖励模型，也不需要使用强化学习优化器。
DAAs 求解的是我们一直在研究的同一个偏好学习问题，而且使用的确实是同一类数据，目的是让语言模型更对齐、更聪明、更易用。
由于没有奖励模型，也没有在线优化，DAAs 的实现要简单得多，可以减少训练计算开销，并让实验更容易开展。
本章会详细介绍推导这些算法所需的复杂数学，然后说明这些有时颇为繁琐的推导最终会得到简单的实现。

最重要的 DAA，也是催生一整场语言模型对齐学术运动的算法，是直接偏好优化（Direct Preference Optimization, DPO）[@rafailov2024direct]。
DPO 的核心是使用梯度上升来求解同一个带约束的 RLHF 目标函数，见第 3 章：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:review_rlhf}

自 2023 年 5 月发布以来，社区曾短暂花了一段时间摸索 DPO 所需的数据和超参数，尤其是出人意料地低的学习率；之后，许多流行模型都使用了 DPO 或其变体，从 2023 年 10 月启动潮流的 Zephyr-$\beta$ [@tunstall2023zephyr]，到 Llama 3 Instruct [@dubey2024llama]、Tülu 2 [@ivison2023camels] 和 3 [@lambert2024t]、Nemotron 4 340B [@adler2024nemotron] 等。
从技术上说，Sequence Likelihood Calibration (SLiC-HF) 是最早发布的现代直接对齐算法 [@zhao2023slic]，但由于多种因素的组合，它并没有流行起来；梳理研究方法为何被采用或未被采用，向来是一件棘手的事。

DPO 和 DAAs 最有影响力的部分，是降低了语言模型后训练实验的门槛：它使用更少计算，更容易从零实现，也更容易在玩具样例和生产样例上跑通。

*本章中，我们用 $x$ 表示 prompts，用 $y$ 表示 completions。这种记号在语言模型文献中很常见，因为这些方法作用于完整的 prompt-completion 对，而不是单个 token。*

## 直接偏好优化

这里我们解释 DPO 如何工作的直觉，并完整重新推导核心方程。

### DPO 如何工作

从表面看，DPO 是直接优化一个策略来求解 RLHF 目标函数。
我们稍后会在推导中重新讨论它的损失函数。该损失函数比较的是：相对于一个参考模型，学到的策略对 chosen completions 和 rejected completions 的概率分别发生了多大偏移。
从 Bradley-Terry 奖励模型推导出的损失函数如下：

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right] $$ {#eq:dpo_core}

在 sigmoid 内部，第一项 $\beta \log \frac{\pi_{\theta}(y_c | x)}{\pi_{\text{ref}}(y_c | x)}$ 衡量策略相对于参考模型把 *chosen* completion 的概率提高了多少；第二项对 *rejected* completion 做同样的衡量。当 chosen 的偏移超过 rejected 的偏移时，损失会下降，也就是说，策略学会偏好正确回复。

在全文中，$\beta$ 是一个超参数，用来平衡奖励优化与最终模型和初始参考模型之间的 KL 散度；换言之，它平衡过度优化，是正确使用 DPO 时的关键超参数。
这依赖 DPO 训练中的隐式奖励，它取代了外部奖励模型。这个隐式奖励是概率的对数比：

$$r(x, y) = \beta  \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ {#eq:dpo_reward}

其中 $\pi_r(y \mid x)$ 是我们要求解的精确最优奖励策略。
它来自相对于最优策略推导 Bradley-Terry 奖励，如 @eq:dpo_opt_policy 所示，也如第 5 章 Bradley-Terry 模型部分所展示。
本质上，正如 DPO 论文所说，这种重参数化给出了“以最优策略而不是奖励模型表示的人类偏好数据概率”，也就是说，我们可以完全绕过显式奖励模型的学习。

让我们看 @eq:dpo_core 中优化器必须降低的损失。
这里，当 chosen 回复的对数比大于 rejected 回复的对数比，并且都经过参考模型归一化时，损失会更低。
在实践中，这就是对数据中整个 token 序列上的模型对数概率求和。
因此，DPO 增大的是 chosen 与 rejected 回复之间的相对对数概率间隔。

有了 @eq:dpo_reward 中的奖励，我们可以写出损失梯度，以进一步解释正在发生什么：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[ w \cdot \left(\nabla_{\theta}\log \pi_{\theta}(y_c \mid x) - \nabla_{\theta}\log \pi_{\theta}(y_r \mid x)\right) \right]$$ {#eq:dpo_gradient}

其中 $w = \sigma\!\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right)$。

这里，梯度通过以下方式求解上述目标：

- sigmoid 函数 $\sigma(\cdot)$ 内的第一项会产生一个从 0 到 1 的参数更新权重。当奖励估计错误时，这个权重更高。当 rejected 样本被偏好得超过 chosen 样本时，权重更新应该更大！
- 其次，内层括号 $[\cdot]$ 中的项会提高 chosen 回复 $y_c$ 的似然，并降低 rejected 回复 $y_r$ 的似然。
- 这些项由 $\beta$ 加权，而 $\beta$ 控制更新如何在正确排序 completions 与 KL 散度之间取得平衡。


核心直觉是，DPO 正在拟合一个隐式奖励模型，而该奖励模型对应的最优策略可以用闭式形式抽取出来（@eq:dpo_opt_policy，这要感谢梯度下降和我们的机器学习工具）。
由于 DPO 损失可以直接微分，计算精确梯度很直接，不需要先训练奖励模型，再采样 completions 来打分并估计梯度。
常被误解的一点是，DPO 的核心确实是在学习一个奖励模型，因此论文副标题才是 *Your Language Model is Secretly a Reward Model*。
这很容易与“DPO 目标函数直接训练策略”混淆，因此学习下面的推导有助于获得完整理解。

通过学习隐式奖励模型，DPO 在给定数据集和目标函数中特定 KL 约束 $\beta$ 的条件下，生成 RLHF 目标函数的最优解。
这里，DPO 针对特定 KL 散度求解精确策略，因为生成不是像策略梯度算法那样在线进行的；这是它与用于偏好调优的 RL 方法之间的核心区别。
在许多方面，相比在线 RL 方法，这让 DPO 中的 $\beta$ 值更容易调节；但关键且直观的是，最优值取决于被训练的模型以及训练它的数据。

在每个偏好数据 batch 中，包含许多 completion 对 $y_{chosen} \succ y_{rejected}$，DPO 会直接朝最优解做梯度步。
它比策略梯度方法简单得多。

![DPO 刚发布时，在研究社区激起了关于如何最好地做 RLHF 和偏好学习的激烈争论。这个 meme 很好地捕捉了当时的情绪：争论常常显得勉强且夸张，但许多刚入门的人和顶级实验室都从 DPO 中获得了巨大收益。DPO simplicity meme，credit Tom Goldstein。](images/dpo_meme.jpeg){#fig:dpo-meme}


### DPO 推导

DPO 推导主要分为两部分。
首先，作者展示了能够最优求解本书一直使用的 RLHF 目标函数的策略形式。
接着，他们展示如何从成对偏好数据，即 Bradley-Terry 模型，得到这个解。

#### 推导最优 RLHF 解

首先，我们再次考虑 RLHF 优化目标函数。这里表示我们希望最大化这个量：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:rlhf_opt_eq_repeat}

这里的双重期望只适用于为了计算期望奖励而进行的采样，KL 项仍然是一个解析表达式。
首先，展开 KL-divergence 的定义。回忆 $\mathcal{D}_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$，其中求和中的 $\pi(y|x)$ 权重成为采样分布。
由于两个项现在都共享对 $y \sim \pi(y|x)$ 的同一个期望，我们可以合并它们：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

接着，把括号中差值前的负号提出来。为此，将其拆成两项：

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

然后乘以 $-1$，把最大化转换为最小化：

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

除以 $\beta$ 并重新合并：

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}


接下来，我们必须引入配分函数 $Z(x)$：

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

配分函数是未归一化密度 $\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 的归一化因子，从而使它对每个固定的 $x$ 都成为 $y$ 上的有效概率函数。随着推导继续，为什么需要它会很快变得清楚。

代入后，我们得到一个中间变换：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

为了看出它如何得到，考虑 @eq:dpo_deriv_4: 中优化括号内的部分：

$$ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_6}

然后，在两侧加上 $\log Z(x) - \log Z(x)$：

$$ = \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) + \log Z(x) - \log Z(x) $$ {#eq:dpo_deriv_7}

然后将项分组：

$$ = \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x) \right) - \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_8}

利用 $\log(x) + \log(y) = \log(x\cdot y)$，并把 $Z$ 移到分母，我们得到：

$$ = \log \frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)}- \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_9}

接着，把 $\frac{1}{\beta}r(x,y)$ 展开为 $\log \exp \frac{1}{\beta}r(x,y)$，并做同样操作，就得到 @eq:dpo_deriv_5。这里我们稍作重写：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}} \left[ \mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} \right] - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

有了这个优化形式，我们需要实际求解最优策略 $\pi^*$。
由于我们引入了配分函数 $Z(x)$，使得 $\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 成为 $y$ 上的有效概率分布，我们可以认出内层期望其实是一个真正的 KL-divergence！

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathcal{D}_{\text{KL}} \left(\pi(y|x) \middle\| \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_11}

由于 $\log Z(x)$ 这一项不依赖于 $\pi$，也就是我们正在优化的策略，我们可以忽略它。剩下的是我们正在学习的策略，与一个由配分函数、$\beta$、奖励和参考策略共同确定的形式之间的 KL 散度。
Gibbs 不等式告诉我们，只有当两个量相等时，这个距离才会在 0 处最小化！
因此，我们得到最优策略：

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}


#### 推导 BT 模型下的 DPO 目标函数

首先，回忆第 5 章奖励建模和第 11 章偏好数据中，人类偏好的 Bradley-Terry 模型形式为：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

通过变形 @eq:dpo_opt_policy，我们可以求解最优奖励。首先，对两边取对数：

$$\log \pi^*(y|x) = \log \left( \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^*(x,y)\right) \right)$$ {#eq:dpo_reward_deriv1}

使用 $\log(abc) = \log a + \log b + \log c$ 展开右侧：

$$\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r^*(x,y)$$ {#eq:dpo_reward_deriv2}

重新排列以求解 $r^*(x,y)$：

$$\frac{1}{\beta}r^*(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)$$ {#eq:dpo_reward_deriv3}

两边乘以 $\beta$：

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

然后，我们可以把这个奖励代入 @eq:bradley_terry_dpo 中的 Bradley-Terry 方程，得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} + \beta \log Z(x)\right)} $$ {#eq:dpo_loss_deriv0}

通过把指数表达式从 $e^{a+b}$ 分解为 $e^a e^b$，然后约去 $e^{\beta \log Z(x)}$ 项，可以简化为：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right)} $$ {#eq:dpo_loss_deriv1}

然后，将分子和分母同时乘以 $\exp\left(-\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)$，得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} $$ {#eq:dpo_loss_deriv2}

最后，利用 sigmoid 函数定义 $\sigma(x) = \frac{1}{1+e^{-x}}$，得到：

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

这就是在给定最优策略 $\pi^*$ 时，Bradley-Terry 模型下偏好数据的似然。回忆第 5 章奖励建模中，我们把 Bradley-Terry 目标函数推导为最大化似然，或者等价地最小化负对数似然，由此得到损失：
$$
\begin{aligned}
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) &= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log p(y_c \succ y_r \mid x)  \right] \\
&= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
\end{aligned}
$${#eq:dpo_loss_deriv4}

这就是 DPO 的损失函数，其形式与 @eq:dpo_core 所示一致。
DPO 论文还给出了 Plackett-Luce Model 下目标函数的额外推导，但它在实践中使用少得多 [@rafailov2024direct]。

#### 推导 BT DPO 梯度

我们在 @eq:dpo_gradient 中使用 DPO 梯度来解释模型如何学习的直觉。
要推导它，我们必须对 @eq:dpo_loss_deriv4 关于模型参数取梯度。

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

首先，这可以改写。
我们知道 sigmoid 函数的导数 $\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))$，对数的导数 $\frac{d}{dx} \log x = \frac{1}{x}$，以及 sigmoid 的性质 $\sigma(-x)=1-\sigma(x)$，因此可以重新整理上面的方程。

先令 $u=\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$，也就是 sigmoid 内的表达式。
那么有：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[\frac{\sigma'(u)}{\sigma(u)}\nabla_{\theta}u\right] $$ {#eq:dpo_grad_2}

展开这一式子，并使用上面关于 sigmoid 和对数的表达式，就得到前面引入的梯度：

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi_{\theta}(y_c|x)-\nabla_{\theta}\log\pi_{\theta}(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## 数值问题、弱点与替代方案

为了解决 DPO 的弱点，人们提出了许多 DPO 算法变体。
例如，如果没有 rollout 供奖励模型对生成结果打分，DPO 会以相同权重处理每一对偏好数据。
现实中，正如第 11 章偏好数据所示，捕捉偏好数据的方式很多，标签可以比二元标签更丰富。
已有多个算法被提出，用来重新平衡优化过程，避免同等对待每一对样本。

- **REgression to RElative REward Based RL (REBEL)** 从奖励模型中加入信号，把 chosen 与 rejected 回复之间的 margin 作为额外信息，而不是只依赖成对偏好数据，从而更准确地求解 RLHF 问题 [@gao2024rebel]。
- **Conservative DPO (cDPO) 和 Identity Preference Optimization (IPO)** 通过假设偏好数据中存在噪声来处理过拟合。cDPO 假设 N percent 的数据被错误标注 [@rafailov2024direct]，IPO 则改变优化方式，使偏好概率变得更柔和，而不是直接从标签优化 [@azar2024general]。实践上，IPO 把偏好概率改为非线性函数，偏离 Bradley-Terry 假设，其中 $\Psi(q) = \log\left(\frac{q}{1-q}\right)$。
- **带 offset 的 DPO (ODPO)** “要求 preferred 和 dispreferred 回复的似然差大于某个 offset 值”[@amini2024direct]，也就是不要同等对待每一对数据；但这可能以更困难的标注环境为代价。

有些 DPO 变体试图通过对损失做小改动来改进学习信号，或者通过降低内存使用来提高应用效率。

- **Odds Ratio Policy Optimization (ORPO)** 直接更新策略模型，使其像指令微调损失一样向 chosen 回复靠拢，同时对 chosen 回复加入一个小惩罚 [@hong2024reference]。这种损失函数变化去掉了参考模型需求，简化了设置。理解 ORPO 的最佳方式是：它受 DPO 启发，而不是 DPO 的派生算法。
- **Simple Preference Optimization (SimPO)** 对 DPO 优化做了一个小改动：对对数概率取平均，而不是求和，或者说加入长度归一化，以提升性能 [@meng2025simpo]。

![DPO 中偏好位移的示意图。](images/dpo_displacement.png){#fig:dpo_issue .center}

DPO 中一个显而易见的核心问题是，优化只是在推动 chosen 和 rejected 回复的概率之间的 margin 增大。
从数值上看，模型会同时降低 chosen 和 rejected 回复的概率，但如 @fig:dpo_issue 所示，*rejected 回复降低得更多*。
直观上，这种机制如何泛化并不清楚，但已有工作提出，它会提高未被覆盖行为的概率，也就是语言模型可以生成、但不在后训练数据集分布中的 token [@razin2024unintentional] [@ren2024learning]。
一些简单方法，例如调整优化过程的 Cal-DPO [@xiao2024cal]，以及修改奖励形状的 AlphaPO [@gupta2025alphapo]，可以缓解这种 **偏好位移**。
实践中，它的确切影响尚不明确，但它指出了在线方法可能优于 vanilla DPO 的一个潜在原因。

另一个主要原因被认为是：DPO 类方法的性能上限低于在线（基于 RL 的）RLHF 方法，因为其训练信号来自过去模型或其他模型的 completions。
在线 DPO 变体通过生成新的 completions，并在训练时纳入偏好信号来缓解这些限制。**Online DPO** [@guo2024direct] 从当前模型采样生成结果，而 **Discriminator-Guided DPO** (D2PO) [@singhal2024d2po] 使用奖励模型重新标注，动态创建新的偏好数据；此外还有许多变体。

其他 DAA 变体还有很长一串，例如 Direct Nash Optimization (DNO) [@rosset2024direct] 或 Binary Classifier Optimization (BCO) [@jung2024binary]，但算法选择远不如初始模型和所用数据重要 [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences]。

## 实现细节

DPO 等 DAAs 的实现方式与策略梯度优化器非常不同。
原始实现中的 DPO 损失大体可以总结如下 [@rafailov2024direct]：

```python
# Log-probability gaps for the policy and the frozen reference model
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# Difference of log-ratios: positive when the policy
# shifts probability toward the chosen completion
logits = pi_logratios - ref_logratios

# DPO loss: negative log-sigmoid drives the policy to
# widen the gap between chosen and rejected
losses = -F.logsigmoid(beta * logits)

# Implicit rewards (detached -- used for logging only)
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

这可以用于标准语言模型训练栈，因为这些信息已经会在模型前向传播期间被整理好，只是需要额外加入一个参考模型。

在大多数方面，DAAs 更简单，也提升了使用体验，但它们也带来一组不同的考量。

1. **KL 散度是静态的**：在 DPO 和其他算法中，KL 散度由 $\beta$ 参数显式设定；$\beta$ 用来平衡到优化目标的距离惩罚。这是因为 DPO 会在给定数据的条件下，朝 RLHF 目标函数的 *最优* 解做梯度步，也就是精确走向由 $\beta$ 项设定的解。另一方面，基于 RL 的优化器会根据 batch 和近期数据做步进。
2. **缓存对数概率**：DPO 的简单实现会为了损失函数的方便，同时对策略模型和参考模型做前向传播。然而，这会使内存使用翻倍，并增加 GPU 使用量。为避免这一点，可以先在训练数据集上计算参考模型的对数概率，然后在每个 batch 计算损失和更新参数时复用这些缓存的参考对数概率，从而把峰值内存使用降低 50%。

## 使用合成偏好数据的 DAAs

如今，用 DAAs 做偏好微调的流行数据集，大多是合成偏好：由一个前沿模型对其他模型的输出进行评分，判断哪个是 winner、哪个是 loser。
突出例子包括 UltraFeedback（这一类别中的第一个）[@cui2023ultrafeedback]、Tülu 3（用扩展的 UltraFeedback 方法构建）[@lambert2024t]、SmolLM 3 的数据 [@bakouch2025smollm3]，以及随 Olmo 3 发布的 Dolci Pref 数据集 [@teamolmo2025olmo3]。

构建这些数据集的最佳实践仍在演化。
Tülu 3 以及围绕其 2024 年 11 月发布的数据集表明，合成的成对偏好数据在某种意义上需要是 “on-policy”：部分 completions 要由你正在微调的模型生成，同时混入更大的模型池。
这种数据的 on-policy 性质保证 DAA 会在模型实际生成的正确 token 空间内优化，因为这些损失函数是对比式的，不像指令微调那样直接。
后来，随着 Olmo 3 和 SmolLM 3 在 2025 年发布，其他工作支持了另一种叫作 Delta Learning 的理论。该理论认为，chosen 与 rejected completions 之间的差异比究竟用哪些模型生成 completions 更重要 [@geng2025the]。
例如，在这两个被引用的模型中，chosen responses 来自 Qwen 3 32B，而 rejected responses 来自 Qwen 3 0.6B；两组作者是同时且独立地开发出这种配对的。

总体而言，考虑到实现简单、相对于基于强化学习的偏好微调方法性能强，使用 DAAs 在合成偏好数据上训练模型，是多数实践者应该首先尝试的地方。
使用大量合成偏好数据时，还存在其他小问题，例如判断 completions 的模型自身偏置。
已知 GPT-4 等前沿模型存在长度偏置 [@dubois2024length]，并且偏好与自身风格相匹配的输出 [@panickssery2024llm]（更多信息见第 12 章），因此数据集中 “chosen” 部分的一段文本略微更可能来自 OpenAI 模型，或来自另一个风格上与它相似的强模型。

作为本节结尾，我们说明这些方法如何改变被训练模型生成结果的直觉。
从高层看，多数 DAAs 优化的是增大 “chosen” 和 “rejected” completions 概率之间的 margin；有些不太流行的算法会略微改变这种动态，但核心不变。
如本章前面所讨论（见 @fig:dpo_issue），这往往意味着两个概率都会下降，只是 rejected 回复下降幅度更大。
序列中的每个 token 都会根据它对整体偏好 margin 的贡献大小，接收不同大小和方向的梯度，从而让优化器识别哪些 token 对结果最重要。

## DAAs 与 RL：在线数据与离线数据

概括地说，这场争论可以归结为一个问题：为了用 RLHF 对齐语言模型，我们是否需要强化学习内部的那些机制，包括价值函数、策略梯度等等？
与大多数这样表述的问题一样，这个问题过于简单。
当然，两类方法都已经很成熟，但重要的是说明二者的根本差异和性能流形分别在哪里。

多份报告得出结论：基于策略梯度和 RL 的方法优于 DPO 及其变体。
这些论证形式不同，有的是在控制数据的情况下用不同算法训练模型 [@ivison2024unpacking] [@xu2024dpo]，有的是研究 RL 优化循环中 on-policy 数据的作用 [@tajwar2024preference]。
在所有这些情况下，DPO 算法都略微落后。

即便存在这种性能差距，DAAs 仍因简单而被领先模型广泛使用。
DAAs 提供了一个受控环境，可以快速迭代训练数据和其他配置；考虑到数据往往远比算法重要，使用 DPO 完全可以成立。

随着主要用 RL 训练的推理模型出现，长期来看，对偏好调优中使用 RL 的进一步投入将会回归。这会改善 RL 基础设施的鲁棒性，并巩固 DAAs 与 RL 在从人类反馈中优化时的性能差距。

## 建议实验

配套代码 `code/direct_alignment/` 会在偏好数据上训练 DPO 和几种相关损失。
这是开始实验偏好调优最容易上手的地方，因为它是离线设置：不需要奖励模型服务器，也不需要 rollout 循环。

1. **在 UltraFeedback 上训练一次小型 DPO 运行。**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --loss dpo --max_samples 1000
   ```

   观察 `loss`、`accuracy`、`margins`、`chosen_rewards` 和 `rejected_rewards`。
   主要 sanity check 是：隐式奖励 margin 应该朝期望方向移动，同时模型的样本生成不应崩塌。

2. **比较 DPO、IPO 和长度归一化 DPO。**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/ipo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo_norm.yaml
   ```

   比较 margin 尺度和学习率敏感性。
   IPO 的损失与 DPO 不在同一数值尺度上，因此应通过 `accuracy` 和 margin 行为来理解，而不是只看原始 loss。

3. **谨慎尝试 reference-free 变体。**
   从配置运行 SimPO 或 ORPO，然后检查训练期间记录的生成样本。
   这些损失对对数概率尺度和学习率更敏感，因此很适合作为调试练习。

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/simpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/orpo.yaml
   ```

4. **先改数据，再改损失。**
   固定损失，改变 `--max_samples`、`--max_length` 或偏好数据集。
   如果结果变化比在 DPO 类目标函数之间切换还大，那就是对偏好调优中一个中心主题的经验提醒：数据通常支配小的算法差异。
