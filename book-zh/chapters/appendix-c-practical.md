<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "超越 \"只是风格\""
prev-url: "appendix-b-style"
page-title: "附录 C：实践问题"
search-title: "附录 C：实践问题"
meta-description: "运行 RLHF 与后训练实验的实践建议，涵盖工程约束和调试。"
next-chapter: "首页"
next-url: "https://rlhfbook.com/"
---

# 实践问题

本附录讨论大规模运行后训练实验时的实践考虑。
它采用经验列表的形式，而不是一段连贯叙事。

## 后训练的计算成本

确定后训练运行成本有两种不同范围。
最大的成本在于开发 recipe，这很容易达到最后几次训练运行计算量的 10 倍到 100 倍。
次要成本更容易衡量，即完整应用某个 recipe 的成本，包括多个 seed、谨慎评估、潜在工程问题等。

以第一个成本为例，为了开发 Tülu 3 [@lambert2024t] 这样的后训练 recipe，团队在得到最终模型之前，在 7B 规模上运行了数千次实验/评估量级的工作。

对于最终运行，Olmo 3 报告详细说明了训练最终 32B Think 模型所涉及的内容 [@teamolmo2025olmo3]：

> Post-training follows a different operational pattern in which we run each stage multiple times, sweeping over learning rates and other hyperparameters. The theory for post-training, particularly, RL, is less developed, so we have to run multiple experiments to identify the optimal hyperparameters for a given base model. We hope to address this in future work.
>
> During post-training, checkpoint evaluation consumes a larger proportion of compute resources, in part due to long generations from reasoning models on core benchmarks. For SFT, we swept over four candidate learning rates, on 256 GPUs each, in parallel for 36 hours. Then approximately 12 hours was spent on evaluation, merging, and checkpoint confirmation, totaling approximately two days. DPO training takes less time per run (about 18 hours for a full learning-rate sweep on 64 GPUs per job) but in practice extended over multiple days due to cluster instability. The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days with at least a day of training time lost due to stability issues. After the initial release of Olmo 3, we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B.

随着扩展强化学习变成更标准的实践，这还会再次变化 [@khatri2025art]。
继续上面的例子，原始 Olmo 3 32B Think 的后训练只花了几周；而为了发布改进后的 Olmo 3.1 32B Think 模型，团队需要再用 RLVR 额外训练 3.5 周。相比总计算量，这更是一项显著的 *时间* 成本。

## 评估方差

后训练中一个被低估的挑战是评估方差。随着推理模型兴起，这一点尤其重要，因为推理模型需要使用高于 0 的 temperature 进行采样，才能得到最佳评估分数。
只要从模型中采样，输出就会变得更有变异性。
不同 benchmark 的稳定性特征差异很大，原因包括 prompt 难度的方差、评估集中 prompt 的数量、被训练模型的脆弱性等。

在 Olmo 3 期间，团队跟踪了用于评估推理模型的不同评估的方差。
下表展示每个评估的标准差，计算方式是：对 14 个模型各运行 3 次，先取每个模型的方差，再按评估求平均：

| 类别 | Benchmark | 标准差 |
|----------|-----------|-----------|
| 高方差 | GPQA | 1.48 |
| | AlpacaEval 3 | 1.24 |
| | IFEval | 0.88 |
| 稳定 | ZebraLogic | 0.56 |
| | Omega | 0.56 |
| | AIME 24 (Avg@32) | 0.54 |
| | HumanEvalPlus | 0.46 |
| | AgiEval | 0.43 |
| | BigBenchHard | 0.39 |
| 非常稳定 | LiveCodeBench (Avg@10) | 0.29 |
| | MBPPPlus | 0.27 |
| | MATH | 0.25 |
| | MMLU | 0.22 |
| | PopQA | 0.16 |

Table: 多次推理运行中各评估 benchmark 的标准差，并按稳定性分类（数据来自 Olmo 3）。 {#tbl:eval_variance}

有些评估，如 LiveCodeBench，既有噪声又便宜（因为评估集中 prompt 较少），因此通过对每个模型重复运行 10 次评估，可以把评估从高方差集合移动到稳定设置。每个评估都可以这样做，但成本很容易膨胀。

我们还会在评估设置中看到方差来源，例如 batch size、vLLM 中的 tensor parallel 设置（例如 baseline 使用 TP=2），以及基础设施上长生成采样时的其他敏感数值问题。对 reasoner 来说，方差无处不在。

## 管理训练性能方差

在本书讨论的所有后训练 recipe 和工具中，最终模型的性能都会受到有意义的方差影响。
理解这种方差的分布、来源和影响，对于创建强模型至关重要。
训练最终模型的目标，是通过改变训练参数和随机 seed 来采样许多点，从而得到尽可能强的模型。
注意，这需要在模型 *实际* 变得更好与只是从评估噪声中重新抽到更好结果之间取得平衡。

上一节关注的是 *evaluation* 噪声，而更棘手的噪声来源是训练不确定性。
评估噪声可以通过在给定 checkpoint 上运行更多测试来管理，从而均匀降低噪声；但模型通常只训练一次，并且可能 *受益于* 一个正向异常值。

实践中，训练团队会采取许多步骤，从训练 recipe 中捕捉尽可能大的价值：

1. 对每次最终模型运行，都 sweep 学习率、batch size 等核心优化值。例如，对于一个新的基础模型，我建议先在较宽范围内运行 10 个学习率，确认最优区间，然后在更窄的最优窗口中重新运行。
2. 在最好的一些设置上运行多个 seed。随机 seed 可以对最终模型产生有意义的影响，值得投入计算。
3. Model merging 已经被确立为创建强模型的关键工具。合并可以有许多方式，从合并同一数据上的不同 checkpoint，到合并特定领域的专门模型。一般来说，合并被视为最终 recipe 中强大而简单的工具，但关于如何在 recipe 中为后续合并准备模型，尚未形成清晰最佳实践 [@yadav2024matters]。

## 识别失败的训练任务

训练模型时，一个需要建立的简单直觉，是区分不同类型的模型问题。
你希望大部分时间都花在这样的问题上：当前数据、算法或 recipe 本身还不够好。
另一方面，在搭建新 recipe 时，也经常会出现某些方法直接坏掉的情况。

理解这一点的最佳方式，是在一个基本静态的评估套件上评估许多模型。然后你会形成直觉，知道哪些测试很难通过后训练干预推动，例如 MMLU 这类知识密集型评估。
当后训练设置中某些东西非常、*非常* 坏时，这些基本稳定的评估往往会在一次训练任务中下降 10 到 20 分。
这是开发工具链时最有用的信号之一。
