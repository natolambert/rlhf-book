# RLHF Book 中文术语表

本术语表用于统一 `book-zh/chapters/` 的翻译。首次出现的重要术语可写作“中文（English）”，之后使用中文或常见缩写。

机器可处理的术语表见 `translation/TERMS.zh.tsv`。如果需要全书批量替换某个译法，使用 `scripts/replace_translation_term.py` 先 dry-run，再 apply。

| English | 中文建议 | 备注 |
| --- | --- | --- |
| reinforcement learning | 强化学习 | 首次可写“强化学习（reinforcement learning, RL）”。 |
| reinforcement learning from human feedback | 基于人类反馈的强化学习 | 首次写“基于人类反馈的强化学习（RLHF）”，之后多用 RLHF。 |
| language model | 语言模型 | LLM 译为“大语言模型”。 |
| large language model | 大语言模型 | 可保留 LLM。 |
| post-training | 后训练 | 指预训练之后的模型适配和优化阶段。 |
| pretraining | 预训练 |  |
| instruction tuning | 指令微调 | SFT 相关上下文中也可写“指令调优”，本书统一“指令微调”。 |
| supervised fine-tuning | 监督微调 | 保留 SFT。 |
| preference tuning | 偏好调优 |  |
| preference data | 偏好数据 |  |
| preference model | 偏好模型 |  |
| reward model | 奖励模型 | 保留 RM。 |
| outcome reward model | 结果奖励模型 | 保留 ORM。 |
| process reward model | 过程奖励模型 | 保留 PRM。 |
| reward signal | 奖励信号 |  |
| policy | 策略 | 数学符号 `policy \pi` 可写“策略 $\pi$”。 |
| value function | 价值函数 |  |
| advantage | 优势 |  |
| trajectory | 轨迹 |  |
| rollout | rollout | 强化学习语境下通常保留英文；必要时译为“采样轨迹”。 |
| on-policy | on-policy | 可解释为“同策略”。本书优先保留英文。 |
| off-policy | off-policy | 可解释为“离策略”。本书优先保留英文。 |
| rejection sampling | 拒绝采样 |  |
| best-of-N | best-of-N | 可解释为“从 N 个候选中选优”。 |
| direct alignment | 直接对齐 | 算法章节标题中使用“直接对齐”。 |
| direct preference optimization | 直接偏好优化 | 保留 DPO。 |
| proximal policy optimization | 近端策略优化 | 保留 PPO。 |
| policy gradient | 策略梯度 |  |
| regularization | 正则化 |  |
| over-optimization | 过度优化 |  |
| evaluation | 评估 |  |
| benchmark | benchmark | 可按语境译为“基准测试”，具体 benchmark 名称不译。 |
| synthetic data | 合成数据 |  |
| distillation | 蒸馏 |  |
| tool use | 工具使用 |  |
| character training | 角色训练 | 如指行业术语，可首现保留英文。 |
| alignment | 对齐 | AI alignment 译为“AI 对齐”。 |
| preference | 偏好 |  |
| completion | 补全 | 聊天语境可译为“回复”。 |
| prompt | prompt | 视语境可译为“提示词”，但技术流程中可保留 prompt。 |
| response | 回复 |  |
| sample | 样本 / 采样 | 名词“样本”，动词“采样”。 |
| dataset | 数据集 |  |
| distribution | 分布 |  |
| loss | 损失 |  |
| objective | 目标函数 | 算法优化语境中使用。 |
| inference | 推理 |  |
| training | 训练 |  |
| deployment | 部署 |  |
| serving | 服务化 |  |
| elicitation | 引出 / 激发 | 视上下文选择。 |
| calibration | 校准 |  |
| annotator | 标注者 |  |
| labeler | 标注员 |  |
| human feedback | 人类反馈 |  |
| human preference | 人类偏好 |  |
| scalable oversight | 可扩展监督 |  |
| reward hacking | 奖励黑客 / 奖励投机 | 视上下文，首次解释。 |
| specification gaming | 规范钻空子 | 可首现保留英文。 |
| hallucination | 幻觉 |  |
| reasoning model | 推理模型 |  |
| verifier | 验证器 |  |
| generator | 生成器 |  |
| classifier | 分类器 |  |
| ranking | 排序 |  |
| ranking model | 排序模型 |  |
| pairwise comparison | 成对比较 |  |
| pairwise preference | 成对偏好 |  |
| Likert scale | Likert 量表 |  |
| fine-tune | 微调 |  |
| open model | 开放模型 |  |
| frontier model | 前沿模型 |  |
| closed model | 闭源模型 |  |
