---
prev-chapter: "Style & Information"
prev-url: "appendix-b-style"
page-title: "Appendix C: Practical Issues"
next-chapter: "Home"
next-url: "https://rlhfbook.com/"
---

# Practical Issues

This appendix covers practical considerations for running post-training experiments at scale.

## Compute Costs of Post-Training

Post-training follows a different operational pattern than pretraining. From the OLMo 3 report [@teamolmo2025olmo3]:

> Post-training follows a different operational pattern in which we run each stage multiple times, sweeping over learning rates and other hyperparameters. The theory for post-training, particularly, RL, is less developed, so we have to run multiple experiments to identify the optimal hyperparameters for a given base model. We hope to address this in future work.
>
> During post-training, checkpoint evaluation consumes a larger proportion of compute resources, in part due to long generations from reasoning models on core benchmarks. For SFT, we swept over four candidate learning rates, on 256 GPUs each, in parallel for 36 hours. Then approximately 12 hours was spent on evaluation, merging, and checkpoint confirmation, totaling approximately two days. DPO training takes less time per run (about 18 hours for a full learning-rate sweep on 64 GPUs per job) but in practice extended over multiple days due to cluster instability. The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days with at least a day of training time lost due to stability issues. After the initial release of Olmo 3, we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B.

## Evaluation Variance

One underappreciated challenge in post-training is evaluation variance. Different benchmarks have vastly different stability characteristics, which can lead practitioners to draw incorrect conclusions from noisy signals [@teamolmo2025olmo3].

The table below shows the standard deviation of each evaluation, computed as the mean of the standard deviation from 3 runs of 14 models (take variance of each model, then average per evaluation):

| Category | Benchmark | Std. Dev. |
|----------|-----------|-----------|
| High Variance | GPQA | 1.48 |
| | AlpacaEval 3 | 1.24 |
| | IFEval | 0.88 |
| Stable | ZebraLogic | 0.56 |
| | Omega | 0.56 |
| | AIME 24 (Avg@32) | 0.54 |
| | HumanEvalPlus | 0.46 |
| | AgiEval | 0.43 |
| | BigBenchHard | 0.39 |
| Very Stable | LiveCodeBench (Avg@10) | 0.29 |
| | MBPPPlus | 0.27 |
| | MATH | 0.25 |
| | MMLU | 0.22 |
| | PopQA | 0.16 |

We also see sources of variance in items like batch size, tensor parallel settings (e.g., TP=2 for baselines), and other sensitive numerics for sampling long generations across infrastructure. Variance is everywhere with reasoners.

## Running Multiple Seeds

<!-- TODO: Add discussion on the importance of running multiple training seeds -->

## Identifying Bad Training Jobs

<!-- TODO: Add discussion on what bad training jobs look like - e.g., sharp drops on stable evals -->

## Model Merging

Model merging has emerged as a practical technique for improving model performance across different training approaches.

<!-- TODO: Add discussion on model merging as performance gain across any training optimizer. Basic linear merging often works best on larger models. -->

**Key references on empirical model merging:**

<!-- TODO: Add references from ChatGPT Pro suggestions -->
