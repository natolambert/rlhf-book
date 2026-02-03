---
prev-chapter: "Style & Information"
prev-url: "appendix-b-style"
page-title: "Appendix C: Practical Issues"
next-chapter: "Home"
next-url: "https://rlhfbook.com/"
---

# Practical Issues

This appendix covers practical considerations for running post-training experiments at scale. 
This takes the form of a list of lessons, rather than a coherent narrative.

## 1. Compute Costs of Post-Training

There are two different ways of scoping costs for post-training runs.
The largest cost is in developing the recipe, which can easily be 10 to 100X the compute of the final few training runs.
The secondary costs, which are easier to measure are the costs to thoroughly apply a recipe, which entails multiple seeds, careful evaluation, potential engineering headaches, etc.

For the first cost, to develop a post-training recipe like Tülu 3 [@lambert2024t], the team ran on the order of thousands of experiments/evaluations at the 7B scale before having the final model.

For final runs, the Olmo 3 report has a detailed accounting of what is involved in training the final 32B Think model [@teamolmo2025olmo3]:

> Post-training follows a different operational pattern in which we run each stage multiple times, sweeping over learning rates and other hyperparameters. The theory for post-training, particularly, RL, is less developed, so we have to run multiple experiments to identify the optimal hyperparameters for a given base model. We hope to address this in future work.
>
> During post-training, checkpoint evaluation consumes a larger proportion of compute resources, in part due to long generations from reasoning models on core benchmarks. For SFT, we swept over four candidate learning rates, on 256 GPUs each, in parallel for 36 hours. Then approximately 12 hours was spent on evaluation, merging, and checkpoint confirmation, totaling approximately two days. DPO training takes less time per run (about 18 hours for a full learning-rate sweep on 64 GPUs per job) but in practice extended over multiple days due to cluster instability. The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days with at least a day of training time lost due to stability issues. After the initial release of Olmo 3, we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B.

As scaling reinforcement learning becomes more standard practice, this will shift yet again [@khatri2025art].
Continuing the above example, where the original Olmo 3 32B Think post-training took only a couple of weeks, to release the improved Olmo 3.1 32B Think model the team needed to train it for an additional 3.5 weeks with RLVR. This is a substantial cost in *time* more than in total compute.

## Evaluation Variance

One underappreciated challenge in post-training is evaluation variance, especially with the rise of reasoning models that need to use sampling with temperatures above 0 to get the best evaluation scores. 
With any sampling from models, the outputs become more variable.
Different benchmarks have vastly different stability characteristics, due to the variance in difficulty of the prompts, the number of prompts in the evaluation set, the brittleness of the models being trained, etc.

During Olmo 3, the team tracked the variance of different evaluations used to evaluate reasoning models.
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

Table: Standard deviation of evaluation benchmarks across multiple inference runs, categorized by stability (data from Olmo 3). {#tbl:eval_variance}

Some evaluations, such as LiveCodeBench, were both noisy and cheap (via few prompts in the set), so by re-running the evaluation 10 times per model, the evaluation could move from the high-variance set to a stable setting. This could be done for every evaluation, but it can easily balloon costs.

We also see sources of variance in evaluation settings like batch size, tensor parallel settings within VLLM (e.g., TP=2 for baselines), and other sensitive numerics for sampling long generations across infrastructure. Variance is everywhere with reasoners.

## Running Multiple Seeds

<!-- TODO: Add discussion on the importance of running multiple training seeds -->

## Identifying Bad Training Jobs

<!-- TODO: Add discussion on what bad training jobs look like - e.g., sharp drops on stable evals -->

## Model Merging

Model merging has emerged as a practical technique for improving model performance across different training approaches.

<!-- TODO: Add discussion on model merging as performance gain across any training optimizer. Basic linear merging often works best on larger models. -->

**Key references on empirical model merging:**

<!-- TODO: Add references from ChatGPT Pro suggestions -->

## Training vs Deployment System Prompts

<!-- TODO: Expand on deployment vs training system prompts - when to bake behavior into weights vs handle at inference time -->

A practical lesson from training models at different scales: identity and persona are easier to adjust via system prompts at deployment time than to retrain into the model weights. For example, with OLMo 3, the 32B model's identity was not baked into the training data, making it easy to adjust via system prompt at demo time. However, the 7B model had identity more embedded in training, and retraining wasn't feasible at that point.

The takeaway: if using synthetic data to shift model identity or persona, you need a meaningful amount of data to make that shift stick. For smaller models or tighter timelines, keeping identity flexible via deployment-time system prompts can be more practical than trying to train it in.
