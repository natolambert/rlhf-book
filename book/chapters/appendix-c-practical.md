---
prev-chapter: "Style & Information"
prev-url: "appendix-b-style"
page-title: "Appendix C: Practical Issues"
search-title: "Appendix C: Practical Issues"
next-chapter: "Home"
next-url: "https://rlhfbook.com/"
---

# Practical Issues and Advice

This appendix covers practical considerations for running post-training experiments at scale. 
This takes the form of a list of lessons, rather than a coherent narrative.

## 1. Compute Costs of Post-Training

There are two different ways of scoping costs for post-training runs.
The largest cost is in developing the recipe, which can easily be 10 to 100X the compute of the final few training runs.
The secondary costs, which are easier to measure, are the costs to thoroughly apply a recipe, which entails multiple seeds, careful evaluation, potential engineering headaches, etc.

For the first cost, to develop a post-training recipe like Tülu 3 [@lambert2024t], the team ran on the order of thousands of experiments/evaluations at the 7B scale before having the final model.

For final runs, the Olmo 3 report has a detailed accounting of what is involved in training the final 32B Think model [@teamolmo2025olmo3]:

> Post-training follows a different operational pattern in which we run each stage multiple times, sweeping over learning rates and other hyperparameters. The theory for post-training, particularly, RL, is less developed, so we have to run multiple experiments to identify the optimal hyperparameters for a given base model. We hope to address this in future work.
>
> During post-training, checkpoint evaluation consumes a larger proportion of compute resources, in part due to long generations from reasoning models on core benchmarks. For SFT, we swept over four candidate learning rates, on 256 GPUs each, in parallel for 36 hours. Then approximately 12 hours was spent on evaluation, merging, and checkpoint confirmation, totaling approximately two days. DPO training takes less time per run (about 18 hours for a full learning-rate sweep on 64 GPUs per job) but in practice extended over multiple days due to cluster instability. The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days with at least a day of training time lost due to stability issues. After the initial release of Olmo 3, we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B.

As scaling reinforcement learning becomes more standard practice, this will shift yet again [@khatri2025art].
Continuing the above example, where the original Olmo 3 32B Think post-training took only a couple of weeks, to release the improved Olmo 3.1 32B Think model the team needed to train it for an additional 3.5 weeks with RLVR. This is a substantial cost in *time* more than in total compute.

## 2. Evaluation Variance

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

## 3. Managing Training Performance Variance

Throughout all the post-training recipes and tools discussed in this book, the final model is subject to meaningful variance in performance.
Understanding the distribution of this variance, its sources, and its effects is crucial to creating strong models.
The goal of training a final model is to sample many points, by varying training parameters and random seeds, in order to get the strongest model possible.
Note that this is a balance between the model *actually* being better, and not just the benefit of re-rolling from evaluation noise.

Where the previous section focuses on *evaluation* noise, the trickier source of noise is training uncertainty.
Where evaluation noise can be managed by running more tests on a given checkpoint (uniformly reducing noise), models are trained once and can *benefit* from a positive outlier.

In practice, training teams take many steps to capture the maximum possible value out of their training recipe:

1. Sweep core optimization values like learning rate, batch size, etc. for every final model run. For example, with a new base model, I'd recommend running 10 learning rates over a wide region to be sure you're in the optimal range, then re-run in the tighter, optimal window.
2. Run multiple seeds on the best few settings. Random seed can have meaningful effects on the final model, and it's worth spending compute on.
3. Model merging is established as a key tool used to create strong models. Merging can be done in many ways, from merging different checkpoints on the same data or specialized models on specific domains. Generally, merging is seen to be a strong and simple tool in final recipes, but clear best practices aren't established on how to prepare a model for later merging in a recipe [@yadav2024matters].

## 4. Identifying Bad Training Jobs

A simple intuition that's important to establish when training models is the different types of model issues. 
You want most of your time to be spent on issues where the current data, algorithm, or recipe just isn't good enough.
On the other hand, there are plenty of times when setting up a new recipe that certain methods are just broken.

The best way to understand this is to evaluate many models on a largely static evaluation suite. Then you develop an intuition for which tests are hard to move with post-training interventions (often knowledge-heavy evaluations such as MMLU).
When something is very, *very* broken in a post-training setup these largely stable evaluations can often drop by 10-20 points in a training job. 
This is one of the most useful signals there are when developing tooling!