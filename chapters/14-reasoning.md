---
prev-chapter: "Constitutional AI & AI Feedback"
prev-url: "13-cai.html"
page-title: Reasoning Training & Models
next-chapter: "Synthetic Data & Distillation"
next-url: "16-synthetic.html"
---

# [Incomplete] Reasoning Training & Models

At the 2016 edition of the Neural Information Processing Systems (NeurIPS) conference, Yann LeCun first introduced his now-famous cake metaphor for where learning happens in modern machine learning systems:

> If intelligence is a cake, the bulk of the cake is unsupervised learning, the icing on the cake is supervised learning, and the cherry on the cake is reinforcement learning (RL).

This analogy is now largely complete with modern language models. Self-supervised learning on vast swaths of internet data makes up the majority of the cake (especially when viewed in compute spent in FLOPs), the beginning of post-training in supervised finetuning (SFT) for instructions tunes the model to a narrower distribution, and finally “pure” reinforcement learning (RL) is the cherry on top. 
We learn just “a few bits” of information with RL in just a few training samples.

This little bit of RL takes many forms. It can be ... TODO


Despite many, many takes that “RL doesn’t work yet” or “RL scaling isn’t ready yet” (and implicit versions of this saying to focus on “RL that Matters”), Yann’s view seems to have been right.

OpenAI’s new Reinforcement Finetuning (RFT) API (just a research program for now), announced on day 2 of the 12 days of OpenAI, is the bridge that brings RL to the masses. This is a very surprising development even for those most faithful to RL. With RFT, one can likely finetune any of OpenAI’s models, while they highlighted o1 mini, it is of obvious value to both standard autoregressive models and reasoning-heavy models. To use RFT, you need three things — 1) training data for your application, 2) validation data for your application to test overfitting, and 3) a definition via OpenAI’s “grader” configuration (more on this later).

Reinforcement Finetuning has been met with excitement and trepidation. The best practices for using existing finetuning APIs, built on instruction tuning infrastructure, are still far from established. The general public of AI builders knows very little about how RL training can change model behavior to improve performance on tasks with minimal overall changes to the model.

In many domains, Reinforcement Finetuning is much more aligned with the goals of developers by being focused on performance rather than behavior. Standard finetuning APIs generally use a parameter-efficient finetuning method such as LoRA with supervised finetuning on instructions. Developers pass in prompts and completions and the model is tuned to match that by updating model parameters to match the completions. OpenAI describes this as increasing the prevalence of “features” in the text of interest.

Reinforcement finetuning is focused on matching answers. Given queries and correct answers, RFT helps the model learn to get the correct answers. While standard instruction tuning is done with 1 or 2 epochs of loss updates over the data, reinforcement finetuning gets its name by doing hundreds or thousands of epochs over the same few data points to give the model time to learn new behaviors. This can be viewed as reinforcing positive behaviors that would work sparingly in the base model version into robust behaviors after RFT.

The impact of reinforcement finetuning’s existence
Reinforcement finetuning signals many changes to the fate of RL and language models, both at OpenAI and elsewhere:

* Stability of RL can be solved: For its entire existence, the limiting factor on RL’s adoption has been stability. This manifests in two ways. First, the learning itself can be fickle and not always work. Second, the training itself is known to be more brittle than standard language model training and more prone to loss spikes, crashes, etc. Releasing an API where any user can train on their own data signals unprecedented stability improvements. This program is still a beta ahead of a full launch, but this signals that OpenAI is more confident about it working for the public rather than not. For example, last year when I heard about large-scale RL runs at frontier AI laboratories, it would be with stories like “they launch multiple seeds at once and only keep running the ones that didn’t crash.” Now, they can be confident in their RL running and accomplishing the task. The final output model is likely automatically detected by running evaluations on the checkpoint to make sure behavior did not dip and or measuring the KL distance from the initial policy. Both of these are signals that researchers rely on heavily in post-training experimentation, so automating decision-making based on it is extremely impactful.

* Open-source versions already “exist”: Our recent work at Ai2 on reinforcement learning with verifiable rewards (RLVR) is extremely similar. The major components, i.e. data format and optimizer type are identical, we just need increased open-source investment to understand many discussion items like which model to start on, which types of data to use, etc. Check out the code we’ve been using at Open Instruct.

* A potential data flywheel for advanced reasoning models: The best speculation is that OpenAI’s o1 is trained mostly with large-scale RL on data with verifiable outputs — much like this API. If this API works as intended, OpenAI could accumulate an extreme dataset for training future versions of their o1 models. The main limitation of these models is the lack of diversity in available domains and by experimenting with training on the targeted domains of many users of OpenAI’s models they can start turning a fruitful flywheel.

The scope of RL training for language models continues to grow: The biggest takeaway from o1 on a fundamental scientific level was that we have even more ways to train language models to potentially valuable behaviors. The more open doors that are available to researchers and engineers, the more optimism we should have about AI’s general trajectory. The RL finetuning API expands the window of permissivity for RL training.

I recall listening to a talk by a prominent OpenAI researcher a year ago (I couldn’t find it), where they said they were excited about RLHF and related methods just because the loss function is more general than autoregressive prediction. We are now living in this world of RL’s impact growing rapidly, and as many people expected, the human feedback piece is not always necessary.

