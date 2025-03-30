---
prev-chapter: "Constitutional AI & AI Feedback"
prev-url: "13-cai.html"
page-title: Reasoning Training & Inference-Time Scaling
next-chapter: "Synthetic Data & Distillation"
next-url: "16-synthetic.html"
---

# Reasoning Training & Inference-Time Scaling

At the 2016 edition of the Neural Information Processing Systems (NeurIPS) conference, Yann LeCun first introduced his now-famous cake metaphor for where learning happens in modern machine learning systems:

> If intelligence is a cake, the bulk of the cake is unsupervised learning, the icing on the cake is supervised learning, and the cherry on the cake is reinforcement learning (RL).

This analogy is now largely complete with modern language models and recent changes to the post-training stack.
In this analogy: 

- Self-supervised learning on vast swaths of internet data makes up the majority of the cake (especially when viewed in compute spent in FLOPs), 
- The beginning of post-training in supervised finetuning (SFT) for instructions tunes the model to a narrower distribution (along with the help of chosen examples for RLHF), and 
- Finally “pure” reinforcement learning (RL) is the cherry on top. 

We learn just “a few bits” of information with RL in just a few training samples.
This little bit of reasoning training emerged with **reasoning models** that use a combination of the post-training techniques discussed in this book to align preferences along with RL training on verifiable domains to dramatically increase capabilities such as reasoning, coding, and mathematics problem solving.

The training method for these models, Reinforcement Learning with Verifiable Rewards (RLVR) [@lambert2024t], proceeds very similarly to RLHF, but it makes the reward model optional in lieu of a scoring function that returns a positive reward when the answer is correct and 0 otherwise.

![RLVR in the form of an RL feedback loop. Instead of a reward model, we use a verification function.](images/rlvr-system){#fig:rlvr}

The first models to successfully deploy this type of training were OpenAI's o1 [@openai2024o1] and the open-weight model DeepSeek R1 [@guo2025deepseek]. 
Soon after, the the entire AI industry prioritized this training process and model style.
The core change here is more of a reallocation of the stages of training and the priority of different behaviors rather than this type of RL setup being entirely new.
Reasoning models brought an era where scaling RL training is expected.

As for the type of behavior these models accrue, consider the following example with DeepSeek V3 0325 versus their reasoning model, DeepSeek R1, on the query `Write me a short poem about a goldfish`. 
DeepSeek V3 is very to the point:

> Goldfish Dreams
> 
> In a bowl of glass so clear and round,
> A tiny goldfish swims without a sound.
> With scales that gleam like sunlight’s kiss,
> He drifts along in watery bliss.
> 
> A flick of fin, a flash so bright,
> Chasing bubbles in the light.
> Does he dream of oceans wide?
> Or is his little world enough—his pride?

Meanwhile, DeepSeek R1's experience is different.
It first generates numerous thinking tokens to craft a response before responding to the user.
The answer appears as:

> `<thinking>` Okay, the user wants a short poem about a goldfish. Let me start by brainstorming some goldfish-related imagery. Goldfish are often in bowls, right? So maybe start with the setting. Words like "gilded bowl" or "crystal globe" could work.
> 
> Next, think about the movement of the goldfish. They swim smoothly, so words like "dart", "glide", or "swirl" come to mind. Maybe describe their fins as "silken" to emphasize delicacy.
> 
> Then, consider the environment around the goldfish. Sunlight through water creates shimmering effects. Words like "sunbeams dance" or "ripples spin" might capture that. Including elements like castles or pebbles adds a playful touch.
> 
> Themes? Maybe the contrast between the goldfish's small world and its perception of freedom. The bowl as a universe, tiny but complete. Ending with a sense of mystery or silent observation could give depth. Need to keep it concise, maybe four stanzas with a rhyme scheme. Let me check the rhythm and flow. Avoid clichés but keep it accessible. Alright, let's piece it together. `</thinking>` 
> 
> Gilded Dreams in Liquid Glass
> 
> A gilded bowl, a crystal globe,
> Where tiny tides in silence flow—
> A flash of orange, a whispered robe,
> You dart through sunlit indigo.
> 
> Your world is small, yet boundless too—
> Each pebble, castle, weed unfurled.
> The surface bends the sky to blue,
> And ripples spin your liquid world.
> 
> You glide through myths we’ll never know,
> A silent monarch, veiled in mist.
> Do dreams drift where the bubbles go?
> Or memories in amethyst?
> 
> We watch you trace your endless sphere—
> A universe within a tear.

The tokens in the `<thinking>` tags is the model's reasoning.
For more complex problems the reasoning stage can take thousands of tokens before producing an answer.
So, long-context language models were a prerequisite to advanced reasoning behavior, but that is not the focus of this chapter.

## Why Does RL Work Now?

Despite many, many takes that “RL doesn’t work yet” or “RL scaling isn’t ready yet” (and implicit versions of this saying to focus on “RL that Matters”), Yann’s view seems to have been right.

* Stability of RL can be solved: For its entire existence, the limiting factor on RL’s adoption has been stability. This manifests in two ways. First, the learning itself can be fickle and not always work. Second, the training itself is known to be more brittle than standard language model training and more prone to loss spikes, crashes, etc. Releasing an API where any user can train on their own data signals unprecedented stability improvements. This program is still a beta ahead of a full launch, but this signals that OpenAI is more confident about it working for the public rather than not. For example, last year when I heard about large-scale RL runs at frontier AI laboratories, it would be with stories like “they launch multiple seeds at once and only keep running the ones that didn’t crash.” Now, they can be confident in their RL running and accomplishing the task. The final output model is likely automatically detected by running evaluations on the checkpoint to make sure behavior did not dip and or measuring the KL distance from the initial policy. Both of these are signals that researchers rely on heavily in post-training experimentation, so automating decision-making based on it is extremely impactful.

* Open-source versions already “exist”: Our recent work at Ai2 on reinforcement learning with verifiable rewards (RLVR) is extremely similar. The major components, i.e. data format and optimizer type are identical, we just need increased open-source investment to understand many discussion items like which model to start on, which types of data to use, etc. Check out the code we’ve been using at Open Instruct.

* A potential data flywheel for advanced reasoning models: The best speculation is that OpenAI’s o1 is trained mostly with large-scale RL on data with verifiable outputs — much like this API. If this API works as intended, OpenAI could accumulate an extreme dataset for training future versions of their o1 models. The main limitation of these models is the lack of diversity in available domains and by experimenting with training on the targeted domains of many users of OpenAI’s models they can start turning a fruitful flywheel.


## RL Training vs. Inference Time Scaling

Training with Reinforcement Learning to illicit reasoning behaviors and performance on verifiable domains is closely linked to the ideas of inference time scaling.
Inference-time scaling, also called test-time scaling, is the general class of methods that use more computational power at inference in order to perform better at a downstream tasks.
Methods for inference-time scaling were studied before the release of DeepSeek R1 and OpenAI's o1, which both massively popularized investment in RL training specifically.
Examples include value-guided sampling [@liu2023don] or repeated random sampling with answer extraction [@brown2024large].

RL training is a short path to inference time scaling laws being used, but in the long-term we will have more methods for eliciting the inference-time tradeoffs we need for best performance.
Training models heavily with RL changes them so that they generate more tokens per response in a way that is strongly correlated with downstream performance. 
This is a substantial shift from the length-bias seen in early RLHF systems [@singhal2023long], where the human preference training had a side effect of increasing response rate for marginal gains on preference rankings.

Downstream of the RL trained models there are many methods being explored to continue to push the limits of reasoning and inference-time compute.
These are largely out of the scope of this book due to their rapidly evolving nature, but they include distilling reasoning behavior from a larger RL trained model to a smaller model via instruction tuning [@muennighoff2025s1], composing more inference calls [@chen2024more], and more.
What is important here is the correlation between downstream performance and an increase in the number of tokens generated -- otherwise it is just wasted energy.


## The Future (Beyond Reasoning) of Reinforcement Finetuning

In many domains, Reinforcement Finetuning is much more aligned with the goals of developers by being focused on performance rather than behavior. Standard finetuning APIs generally use a parameter-efficient finetuning method such as LoRA with supervised finetuning on instructions. Developers pass in prompts and completions and the model is tuned to match that by updating model parameters to match the completions. OpenAI describes this as increasing the prevalence of “features” in the text of interest.

Reinforcement finetuning is focused on matching answers. Given queries and correct answers, RFT helps the model learn to get the correct answers. While standard instruction tuning is done with 1 or 2 epochs of loss updates over the data, reinforcement finetuning gets its name by doing hundreds or thousands of epochs over the same few data points to give the model time to learn new behaviors. This can be viewed as reinforcing positive behaviors that would work sparingly in the base model version into robust behaviors after RFT.

The impact of reinforcement finetuning’s existence
Reinforcement finetuning signals many changes to the fate of RL and language models, both at OpenAI and elsewhere:

The scope of RL training for language models continues to grow: The biggest takeaway from o1 on a fundamental scientific level was that we have even more ways to train language models to potentially valuable behaviors. The more open doors that are available to researchers and engineers, the more optimism we should have about AI’s general trajectory. The RL finetuning API expands the window of permissivity for RL training.

I recall listening to a talk by a prominent OpenAI researcher a year ago (I couldn’t find it), where they said they were excited about RLHF and related methods just because the loss function is more general than autoregressive prediction. We are now living in this world of RL’s impact growing rapidly, and as many people expected, the human feedback piece is not always necessary.

