---
prev-chapter: "Evaluation & Prompting"
prev-url: "16-evaluation.html"
next-chapter: "Style and Information"
next-url: "18-style.html"
---

# Over Optimization

In the RLHF literature and discourse, there are three directions that over-optimization can emerge:

1. **Quantitative research** on the technical notion of over-optimization of reward, 
2. **Qualitative observations** that "overdoing" RLHF can result in worse models.
3. **Misalignment** where overdoing RLHF or related techniques can make a language model behave against its design.

This chapter provides a cursory introduction to both. We begin with the latter, qualitative, because it motivates the problem to study further.

## Qualitative (behavioral) over-optimization

*Note: This section draws on two [blog](https://www.interconnects.ai/p/llama-2-part-2) [posts](https://www.interconnects.ai/p/specifying-objectives-in-rlhf) from Interconnects.ai. It can also be viewed as an "objective mismatch" [@lambert2023alignment] [@lambert2020objective].*

### Managing proxy objectives

The thing about RLHF that should be more obvious is that we don't have a good reward function for chatbots. 
RLHF has been driven into the forefront because of its impressive performance at making chatbots a bit better to use (from both eliminating bad stuff and a bit of adding capabilities), which is entirely governed by a proxy objective — thinking that the rewards measured from human labelers in a controlled setting mirror those desires of downstream users. 
Post-training generally has emerged to include training on explicitly verifiable rewards, but standard learning from preferences alone also improves performance on domains such as mathematical reasoning and coding.

The proxy reward in RLHF is the score returned by a trained reward model to the RL algorithm itself because it is known to only be at best correlated with chatbot performance [@schulman2023proxy].
Therefore, it's been shown that applying too much optimization power to the RL part of the algorithm will actually decrease the usefulness of the final language model. 
And over-optimization, put simply by John, is "when optimizing the proxy objective causes the true objective to get better, then get worse." 
A curve where the training loss goes up, slowly levels off, then goes down. 
This is different from overfitting, where the model accuracy keeps getting better on the training distribution. 
Over-optimization of a proxy reward is much more subtle (and linked to the current [evaluation fog](https://www.interconnects.ai/t/evaluation) in NLP, where it's hard to know which models are actually "good").

The general notion captured by this reasoning follows from Goodhart's law, which is colloquially the notion that "When a measure becomes a target, it ceases to be a good measure." 
This adage is derived from Goodhart's writing [@goodhart1984problems]:

> Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes.

The insight here builds on the fact that we have optimizations we are probably incorrectly using ML losses as ground truths in these complex systems. 
In reality, the loss functions we use are designed (and theoretically motivated for) local optimizations. 
The global use of them is resulting in challenges with the RLHF proxy objective.

Common signs of over-optimization in early chat models emerged as:

- "As an AI language model..."
- "Certainly!..."
- Repetitiveness, hedging, ...
- Self-doubt, sycophancy [@sharma2023towards], and over apologizing
- Over refusals (more below)

Technically, it is an open question on which types of error in the training process result in these failures.
Many sources of error exist [@schulman2023proxy]: Approximation error from reward models not being able to fit to preferences, estimation error from overfitting during training the RM, optimization error in training the language model policy, etc.
This points to a fundamental question as to the limits of optimization the intents of data contractors relative to what downstream users want.

A potential solution is that *implicit* feedback will be measured from users of chatbots and models to tune performance.
Implicit feedback is actions taken by the user, such as re-rolling an output, closing the tab, or writing an angry message that indicates the quality of the previous response. The challenge here, and with most optimization changes to RLHF, is that there's a strong risk of losing stability when making the reward function more specific. RL, as a strong optimizer, is increasingly likely to exploit the reward function when it is a smooth surface (and not just pairwise human values). The expected solution to this is that future RLHF will be trained with both pairwise preference data and additional steering loss functions. There are also a bunch of different loss functions that can be used to better handle pairwise data, such as Mallow's model [@lu2011learning] or Plackett-Luce [@liu2019learning].

### Llama 2 and "too much RLHF"

### An aside on "undercooking" RLHF

As training practices for language models have matured, there are also prominent cases where strong models do not have an amount of post-training that most users expect, resulting in models that are harder to use than their evaluation scores would suggest.

TODO add references to minimax model? See tweets etc?

## Quantitative over-optimization

KL is the primary metric,

Put simply, the solution that will most likely play out is to use bigger models. Bigger models have more room for change in the very under-parameterized setting of a reward model (sample efficient part of the equation), so are less impacted. 
DPO may not benefit from this as much, the direct optimization will likely change sample efficiency one way or another.


[@gao2023scaling]

## Misalignment

Consequences [@zhuang2020consequences] or sycophancy [@sharma2023towards]