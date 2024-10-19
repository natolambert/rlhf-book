# Reward Modeling

Reward models are core to the modern approach to RLHF.
Reward models broadly have been used extensively in reinforcement learning research as a proxy for environment rewards [@sutton2018reinforcement].
The practice is closely related to inverse reinforcement learning, where the problem is to approximate an agent's reward function given trajectories of behavior [@ng2000algorithms], and other areas of deep reinforcement learning.
Reward models were proposed, in their modern form, as a tool for studying the value alignment problem [@leike2018scalable].

## Training Reward Models

There are two popular expressions for how to train a reward model -- they are numerically equivalent. 
The canonical implementation is derived from the Bradley-Terry model of preference [@BradleyTerry].
A Bradley-Terry model of preferences measures the probability that the pairwise comparison for two events drawn from the same distribution, say $i$ and $j$, satisfy the following relation, $i > j$:
$$P(i > j) = \frac{p_i}{p_i + p_j}$$ {eq:bradterry}

To train a reward model, we must formulate a loss function that satisfies the above relation.
The first structure applied is to convert a language model into a model that outputs a scalar value, often in the form of a single classification probability logit.
Thus, we can take the score of this model with two samples, the $i$ and $j$ above are now completions, $y_1$ and $y_2$, to one prompt, $x$ and score both of them with respect to the above model, $r_\theta$.

The probability of success for a given reward model in a pairwise comparison, becomes:

$$P(y_1 > y_2) = \frac{\exp(r(y_1))}{\exp(r(y_1)) + \exp(r(y_2))}$$ {eq:bradterryrm}

Then, by taking the gradient with respect to the model parameters, we can arrive at the loss function to train a reward model.
The first form, as in [@ouyang2022training] and other works:
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) \right) \right)$$ {#eq:rewardmodeling1}

Second, as in [@askell2021general] and other works:
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(x, y_l)}  - e^{r_{\theta}(x, y_w)} \right)$$ {#eq:rewardmodeling2}


## Implementation Example

Implementing the reward modeling loss is quite simple.
More of the implementation challenge is on setting up a separate data loader and inference pipeline.
```python
import torch.nn as nn
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

## Variants

### Margin Loss

Llama 2

### Prompt Balancing

InstructGPT

### K-wise loss function

Starling https://arxiv.org/abs/2301.11270

## Further Reading

reward modeling reading list imo

RewardBench (biased, but gives a good overview): https://arxiv.org/abs/2403.13787
ArmorRM: https://arxiv.org/abs/2406.12845
HelpSteer2: https://arxiv.org/html/2406.08673v1
HelpSteer2-Preference: https://arxiv.org/abs/2410.01257
Nemotron 340: https://arxiv.org/abs/2406.11704
Llama 2: https://arxiv.org/abs/2307.09288
Interconnects 1: https://www.interconnects.ai/p/why-reward-models-matter
Interconnects 2: https://www.interconnects.ai/p/open-rlhf-reward-models
The o.g. paper: https://arxiv.org/abs/1811.07871
Critique out loud RMs: https://arxiv.org/abs/2408.11791

## Recommendations

Strong tendency in the literature to train for only one epoch, otherwise it overfits