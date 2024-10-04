# Reward Modeling

TODO: Have both the InstructGPT and Anthropic loss formulations, which are slightly different

## Training Reward Models

There are two popular expressions for how to train a reward model -- they are numerically equivalent.

$$
\mathcal{L}(\theta) = - \left[ \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) \right) \right) \right]
$$
[@ouyang2022training]

$$
\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(x, y_l)}  - e^{r_{\theta}(x, y_w)} \right)
$$
[@askell2021general]

## Implementation Example

Implementing the reward modeling loss is quite simple.
More of the implementation challenge is on setting up a separate data loader and inference pipeline.
```python
import torch.nn as nn
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

### Further Reading

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