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