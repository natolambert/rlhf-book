# Regularization

Throughout the RLHF optimization, many regularization steps are used to prevent over-optimization of the reward model.
The most popular variant, used in most RLHF implementations at the time of writing, is a KL Distance from the current policy to a reference policy across the generated samples.
Many other regularization techniques have emerged in the literature to then disappear in the next model iteration in that line of research.
The general formulation, when used in an RLHF framework with a reward model, $r_\theta$ is as follows:

$$ r = r_\theta - \lambda r_{\text{reg.}} $$

With the reference implementation being:

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{Ref.}}(y \mid x) \right)
$$

## KL Distances

For mathematical definitions, see Chapter 5 on Problem Setup.
Recall that KL distance is defined as follows:

$$ D_{KL}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$

### Reference Policy

The most common implementation of KL penalities are by comparing the distance between the generated tokens 

### Reference Dataset

### KL Controllers

### Implementation Example

In practice, the implementation of KL distance is often approximated [@schulman2016klapprox], making the implementation far simpler.
With the above definition, the summation of KL can be converted to an expectation when sampling directly from the distribution $P(X)$.
In this case, the distribution $P(X)$ is the generative distribution of the model currently being trained (i.e. not the reference model).
Then, the computation for KL distance changes to the following:

$$
D_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$

This mode is far simpler to implement, particularly when dealing directly with log probabilities used frequently in language model training.

```python
import torch.nn.functional as F
# Step 1: Generate tokens using the trained model's policy
generated_tokens = model.generate(inputs)

# Step 2: Get logits for both models using the generated tokens as context
logits = model.forward(inputs) # technically redundant
ref_logits = ref_model.forward(inputs)
logprobs = convert_to_logpbs(logits) # softmax and normalize
ref_logprobs = convert_to_logpbs(ref_logits)

kl_approx = logprob - ref_logprob
kl_full = F.kl_div(ref_logprob, logprob) # alternate computation
```
Some example implementations include [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) and [Hamish Ivison's Jax Code]https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)

## Likelihood Penalty

- https://arxiv.org/abs/2404.19733 on DPO loss

## Reward Bonuses

- Nemotron

## Margin Losses

- Llama 2
- Rebel
- Reward Preference Optimization (Nemotron)