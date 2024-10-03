# Regularization

Throughout the RLHF optimization, many regularization steps are used to prevent over-optimization of the reward model.
Over-optimization in these contexts looks like models that output nonsensical text.
Some examples of optimization ``off the rails'' are that models can output followable math reasoning with extremely incorrect answers, repeated text, switching languages, or excessive special characters.

The most popular variant, used in most RLHF implementations at the time of writing, is a KL Distance from the current policy to a reference policy across the generated samples.
Many other regularization techniques have emerged in the literature to then disappear in the next model iteration in that line of research.
That is to say that regularization outside the core KL distance from generations is often used to stabilize experimental setups that can then be simplified in the next generations.
Still, it is important to understand tools to constrain optimization in RLHF.

The general formulation, when used in an RLHF framework with a reward model, $r_\theta$ is as follows:

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {eq:rl_start}

With the reference implementation being:

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi^{\text{RL}}(y \mid x) \, \| \, \pi^{\text{Ref.}}(y \mid x) \right)
$$ {#eq:kl_standard}

## KL Distances

For mathematical definitions, see Chapter 5 on Problem Setup.
Recall that KL distance is defined as follows:

$$ D_{KL}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$

### Reference Model to Generations

The most common implementation of KL penalities are by comparing the distance between the generated tokens during training to a static reference model.
The intuition is that the model you're training from has a style that you would like to stay close to.
This reference model is most often the instruction tuned model, but can also be a previous RL checkpoint.
With simple substitution, the model we are sampling from becomes $P(x)^{\text{RL}}$ and $P(x)^{\text{Ref.}}, shown above in @eq:kl_standard.
Such KL distance was first applied to dialogue agents well before the popularity of large language models [@jaques2017sequence], yet KL control was quickly established as a core technique for fine-tuning pretrained models [@jaques2020human].

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

## Pretraining Gradients

Another way of viewing regularization is that you may have a *dataset* that you want the model to remain close to, as done in InstructGPT [@ouyang2022training] ``in order to fix the
performance regressions on public NLP datasets''.
To implement this, they modify the training objective for RLHF.
Taking @eq:rl_start, we can transform this into an objective function to optimize by sampling from the RL policy model, completions $y$ from prompts $x$, which yields:
$$
\text{objective} (\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi^{\text{RL}}_{\theta}}} \left[ r_{\theta}(x, y) - \lambda r_{\text{reg.}} \right]
$$
Then, we can add an additional reward for higher probabilities on pretraining accuracy:
$$
\text{objective} (\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi^{\text{RL}}_{\theta}}} \left[ r_{\theta}(x, y) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi^{\text{RL}}_{\theta}(x)) \right]
$$

## Likelihood Penalty

- https://arxiv.org/abs/2404.19733 on DPO loss

## Reward Bonuses

- Nemotron

## Margin Losses

- Llama 2
- Rebel
- Reward Preference Optimization (Nemotron)