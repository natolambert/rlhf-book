---
prev-chapter: "Reward Modeling"
prev-url: "07-reward-models"
page-title: Regularization
next-chapter: "Instruction Tuning"
next-url: "09-instruction-tuning"
---

# Regularization

Throughout the RLHF optimization, many regularization steps are used to prevent over-optimization of the reward model.
Over-optimization in these contexts looks like models that output nonsensical text.
Some examples of optimization "off the rails" are that models can output followable math reasoning with extremely incorrect answers, repeated text, switching languages, or excessive special characters.

The most popular variant, used in most RLHF implementations at the time of writing, is a KL distance from the current policy to a reference policy across generated samples.
“KL distance” is a colloquial term for expressing the *optimization distance* within the training process, even though KL divergence—the underlying mathematical method for measuring the separation of two probability distributions—does not satisfy the formal properties required to be a true distance metric (it is simply easier to call the number a distance than a numeric measure of distributional difference).
Many other regularization techniques have emerged in the literature to then disappear in the next model iteration in that line of research.
That is to say that regularization outside the core KL distance from generations is often used to stabilize experimental setups that can then be simplified in the next generation.
Still, it is important to understand tools to constrain optimization in RLHF.

The general formulation, when used in an RLHF framework with a reward model, $r_\theta$ is as follows:

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

With the reference implementation being:

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi^{\text{RL}}(y \mid x) \, \| \, \pi^{\text{Ref.}}(y \mid x) \right)
$$ {#eq:kl_standard}

## KL Divergences in RL Optimization

For mathematical definitions, see Chapter 3 on Problem Setup.
Recall that a KL divergence measure of probability difference is defined as follows:

$$ D_{KL}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

In RLHF, the two distributions of interest are often the distribution of the new model version, say $P(x)$, and a distribution of the reference policy, say $Q(x)$.
Different optimizers use different KL directions. Throughout this book, the most common "KL Penalty" that is used is called the reverse KL to the reference policy. In practice, this reduces to a Monte Carlo estimate that samples tokens from the RL model and computes probabilities from the reference model. Intuitively, this reverse KL has a numerical property that applies a large penalty when the new model, $P$ or $\pi_\text{RL}$, puts substantial probability mass where the original reference model assigns low probability.

The other KL direction is still often used in ML, e.g. in the internal trust region calculation of some RL algorithms. This penalty intuitively penalizes the new model when its update does *not* apply probability to a high-likelihood region in $Q$ or $\pi_\text{Ref.}$. This is closer to an objective used for distillation or behavioral cloning.

### Reference Model to Generations

KL penalties are most commonly implemented by comparing the distance between the generated tokens during training to a static reference model.
The intuition is that the model you're training from has a style that you would like to stay close to.
This reference model is most often the instruction tuned model, but can also be a previous RL checkpoint.
With simple substitution, the model we are sampling from becomes $\pi^{\text{RL}}(x)$ and $\pi^{\text{Ref.}}(x)$, shown above in @eq:kl_standard (often $P$, and $Q$, in standard definitions, when applied for RL KL penalties).
Such a KL divergence penalty was first applied to dialogue agents well before the popularity of large language models [@jaques2017sequence], yet KL control was quickly established as a core technique for fine-tuning pretrained models [@jaques2020human].

### Implementation Example

In practice, the implementation of KL divergence is often approximated [@schulman2016klapprox], making the implementation far simpler.
With the above definition, the summation of KL can be converted to an expectation when sampling directly from the distribution $P(x)$.
In this case, the distribution $P(x)$ is the generative distribution of the model currently being trained (i.e. not the reference model).
Then, the computation for KL divergence changes to the following:

$$
D_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

This mode is far simpler to implement, particularly when dealing directly with log probabilities used frequently in language model training.

```python
# Step 1: sample (or otherwise generate) a sequence from your policy
generated_tokens = model.generate(inputs)

# Step 2: score that generated sequence under both models
#    for autoregressive LMs, you usually do:
#      inputs_for_scoring = generated_tokens[:, :-1]
#      labels           = generated_tokens[:, 1:]
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# convert to log-probs, then align labels to index into the logits
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# gather the log-probs of the actual next tokens
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# now you can sum (or average) those to get the sequence log-prob,
# and compute KL:
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

Some example implementations include [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) and [Hamish Ivison's Jax Code](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278).

## Pretraining Gradients

Another way of viewing regularization is that you may have a *dataset* that you want the model to remain close to, as done in InstructGPT [@ouyang2022training] "in order to fix the performance regressions on public NLP datasets".
To implement this, they modify the training objective for RLHF.
Taking @eq:rl_start, we can transform this into an objective function to optimize by sampling from the RL policy model, completions $y$ from prompts $x$ in the RL dataset used for RLHF, which yields:
$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi^{\text{RL}}_{\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

Then, we can add an additional reward for higher probabilities on the standard autoregressive next-token prediction loss used at pretraining, over a set of documents sampled from the pretraining corpus (or another dataset) to maintain textual coherence:

$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi^{\text{RL}}_{\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi^{\text{RL}}_{\theta}(x)) \right]
$$ {#eq:objective_pretraining}

Recent work proposed using a negative log-likelihood term to balance the optimization of Direct Preference Optimization (DPO) [@pang2024iterative].
Given the pairwise nature of the DPO loss, the same loss modification can be made to reward model training, constraining the model to predict accurate text (rumors from laboratories that did not publish the work).

The optimization follows as a modification to DPO.
$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{P_\theta(c_i^w, y_i^w \mid x_i)}{P_{\text{ref.}}(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{P_\theta(c_i^l, y_i^l \mid x_i)}{P_{\text{ref.}}(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log P_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|},
$$ {#eq:dpo_nll_expanded}

where $P_{\theta}$ is the trainable policy model, $P_{\text{ref.}}$ is a fixed reference model (often the SFT checkpoint), and $(c_i^w, y_i^w)$ and $(c_i^l, y_i^l)$ denote the winning and losing completions for prompt $x_i$.
The first term is the standard DPO logistic loss: it increases the margin between the win and loss using the difference of log-likelihood ratios, $\log \tfrac{P_{\theta}}{P_{\text{ref.}}}$, and $\beta$ controls how strongly this preference signal pulls away from the reference.
The second term is a length-normalized negative log-likelihood penalty on the winning completion, weighted by $\alpha$, which helps keep the preferred text high-likelihood in an absolute language modeling sense rather than only relatively better than the rejected sample.

## Other Regularization

Controlling the optimization is less well defined in other parts of the RLHF stack.
Most reward models have no regularization beyond the standard contrastive loss function.
Direct Alignment Algorithms handle regularization to KL divergences differently, through the $\beta$ parameter (see the chapter on Direct Alignment).

Llama 2 proposed a margin loss for reward model training [@touvron2023llama]:

$$
\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)
$$ {#eq:margin_loss}

where $m(y_c, y_r)$ is the margin between two datapoints $y_c$ and $y_r$ representing numerical difference in delta between the ratings of two annotators.
This is either achieved by having annotators rate the outputs on a numerical scale or by using a quantified ranking method, such as [Likert scales](https://en.wikipedia.org/wiki/Likert_scale).

Reward margins have been used heavily in the direct alignment literature, such as Reward weighted DPO, ''Reward-aware Preference Optimization'' (RPO), which integrates reward model scores into the update rule following a DPO loss [@adler2024nemotron], or REBEL [@gao2024rebel] that has a reward delta weighting in a regression-loss formulation.
