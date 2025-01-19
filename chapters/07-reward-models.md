---
prev-chapter: "Preference Data"
prev-url: "06-preference-data.html"
next-chapter: "Regularization"
next-url: "08-regularization.html"
---

# Reward Modeling

Reward models are core to the modern approach to RLHF.
Reward models broadly have been used extensively in reinforcement learning research as a proxy for environment rewards [@sutton2018reinforcement].
The practice is closely related to inverse reinforcement learning, where the problem is to approximate an agent's reward function given trajectories of behavior [@ng2000algorithms], and other areas of deep reinforcement learning.
Reward models were proposed, in their modern form, as a tool for studying the value alignment problem [@leike2018scalable].

## Training Reward Models

There are two popular expressions for how to train a reward model -- they are numerically equivalent. 
The canonical implementation is derived from the Bradley-Terry model of preference [@BradleyTerry].
A Bradley-Terry model of preferences measures the probability that the pairwise comparison for two events drawn from the same distribution, say $i$ and $j$, satisfy the following relation, $i > j$:

$$P(i > j) = \frac{p_i}{p_i + p_j}$$ {#eq:bradterry}

To train a reward model, we must formulate a loss function that satisfies the above relation.
The first structure applied is to convert a language model into a model that outputs a scalar value, often in the form of a single classification probability logit.
Thus, we can take the score of this model with two samples, the $i$ and $j$ above are now completions, $y_1$ and $y_2$, to one prompt, $x$ and score both of them with respect to the above model, $r_\theta$.

The probability of success for a given reward model in a pairwise comparison, becomes:

$$P(y_1 > y_2) = \frac{\exp(r(y_1))}{\exp(r(y_1)) + \exp(r(y_2))}$$ {#eq:bradterryrm}

Then, by taking the gradient with respect to the model parameters, we can arrive at the loss function to train a reward model.
The first form, as in [@ouyang2022training] and other works:
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) \right) \right)$$ {#eq:rewardmodeling1}

Second, as in [@askell2021general] and other works:
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(x, y_l)}  - e^{r_{\theta}(x, y_w)} \right)$$ {#eq:rewardmodeling2}

## Architecture

The most common way reward models are implemented is through an abstraction similar to Transformer's `AutoModelForSequenceClassification`, which appends a small linear head to the language model that performs classification between two outcomes -- chosen and rejected.
At inference time, the model outputs the *probability that the piece of text is chosen* as a single logit from the model.

Other implementation options exist, such as just taking a linear layer directly from the final embeddings, but they are less common in open tooling.

## Implementation Example

Implementing the reward modeling loss is quite simple.
More of the implementation challenge is on setting up a separate data loader and inference pipeline.
Given the correct dataloader, the loss is implemented as:
```python
import torch.nn as nn
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

## Variants

Reward modeling is a relatively under-explored area of RLHF.
The traditional reward modeling loss has been modified in many popular works, but the modifications have no solidified into a single best practice.

### Preference Margin Loss

In the case where annotators are providing either scores or rankings on a Likert Scale, the magnitude of the relational quantities can be used in training.
The most common practice is to binarize the data direction, implicitly scores of 1 and 0, but the additional information has been used to improve model training.
Llama 2 proposes using the margin between two datapoints, $m(r)$, to distinguish the magnitude of preference:

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) - m(r) \right) \right)$$ {#eq:rewardmodelingmargin}

### Balancing Multiple Comparisons Per Prompt

InstructGPT studies the impact of using a variable number of completions per prompt, yet balancing them in the reward model training [@ouyang2022training].
To do this, they weight the loss updates per comparison per prompt.
At an implementation level, this can be done automatically by including all examples with the same prompt in the same training batch, naturally weighing the different pairs -- not doing this caused overfitting to the prompts.
The loss function becomes:

$$\mathcal{L}(\theta) = - \frac{1}{(\frac{K}{2})} \mathbb{E}_{(x, y_w, y_l)\sim D} \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) \right) \right)$$ {#eq:rewardmodelinginstructgpt}


### K-wise Loss Function

Starling [@zhu2023principled] https://arxiv.org/abs/2301.11270

## Outcome Reward Models

The majority of *preference tuning* for language models and other AI systems is done with the Bradley Terry models discussed above.
For reasoning heavy tasks, one can use an Outcome Reward Model (ORM).
The training data for an ORM is constructed in a similar manner to standard preference tuning.
Here, we have a problem statement or prompt, $x$ and two completions $y_1$ and $y_2$. 
The inductive bias used here is that one completion should be a correct solution to the problem and one incorrect, resulting in $(y_c,y_{ic})$.

The shape of the models used is very similar to a standard reward model, with a linear layer appended to a model that can output a single logit (in the case of an RM) -- with an ORM, the training objective that follows is slightly different [@cobbe2021training]:

> [We] train verifiers with a joint objective where the model
learns to label a model completion as correct or incorrect, in addition to the original language modeling objective. 
> Architecturally, this means our verifiers
are language models, with a small scalar head that outputs predictions on a per-token basis. 
> We implement this scalar head as a single bias parameter and single gain parameter that operate on the logits outputted by the language model’s final unembedding layer.

These models have continued in use, but are less supported in open-source RLHF tools. 
For example, the same type of ORM was used in the seminal work *Let's Verify Step by Step* [@lightman2023let], but without the language modeling prediction piece of the loss.
Then, the final loss is a cross entropy loss on every token predicting if the final answer is correct.

## Process Reward Models

Process Reward Models (PRMs), originally called Process-supervised Reward Models, are reward models trained to output scores at every *step* in a chain of thought reasoning process. 
These differ from a standard RM that outputs a score only at an EOS token or a ORM that outputs a score at every token.
Process Reward Models require supervision at the end of each reasoning step, and then are trained similarly where the tokens in the step are trained to their relevant target -- the target is the step in PRMs and the entire response for ORMs.

Here's an example of how this per-step label can be packaged in a trainer, from HuggingFace's TRL [@vonwerra2022trl]:
```
# Get the ID of the separator token and add it to the completions
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# Create the label 
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

## Generative Reward Modeling

[@mahan2024generative], [@zhang2024generative], [@lambert2023entangled], generative and classifer [@ankner2024critique]

Related to LLM-as-a-judge and other evaluator models, which are very popular

## Further Reading

The academic literature for reward modeling established itself in 2024. 
The bulk of progress in reward modeling early on has been in establishing benchmarks and identifying behavior modes.
The first RM benchmark, RewardBench, provided common infrastructure for testing reward models
RewardBench (biased, but gives a good overview): [@lambert2023entangled] [@zhou2024rmb]

New reward model training methods, with aspect-conditioned models [@wang2024interpretable], high quality human datasets [@wang2024helpsteer2] [@wang2024helpsteer2p], scaling [@adler2024nemotron], extensive experimentation [@touvron2023llama], debiasing data [@park2024offsetbias],

Evaluations

## Recommendations

Strong tendency in the literature to train for only one epoch, otherwise it overfits