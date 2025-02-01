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

Note, when training reward models, the most common practice is to train for only 1 epoch to avoid overfitting.

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

There are many other formulations that can create suitable models of human preferences for RLHF.
One such example, used in the popular, early RLHF'd models Starling 7B and 34B [@zhu2024starling], is a K-wise loss function based on the Plackett-Luce model [@liu2019learning].

Following Zhu et al. 2023 formalizes the setup [@zhu2023principled], following as follows.
With a prompt, or state, $s^i$, $K$ actions $(a_0^i, a_1^i, \cdots, a_{K-1}^i)$ are sampled from $P(a_0,\cdots,a_{K-1}|s^i)$.
Then, labelers are used to rank preferences with $\sigma^i: [K] \mapsto [K]$ is a function representing action rankings, where $\sigma^i(0)$ is the most preferred action. This yields a preference model capturing the following:

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$

When $K = 2$, this reduces to the Bradley-Terry (BT) model for pairwise comparisons.
Regardless, once trained, these models are used similarly to other reward models during RLHF training.

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

To translate, this is implemented as as a language modeling head that can predict two classes per token (1 for correct, 0 for incorrect), rather than a classification head of a traditional RM that outputs one token for the entire sequence.

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

TODO comment on how they are often trained with LM heads and have 3 classes, +, neutral, -

## Reward Models vs. Outcome RMs vs. Process RMs vs. Value Functions

## Generative Reward Modeling

With the cost of preference data, a large research area emerged to use existing language models as a judge of human preferences or in other evaluation settings [@zheng2023judging].
The core idea is to prompt a language model with instructions on how to judge, a prompt, and two completions (much as would be done with human labelers). 
An example prompt, from one of the seminal works here for the chat evaluation MT-Bench [@zheng2023judging], follows:

```
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
```

Given the efficacy of LLM-as-a-judge for evaluation, spawning many other evaluations such as AlpacaEval [@dubois2024length], Arena-Hard [@li2024crowdsourced], and WildBench [@lin2024wildbench], many began using LLM-as-a-judge instead of reward models to create and use preference data.

An entire field of study has emerged to study how to use so called "Generative Reward Models" [@mahan2024generative]
[@zhang2024generative] [@ankner2024critique] (including models trained *specifically* to be effective judges [@kim2023prometheus]), but on RM evaluations they tend to be behind existing reward models, showing that reward modeling is an important technique for current RLHF.

## Further Reading

The academic literature for reward modeling established itself in 2024. 
The bulk of progress in reward modeling early on has been in establishing benchmarks and identifying behavior modes.
The first RM benchmark, RewardBench, provided common infrastructure for testing reward models [@lambert2024rewardbench].
Since then, RM evaluation has expanded to be similar to the types of evaluations available to general post-trained models, where some evaluations test the accuracy of prediction on domains with known true answers [@lambert2024rewardbench] or those more similar to "vibes" performed with LLM-as-a-judge or correlations to other benchmarks [@wen2024rethinking].

Examples of new benchmarks include multilingual reward bench (M-RewardBench) [@gureja2024m], RAG-RewardBench [@jin2024rag], RM-Bench [@zhou2024rmb], Preference Proxy Evaluations [@frick2024evaluate], and RewardMATH [@kim2024evaluating].

To understand progress on *training* reward models, one can reference new reward model training methods, with aspect-conditioned models [@wang2024interpretable], high quality human datasets [@wang2024helpsteer2] [@wang2024helpsteer2p], scaling [@adler2024nemotron], extensive experimentation [@touvron2023llama], or debiasing data [@park2024offsetbias].
