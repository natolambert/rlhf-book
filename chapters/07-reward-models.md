---
prev-chapter: "Preference Data"
prev-url: "06-preference-data"
page-title: Reward Modeling
next-chapter: "Regularization"
next-url: "08-regularization"
---

# Reward Modeling

Reward models are core to the modern approach to RLHF.
Reward models broadly have been used extensively in reinforcement learning research as a proxy for environment rewards [@sutton2018reinforcement].
The practice is closely related to inverse reinforcement learning, where the problem is to approximate an agent's reward function given trajectories of behavior [@ng2000algorithms], and other areas of deep reinforcement learning.
Reward models were proposed, in their modern form, as a tool for studying the value alignment problem [@leike2018scalable].

The most common reward model predicts the probability that a piece of text was close to a "preferred" piece of text from the training comparisons.
Later in this section we also compare these to Outcome Reward Models (ORMs) that predict the probability that a completion results in a correct answer or a Process Reward Model (PRM) that assigns a score to each step in reasoning.
When not indicated, the reward models mentioned are those predicting preference between text.

## Training Reward Models

There are two popular expressions for how to train a standard reward model for RLHF -- they are numerically equivalent. 
The canonical implementation is derived from the Bradley-Terry model of preference [@BradleyTerry].
A Bradley-Terry model of preferences measures the probability that the pairwise comparison for two events drawn from the same distribution, say $i$ and $j$, satisfy the following relation, $i > j$:

$$P(i > j) = \frac{p_i}{p_i + p_j}$$ {#eq:bradterry}

To train a reward model, we must formulate a loss function that satisfies the above relation.
The first structure applied is to convert a language model into a model that outputs a scalar value, often in the form of a single classification probability logit.
Thus, we can take the score of this model with two samples, the $i$ and $j$ above are now completions, $y_1$ and $y_2$, to one prompt, $x$ and score both of them with respect to the above model, $r_\theta$. We denote the conditional scores as $r_\theta(y_i \mid x)$.

The probability of success for a given reward model in a pairwise comparison becomes:

$$P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}$$ {#eq:bradterryrm}

We denote the preferred completion as $y_c$ (chosen) and the rejected completion as $y_r$.

Then, by maximizing the log-likelihood of the above function (or alternatively minimizing the negative log-likelihood), we can arrive at the loss function to train a reward model:

$$
\begin{aligned}
\theta^* = \arg\max_\theta P(y_c > y_r \mid x) &= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right) + \exp\left(r_\theta(y_r \mid x)\right)} \\
&= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)\left(1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}\right)} \\
&= \arg\max_\theta \frac{1}{1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}} \\ 
&= \arg\max_\theta \frac{1}{1 + \exp\left(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x))\right)} \\
&= \arg\max_\theta \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right) \\
&= \arg\min_\theta - \log \left( \sigma \left(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)\right) \right)
\end{aligned}
$$ {#eq:bradterryrm_deriv}

The first form, as in [@ouyang2022training] and other works:
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodeling1}

Second, as in [@askell2021general] and other works:
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$ {#eq:rewardmodeling2}

## Architecture

The most common way reward models are implemented is through an abstraction similar to Transformer's `AutoModelForSequenceClassification`, which appends a small linear head to the language model that performs classification between two outcomes -- chosen and rejected.
At inference time, the model outputs the *probability that the piece of text is chosen* as a single logit from the model.

Other implementation options exist, such as just taking a linear layer directly from the final embeddings, but they are less common in open tooling.

## Implementation Example

Implementing the reward modeling loss is quite simple.
More of the implementation challenge is on setting up a separate data loader and inference pipeline.
Given the correct dataloader with tokenized, chosen and rejected prompts with completions, the loss is implemented as:
```python
import torch.nn as nn
# inputs_chosen / inputs_rejected include the prompt tokens x and the respective
# completion tokens (y_c or y_r) that the reward model scores jointly.
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

As for the bigger picture, this is often within a causal language model that has an additional head added (and learned with the above loss) that transitions from the final hidden state to the score of the inputs.
This model will have a structure as follows:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryRewardModel(nn.Module):
    """
    Standard scalar reward model for Bradley-Terry preference learning.

    Usage (pairwise BT loss):
        rewards_chosen = model(**inputs_chosen)    # (batch,)
        rewards_rejected = model(**inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def _sequence_rep(self, hidden, attention_mask):
        """
        Get a single vector per sequence to score.
        Default: last non-padding token (EOS token); if no mask, last token.
        hidden: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """

        # Index of last non-pad token in each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        A forward pass designed to show inference structure of a standard reward model.
        To train one, this function will need to be modified to compute rewards from both
         chosen and rejected inputs, applying the loss above.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]

        # One scalar reward per sequence: (batch,)
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.head(seq_repr).squeeze(-1)

        return rewards
```

In this section and what follows, most of the implementation complexity for reward models (and much of post-training) is around constructing the data-loaders correctly and distributed learning systems.
Note, when training reward models, the most common practice is to train for only 1 epoch to avoid overfitting.

## Variants

Reward modeling is a relatively under-explored area of RLHF.
The traditional reward modeling loss has been modified in many popular works, but the modifications have not solidified into a single best practice.

### Preference Margin Loss

In the case where annotators are providing either scores or rankings on a Likert Scale, the magnitude of the relational quantities can be used in training.
The most common practice is to binarize the data direction, implicitly scores of 1 and 0, but the additional information has been used to improve model training.
Llama 2 proposes using the margin between two datapoints, $m(y_c, y_r)$, to distinguish the magnitude of preference:

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)$$ {#eq:rewardmodelingmargin}

For example, each completion is often given a ranking from 1 to 5 in terms of quality.
In the case where the chosen sample was assigned a score of 5 and rejected a score of 2, the margine $m(y_c, y_r)= 5 - 2 = 3$. 
Other functions for computing margins can be explored.

Note that in Llama 3 the margin term was removed as the team observed diminishing improvements after scaling.

### Balancing Multiple Comparisons Per Prompt

InstructGPT studies the impact of using a variable number of completions per prompt, yet balancing them in the reward model training [@ouyang2022training].
To do this, they weight the loss updates per comparison per prompt.
At an implementation level, this can be done automatically by including all examples with the same prompt in the same training batch, naturally weighing the different pairs -- not doing this caused overfitting to the prompts.
The loss function becomes:

$$\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodelinginstructgpt}


### K-wise Loss Function

There are many other formulations that can create suitable models of human preferences for RLHF.
One such example, used in the popular, early RLHF'd models Starling 7B and 34B [@zhu2024starling], is a K-wise loss function based on the Plackett-Luce model [@liu2019learning].

Zhu et al. 2023 [@zhu2023principled] formalizes the setup as follows.
With a prompt, or state, $s^i$, $K$ actions $(a_0^i, a_1^i, \cdots, a_{K-1}^i)$ are sampled from $P(a_0,\cdots,a_{K-1}|s^i)$.
Then, labelers are used to rank preferences with $\sigma^i: [K] \mapsto [K]$ is a function representing action rankings, where $\sigma^i(0)$ is the most preferred action. This yields a preference model capturing the following:

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$ {#eq:kwise_rm}

When $K = 2$, this reduces to the Bradley-Terry (BT) model for pairwise comparisons.
Regardless, once trained, these models are used similarly to other reward models during RLHF training.


## Outcome Reward Models

<!-- Huge thanks to Hangliang Ren, graduate student at Northeastern University for helping with this section (and PRMs), see https://github.com/myhott163com/RLHF_ORM_PRM -->

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
> We implement this scalar head as a single bias parameter and single gain parameter that operate on the logits outputted by the language model's final unembedding layer.

To translate, this is implemented as a language modeling head that can predict two classes per token (1 for correct, 0 for incorrect), rather than a classification head of a traditional RM that outputs one logit for the entire sequence.
Formally, following [@lyu2025exploring] this can be shown as:

$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))]$$ {#eq:orm_loss}

where $r \in {0,1}$ is a binary label where 1 applies to a correct answer to a given prompt and 0 applies to an incorrect, and $p_\theta(s)$ is the scalar proportional to predicted probability of correctness from the model being trained.

Implementing an outcome reward model (and other types, as we'll see with the Process Reward Model) involves applying the cross-entropy loss per-token based on if the completion is a correct sample. 
This is far closer to the language modeling loss, where it does not need the structured chosen-rejected nature of standard Bradley-Terry reward models.

The model structute could follw as:

```python
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        The input data here will be tokenized prompts and completions along with labels
         per prompt for correctness.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One scalar logit per token: (batch, seq_len)
        logits = self.head(hidden).squeeze(-1)

        # Only compute loss on completion tokens (labels 0 or 1)
        # Prompt tokens have labels = -100
        mask = labels != -100
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float()
            )
        return loss, logits
```

A simplified version of the loss follows:

```python
# Assume model already has: model.lm (backbone) + model.head
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)
# This will sometimes be compressed as model.forward() in other implementations

# Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits_per_token[mask], labels[mask].float()
)
```

The important intuition here is that an ORM will output a probability of correctness at every token in the sequence.
This can be a noisy process, as the updates and loss propagates per token depending on outcomes and attention mappings.
<!-- On the other hand, this process is more computationally intensive. [@cobbe2021training] posits a few potential benefits to these models, such as (1) implementation of ORMs often being done with both the standard next-token language modelling loss and the reward modelling loss above in @eq:orm_loss and (2) the ORM design as a token-level loss outperforms completion-level loss calculation used in standard RMs. -->

These models have continued in use, but are less supported in open-source RLHF tools. 
For example, the same type of ORM was used in the seminal work *Let's Verify Step by Step* [@lightman2023let], but without the language modeling prediction piece of the loss.
Then, the final loss is a cross entropy loss on every token predicting if the final answer is correct.

Given the lack of support, the term outcome reward model (ORM) has been used in multiple ways. 
Some literature, e.g. [@lyu2025exploring], continues to use the original definition from Cobbe et al. 2021. 
Others do not.


## Process Reward Models

Process Reward Models (PRMs), originally called Process-supervised Reward Models, are reward models trained to output scores at every *step* in a chain of thought reasoning process. 
These differ from a standard RM that outputs a score only at an EOS token or a ORM that outputs a score at every token.
Process Reward Models require supervision at the end of each reasoning step, and then are trained similarly where the tokens in the step are trained to their relevant target -- the target is the step in PRMs and the entire response for ORMs.

Following [@lightman2023let], a binary-labeled PRM is commonly optimized with a per-step cross-entropy loss:

$$\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_{s_i} \log r_\theta(s_i \mid x) + (1 - y_{s_i}) \log \left(1 - r_\theta(s_i \mid x)\right) \right] $$ {#eq:prm_loss}

where $s$ is a sampled chain-of-thought with $K$ annotated steps, $y_{s_i} \in \{0,1\}$ denotes whether the $i$-th step is correct, and $r_\theta(s_i \mid x)$ is the PRM's predicted probability that step $s_i$ is valid conditioned on the original prompt $x$.

Here's an example of how this per-step label can be packaged in a trainer, from HuggingFace's TRL [@vonwerra2022trl]:

```
# Get the ID of the separator token and add it to the completions
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# Create the label 
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

Traditionally PRMs are trained with a language modeling head that outputs a token only at the end of a reasoning step, e.g. at the token corresponding to a double new line or other special token.
These predictions tend to be -1 for incorrect, 0 for neutral, and 1 for correct.
These labels do not necessarily tie with whether or not the model is on the right path, but if the step is correct.

An example construction of a PRM is shown below.

```python
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        The inputs are tokenizer prompts and completions, where the the end of a 
         "reasoning step" is denoted by another non-padding token. 
        labels will be a list of labels, True, False, and Neutral (3 labels) which
         will be predicted by the model.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One logit vector per token: (batch, seq_len, num_classes)
        logits = self.head(hidden)

        # Only compute loss at step boundaries (where labels != -100)
        # Labels map: -1 -> 0, 0 -> 1, 1 -> 2 (class indices)
        mask = labels != -100
        if mask.any():
            loss = F.cross_entropy(
                logits[mask], labels[mask]
            )
        return loss, logits
```

The core loss function looks very similar to outcome reward models, with the labels being applied at different intervals.
```python
# Assume model outputs 3-class logits per token
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = model.head(hidden)  # (batch, seq_len, 3)

# 3-class labels at step boundaries only: 0=-1, 1=0, 2=1 (others masked as -100)
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

## Reward Models vs. Outcome RMs vs. Process RMs vs. Value Functions

The various types of reward models covered indicate the spectrum of ways that "quality" can be measured in RLHF and other post-training methods.
Below, a summary of what the models predict and how they are trained.

::: {.table-wrap}
| Model Class | What They Predict | How They Are Trained | LM structure |
|------------|------------------|---------------------|--------------|
| **Reward Models** | Quality of text via probability of chosen response at EOS token | Contrastive loss between pairwise (or N-wise) comparisons between completions | Regression or classification head on top of LM features |
| **Outcome Reward Models** | Probability that an answer is correct per-token | Labeled outcome pairs (e.g., success/failure on verifiable domains) | Language modeling head per-token cross-entropy, where every label is the outcome level label |
| **Process Reward Models** | A reward or score for intermediate steps at end of reasoning steps | Trained using intermediate feedback or stepwise annotations (trained per token in reasoning step) | Language modeling head only running inference per reasoning step, predicts three classes -1, 0, 1 |
| **Value Functions** | The expected return given the current state | Trained via regression to each point in sequence | A classification with output per-token |
Table: Comparing types of reward models. {#tbl:rm_compare}
:::

Some notes, given the above table has a lot of edge cases.

* Both in preference tuning and reasoning training, the value functions often have a discount factor of 1, which makes a value function even closer to an outcome reward model, but with a different training loss.
* A process reward model can be supervised by doing rollouts from an intermediate state and collecting outcome data. This blends multiple ideas, but if the *loss* is per reasoning step labels, it is best referred to as a PRM.

## Generative Reward Modeling

With the cost of preference data, a large research area emerged to use existing language models as a judge of human preferences or in other evaluation settings [@zheng2023judging].
The core idea is to prompt a language model with instructions on how to judge, a prompt, and two completions (much as would be done with human labelers). 
An example prompt, from one of the seminal works here for the chat evaluation MT-Bench [@zheng2023judging], follows:

```
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user's instructions and answers the user's question better. Your evaluation
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
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
```

Given the efficacy of LLM-as-a-judge for evaluation, spawning many other evaluations such as AlpacaEval [@dubois2024length], Arena-Hard [@li2024crowdsourced], and WildBench [@lin2024wildbench], many began using LLM-as-a-judge instead of reward models to create and use preference data.

An entire field of study has emerged to study how to use so called "Generative Reward Models" [@mahan2024generative]
[@zhang2024generative] [@ankner2024critique] (including models trained *specifically* to be effective judges [@kim2023prometheus]), but on RM evaluations they tend to be behind existing reward models, showing that reward modeling is an important technique for current RLHF.

A common trick to improve the robustness of LLM-as-a-judge workflows is to use a sampling temperature of 0 to reduce variance of ratings.

## Further Reading

The academic literature for reward modeling established itself in 2024. 
The bulk of progress in reward modeling early on has been in establishing benchmarks and identifying behavior modes.
The first RM benchmark, RewardBench, provided common infrastructure for testing reward models [@lambert2024rewardbench].
Since then, RM evaluation has expanded to be similar to the types of evaluations available to general post-trained models, where some evaluations test the accuracy of prediction on domains with known true answers [@lambert2024rewardbench] or those more similar to "vibes" performed with LLM-as-a-judge or correlations to other benchmarks [@wen2024rethinking].

Examples of new benchmarks include multilingual reward bench (M-RewardBench) [@gureja2024m], RAG-RewardBench [@jin2024rag], RMB [@zhou2024rmb] or RM-Bench [@liu2024rm] for general chat, ReWordBench for typos [@wu2025rewordbench], MJ-Bench [@chen2024mj], Multimodal RewardBench [@yasunaga2025multimodal], VL RewardBench [@li2024vlrewardbench], or VLRMBench [@ruan2025vlrmbench] for vision language models, Preference Proxy Evaluations [@frick2024evaluate], and RewardMATH [@kim2024evaluating].
Process reward models (PRMs) have their own emerging benchmarks, such as PRM Bench [@song2025prmbench] and visual benchmarks of VisualProcessBench [@wang2025visualprm] and ViLBench [@tu2025vilbench].

To understand progress on *training* reward models, one can reference new reward model training methods, with aspect-conditioned models [@wang2024interpretable], high quality human datasets [@wang2024helpsteer2] [@wang2024helpsteer2p], scaling [@adler2024nemotron], extensive experimentation [@touvron2023llama], or debiasing data [@park2024offsetbias].
