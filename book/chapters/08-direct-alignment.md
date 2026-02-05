---
prev-chapter: "Reasoning"
prev-url: "07-reasoning"
page-title: Direct Alignment
next-chapter: "Rejection Sampling"
next-url: "09-rejection-sampling"
---

# Direct Alignment Algorithms

Direct Alignment Algorithms (DAAs) allow one to update models to solve the same RLHF objective, shown again in @eq:review_rlhf, without ever training an intermediate reward model or using reinforcement learning optimizers. 
DAAs solve the same preference learning problem we've been studying (with literally the same data!), in order to make language models more aligned, smarter, and easier to use.
The lack of a reward model and online optimization makes DAAs far simpler to implement, reducing compute spent during training and making experimentation easier.
This chapter details the complex mathematics done to derive these algorithms, and then shows that the sometimes tedious derivations result in simple implementations.
 
The most prominent DAA and one that catalyzed an entire academic movement of aligning language models is Direct Preference Optimization (DPO) [@rafailov2024direct].
At its core, DPO is using gradient ascent to solve the same constrained RLHF objective (see Chapter 3):

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:review_rlhf}

Since its release in May of 2023, after a brief delay where the community figured out the right data and hyperparameters to use DPO with (specifically, surprisingly low learning rates), many popular models have used DPO or its variants, from Zephyr-$\beta$ kickstarting it in October of 2023 [@tunstall2023zephyr], Llama 3 Instruct [@dubey2024llama], Tülu 2 [@ivison2023camels] and 3 [@lambert2024t], Nemotron 4 340B [@adler2024nemotron], and others.
Technically, Sequence Likelihood Calibration (SLiC-HF) was the first, modern direct alignment algorithm released [@zhao2023slic], but it did not catch on due to a combination of factors (unwinding the adoption of research methods is always a tricky task).

The most impactful part of DPO and DAAs is lowering the barrier of entry to experimenting with language model post-training -- it uses less compute, is easier to implement from scratch, and is easier to get working on both toy and production examples.

*Throughout this chapter, we use $x$ to denote prompts and $y$ to denote completions. This notation is common in the language model literature, where methods operate on full prompt-completion pairs rather than individual tokens.*

## Direct Preference Optimization (DPO)

Here we explain intuitions for how DPO works and re-derive the core equations fully. 

### How DPO Works

DPO directly optimizes a policy to solve the RLHF objective. The loss function, which we derive below, centers around the relationship between a pair of log-probabilities with respect to a learned policy and a starting reference model. Derived from a Bradley-Terry preference model, the DPO loss is:

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right] $$ {#eq:dpo_core}

Throughout, $\beta$ is a hyperparameter balancing the reward optimization to the KL distance between the final model and the initial reference (i.e. balancing over-optimization, a crucial hyperparameter when using DPO correctly).
This relies on the implicit reward for DPO training that replaces using an external reward model, which is a log-ratio of probabilities:

$$r(x, y) = \beta  \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ {#eq:dpo_reward}

where $\pi_r(y \mid x)$ is the exact, optimal reward policy that we are solving for.
This comes from deriving the Bradley-Terry reward with respect to an optimal policy (shown in @eq:dpo_opt_policy), as shown in the Bradley-Terry model section of Chapter 5. 
Essentially, the implicit reward model, as stated in the DPO paper, gives us "the probability of human preference data in terms of the optimal policy rather than the reward model" -- meaning we can bypass learning an explicit reward model entirely.

Let us consider the loss shown in @eq:dpo_core that the optimizer must decrease. 
Here, the loss will be lower when the log-ratio of the chosen response is bigger than the log-ratio of the rejected response (normalized by the reference model).
In practice, this is a sum of log-probabilities of the model across the sequence of tokens in the data presented.
Hence, DPO is increasing the delta in probabilities between the chosen and rejected responses.

With the reward in @eq:dpo_reward, we can write the gradient of the loss to further interpret what is going on:

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[ \sigma\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right) \left(\nabla_{\theta}\log \pi(y_c \mid x) - \nabla_{\theta}\log \pi(y_r \mid x)\right) \right] $$ {#eq:dpo_gradient}

Here, the gradient solves the above objective by doing the following:

- The first term within the sigmoid function, $\sigma(\cdot)$, creates a weight of the parameter update from 0 to 1 that is higher when the reward estimate is incorrect. When the rejected sample is preferred over the chosen, the weight update should be larger!
- Second, the terms in the inner brackets $[\cdot]$ increase the likelihood of the chosen response $y_c$ and decrease the likelihood of the rejected $y_r$.
- These terms are weighted by $\beta$, which controls how the update balances ordering the completions correctly relative to the KL distance.


The core intuition is that DPO is fitting an implicit reward model whose corresponding optimal policy can be extracted in a closed form (thanks to gradient descent and our ML tools).
The closed form of the equation means that it is straightforward to implement the exact gradient, rather than needing to reach it by proxy of training a reward model and sampling completions to score.
What is often misunderstood is that DPO is learning a reward model at its core, hence the subtitle of the paper *Your Language Model is Secretly a Reward Model.* 
It is easy to confuse this with the DPO objective training a policy directly, hence studying the derivations below is good for a complete understanding.

With the implicit reward model learning, DPO is generating an optimal solution to the RLHF objective given the data in the dataset and the specific KL constraint in the objective $\beta$. 
Here, DPO solves for the exact policy given a specific KL distance because the generations are not online as in policy gradient algorithms -- a core difference from the RL methods for preference tuning.
In many ways, this makes the $\beta$ value easier to tune with DPO relative to online RL methods, but crucially and intuitively the optimal value depends on the model being trained and the data training it.

At each batch of preference data, composed of many pairs of completions $y_{chosen} \succ y_{rejected}$, DPO takes gradient steps directly towards the optimal solution.
It is far simpler than policy gradient methods.

![When DPO first released it sparked a fierce debate in the research community about how to best do RLHF and preference learning. This meme is a great job capturing the sentiment, where the debate often felt forced and over the top, but many people both getting started and in top labs were getting immense benefit out of DPO. DPO simplicity meme, credit Tom Goldstein.](images/dpo_meme.jpeg){#fig:dpo-meme}


### DPO Derivation

The DPO derivation takes two primary parts. 
First, the authors show the form of the policy that optimally solved the RLHF objective used throughout this book.
Next, they show how to arrive at that solution from pairwise preference data (i.e. a Bradley Terry model).

#### 1. Deriving the Optimal RLHF Solution

To start, we should consider the RLHF optimization objective once again, here indicating we wish to maximize this quantity:

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:rlhf_opt_eq_repeat}

Here, the dual expectation only applies to the sampling to compute the expected reward, as the KL term is still an analytical expression.
First, let us expand the definition of KL-divergence. Recall that $\mathcal{D}_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$, where the $\pi(y|x)$ weighting in the sum becomes the sampling distribution. 
Since both terms now share the same expectation over $y \sim \pi(y|x)$, we can combine them:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

Next, pull the negative sign out of the difference in brackets. To do this, split it into two terms:

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

Then, remove the factor of $-1$ and $\beta$,

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

Divide by $\beta$ and recombine:

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}


Next, we must introduce a partition function, $Z(x)$:

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

The partition function acts as a normalization factor for the unnormalized density $\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$, thereby making it a valid probability distribution over $y$ for each fixed $x$. The exact need for this will become clear shortly as we proceed with the derivation.

With this substituted in, we obtain our intermediate transformation:

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

To see how this is obtained, consider the internal part of the optimization in brackets of @eq:dpo_deriv_4:

$$ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_6}

Then, add $\log Z(x) - \log Z(x)$ to both sides:

$$ = \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) + \log Z(x) - \log Z(x) $$ {#eq:dpo_deriv_7}

Then, we group the terms:

$$ = \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x) \right) - \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_8}

With $\log(x) + \log(y) = \log(x\cdot y)$ (and moving $Z$ to the denominator), we get:

$$ = \log \frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)}- \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_9}

Next, we rewrite $\frac{1}{\beta}r(x,y)$ as $\log \exp\left(\frac{1}{\beta}r(x,y)\right)$. This lets us use the log subtraction rule again: $\log a - \log b = \log(a/b)$, folding the $\exp\left(\frac{1}{\beta}r(x,y)\right)$ term into the existing denominator to obtain @eq:dpo_deriv_5, which we slightly rewrite here:

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}} \left[ \mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} \right] - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

With this optimization form, we need to actually solve for the optimal policy $\pi^*$.
Since we introduced the partition function $Z(x)$, thereby making the term $\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ a valid probability distribution over $y$, we can recognize that the inner expectation is in fact a proper KL divergence:

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathcal{D}_{\text{KL}} \left(\pi(y|x) \middle\| \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_11}

Since the partition function $Z(x)$ does not depend on $\pi$ (the policy we are optimizing), we can ignore it for the minimization. This leaves us with just the KL distance
between the policy we are learning and a form relating the partition, $\beta$, reward, and reference policy.
Gibbs' inequality states that KL divergence is always non-negative and equals zero only when the two distributions are identical. Thus, the minimum is achieved when $\pi$ equals the target distribution exactly.
Hence (assuming the target distribution is fixed and lies in the feasible policy class), we get an optimal policy for the $\pi(y|x)$ we've been optimizing:

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}


#### 2. Deriving DPO Objective for Bradley Terry Models

To start, recall from Chapter 5 on Reward Modeling and Chapter 11 on Preference Data that a Bradley-Terry model of human preferences is formed as:

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

By manipulating @eq:dpo_opt_policy, we can solve for the optimal reward. First, take the logarithm of both sides:

$$\log \pi^*(y|x) = \log \left( \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^*(x,y)\right) \right)$$ {#eq:dpo_reward_deriv1}

Expanding the right-hand side using $\log(abc) = \log a + \log b + \log c$:

$$\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r^*(x,y)$$ {#eq:dpo_reward_deriv2}

Rearranging to solve for $r^*(x,y)$:

$$\frac{1}{\beta}r^*(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)$$ {#eq:dpo_reward_deriv3}

Multiplying both sides by $\beta$:

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

We then can substitute the reward into the Bradley-Terry equation shown in @eq:bradley_terry_dpo to obtain:

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} + \beta \log Z(x)\right)} $$ {#eq:dpo_loss_deriv0}

By decomposing the exponential expressions from $e^{a+b}$ to $e^a e^b$ and then cancelling out the terms $e^{\log(Z(x))}$, this simplifies to:

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right)} $$ {#eq:dpo_loss_deriv1}

Then, multiply the numerator and denominator by $\exp\left(-\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)$ to obtain:

$$p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} $$ {#eq:dpo_loss_deriv2}

Finally, with the definition of a sigmoid function as $\sigma(x) = \frac{1}{1+e^{-x}}$, we obtain:

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

This is the likelihood of preference data under the Bradley-Terry model, given the optimal policy $\pi^*$. Recall from Chapter 5 on Reward Modeling that we derived the Bradley-Terry objective as maximizing the likelihood, or equivalently minimizing the negative log-likelihood, which gives us the loss:
$$
\begin{aligned}
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) &= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log p(y_c \succ y_r \mid x)  \right] \\
&= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
\end{aligned}
$${#eq:dpo_loss_deriv4}

This is the loss function for DPO, in a form as shown in @eq:dpo_core. 
The DPO paper has an additional derivation for the objective under a Plackett-Luce Model, which is far less used in practice [@rafailov2024direct].

#### 3. Deriving the Bradley Terry DPO Gradient

We used the DPO gradient shown in @eq:dpo_gradient to explain intuitions for how the model learns.
To derive this, we must take the gradient of @eq:dpo_loss_deriv4 with respect to the model parameters.

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

To start, this can be rewritten.
We know that the derivative of a sigmoid function $\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))$, the derivative of logarithm $\frac{d}{dx} \log x = \frac{1}{x}$, and properties of sigmoid $\sigma(-x)=1-\sigma(x)$, so we can reformat the above equation. 

First, let $u=\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$ (the expression inside the sigmoid).
Then, we have

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[\frac{\sigma'(u)}{\sigma(u)}\nabla_{\theta}u\right] $$ {#eq:dpo_grad_2}

Expanding this and using the above expressions for sigmoid and logarithms results in the gradient introduced earlier:

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi(y_c|x)-\nabla_{\theta}\log\pi(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## Numerical Concerns, Weaknesses, and Alternatives

Many variants of the DPO algorithm have been proposed to address weaknesses of DPO.
For example, without rollouts where a reward model can rate generations, DPO treats every pair of preference data with equal weight. 
In reality, as seen in Chapter 11 on Preference Data, there are many ways of capturing preference data with a richer label than binary.
Multiple algorithms have been proposed to re-balance the optimization away from treating each pair equally.

- **REgression to RElative REward Based RL (REBEL)** adds signal from a reward model, as a margin between chosen and rejected responses, rather than solely the pairwise preference data to more accurately solve the RLHF problem [@gao2024rebel].
- **Conservative DPO (cDPO) and Identity Preference Optimization (IPO)** address overfitting by assuming noise in the preference data. cDPO assumes N percent of the data is incorrectly labeled [@rafailov2024direct] and IPO changes the optimization to soften the probability of preference rather than optimize directly from a label [@azar2024general]. Practically, IPO changes the preference probability to a nonlinear function, moving away from the Bradley-Terry assumption, with $\Psi(q) = \log\left(\frac{q}{1-q}\right)$.
- **DPO with an offset (ODPO)** "requires the difference between the likelihood of the preferred and dispreferred response to be greater than an offset value" [@amini2024direct] -- do not treat every data pair equally, but this can come at the cost of a more difficult labeling environment.

Some variants to DPO attempt to either improve the learning signal by making small changes to the loss or make the application more efficient by reducing memory usage.

- **Odds Ratio Policy Optimization (ORPO)** directly updates the policy model with a pull towards the chosen response, similar to the instruction fine-tuning loss, with a small penalty on the chosen response [@hong2024reference]. This change of loss function removes the need for a reference model, simplifying the setup. The best way to view ORPO is DPO inspired, rather than a DPO derivative.
- **Simple Preference Optimization (SimPO)** makes a minor change to the DPO optimization, by averaging the log-probabilities rather than summing them (SimPO) or adding length normalization, to improve performance [@meng2025simpo].

![Sketch of preference displacement in DPO.](images/dpo_displacement.png){#fig:dpo_issue .center}

One of the core issues *apparent* in DPO is that the optimization drives only to increase the margin between the probability of the chosen and rejected responses.
Numerically, the model reduces the probability of both the chosen and rejected responses, but the *rejected response is reduced by a greater extent* as shown in @fig:dpo_issue.
Intuitively, it is not clear how this generalizes, but work has posited that it increases the probability of unaddressed behaviors -- i.e. tokens that the language model could generate, but are not in the distribution of the post-training datasets [@razin2024unintentional] [@ren2024learning]. 
Simple methods---such as Cal-DPO [@xiao2024cal], which adjusts the optimization process, and AlphaPO [@gupta2025alphapo], which modifies the reward shape---mitigate this **preference displacement**.
In practice, the exact impact of this is not well known, but points to a potential reason why online methods can outperform vanilla DPO.

The largest other reason that is posited for DPO-like methods to have a lower ceiling on performance than online (RL based) RLHF methods is that the training signal comes from completions from previous or other models.
Online variants of DPO alleviate these limitations by generating new completions and incorporating a preference signal at training time. **Online DPO** [@guo2024direct] samples generations from the current model, while **Discriminator-Guided DPO** (D2PO) [@singhal2024d2po] uses reward model relabelling to create new preference data on the fly, and many more variants exist.

There is a long list of other DAA variants, such as Direct Nash Optimization (DNO) [@rosset2024direct] or Binary Classifier Optimization (BCO) [@jung2024binary], but the choice of algorithm is far less important than the initial model and the data used [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences].

## Implementation Considerations

DAAs such as DPO are implemented very differently than policy gradient optimizers.
The DPO loss, taken from the original implementation, largely can be summarized as follows [@rafailov2024direct]:

```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

losses = -F.logsigmoid(beta * logits)

chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

This can be used in standard language model training stacks as this information is already collated during the forward pass of a model (with the addition of a reference model).

In most ways, DAAs are simpler and a quality of life improvement, but they also offer a different set of considerations.

1. **KL distance is static**: In DPO and other algorithms, the KL distance is set explicitly by the $\beta$ parameter that balances the distance penalty to the optimization. This is due to the fact that DPO takes gradient steps towards the *optimal* solution to the RLHF objective given the data -- it steps exactly to the solution set by the $\beta$ term. On the other hand, RL based optimizers take steps based on the batch and recent data.
2. **Caching log-probabilities**: Simple implementations of DPO do the forward passes for the policy model and reference models at the same time for convenience with respect to the loss function. Though, this doubles the memory used and results in increased GPU usage. To avoid this, one can compute the log-probabilities of the reference model over the training dataset first, then reference it when computing the loss and updating the parameters per batch, reducing the peak memory usage by 50%.

## DAAs with Synthetic Preference Data

Most of the popular datasets for performing preference fine-tuning with DAAs these days are synthetic preferences where a frontier model rates outputs from other models as the winner or the loser. 
Prominent examples include UltraFeedback (the first of this category) [@cui2023ultrafeedback], Tülu 3 (built with an expanded UltraFeedback methodology) [@lambert2024t], SmolLM 3's data [@bakouch2025smollm3], or the Dolci Pref dataset released with Olmo 3 [@teamolmo2025olmo3].

The best-practices for constructing these datasets is still evolving.
Tülu 3 and datasets around its release in November of 2024 demonstrated that synthetic, pairwise preference data needs to be "on-policy" in a sense that some completions are generated from the model you're fine-tuning (while being mixed in a bigger model pool).
This on-policy nature of the data ensured that the DAA would optimize the correct token space within which the model generates -- as the loss functions are contrastive and less direct than instruction fine-tuning.
Later, with the release of Olmo 3 and SmolLM 3 in 2025, other works supported a different theory called Delta Learning, which argues that the difference between the chosen and rejected completions is more important to learning than exactly which models are used for the completions [@geng2025the].
For example, in both of these two referenced models, the chosen responses are from Qwen 3 32B and the rejected responses are from Qwen 3 0.6B -- both authors developed this pairing concurrently and independently.

Overall, training models on synthetic preference data with DAAs is the place most practitioners should start with given the simplicity of implementation and strong performance relative to preference fine-tuning with reinforcement learning based methods.
Other minor issues exist when using extensive, synthetic preference data, such as biases of the model judging between completions.
Given that frontier models such as GPT-4 are known to have length bias [@dubois2024length] and a preference for outputs that match themselves [@panickssery2024llm] (see Chapter 12 for more information), it is slightly more likely for a piece of text in the "chosen" section of the dataset to be either from an OpenAI model or another strong model that is stylistically similar to it. 

To conclude this section, we'll cover an intuition for how these methods change the generations of the model being trained.
At a high level, most DAAs optimize to increase the margin between the probability of "chosen" and "rejected" completions (some less popular algorithms are designed to slightly change these dynamics, but the core remains).
As discussed earlier in this chapter (see @fig:dpo_issue), this often means both probabilities decrease, but the rejected response decreases by a greater extent.
Each token in a sequence receives a different gradient (magnitude and direction) based on how much it contributed to the overall preference margin, allowing the optimizer to identify which tokens matter most to the outcome.

## DAAs vs. RL: Online vs. Offline Data

Broadly, the argument boils down to one question: Do we need the inner workings of reinforcement learning, with value functions, policy gradients, and all, to align language models with RLHF? 
This, like most questions phrased this way, is overly simple. 
Of course, both methods are well-established, but it is important to illustrate where the fundamental differences and performance manifolds lie.

Multiple reports have concluded that policy-gradient based and RL methods outperform DPO and its variants.
The arguments take different forms, from training models with different algorithms but controlled data [@ivison2024unpacking] [@xu2024dpo] or studying the role of on-policy data within the RL optimization loop [@tajwar2024preference].
In all of these cases, DPO algorithms are a hair behind.

Even with this performance delta, DAAs are still used extensively in leading models due to their simplicity.
DAAs provide a controlled environment where iterations on training data and other configurations can be made rapidly, and given that data is often far more important than algorithms, using DPO can be fine.

With the emergence of reasoning models that are primarily trained with RL, further investment will return to using RL for preference-tuning, which in the long-term will improve the robustness of RL infrastructure and cement this margin between DAAs and RL for optimizing from human feedback.
