---
prev-chapter: "Policy Gradients"
prev-url: "11-policy-gradients.html"
page-title: Direct Alignment Algorithms
next-chapter: "Constitutional AI"
next-url: "13-cai.html"
---

# [Incomplete] Direct Alignment Algorithms

Direct Alignment Algorithms (DAAs) allow one to update models to solve the  same RLHF objective without ever training an intermediate reward model or using reinforcement learning optimizers.
The most prominent DAA and one that catalyzed an entire academic movement of aligning language models is Direct Preference Optimization (DPO) [@rafailov2024direct].
At its core, DPO is using gradient ascent to solve the same constrained RLHF objective.
Since its release in May of 2023, after a brief delay where the community figured out the right data and hyperparameters to use DPO with (specifically, surprisingly low learning rates), many popular models have used DPO or its variants, from Zephyr-$\beta$ kickstarting it in October of 2024 [@tunstall2023zephyr], Llama 3 Instruct [@dubey2024llama], Tülu 2 [@ivison2023camels] and 3 [@lambert2024t], Nemotron 4 340B [@adler2024nemotron], and others.
Technically, Sequence Likelihood Calibration (SLiC-HF) was released first [@zhao2023slic], but it did not catch on due to a combination of luck and effectiveness.

The most impactful part of DPO and DAAs is lowering the barrier of entry to experimenting with language model post-training.

## Direct Preference Optimization (DPO)

Here we will re-derive DPO completely and explain intuitions for how it works. 

### DPO Derivation

#### Deriving Optimal RLHF Solution

To start, we should consider the RLHF optimization objective once again, here indicating we wish to maximize this quantity:

$$ \max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t)\right] - \beta  \mathcal{D}_{KL}(\pi^{\text{RL}}(\cdot|s_t) \| \pi^{\text{ref}}(\cdot|s_t)).$$ {#eq:rlhf_opt_eq_repeat}

First, let us expand the definition of KL-divergence,

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

Next, pull the negative sign out of the difference in brackets. To do this, split it into two terms:

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

Then, remove the factor of $-1$ and $\beta$,

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

Divide by $\beta$ and recombine:

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}


Next, we must introduce a partition function, $Z(x)$:

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

The partition function acts as a normalization factor over the reference policy, summing over all possible responses $y$ to a prompt $x$.
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

Next, we expand $\frac{1}{\beta}r(x,y)$ to $\log \exp \frac{1}{\beta}r(x,y)$ and do the same operation to get @eq:dpo_deriv_5.
With this optimization form, we need to actually solve for the optimal policy $\pi^*$.
To do so, let us consider the above optimization as a KL distance:

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathbb{D}_\text{KL} \left(\pi(y|x)||\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

Since the partition function $Z(x)$ does not depend on the final answer, we can ignore it. This leaves us with just the KL distance between our policy we are learning and a form relating the partition, $\beta$, reward, and reference policy.
The Gibb's inequality tells this is minimized at a distance of 0, only when the two quantities are equal!
Hence, we get an optimal policy:

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$


#### Deriving DPO Objective for Bradley Terry Models

TODO

The DPO has an additional derivation for the objective under a Plackett-Luce Model.

### How DPO Works

## Numerical Concerns, Weaknesses, and Alternatives

Many variants of the DPO algorithm have been proposed to address weaknesses of DPO.
For example, without rollouts where a reward model can rate generations, DPO treats every pair of preference data with equal weight. 
In reality, as seen in Chapter 6 on Preference Data, there are many ways of capturing preference data with a richer label than binary.
Multiple algorithms have been proposed to re-balance the optimization away from treating each pair equally.

- **REgression to RElative REward Based RL (REBEL)** adds signal from a reward model, as a margin between chosen and rejected responses, rather than solely the pairwise preference data to more accurately solve the RLHF problem [@gao2024rebel].
- **Conservative DPO (cDPO) and Identity Preference Optimization (IPO)** address the overfitting by assuming noise in the preference data. cDPO assumes N percent of the data is incorrectly labelled [@rafailov2024direct] and IPO changes the optimization to soften probability of preference rather than optimize directly from a label [@azar2024general]. Practically, IPO changes the preference probability to a nonlinear function, moving away from the Bradley-Terry assumption, with $\Psi(q) = \log\left(\frac{q}{1-q}\right)$.
- **DPO with an offset (ODPO)** "requires the difference between the likelihood of the preferred and dispreferred response to be greater than an offset value" [@amini2024direct] -- do not treat every data pair equally, but this can come at the cost of a more difficult labeling environment.

Some variants to DPO attempt to either improve the learning signal by making small changes to the loss or make the application more efficient by reducing memory usage.

- **Odds Ratio Policy Optimization (ORPO)** directly updates the policy model with a pull towards the chosen response, similar to the instruction finetuning loss, with a small penalty on the chosen response [@hong2024reference]. This change of loss function removes the need for a reference model, simplifying the setup. The best way to view ORPO is DPO inspired, rather than a DPO derivative.
-- **Simple Preference Optimization SimPO** makes a minor change to the DPO optimization, by averaging the log-probabilities rather than summing them (SimPO) or adding length normalization, to improve performance [@meng2025simpo].


TODO - figure on preference displacement

![Sketch of preference displacement in DPO.](images/DPO-displacement.png){#fig:dpo_issue .center}

One of the core issues *apparent* in DPO is that the optimization drives only to increase the margin between the probability of the chosen and rejected responses.
Numerically, the model reduces the probabiltiy of both the chosen and rejected responses, but the *rejected response is reduced by a greater extent* as shown in @fig:DPO-displacement.
Intuitively, it is not clear how this generalizes, but work has posited that it increases the probability of unaddressed for behaviors [@razin2024unintentional]. 
Simple methods, such as Cal-DPO [@xiao2024cal], adjust the optimization so that this **preference displacement** does not occur.
In practice, the exact impact of this is not well known, but points are a potential reason why online methods can outperform vanilla DPO.

The largest other reason that is posited for DPO-like methods to have a lower ceiling on performance than online (RL based) RLHF methods is that the training signal comes from completions from previous or other models.
Online variants that sample generations from the model, e.g. **Online DPO** [@guo2024direct], even with regular reward model relabelling of newly created creations **Discriminator-Guided DPO** (D2PO) [@singhal2024d2po], alleviate these by generating new completions for the prompt and incorporating a preference signal at training time.

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

In most ways, this is simpler and an quality of life improvement, but also they offer a different set of considerations.

1. **KL distance is static**: In DPO and other algorithms, the KL distance is set explicitly by the $\beta$ parameter that balances the distance penalty to the optimization. This is due to the fact that DPO takes gradient steps towards the *optimal* solution to the RLHF objective given the data -- it steps exactly to the solution set by the $\beta$ term. On the other hand, RL based optimizers take steps based on the batch and recent data.
2. **Caching log-probabilities**: Simple implementations of DPO do the forward passes for the policy model and reference models at the same time for conveniences with respect to the loss function. Though, this doubles the memory used and results in increased GPU usage. To avoid this, one can compute the log-probabilities of the reference model over the training dataset first, then reference it when computing the loss and updating the parameters per batch, reducing the peak memory usage by 50%.


## DAAs vs. RL: Online vs. Offline Data

Broadly, the argument boils down to one question: Do we need the inner workings of reinforcement learning, with value functions, policy gradients, and all, to align language models with RLHF? 
This, like most questions phrased this way, is overly simplistic. 
Of course, both methods are well-established, but it is important to illustrate where the fundamental differences and performance manifolds lie.

Multiple reports have concluded that policy-gradient based and RL methods outperform DPO and its variants.
The arguments take different forms, from training models with different algorithms but controlled data[@ivison2024unpacking] [@xu2024dpo] or studying the role of on-policy data within the RL optimization loop [@tajwar2024preference].
In all of these cases, DPO algorithms are a hair behind.

Even with this performance delta, DAA are still used extensively in leading models due to its simplicity.
DAAs provide a controlled environment where iterations on training data and other configurations can be made rapidly, and given that data is often far more important than algorithms, using DPO can be fine.

With the emergence of reasoning models that are primarily trained with RL, further investment will return to using RL for preference-tuning, which in the long-term will improve the robustness of RL infrastructure and cement this margin between DAAs and RL for optimizing from human feedback.