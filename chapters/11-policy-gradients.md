---
prev-chapter: "Rejection Sampling"
prev-url: "10-rejection-sampling.html"
next-chapter: "Direct Alignment Algorithms"
next-url: "12-direct-alignment.html"
---

# Policy Gradient Algorithms


The algorithms that popularized RLHF for language models were policy-gradient reinforcement learning algorithms. 
These algorithms, such as PPO and Reinforce, use recently generated samples to update their model rather than storing scores in a replay buffer.
In this section we will cover the fundamentals of the policy gradient algorithms and how they are used in the modern RLHF framework.

For definitions of symbols, see the problem setup chapter.

## Policy Gradient Algorithms

Reinforcement learning algorithms are designed to maximize the future, discounted reward across a trajectory of states, $s \in \mathcal{S}$, and actions, $a \in \mathcal{A}$ (for more notation, see Chapter 3, Definitions).
The objective of the agent, often called the *return*, is the sum of discounted, future rewards (where $\gamma\in [0,1)$ is a factor that prioritizes near term rewards) at a given time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=o}^\infty \gamma^k R_{t+k+1}.$$

The return definition can also be estimated as:
$$G_{t} = \gamma{G_{t+1}} + R_{t}.$$

This return is the basis for learning a value function $V(s)$ that is the estimated future return given a current state:

$$V(s) = \mathbb{E}\big[G_t | S_t = s \big].$$

All policy gradient algorithms solve an objective for such a value function induced from a specific policy, $\pi(s|a)$. 

The optimization is defined as:
$$
J(\theta)
\;=\;
\sum_{s} d_\pi(s) V_\pi(s),
$$

The core of policy gradient algorithms is computing the gradient with respect to the finite time expected return over the current policy. 
With this expected return, $J$, the gradient can be computed as follows, where $\alpha$ is the learning rate: 

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

The core implementation detail is how to compute said gradient.
Schulman et al. 2015 provides an overview of the different ways that policy gradients can be computed [@schulman2015high].
The goal is to *estimate* the exact gradient $g := \nabla_\theta \mathbb{E}[\sum_{t=0}^\infty r_t]$, of which, there are many forms similar to:

$$ g = \mathbb{E}\Big[\sum_{t=0}^\infty \Psi_t \nabla_\theta \text{log} \pi_\theta(a_t|s_t) \Big], $$

Where $\Psi_t$ can be the following:

1. $\sum_{t=0}^{\infty} r_t$: total reward of the trajectory.
2. $\sum_{t'=t}^{\infty} r_{t'}$: reward following action $a_t$.
3. $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$: baselined version of previous formula.
4. $Q^{\pi}(s_t, a_t)$: state-action value function.
5. $A^{\pi}(s_t, a_t)$: advantage function.
6. $r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$: TD residual.

The *baseline* is a value used to reduce variance of policy updates (more on this below).

For language models, some of these concepts do not make as much sense.
For example, we know that for a deterministic policy the value function is defined as $V(s) = \max_a Q(s,a)$ or for a stochastic policy as $V(s) = \mathbb{E}_{a \sim \pi(a|s)}[Q(s,a)]$.
If we define $s+a$ as the continuation $a$ to the prompt $s$, then $Q(s, a) = V(s+a)$, which gives a different advantage trick:

$$A(s,a) = Q(s,a) - V(s) = V(s + a) - V(s) = r + \gamma V(s + a) - V(s)$$

Which is a combination of the reward, the value of the prompt, and the discounted value of the entire utterance.

### Vanilla Policy Gradient

The vanilla policy gradient implementation optimizes the above expression for $J(\theta)$ by differentiating with respect to the policy parameters.
A simple version, with respect to the overall return, is:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]$$

A common problem with vanilla policy gradient algorithms is the high variance in gradient updates, which can be mitigated in multiple ways.
In order to alleviate this,  various techniques are used to normalize the value estimation, called *baselines*. 
Baselines accomplish this in multiple ways, effectively normalizing by the value of the state relative to the downstream action (e.g. in the case of Advantage, which is the difference between the Q value and the value). 
The simplest baselines are averages over the batch of rewards or a moving average.
Even these baselines can de-bias the gradients so $\mathbb{E}_{a \sim \pi(a|s)}[\nabla_\theta \log \pi_\theta(a|s)] = 0$, improving the learning signal substantially.

Many of the policy gradient algorithms discussed in this chapter build on the advantage formulation of policy gradient:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]$$

A core piece of the policy gradient implementation involves taking the derivative of the probabilistic policies. 
This comes from:

$$\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

Which is derived from the chain rule:

$$\nabla_\theta \log x = \frac{1}{x} \nabla_\theta x$$

We will use this later on in the chapter.


### REINFORCE

The algorithm REINFORCE is likely a backronym, but the components of the algorithms it represents are quite relevant for modern reinforcement learning algorithms. 
Defined in the seminal paper *Simple statistical gradient-following algorithms for connectionist reinforcement learning* [@williams1992simple]:

> The name is an acronym for "REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility."

The three components of this are how to do the *reward increment*, a.k.a. the policy gradient step.
It has three pieces to the update rule:

1. Nonnegative factor: This is the learning rate (step size) that must be a positive number.
2. Offset Reinforcement: This is a baseline $b$ or other normalizing factor of the reward to improve stability.
3. Characteristic Eligibility: This is how the learning becomes attributed per token. It can be a general value, $e$ per parameter, but is often log probabilities of the policy in modern equations.

Thus, the form looks quite familiar:

$$ \Delta_\theta = \alpha(r - b)e $$ {#eq:REINFORCE_BASIC}

With more modern notation and the generalized return $G$, the REINFORCE operator appears as:

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\Big[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,G_t
\Big],
$$


REINFORCE is a specific implementation of vanilla policy gradient that uses a Monte Carlo estimator of the gradient.
[@ahmadian2024back]

REINFORCE can be run without value network -- the value network is for the baseline in the policy gradient. 
PPO on the other hand needs the value network to accurately compute the advantage function.

#### REINFORCE Leave One Out (RLOO)

[@huang2024putting], [@ahmadian2024back]

Note that for verifiable domains like reasoning, RLOO may not because it averages over outcomes to update parameters. 
This reduces credit assignment to the batch level and will make it harder for the model to attribute outcomes to specific behaviors within one sample.

Other implementations of REINFORCE algorithms have been designed for language models, such as ReMax [@li2023remax], which implements a baseline normalization designed specifically to accommodate the sources of uncertainty from reward model inference.

### Proximal Policy Optimization

*This section follows similar to [@achiam2018spinning].*

Proximal Policy Optimization (PPO) [@schulman2017proximal] is one of the foundational algorithms to Deep RL's successes (such as OpenAI's DOTA 5 [@berner2019dota] and large amounts of research).

For now, see: https://spinningup.openai.com/en/latest/algorithms/ppo.html


#### Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) is introduced in DeepSeekMath [@shao2024deepseekmath], and used in other DeepSeek works, e.g. DeepSeek-V3 [@liu2024deepseek] and DeepSeek-R1 [@guo2025deepseek].
GRPO can be viewed as PPO-inspired algorithm with a very similar surrogate loss, but it avoids learning a value function with another copy of the original policy language model (or another checkpoint for initialization). 
This brings two posited benefits:

1. Avoiding the challenge of learning a value function from a LM backbone, where research hasn't established best practices.
2. Saves memory by not needing to keep another set of model weights in memory.

GRPO does this by simplifying the value estimation and assigning the same value to every token in the episode (i.e. in the completion to a prompt, each token gets assigned the same value rather than discounted rewards in a standard value function) by estimating the advantage or baseline.
The estimate is done by collecting multiple completions ($a_i$) and rewards ($r_i$), i.e. a Monte Carlo estimate, from the same initial state / prompt ($s$).

To state this formally, the GRPO objective is very similar to the PPO objective above:

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{old}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{old}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta D_{KL}(\pi_\theta||\pi_{ref})\right).$$

With the advantage computation for the completion index $i$:

$$A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}. \quad (3)$$ {#eq:GRPO_ADV}

@eq:GRPO_ADV is the implementation of GRPO when working with outcome supervision (either a standard reward model or a single verifiable reward) and a different implementation is needed with process supervision.
In this case, GRPO computes the advantage as the sum of the normalized rewards for the following reasoning steps.
To do so, the rewards are accumulated with additional tracking of a reasoning index $j$, and then computed step wise as TODO, ref paper

Finally, GRPO's advantage estimation can also be applied without the PPO clipping to more vanilla versions of policy gradient (e.g. REINFORCE), but it is not the canonical form.

#### Generalized Advantage Estimation (GAE)

## Implementation

- Only score a response with a reward model with the `eos_token` is generated, otherwise the response is truncated.

TODO. Cite:
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#

https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

For more details on implementation details for RLHF, see [@huang2024n]. For further information on the algorithms, see [@weng2018PG].

### Policy Gradient

A simple implementation of policy gradient, using advantages to estimate the gradient to prepare for advanced algorithms such as PPO and GRPO follows:
```python
pg_loss = -advantages * ratio
```
Ratio here is the logratio of the new policy model probabilities relative to the reference model.

In order to understand this equation it is good to understand different cases that can fall within a batch of updates. 
Remember that we want the loss to *decrease* as the model gets better at the task.

Case 1: Positive advantage, so the action was better than the expected value of the state. We want to reinforce this. In this case, the model will make this more likely with the negative sign. To do so it'll increase the logratio. A positive logratio, or sum of log probabilties of the tokens, means that the model is more likely to generate those tokens.

Case 2: Negative advantage, so the action was worse than the expected value of the state. This follows very similarly. Here, the loss will be positive if the new model was more likely, so the model will try to make it so the policy parameters make this completion less likely.

Case 3: Zero advantage, so no update is needed. The loss is zero, don't change the policy model.

### Proximal Policy Optimization

There are many, many implementations of PPO available. 
The core *loss* computation is shown below. 
Crucial to stable performance is also the *value* computation, where multiple options exist (including multiple options for the *value model* loss).

Note that the reference policy (or old logprobs) here are from the time the generations were sampled and not necessarily the reference policy. 
The reference policy is only used for the KL distance constraint/penalty.

```python
# B: Batch Size, L: Sequence Length, G: Num of Generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: (B*G, L)

# Get value predictions
values = value_net(completions)  # Shape: (B*G, L)

# Compute simple advantages
advantages = rewards - values.detach()  # Shape: (B*G, L)

# Normalize advantages (optional but stable)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
advantages = advantages.unsqueeze(1)  # Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# Simple value function loss
vf_loss = 0.5 * ((rewards - values) ** 2)  # Shape: (B*G, L)

# Combine policy and value losses
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute metrics for logging
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = 0.5 * ((new_per_token_logps - per_token_logps)**2).mean()
    
    # Compute value loss for logging
    value_loss = vf_loss.mean()
```

The core piece to understand with PPO is how the policy gradient loss is updated.
Focus on these three lines:
```python
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)
```
`pg_losses1` is the same as the vanilla advantage-based PR loss above, which is included in PPO, but the loss (and gradient update) can be clipped.
Though, PPO is controlling the update size to not be too big. Because losses can be negative, we must create a more conservative version of the vanilla policy gradient update rule.

We know that if we *do not* constrain the loss, the policy gradient algorithm will update the weights exactly to the new probability distribution. 
Hence, by clamping the logratio's, PPO is limiting the distance that the update can move the policy parameters.

Finally, the max of two is taken as mentioned above, in order to take the more conservative loss update.

For PPO, all of this happens *while* learning a value function, which opens more complexity, but this is the core logic for the parameter update.

#### PPO/GRPO simplification with 1 gradient step per sample

PPO (and GRPO) implementations can be handled much more elegantly if the hyperparameter "number of gradient steps per sample" is equal to 1.
Many normal values for this are from 2-4 or higher.
In the main PPO or GRPO equations, see @eq:TODO_PPO, the "reference" policy is the previous parameters -- those used to generate the completions or actions.
Thus, if only one gradient step is taken, $\pi_\theta = \pi_{\theta_{old}}$, and the update rule reduces to the following (the notation $[]_\nabla$ indicates a stop gradient):

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\frac{\pi_\theta(a_i|s)}{\left[\pi_{\theta}(a_i|s)\right]_\nabla}A_i - \beta D_{KL}(\pi_\theta||\pi_{ref})\right). $$ {#eq:ppo_1step}

This leads to PPO or GRPO implementations where the second policy gradient and clipping logic can be omitted, making the optimizer far closer to standard policy gradient.


### Group Relative Policy Optimization

The DeepSeekMath paper details some implementation details of GRPO that differ from PPO [@shao2024deepseekmath], especially if comparing to a standard application of PPO from Deep RL rather than language models.
For example, the KL penalty within the RLHF optimization (recall the KL penalty is also used when training reasoning models on verifiable rewards without a reward model) is applied directly in the loss update rather to the reward function.
Where the standard KL penalty application for RLHF is applied as $r=r_\theta + \beta D_{KL}$, the GRPO implementation is along the lines of:

$$ L = L_{\text{policy gradient}} - \beta * D_{KL} $$

Though, there are multiple ways to implement this.
Traditionally, the KL distance is computed with respect to each token in the completion to a prompt $s$.
For reasoning training, multiple completions are sampled from one prompt, and there are multiple prompts in one batch,
so the KL distance will have a shape of [B, L, N], where B is the batch size, L is the sequence length, and N is the number of completions per prompt.
The question when implementing GRPO is: How do you sum over the KL distance and loss to design different types of value-attribution. 
In the below implementation, the loss is summed over the tokens in the completion, but mean could be an alternative.

```python
# B: Batch Size, L: Sequence Length, G: Number of Generations
# Compute grouped-wise rewards # Shape: (B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)    


# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# Shape: (B*G,)

# Compute advantages
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# important to GRPO -- PPO applies this in reward traditionally
# Combine with KL penalty
per_token_loss = pg_loss_max + self.beta * per_token_kl  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute core metric for logging (KL, reward, etc. also logged)
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = 0.5 * ((new_per_token_logps - per_token_logps)**2).mean()
```

For more details on how to interpret this code, see the PPO section above.


## KL Controllers

TODO: adaptive vs static KL control 

See tAble 10 for impelementation details in tulu 2.5 paper

## Double regularization

Many popular policy gradient algorithms from Deep Reinforcement Learning originated due to the need to control the learning process of the agent.
In RLHF, as discussed extensively in Chapter 8 on Regularization and in Chapter 4 on Problem Formulation, there is a built in regularization term via the distance penalty relative to the original policy one is finetuning.
In this view, a large part of the difference between algorithms like PPO (which have internal step-size regularization) and REINFORCE (which is simpler, and PPO under certain hyperparameters reduces to) is far less meaningful for finetuning language models than training agents from scratch.

In PPO, the objective that handles capping the step-size of the update is known as the [surrogate objective](https://huggingface.co/blog/deep-rl-ppo#introducing-the-clipped-surrogate-objective). 
To monitor how much the PPO regularization is impacting updates in RLHF, one can look at the clip fraction variable in many popular implementations, which is the percentage of samples in the batch where the gradients are clipped by this regularizer in PPO. These gradients are *reduced* to a maximum value.

## Training Value Networks

TODO BCELoss loss because continuous between 0 and 1.
If any range, they could be a MSELoss, I think.