---
prev-chapter: "Rejection Sampling"
prev-url: "10-rejection-sampling.html"
next-chapter: "Direct Alignment Algorithms"
next-url: "12-direct-alignment.html"
---

# [Incomplete] Policy Gradient Algorithms


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

TODO baselines, explain advantage

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]$$

TODO cite further reading

### Reinforce

The REINFORCE update is as follows:
$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\Big[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,G_t
\Big],
$$


Reinforce is a specific implementation of vanilla policy gradient that uses a Monte Carlo estimator of the gradient.
[@ahmadian2024back]

REINFORCE can be run without value network -- the value network is for the baseline in the policy gradient. PPO on the other hand needs the value network to accurately compute the advantage funciton

### Proximal Policy Optimization

### REINFORCE Leave One Out (RLOO)

[@huang2024putting], [@ahmadian2024back]

Note that for verifiable domains like reasoning, RLOO may not because it averages over outcomes to update parameters. 
This reduces credit assignment to the batch level and will make it harder for the model to attribute outcomes to specific behaviors within one sample.

### Group Relative Policy Optimization

Introduced in DeepSeekMath [@shao2024deepseekmath], used in others e.g. DeepSeek-V3 [@liu2024deepseek]

## Computing Policy Gradients with a Language Model

## Implementation Tricks

- Only score a response with a reward model with the `eos_token` is generated, otherwise the response is truncated.

TODO. Cite:
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#

https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

### KL Controllers

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