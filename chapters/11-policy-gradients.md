# Policy Gradient Algorithms

The algorithms that popularized RLHF for language models were policy-gradient reinforcement learning algoritms. 
These algorithms, such as PPO and Reinforce, use recently generated samples to update their model rather than storing scores in a replay buffer.
In this section we will cover the fundamentals of the policy gradient algorithms and how they are used in the modern RLHF framework.

For definitions of symbols, see the problem setup chapter.

## Policy Gradient Algorithms

The core of policy gradient algorithms is computing the gradient with respect to the finite time expected return over the current policy. 
With this expected return, $J$, the gradient can be computed as follows, where $\alpha$ is the learning rate: 

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$


### Vanilla Policy Gradient

The vanilla policy gradient implementation optimizes the following expectation:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]$$

### Reinforce

Reinforce is a specific implementation of vanilla policy gradient that uses a Monte Carlo estimator of the gradient.
[@ahmadian2024back]
### Proximal Policy Optimization

## Computing Policy Gradients with a Language Model

## Implementation Tricks

- Only score a response with a reward model with the `eos_token` is generated, otherwise the response is truncated.

TODO. Cite:
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#

https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

### KL Controllers

TODO: adaptive vs static KL control 

See tAble 10 for impelementation details in tulu 2.5 paper
