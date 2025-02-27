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
Since its release in May of 2023, after a brief delay where the community figured out the right data and hyperparameters to use DPO with, many popular models have used DPO or its variants, from Zephyr-$\beta$ kickstarting it in October of 2024 [@tunstall2023zephyr], Llama 3 Instruct [@dubey2024llama], Tülu 2 [@ivison2023camels] and 3 [@lambert2024t], Nemotron 4 340B [@adler2024nemotron], and others.
Technically, Sequence Likelihood Calibration (SLiC-HF) was released first [@zhao2023slic], but it did not catch on due to a combination of luck and effectiveness.

The most impactful part of DPO and DAAs is lowering the barrier of entry to experimenting with language model post-training.

## Direct Preference Optimization

## Numerical Concerns, Weaknesses, and Alternatives

Many variants of the DPO algorithm have been proposed to address weaknesses of DPO.
For example, without rollouts where a reward model can rate generations, DPO treats every pair of preference data with equal weight. 
In reality, as seen in Chapter 6 on Preference Data, there are many ways of capturing preference data with a richer label than binary.
Multiple algorithms have been proposed to re-balance the optimization away from treating each pair equally.

- **REgression to RElative REward Based RL (REBEL)** adds signal from a reward model, as a margin between chosen and rejected responses, rather than solely the pairwise preference data to more accurately solve the RLHF problem [@gao2024rebel].
- **Conservative DPO (cDPO) and Identity Preference Optimization (IPO)** address the overfitting by assuming noise in the preference data. cDPO assumes N percent of the data is incorrectly labelled [@rafailov2024direct] and IPO changes the optimization to soften probability of preference rather than optimize directly from a label [@azar2024general]. Practically, IPO changes the preference probability to a nonlinear function, moving away from the Bradley-Terry assumption, with $\Psi(q) = \log\left(\frac{q}{1-q}\right)$.
- **DPO with an offset (ODPO)** "requires the difference between the likelihood of the preferred and dispreferred response to be greater than an offset value" [@amini2024direct] -- do not treat every data pair equally, but this can come at the cost of a more difficult labeling environment.



variants without a reference model by changing the regularization, such as Odds Ratio Policy Optimization (ORPO) [@hong2024reference]

Minor changes to the optimization, such as averaging the log-probabilities rather than summing them (SimPO) or adding length normalization, to improve performance [@meng2025simpo]

Online variants that sample generations from the model, e.g. Online DPO [@guo2024direct], even with regular reward model relabelling of newly created creations (D2PO) [@singhal2024d2po]

And others, such as Direct Nash Optimization (DNO) [@rosset2024direct] or Binary Classifier Optimization (BCO) [@jung2024binary]

Regardless, the choice of algorithm is far less important than the initial model and the data used -- prompts and completions [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences].


DPO does some weird things to the models, but is not fully understood. E.g. decreasing the likelihood of chosen responses, but decreasing rejected more.
[@razin2024unintentional] (and methods have been proposed to address it [@xiao2024cal])



## DAAs vs. RL: Online vs. Offline Data

Tülu 2.5 [@ivison2024unpacking], should use on-policy data [@tajwar2024preference], Online DPO above