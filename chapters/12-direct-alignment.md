---
prev-chapter: "Policy Gradients"
prev-url: "11-policy-gradients.html"
page-title: Direct Alignment Algorithms
next-chapter: "Constitutional AI"
next-url: "13-cai.html"
---

# [Incomplete] Direct Alignment Algorithms

Direct Alignment Algorithms (DAAs) allow one to update models to solve the  same RLHF objective without ever training an intermediate reward model or using reinforcement learning optimizers.
DPO [@rafailov2024direct]

Many popular models have used DPO or its variants, from Zephyr, Llama 3 Instruct, Tülu 3, Nemotron 4 340B

(technically, Sequence Likelihood Calibration (SLiC) was released first [@zhao2023slic], but it did not catch on )

REBEL - regression from rewards rather than solely the pairwise preference data [@gao2024rebel]
IPO (Phi-PO) [@azar2024general]
conservative DPO (cDPO) [@rafailov2024direct]

DPO with an offset (ODPO) [@amini2024direct] -- do not treat every data pair equally,
variants without a reference model by changing the regularization, such as Odds Ratio Policy Optimization (ORPO) [@hong2024reference]

Minor changes to the optimization, such as averaging the log-probabilities rather than summing them (SimPO) or adding length normalization, to improve performance [@meng2025simpo]

Online variants that sample generations from the model, e.g. Online DPO [@guo2024direct], even with regular reward model relabelling of newly created creations (D2PO) [@singhal2024d2po]

And others, such as Direct Nash Optimization (DNO) [@rosset2024direct] or Binary Classifier Optimization (BCO) [@jung2024binary]

Regardless, the choice of algorithm is far less important than the initial model and the data used -- prompts and completions [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences].

## Numerical Concerns

DPO does some weird things to the models, but is not fully understood. E.g. decreasing the likelihood of chosen responses, but decreasing rejected more.
[@razin2024unintentional] (and methods have been proposed to address it [@xiao2024cal])



## DAAs vs. RL: Online vs. Offline Data

Tülu 2.5 [@ivison2024unpacking], should use on-policy data [@tajwar2024preference], Online DPO above