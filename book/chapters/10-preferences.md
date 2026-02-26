---
prev-chapter: "Rejection Sampling"
prev-url: "09-rejection-sampling"
page-title: What are Preferences
search-title: "Chapter 10: What are Preferences"
next-chapter: "Preference Data"
next-url: "11-preference-data"
---

# The Nature of Preferences

Reinforcement learning from human feedback, also referred to as reinforcement learning from human preferences in early literature, emerged to optimize machine learning models in domains where specifically designing a reward function is hard.
The word *preferences*, which was present in early literature, is at the center of the RLHF process -- human preferences are what is trying to be modeled and what fuels the data for training.
In order to understand the scope of the challenge in modeling and measuring human preferences, a broader context is needed in understanding what a preference is, how our understanding of them emerged, and how multiple fields of economics, psychology, philosophy, and more inform modern RLHF.

Consider an example: how do you decide which of these two poems is better?
On February 26th, 2025, I asked both Claude 3.7 Sonnet and ChatGPT with GPT-4o the following:

> **User**: Write me a short poem about an optimistic goldfish.

Example 1:

> **Response**: The Optimistic Goldfish
>
> A little goldfish, bright and bold,
> Swam in circles, dreams untold.
> Though the bowl was small and round,
> Endless wonders still abound.
> 
> "The world is vast," she'd always say,
> "I'll find new joys in every day!"
> With every ripple, every gleam,
> She'd chase the sunlight, chase a dream.
> 
> No walls could dim her hopeful view,
> For in her heart, the ocean grew.
> And though she swam in just one place,
> She met each day with love and grace.

Example 2:

> **Response**: The Optimistic Goldfish
>
> In a bowl of curved glass,
> Swims a goldfish, bright and bold.
> Three seconds of memory, they say,
> Yet each circuit feels brand new and gold.
> 
> "Today might bring fresh flakes,
> Or perhaps a castle tall.
> The water seems especially clear---
> What adventure awaits beyond this wall?"
> 
> Round and round with gleaming scales,
> Forever finding joy anew.
> Each moment fresh, each circle bright,
> The optimist in golden hue.

Which is better? Which came from which model? How would you tell? Particularly, compare this to a question like "Who is the president of the United States?" There is a clear right and wrong answer.
The motivation for using humans as the reward signals is to obtain an indirect metric for the target reward and *align* the downstream model to human preferences.
In practice, the implementation is challenging and there is a substantial grey area to interpret the best practices.

The use of human-labeled feedback data integrates the history of many fields.
Using human data alone is a well-studied problem, but in the context of RLHF it is used at the intersection of multiple long-standing fields of study [@lambert2023entangled].

As an approximation, modern RLHF is the convergence of three areas of development:

1. Philosophy, psychology, economics, decision theory, and the nature of human preferences;
2. Optimal control, reinforcement learning, and maximizing utility; and
3. Modern deep learning systems.

Together, each of these areas brings specific assumptions about what a preference is and how it can be optimized, which dictates the motivations and design of RLHF problems.
In practice, RLHF methods are motivated and studied from the perspective of empirical alignment -- maximizing model performance on specific skills instead of measuring the calibration to specific values.
Still, the origins of value alignment for RLHF methods continue to be studied through research on methods to solve for "pluralistic alignment" across populations, such as position papers [@conitzer2024social], [@mishra2023ai], new datasets [@kirk2024prism], and personalization methods [@poddar2024personalizing].

The goal of this chapter is to illustrate how complex motivations result in presumptions about the nature of tools used in RLHF that often do not apply in practice.
The specifics of obtaining data for RLHF are discussed further in Chapter 11 and using it for reward modeling in Chapter 5.
<!-- For an extended version of this chapter, see [@lambert2023entangled]. -->

<!-- ## The path to optimizing preferences

A popular phrasing for the design of Artificial Intelligence (AI) systems is that of a rational agent maximizing a utility function [@russell2016artificial].
The inspiration of a **rational agent** is a lens of decision making, where said agent is able to act in the world and impact its future behavior and returns, as a measure of goodness in the world.

The lens of study of **utility** began in the study of analog circuits to optimize behavior on a finite time horizon [@widrow1960adaptive].
Large portions of optimal control adopted this lens, often studying dynamic problems under the lens of minimizing a cost function on a certain horizon -- a lens often associated with solving for a clear, optimal behavior.
Reinforcement learning, inspired from literature in operant conditioning, animal behavior, and the *Law of Effect* [@skinner2019behavior],[@thorndike1927law], studies how to elicit behaviors from agents via reinforcing positive behaviors.

Reinforcement learning from human feedback combines multiple lenses by building the theory of learning and change of RL, i.e. that behaviors can be learned by reinforcing behavior, with a suite of methods designed for quantifying preferences.

### Quantifying preferences

The core of RLHF's motivation is the ability to optimize a model of human preferences, which therefore needs to be quantified.
To do this, RLHF builds on extensive literature with assumptions that human decisions and preferences can be quantified.
Early philosophers discussed the existence of preferences, such as Aristotle's Topics, Book Three, and substantive forms of this reasoning emerged later with *The Port-Royal Logic* [@arnauld1861port]:

> To judge what one must do to obtain a good or avoid an evil, it is necessary to consider not only the good and evil in itself, but also the probability that it happens or does not happen.

Progression of these ideas continued through Bentham's *Hedonic Calculus* [@bentham1823hedonic] that proposed that all of life's considerations can be weighed, and Ramsey's *Truth and Probability* [@ramsey2016truth] that applied a quantitative model to preferences.
This direction, drawing on advancements in decision theory, culminated in the Von Neumann-Morgenstern (VNM) utility theorem which gives credence to designing utility functions that assign relative preference for an individual that are used to make decisions.

This theorem is core to all assumptions that pieces of RLHF are learning to model and dictate preferences.
RLHF is designed to optimize these personal utility functions with reinforcement learning.
In this context, many of the presumptions around RL problem formulation break down to the difference between a preference function and a utility function.

### On the possibility of preferences

Across fields of study, many critiques exist on the nature of preferences. 
Some of the most prominent critiques are summarized below:

- **Arrow's impossibility theorem** [@arrow1950difficulty] states that no voting system can aggregate multiple preferences while maintaining certain reasonable criteria.
- **The impossibility of interpersonal comparison** [@harsanyi1977rule] highlights how different individuals have different relative magnitudes of preferences and they cannot be easily compared (as is done in most modern reward model training).
- **Preferences can change over time** [@pettigrew2019choosing].
- **Preferences can vary across contexts**.
- **The utility functions derived from aggregating preferences can reduce corrigibility** [@soares2015corrigibility] of downstream agents (i.e. the possibility of an agents' behavior to be corrected by the designer). -->

## The Origins of RLHF and Preferences

Breaking down the complex history inspiring the modern use of RLHF requires investigation into the intellectual foundations of quantifying human values, reinforcement learning and optimality, as well as behavioral economics as it relates to measuring preferences.
The notion of using reinforcement learning to optimize a reward model of preferences combines the history of various once-distanced fields into an intimate optimization built on variegated assumptions about human nature.
A high level timeline illustrating the history of this foundational content is shown in @fig:tree.

Our goal is to unspool the types of uncertainty that designers have grafted to system architectures at various stages of their intellectual history.
Modern problem specifications have repeatedly stepped away from domains where optimal solutions are possible and deployed under-specified models as approximate solutions.

<!-- Throughout, we distinguish between a series of *assumptions* accepted within theoretically-grounded academic literatures, and relevant *presumptions* which are common methods of practice for particular subject areas.
As we shall see, the unresolved tensions between these assumptions and presumptions are responsible for the current state and outstanding questions of RLHF research.
This section does not set out to be a survey but rather interrelates core references to illustrate the modus operandi of RLHF and preference modeling. -->

To begin, all of the following operates on the assumption that human preferences exist in any form, which emerged in early philosophical discussions, such as Aristotle's Topics, Book Three.

![The timeline of the integration of various subfields into the modern version of RLHF. The direct links are continuous developments of specific technologies, and the arrows indicate motivations and conceptual links.](images/rlhf-tree.png){#fig:tree width=100% .center}

### Specifying objectives: from logic of utility to reward functions

The optimization of RLHF explicitly relies only on reward models. 
In order to use rewards as an optimization target, RLHF presupposes the convergence of ideas from preferences, rewards, and costs.
Models of preference, reward functions, and cost landscapes all are tools used by different fields to describe a notion of relative goodness of specific actions and/or states in the domain. 
The history of these three framings dates back to the origins of probability theory and decision theory.
In 1662, *The Port Royal Logic* introduced the notion of decision making quality [@arnauld1861port]:

> To judge what one must do to obtain a good or avoid an evil, it is necessary to consider not only the good and evil in itself, but also the probability that it happens or does not happen.

This theory has developed along with modern scientific thinking, starting with Bentham's utilitarian *Hedonic Calculus*, arguing that everything in life could be weighed [@bentham1823hedonic].
The first quantitative application of these ideas emerged in 1931 with Ramsey's *Truth and Probability* [@ramsey2016truth].

Since these works, quantifying, measuring, and influencing human preferences has been a lively topic in the social and behavioral sciences.
These debates have rarely been settled on a theoretical level; rather, different subfields and branches of social science have reached internal consensus on methods and approaches to preference measurement even as they have specialized relative to each other, often developing their own distinct semantics in the process.

A minority of economists posit that preferences, if they do exist, are prohibitively difficult to measure because people have preferences over their own preferences, as well as each others' preferences [@hirschman1984against].
In this view, which is not reflected in the RLHF process, individual preferences are always embedded within larger social relations, such that the accuracy of any preference model is contingent on the definition and context of the task.
Some behavioral economists have even argued that preferences don't exist--they may be less an ontological statement of what people actually value than a methodological tool for indirectly capturing psychological predispositions, perceived behavioral norms and ethical duties, commitments to social order, or legal constraints [@hadfield2014microfoundations].
We address the links of this work to the Von Neumann-Morgenstern (VNM) utility theorem and countering impossibility theorems around quantifying preference later in this chapter.

On the other hand, the reinforcement learning optimization methods used today are conceptualized around optimizing estimates of reward-to-go in a trial [@sutton2018reinforcement], which combines the notion of reward with multi-step optimization.
The term *reward* emerged from the study of operant conditioning, animal behavior, and the *Law of Effect* [@thorndike1927law; @skinner2019behavior], where a reward is a scale of "how good an action is" (higher means better).

Reward-to-go follows the notion of utility, which is a measure of rationality [@briggs2014normative], modified to measure or predict the reward coming in a future time window.
In the context of the mathematical tools used for reinforcement learning, utility-to-go was invented in control theory, specifically in the context of analog circuits in 1960 [@widrow1960adaptive].
These methods are designed around systems with clear definitions of optimality, or numerical representations of goals of an agent.
Reinforcement learning systems are well known for their development with a discount factor, a compounding multiplicative factor, $\gamma \in [0,1]$, for re-weighting future rewards.
Both the original optimal control systems stand and early algorithms for reward stand in heavy contrast to reward models that aggregate multimodal preferences.
Specifically, RL systems expect rewards to behave in a specific manner, quoting [@singh2009rewards]:

> Rewards in an RL system correspond to primary rewards, i.e., rewards that in animals have been hard-wired by the evolutionary process due to their relevance to reproductive success.
> ... Further, RL systems that form value functions, ... effectively create conditioned or secondary reward processes whereby predictors of primary rewards act as rewards themselves...
> The result is that the local landscape of a value function gives direction to the system's preferred behavior: decisions are made to cause transitions to higher-valued states.
> A close parallel can be drawn between the gradient of a value function and incentive motivation [@mcclure2003computational].

To summarize, rewards are used in RL systems as a signal to tune behavior towards clearly defined goals.
The core thesis is that a learning algorithm's performance is closely coupled with notions of *expected fitness*, which permeates the popular view that RL methods are *agents* that act in environments.
This view is linked to the development of reinforcement learning technology, exemplified by claims of the general usefulness of the reward formulation [@silver2021reward], but is in conflict when many individual desires are reduced to a single function.

### Implementing optimal utility

Modern reinforcement learning methods depend strongly on the Bellman equation [@bellman1957markovian; @howard1960dynamic] to recursively compute estimates of reward-to-go, derived within closed environments that can be modeled as a Markov Decision Process (MDP) [@sutton2018reinforcement].
These origins of RL are inspired by dynamic programming methods and were developed solely as optimal control techniques (i.e. RL did not yet exist).
The MDP formulation provides theoretical guarantees of performance by structuring the environment as one with a non-changing distribution of state-actions.

The term reinforcement, coming from the psychology literature, became intertwined with modern methods afterwards in the 1960s as *reinforcement learning* [@MENDEL1970287; @waltz1965].
Early work reinforcement learning utilized supervised learning of reward signals to solve tasks.
Work from Harry Klopf reintroduced the notion of trial-and-error learning [@klopf1972brain], which is crucial to the success the field saw in the 1980s and on.

Modern RL algorithms build within this formulation of RL as a tool to find optimal behaviors with trial-and-error, but under looser conditions.
The notion of temporal-difference (TD) learning was developed to aid agents in both the credit assignment and data collection problems, by directly updating the policy as new data was collected [@sutton1988learning], a concept first applied successfully to Backgammon [@tesauro1995temporal] (rather than updating from a large dataset of cumulative experience, which could be outdated via erroneous past value predictions).
The method Q-learning, the basis for many modern forms of RL, learns a model via the Bellman equation that dictates how useful every state-action pair is with a TD update [@watkins1992q].^[The term "Q" is used in Q-learning to refer to a technical concept the Q-function, which maps from any state-action to a scalar estimate of future reward. A value-function maps from states to this same estimate.]
Crucially, these notions of provable usefulness through utility have only been demonstrated for domains cast as MDPs or addressed in tasks with a single closed-form reward function, such as prominent success in games with deep learning (DQN) [@mnih2013playing].
Deep learning allowed the methods to ingest more data and work in high dimensionality environments.

As the methods became more general and successful, most prominent developments before ChatGPT had remained motivated within the context of adaptive control, where reward and cost functions have a finite notion of success [@golnaraghi2017automatic], e.g. a minimum energy consumption across an episode in a physical system.
Prominent examples include further success in games [@silver2017mastering], controlling complex dynamic systems such as nuclear fusion reactors [@degrave2022magnetic], and controlling rapid robotic systems [@Kaufmann2023fpv].
Most reward or cost functions can return an explicit optimal behavior, whereas models of human preferences cannot.

Given the successes of deep RL, it is worth noting that the mechanistic understanding of how the methods succeed is not well documented.
The field is prone to mistakes of statistical analysis as the methods for evaluation grow more complex [@agarwal2021deep].
In addition, there is little mention of the subfield of inverse reinforcement learning (IRL) in the literature of RLHF.
IRL is the problem of learning a reward function based on an agent's behavior [@ng2000algorithms] and highly related to learning a reward model.
This primarily reflects the engineering path by which a stable approach to performing RLHF emerged, and motivates further investment and comparison to IRL methods to scale them to the complexity of open-ended conversations.

<!-- ![The history that creates the assumptions and presumptions central to the current deployments of RLHF. The assumptions indicate core theoretical foundations which RLHF builds upon, transposes, prioritizes, or defers to another development stage. The presumptions represent ideas and practices required to build the current renditions of the technology.](images/timeline-v2.png){#fig:history} -->

### Steering preferences

The context in which reinforcement learning was designed means that rewards and costs are assumed to be stable and determinative. Both rewards and costs are expected to be functions, such that if the agent is in a specific state-action pair, then it will be returned a certain value.
As we move into preferences, this is no longer the case, as human preferences constantly drift temporally throughout their experiences.
The overloading of the term "value" within these two contexts complicates the literature of RLHF that is built on the numerical value updates in Bellman equations with the very different notion of what is a human value, which often refers to moral or ethical principles, but is not well defined in technical literature.
An example of where this tension can be seen is how reward models are attempting to map from the text on the screen to a scalar signal, but in reality, dynamics not captured in the problem specification influence the true decision [@salha2011aesthetics; @gilbert2022choices], such as preference shift when labeling many examples sequentially and assuming they are independent.
Therein, modeling preferences is at best compressing a multi-reward environment to a single function representation.

In theory, the Von Neumann-Morgenstern (VNM) utility theorem gives the designer license to construct such functions, because it ties together the foundations of decision theory under uncertainty, preference theory, and abstract utility functions [@von1947theory]; together, these ideas allow preferences to be modeled in terms of expected value to some individual agent.
The MDP formulation used in most RL research has been shown in theory to be modifiable to accommodate the VNM theorem [@pitis2019rethinking], but this is rarely used in practice.
Specifically, the Markovian formulation is limited in its expressivity [@pitis2023consistent] and the transition to partially-observed processes, which is needed for language, further challenges the precision of problem specification [@abel2021expressivity].

However, the VNM utility theorem also invokes a number of assumptions about the nature of preferences and the environment where preferences are being measured that are challenged in the context of RLHF.
Human-computer interaction (HCI) researchers, for example, have emphasized that any numerical model of preference may not capture all the relevant preferences of a scenario.
For example, how choices are displayed visually influences people's preferences [@salha2011aesthetics].
This means that representing preferences may be secondary to how that representation is integrated within a tool available for people to use.
Work from development economics echoes this notion, showing that theories of revealed preferences may just recapitulate *Hume's guillotine* (you can't extract an "ought" from an "is"), and in particular the difference between choice (what do I want?) and preference (is X better than Y?) [@sen1973behaviour].

On a mathematical level, well-known impossibility theorems in social choice theory show that not all fairness criteria can be simultaneously met via a given preference optimization technique [@arrow1950difficulty; @maskin2014arrow].
Theoretical challenges to these theorems exist, for example by assuming that interpersonal comparison of utility is viable [@harsanyi1977rule].
That assumption has inspired a rich line of work in AI safety and value alignment inspired by the principal-agent problem in behavioral economics [@hadfield2016cooperative], and may even include multiple principals [@fickinger2020multi].
However, the resulting utility functions may come into tension with desiderata for corrigibility, i.e. an AI system's capacity to cooperate with what its creators regard as corrective interventions [@soares2015corrigibility].
Philosophers have also highlighted that preferences change over time, raising fundamental questions about personal experiences, the nature of human decision-making, and distinct contexts [@pettigrew2019choosing].
These conflicts around the preference aggregation across people, places, or diverse situations is central to modern RLHF dataset engineering.

In practice, the VNM utility theorem ignores the possibility that preferences are also uncertain because of the inherently dynamic and indeterminate nature of value---human decisions are shaped by biology, psychology, culture, and agency in ways that influence their preferences, for reasons that do not apply to a perfectly rational agent.
As a result, there are a variety of paths through which theoretical assumptions diverge in practice:

- measured preferences may not be transitive or comparable with each other as the environment where they are measured is made more complex;
- proxy measurements may be derived from implicit data (page view time, closing tab, repeating question to language model), without interrogating how the measurements may interact with the domain they're collected in via future training and deployment of the model;
- the number and presentation of input sources may vary the results, e.g. allowing respondents to choose between more than two options, or taking in inputs from the same user at multiple times or in multiple contexts;
- relatively low accuracy across respondents in RLHF training data, which may mask differences in context between users that the preference model can aggregate or optimize without resolving.