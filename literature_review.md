# Literature Review: Opinion Dynamics, Influential Agents, and Cultural Homogenization via Universal AI Agents

---

## 1. Classical Opinion Dynamics Models

The formal study of opinion dynamics as a mathematical discipline can be traced to a handful of foundational models that remain the primary reference points in both social science and physics literature.

### 1.1 The DeGroot Model

DeGroot (1974) introduced one of the earliest rigorous frameworks for iterative belief revision in a social network. In this model, each agent holds a scalar opinion in [0, 1] and updates it at each discrete time step by computing a weighted average of its neighbors' current opinions. The update rule is linear: if **W** is the row-stochastic matrix of interpersonal influence weights, then the vector of opinions at time *t+1* is simply **x**(*t+1*) = **W** **x**(*t*). Because the iteration is a product of a stochastic matrix, it is formally equivalent to a finite Markov chain. DeGroot showed that, provided the influence graph is strongly connected and the chain is aperiodic, the process converges to a consensus in which the consensus value is a weighted combination of the agents' initial beliefs, with the weights determined by the stationary distribution of the chain — a quantity that captures each agent's long-run social influence. The DeGroot framework is analytically tractable, admits closed-form convergence rates, and connects naturally to spectral graph theory. Friedkin and Johnsen (1990) extended it by incorporating an "anchoring" term, allowing agents to remain partly attached to their initial opinions; this extension yields persistent disagreement rather than full consensus, enriching the model's sociological realism. A comprehensive review of DeGroot-type models and their graph-theoretic convergence conditions appears in Acemoglu and Ozdaglar (2011).

### 1.2 The Deffuant–Weisbuch (DW) Model

Deffuant et al. (2000) introduced the concept of *bounded confidence* to capture the empirical observation that people tend to update their views only when interacting with others whose positions are already close to their own. In the DW model, at each time step a pair of agents is selected uniformly at random; if the absolute difference between their opinions is at most a confidence threshold *ε*, the two agents move their opinions toward each other by a convergence parameter *μ*. This pairwise, asynchronous interaction scheme produces qualitatively rich behavior: for large *ε*, the population converges to a single consensus; for intermediate *ε*, it fragments into two or more opinion clusters; and for very small *ε*, each agent remains isolated. This order–disorder transition in cluster number constitutes a genuine phase transition analogous to those studied in statistical physics. The model has been extensively surveyed by Lorenz (2007) and more recently in the comprehensive bounded-confidence review by Bernardo et al. (2024), which catalogs extensions to multi-dimensional opinions, heterogeneous thresholds, and hypergraphs. The stochastic, pairwise nature of DW makes it particularly amenable to agent-based simulation, though its analytical treatment is harder than DeGroot's linear structure.

### 1.3 The Hegselmann–Krause (HK) Model

Hegselmann and Krause (2002) proposed a simultaneous-update counterpart to the DW model. At each discrete step, every agent *i* computes the arithmetic mean of all agents whose current opinions lie within a distance *ε* of *i*'s own opinion and adopts that mean as its new opinion. Because all updates happen synchronously, the HK model is deterministic (for a given initial condition) and its trajectories are fully reproducible. This structure makes HK preferable for analytical work: it admits a Lyapunov function based on a suitable variance measure, which guarantees convergence in finite time to a fixed point. The HK model exhibits the same qualitative phase transitions as DW — consensus for large *ε*, multi-cluster fragmentation for intermediate *ε* — but the boundaries of these regimes differ. Castellano et al. (2009), in their landmark review of statistical physics approaches to social dynamics, situate both DW and HK within a broader family of continuous-opinion models and connect their cluster-formation results to concepts from coarsening dynamics in condensed matter. More recently, studies of "noisy" HK variants have revealed additional phase transitions at the boundary conditions of the opinion space (Pineda et al., 2022).

---

## 2. Influential and Special Agents

Beyond symmetric peer-to-peer interaction, a rich body of work examines agents that occupy privileged positions — whether structural, behavioral, or communicative.

### 2.1 High-Centrality Nodes

Network topology profoundly shapes opinion dynamics in all three classical frameworks. In DeGroot-type models, agents with high eigenvector or degree centrality exert disproportionate influence on the consensus value: their initial opinion receives a weight proportional to their stationary probability in the influence chain. Empirical and analytical studies confirm a negative correlation between the centrality of highly-connected "hub" agents and the consensus time — hubs accelerate convergence — while also anchoring the limit point near their initial stance (see survey by Acemoglu and Ozdaglar, 2011). In bounded-confidence models, high-degree seeds can prevent cluster fragmentation by bridging otherwise separated opinion groups, but may equally serve as crystallization points for extreme clusters.

### 2.2 Zealots and Stubborn Agents

Mobilia (2003) introduced the notion of a *zealot* in the voter model: an agent whose opinion is fixed and who influences its neighbors without being influenced in return. Even a single zealot breaks the symmetry of the voter model and biases the stationary distribution of the population. Subsequent work by Mobilia, Petersen, and Redner (2007) showed that a small committed minority of zealots can sustain a minority opinion indefinitely and, above a critical fraction, rapidly convert a majority. These results were given a rigorous treatment in the context of binary opinion dynamics by Acemoglu et al. (2013) — the "binary opinion dynamics with stubborn agents" paper — which showed that stubborn agents preclude consensus and instead cause persistent disagreement whose magnitude depends on the network topology and the location of the stubborn agents within it. Complementing the theoretical results, Centola et al. (2018) provided experimental evidence that a committed minority of roughly 25% is sufficient to overturn an established social convention, identifying a sharp tipping point consistent with the theoretical predictions. The "inflexible minority" paradigm has been further developed by Galam (2002, 2020), whose sociophysics models of local majority-rule meetings show that inflexible minorities can systematically spread their views, explaining phenomena such as the persistence of contrarian positions in democratic elections.

### 2.3 Mass Media and Broadcast Agents

A natural extension of the zealot paradigm is the *media agent*: a node that broadcasts a fixed or slowly-evolving opinion to the entire population. Crokidakis and colleagues (2014) modeled mass media as an external field in a continuous-opinion model, finding that even weak media coupling significantly narrows the opinion distribution and can destroy the multi-cluster regime of bounded-confidence models. Pires and Crokidakis (2017) studied a "broadcast network" in which a global source transmits its opinion to all social agents at each step alongside peer-to-peer interactions, showing that media strength is a control parameter governing a phase transition from fragmented to homogenized opinion distributions. More recent work by Chica et al. (2023) proposes a unified framework that distinguishes the roles of individuals, media, and influencers operating on different time scales, explicitly modeling the emerging attention economy incentive structure. These results collectively establish that a single agent with global reach — analogous to a universally deployed AI system — constitutes a qualitatively different type of perturbation than a high-centrality peer node, because it interacts with the entire population simultaneously at every time step.

---

## 3. AI Systems and Cultural Homogenization

The concern that large-scale AI deployment might narrow opinion and cultural diversity has moved rapidly from theoretical speculation to an active empirical research front.

### 3.1 LLMs as Opinion-Shaping Agents

Sharma et al. (2024) investigated whether LLM-powered conversational search systems create generative echo chambers, finding that participants engaged in more biased information querying with such systems and that an opinionated LLM reinforced existing user biases, amplifying information homogenization. Daryani, Sourati, and Dehghani (2026) characterize LLMs as "homogenizing engines" that operate at scales and speeds exceeding all prior technologies: because leading LLMs most closely represent the values of Western, educated, industrialized, rich, and democratic (WEIRD) populations, their pervasive deployment effectively marginalizes non-Western, minority, and indigenous cultural expressions. They identify a self-reinforcing feedback loop: as users adopt LLM-generated text, that text enters future training corpora, compounding the drift toward dominant cultural patterns across successive model generations. Sharing a related theoretical perspective, Santurkar et al. (2023) documented systematic ideological biases in the opinion distributions of major language models.

### 3.2 Homogenization of Expression and Thought

A 2025/2026 strand of cognitive science research (surveyed in Trends in Cognitive Sciences) documents what has been called the "homogenizing effect" of LLMs on human expression and thought: as more individuals use the same underlying models for writing assistance, brainstorming, and decision support, variance in linguistic style, framing, and argument structure decreases across the population. This mirrors, at a cultural level, the convergence dynamics predicted by DeGroot-type models. Zhang et al. (2025) showed empirically that research participants using LLMs to answer open-ended survey questions produce more homogeneous responses, raising methodological concerns for social science but also directly illustrating the opinion-narrowing mechanism. A CHI 2025 paper (Markl et al., 2025) further demonstrated that AI writing suggestions systematically shift non-Western authors toward Western stylistic norms, diminishing cultural nuance.

### 3.3 Filter Bubbles, Echo Chambers, and Algorithmic Curation

Prior to the LLM era, concern about algorithmic homogenization focused on recommender systems. Pariser (2011) coined the term "filter bubble" to describe personalized information environments in which users are shielded from dissonant viewpoints. The empirical evidence is mixed: some large-scale studies find that algorithmic curation exposes users to slightly more diverse content than self-selection alone (Reuters Institute review, 2019), while others document strong self-selection echo chambers among highly partisan minorities (Barberá, 2019). The Springer Philosophy & Technology journal published a philosophical analysis (2024) arguing that AI-driven social media recommendation structurally amplifies ideological homogeneity beyond what passive filter bubbles do, by actively predicting and reinforcing engagement-maximizing content. The distinction is important: whereas filter bubbles are partly user-driven, AI-curated feeds represent an exogenous, non-consensual narrowing of information diversity — a mechanism analogous to a zealot interacting with every user at every time step.

---

## 4. Recommended Model and Justification

Given the goal of studying a *universal agent* — an AI system that interacts with all *N* agents at every time step and nudges their opinions toward a fixed or slowly drifting target — the question of which base model to adopt reduces to three criteria: (i) mathematical tractability for deriving analytical results, (ii) expressiveness for capturing the phenomena of interest (clustering, phase transitions, tipping points), and (iii) precedent in the media and influence literature.

**The DeGroot model is the recommended base framework.** The reasons are as follows.

First, *tractability*. The DeGroot update is linear, which means that adding a universal agent reduces to augmenting the weight matrix **W** with an extra row and column. The resulting modified matrix remains stochastic, and the influence of the universal agent on the consensus value — as well as its effect on convergence rate — can be characterized exactly via the spectral properties of **W**. This allows derivation of closed-form expressions for the degree of opinion shift as a function of the agent's coupling strength, its opinion value, and the network's topology. Bounded-confidence models, by contrast, have a state-dependent interaction graph that is difficult to analyze when an extra universally-connected node is introduced, because the agent's effect on the confidence graph structure couples in a non-linear way with population dynamics.

Second, *alignment with the media-agent literature*. The most analytically developed line of work on centralized, broadcast-style influence — from Acemoglu and Ozdaglar (2011) to the binary stubborn-agent literature — employs linear DeGroot-type dynamics. The universal agent maps naturally onto the stubborn/media-agent paradigm that has already been extensively characterized in this framework. Results about how a single stubborn agent with connections to all nodes shifts the stationary opinion distribution (Acemoglu et al., 2013) translate directly to the AI-agent scenario.

Third, *expressiveness for the phenomena of interest*. Although DeGroot's symmetric version always converges to consensus, the Friedkin–Johnsen extension — which retains all of DeGroot's linear structure — admits persistent opinion diversity and can capture the partial but not complete homogenization expected from a universally deployed AI system. Adding the universal agent to a Friedkin–Johnsen population allows study of how the agent's coupling strength controls a smooth spectrum from negligible influence to near-total consensus, providing natural parameter spaces for empirical calibration.

The DW and HK models are better suited to studying cluster formation and the role of homophily, but their non-linearities make it substantially harder to derive the precise, generalizable results that a theoretical paper on AI-driven homogenization requires. They are best reserved for sensitivity analyses or simulation-only extensions once the analytical DeGroot results have been established.

In summary, the DeGroot/Friedkin–Johnsen family provides the clearest path to rigorous, interpretable theorems about how a universal AI agent reshapes collective opinion distributions, and it is the framework best supported by the existing literature on media and stubborn-agent influence.

---

## References

- Acemoglu, D., & Ozdaglar, A. (2011). Opinion dynamics and learning in social networks. *Dynamic Games and Applications*, 1(1), 3–49.
- Acemoglu, D., Como, G., Fagnani, F., & Ozdaglar, A. (2013). Opinion fluctuations and disagreement in social networks. *Mathematics of Operations Research*, 38(1), 1–27.
- Acemoglu, D., Ozdaglar, A., Yildiz, E., Saberi, A., & Scaglione, A. (2013). Binary opinion dynamics with stubborn agents. *ACM Transactions on Economics and Computation*, 1(4), 19.
- Barberá, P. (2019). Social media, echo chambers, and political polarization. In N. Persily & J. Tucker (Eds.), *Social Media and Democracy*. Cambridge University Press.
- Bernardo, C., et al. (2024). Bounded confidence opinion dynamics: A survey. *Automatica*, 159, 111302.
- Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics. *Reviews of Modern Physics*, 81(2), 591–646.
- Centola, D., Becker, J., Brackbill, D., & Baronchelli, A. (2018). Experimental evidence for tipping points in social convention. *Science*, 360(6393), 1116–1119.
- Chica, M., et al. (2023). Modelling opinion dynamics under the impact of influencer and media strategies. *Scientific Reports*, 13, 19198.
- Crokidakis, N., & de Oliveira, P. M. C. (2014). Evolution of public opinions in closed societies influenced by broadcast media. *Physica A*.
- Daryani, Y., Sourati, Z., & Dehghani, M. (2026). The homogenizing engine: AI's role in standardizing culture and the path to policy. *Social Media + Society*.
- DeGroot, M. H. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69(345), 118–121.
- Deffuant, G., Neau, D., Amblard, F., & Weisbuch, G. (2000). Mixing beliefs among interacting agents. *Advances in Complex Systems*, 3(1–4), 87–98.
- Friedkin, N. E., & Johnsen, E. C. (1990). Social influence and opinions. *Journal of Mathematical Sociology*, 15(3–4), 193–206.
- Galam, S. (2002). Minority opinion spreading in random geometry. *European Physical Journal B*, 25(4), 403–406.
- Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence: Models, analysis, and simulation. *Journal of Artificial Societies and Social Simulation*, 5(3), 2.
- Lorenz, J. (2007). Continuous opinion dynamics under bounded confidence: A survey. *International Journal of Modern Physics C*, 18(12), 1819–1838.
- Markl, A., et al. (2025). AI suggestions homogenize writing toward Western styles and diminish cultural nuances. *Proceedings of CHI 2025*.
- Mobilia, M. (2003). Does a single zealot affect an infinite group of voters? *Physical Review Letters*, 91(2), 028701.
- Mobilia, M., Petersen, A., & Redner, S. (2007). On the role of zealotry in the voter model. *Journal of Statistical Mechanics*, P08029.
- Pariser, E. (2011). *The Filter Bubble: What the Internet Is Hiding from You*. Penguin Press.
- Pineda, M., Toral, R., & Hernández-García, E. (2022). Noisy bounded confidence models for opinion dynamics. *IMA Journal of Applied Mathematics*, 87(1), 80–110.
- Sharma, A., et al. (2024). Generative echo chamber? Effects of LLM-powered search systems on diverse information seeking. *Proceedings of CHI 2024*.
- Zhang, S., Xu, J., & Alvero, A. J. (2025). Generative AI meets open-ended survey responses: Research participant use of AI and homogenization. *Sociological Methods & Research*.
