# Formal Model Specification: Cultural Homogenization via a Universal Agent

---

## 3. Model

We develop the model in two stages. Stage A establishes a baseline population of interacting agents without any distinguished external influence. Stage B introduces the Universal Agent (UA) — a single entity that interacts with every member of the population simultaneously — and analyzes how the coupling strength to the UA reshapes collective opinion dynamics.

---

### 3.1 Stage A — Baseline Model (No Universal Agent)

#### Agents and State Space

Let there be $N$ agents indexed $i = 1, \ldots, N$. Each agent $i$ holds an opinion vector $x_i(t) \in \mathbb{R}^d$ at discrete time $t = 0, 1, 2, \ldots$ For analytical tractability we begin with $d = 1$ (scalar opinions); the extension to $d > 1$ is discussed in Section 3.4. Denote the population mean at time $t$ by $\bar{x}(t) = \frac{1}{N} \sum_{i=1}^{N} x_i(t)$.

Agent interactions are governed by a weighted network. Let $G = (V, E)$ be a directed graph on $N$ nodes, and let $W = [w_{ij}]$ be the associated $N \times N$ weight matrix. We impose $w_{ij} \geq 0$ for all $i, j$ and $\sum_{j=1}^{N} w_{ij} = 1$ for all $i$ (row-stochastic), so that $w_{ij}$ represents the fractional trust or attention agent $i$ grants to agent $j$.

#### A1. DeGroot Variant

Under the DeGroot model, opinions evolve according to a linear consensus protocol:

$$x_i(t+1) = \sum_{j=1}^{N} w_{ij} \, x_j(t), \qquad i = 1, \ldots, N.$$

In matrix form, letting $\mathbf{x}(t) = (x_1(t), \ldots, x_N(t))^\top$,

$$\mathbf{x}(t+1) = W \, \mathbf{x}(t).$$

**Baseline network.** We begin with the complete, uniform graph: $w_{ij} = 1/N$ for all $i, j$. This collapses to $x_i(t+1) = \bar{x}(t)$ for every agent, so consensus is reached in a single step. We later extend to arbitrary topologies — ring, small-world, and scale-free networks — to test Hypothesis H4.

**Convergence (standard DeGroot theorem).** If $W$ is row-stochastic, strongly connected, and aperiodic, then the opinion trajectory converges:

$$\lim_{t \to \infty} \mathbf{x}(t) = \mathbf{1} \, \pi^\top \mathbf{x}(0),$$

where $\pi$ is the unique left eigenvector of $W$ corresponding to eigenvalue 1 (i.e., $\pi^\top W = \pi^\top$, $\mathbf{1}^\top \pi = 1$). The consensus value is therefore a fixed weighted average of initial opinions, with weights determined by the network's stationary distribution.

#### A2. Bounded Confidence Variant (Deffuant / Hegselmann–Krause Style)

To capture the empirically observed phenomenon of opinion clustering, we adopt a Hegselmann–Krause (HK) synchronous update rule. Let $\varepsilon > 0$ be the confidence bound. Define the neighborhood of agent $i$ at time $t$ as

$$\mathcal{N}_i(t) = \{ j \in \{1, \ldots, N\} : |x_j(t) - x_i(t)| < \varepsilon \}.$$

The update rule is

$$x_i(t+1) = \frac{1}{|\mathcal{N}_i(t)|} \sum_{j \in \mathcal{N}_i(t)} x_j(t), \qquad i = 1, \ldots, N.$$

Note that agent $i$ always lies in its own neighborhood ($i \in \mathcal{N}_i(t)$), so $|\mathcal{N}_i(t)| \geq 1$. Because interactions are restricted to agents within the confidence radius, small values of $\varepsilon$ allow the population to fragment into multiple persistent opinion clusters. The number of clusters and their locations depend jointly on $\varepsilon$ and the initial opinion distribution.

---

### 3.2 Stage B — Universal Agent Extension

We now introduce a single distinguished Universal Agent (UA), with state $x_{\mathrm{UA}}(t) \in \mathbb{R}^d$, that interacts with every agent $i$ at every time step. A scalar coupling parameter $\alpha \in [0, 1]$ controls the weight each agent allocates to the UA relative to peer interactions. At $\alpha = 0$ the system reduces exactly to Stage A; at $\alpha = 1$ every agent adopts the UA's opinion unconditionally.

#### Modified Update Rules

**B-DeGroot.** The update rule for agent $i$ under the DeGroot dynamics becomes

$$x_i(t+1) = (1 - \alpha) \sum_{j=1}^{N} w_{ij} \, x_j(t) \;+\; \alpha \, x_{\mathrm{UA}}(t).$$

**B-HK (Bounded Confidence).** For agent $i$ under HK dynamics,

$$x_i(t+1) = (1 - \alpha) \cdot \frac{1}{|\mathcal{N}_i(t)|} \sum_{j \in \mathcal{N}_i(t)} x_j(t) \;+\; \alpha \, x_{\mathrm{UA}}(t).$$

In both cases, the UA acts as an additional, universally accessible "virtual neighbor" given weight $\alpha$, while the remaining weight $(1 - \alpha)$ is distributed over the peer term as before.

#### B1. Fixed UA

The UA maintains a constant opinion for all time:

$$x_{\mathrm{UA}}(t) = x_{\mathrm{UA}} \in \mathbb{R}^d, \qquad \forall\, t \geq 0.$$

**Equilibrium analysis (B1-DeGroot).** Substituting $x_i^* = x_i(t+1) = x_i(t)$ into the B-DeGroot update and using the row-stochastic property of $W$, one finds that under any strongly connected, aperiodic $W$, the unique equilibrium satisfies

$$x_i^* = x_{\mathrm{UA}}, \qquad \forall\, i, \quad \text{for any } \alpha > 0.$$

The convergence rate is governed by the spectral gap: the second-largest eigenvalue of the effective update operator $((1-\alpha)W)$ is $(1-\alpha)\lambda_2(W)$, so larger $\alpha$ accelerates consensus geometrically. For the complete graph, the equilibrium is reached in finite time proportional to $\log(1/\alpha)$ in a suitable sense. For the bounded confidence variant, the equilibrium cluster structure collapses to a single cluster near $x_{\mathrm{UA}}$ when $\alpha$ exceeds a critical value $\alpha^*$ (Hypothesis H1).

#### B2. Adaptive UA

The UA updates its own state as an exponential moving average of the current population mean:

$$x_{\mathrm{UA}}(t+1) = (1 - \beta)\, x_{\mathrm{UA}}(t) \;+\; \beta \cdot \frac{1}{N} \sum_{i=1}^{N} x_i(t),$$

where $\beta \in [0, 1]$ is the UA's learning rate. At $\beta = 0$ the UA is frozen (reducing to B1); at $\beta = 1$ the UA tracks the instantaneous population mean exactly.

**Co-evolutionary fixed point.** A fixed point $(x^*, x_{\mathrm{UA}}^*)$ of the coupled system satisfies simultaneously the agent equilibrium conditions and $x_{\mathrm{UA}}^* = (1-\beta) x_{\mathrm{UA}}^* + \beta \bar{x}^*$, which gives $x_{\mathrm{UA}}^* = \bar{x}^*$ for $\beta > 0$. Combined with the agent equilibrium, the fixed point lies at the value jointly consistent with the network dynamics and the UA's tracking. The result is a co-evolutionary attractor that is shifted toward (but not necessarily equal to) the initial UA position, with the degree of shift depending on $\alpha / \beta$. When $\alpha \gg \beta$, the population is pulled strongly toward the UA before the UA can adapt, yielding convergence near $x_{\mathrm{UA}}(0)$. When $\beta \gg \alpha$, the UA tracks the population closely and the system converges near the initial population mean — qualitatively similar to Stage A.

#### B3. Heterogeneous Coupling

In this sub-variant, each agent $i$ is assigned its own coupling parameter $\alpha_i$, drawn independently from a distribution $P(\alpha)$ over $[0, 1]$ at $t = 0$ and held fixed thereafter. The update rule becomes

$$x_i(t+1) = (1 - \alpha_i) \cdot [\text{peer term}] \;+\; \alpha_i \, x_{\mathrm{UA}}(t),$$

where the peer term is the appropriate DeGroot or HK expression. High-$\alpha_i$ agents (heavy UA users) converge rapidly toward $x_{\mathrm{UA}}$, while low-$\alpha_i$ agents retain stronger peer-driven dynamics. This disparity generates cultural stratification: the population bifurcates into a "core" cluster tightly aligned with $x_{\mathrm{UA}}$ and one or more "peripheral" clusters whose positions depend on the local network dynamics and the residual peer interactions (Hypothesis H3).

---

### 3.3 Parameter Table

| Symbol | Description | Range / Units | Default |
|---|---|---|---|
| $N$ | Number of agents | $\mathbb{Z}_{>0}$ | 200 |
| $d$ | Dimension of opinion vector | $\mathbb{Z}_{>0}$ | 1 |
| $\varepsilon$ | Confidence bound (HK variant) | $(0, \infty)$ | 0.3 |
| $\alpha$ | Coupling weight to the UA (uniform) | $[0, 1]$ | varied |
| $\alpha_i$ | Agent-specific coupling weight (B3) | $[0, 1]$, drawn from $P(\alpha)$ | — |
| $\beta$ | UA learning rate (B2 only) | $[0, 1]$ | 0.1 |
| $x_{\mathrm{UA}}$ | Initial (or fixed) UA opinion | $\mathbb{R}^d$ | 0.8 |
| $x_i(0)$ | Initial opinion of agent $i$ | $\mathbb{R}^d$, drawn from $P_0$ | $\mathcal{U}[0,1]$ |
| $P_0$ | Initial opinion distribution | Probability distribution on $\mathbb{R}^d$ | $\mathcal{U}[0,1]$ |
| $W$ | Peer weight matrix | Row-stochastic, $N \times N$ | Complete graph |
| Topology | Network structure of $G$ | Complete, ring, small-world, scale-free | Complete |

---

### 3.4 Formal Statement of Hypotheses

Using the notation established above, the four central hypotheses are stated as follows.

**H1 (Phase Transition — Fixed UA).** There exists a critical coupling $\alpha^* \in (0, 1)$, depending on $\varepsilon$ and $P_0$, such that for all $\alpha > \alpha^*$ the HK-based dynamics (B1-HK) satisfy

$$\lim_{t \to \infty} \#\text{clusters}(\mathbf{x}(t)) = 1 \quad \text{and} \quad \lim_{t \to \infty} \bar{x}(t) = x_{\mathrm{UA}},$$

for almost every initial condition $\mathbf{x}(0)$ drawn from $P_0^{\otimes N}$. For $\alpha < \alpha^*$, the system generically converges to multiple clusters with $\lim_{t\to\infty} \bar{x}(t) \neq x_{\mathrm{UA}}$.

**H2 (Sharpness of Transition — Fixed vs. Adaptive).** Let $\Phi_{\mathrm{fixed}}(\alpha)$ and $\Phi_{\mathrm{adapt}}(\alpha, \beta)$ denote the asymptotic opinion variance $\mathrm{Var}(\infty)$ as a function of $\alpha$ under the B1 and B2 dynamics, respectively. Then

$$\left|\frac{d\Phi_{\mathrm{fixed}}}{d\alpha}\right|_{\alpha = \alpha^*} > \left|\frac{\partial \Phi_{\mathrm{adapt}}}{\partial \alpha}\right|_{\alpha = \alpha^*}, \qquad \text{for any } \beta \in (0, 1].$$

That is, the transition in opinion variance is steeper (more abrupt) when the UA is fixed than when it adapts to the population.

**H3 (Cultural Stratification — Heterogeneous Coupling).** Under B3 dynamics with $P(\alpha)$ having non-degenerate support, the asymptotic distribution of opinions $\{x_i(\infty)\}$ exhibits at least two distinct clusters for a set of parameters $(\varepsilon, \bar{\alpha})$ (mean of $P(\alpha)$) under which the homogeneous-coupling model (B1 or B2 with $\alpha = \bar{\alpha}$) produces a single cluster. Formally, agents with $\alpha_i > \alpha^*$ converge to a cluster near $x_{\mathrm{UA}}$, while agents with $\alpha_i < \alpha^*$ may form one or more separate clusters determined by their mutual confidence neighborhoods.

**H4 (Topology Irrelevance at High Coupling).** Let $\mathrm{Var}_G(\infty, \alpha)$ denote the asymptotic opinion variance under network topology $G$. For any two topologies $G_1, G_2$ (e.g., complete graph vs. scale-free),

$$\lim_{\alpha \to 1} \left| \mathrm{Var}_{G_1}(\infty, \alpha) - \mathrm{Var}_{G_2}(\infty, \alpha) \right| = 0.$$

Equivalently, the influence of peer network structure on collective outcomes vanishes as $\alpha \to 1$, because the UA's universally shared signal dominates the local topology.

---

### 3.5 Key Quantities Tracked in Simulation

To evaluate the hypotheses empirically across simulation runs, we monitor the following quantities at each time step $t$:

**Opinion variance.** The population-level spread of opinions is measured by

$$\mathrm{Var}(t) = \frac{1}{N} \sum_{i=1}^{N} \| x_i(t) - \bar{x}(t) \|^2.$$

This serves as the primary order parameter for detecting the phase transition in H1 and H2: a sharp drop in $\mathrm{Var}(\infty)$ as $\alpha$ crosses $\alpha^*$ signals consensus formation.

**Number of clusters.** For the bounded confidence variant, we count the number of distinct opinion clusters at stationarity. Two agents $i, j$ belong to the same cluster at time $t$ if $|x_i(t) - x_j(t)| < \varepsilon$. We report the cluster count $K(t)$ using a single-linkage criterion applied to the sorted opinion sequence.

**Convergence time.** We define the convergence time $T^*$ as the first time $t$ at which the system has effectively stabilized:

$$T^* = \min\left\{ t : \max_{i} |x_i(t+1) - x_i(t)| < \delta \right\},$$

for a small tolerance $\delta > 0$ (typically $\delta = 10^{-4}$). This quantity is used to compare how quickly consensus is achieved under fixed versus adaptive UA dynamics, and across topologies.

---

*This model specification defines the complete mathematical framework for simulation and analysis in subsequent sections. All experiments are conducted within this parameter space; specific values are reported in the Simulation section (Section 5).*
