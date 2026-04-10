# Analytical Results: Opinion Homogenization via a Universal Agent

---

## Overview

This section derives the key analytical properties of the Universal Agent (UA) model variants defined in the model specification. We proceed from the linear DeGroot setting — where exact results are obtainable — to the nonlinear bounded confidence setting, where results are necessarily heuristic or approximate. Throughout, we flag clearly which arguments are rigorous and which are informal.

---

## Proposition 1: Convergence of the Fixed-UA DeGroot System

**Statement.** For any $\alpha \in (0, 1]$ and any row-stochastic, strongly connected, aperiodic weight matrix $W$, the B1-DeGroot system converges geometrically to the unique fixed point $x_i^* = x_{\mathrm{UA}}$ for all $i$, regardless of initial conditions $\mathbf{x}(0)$. The convergence rate is $(1-\alpha)\lambda_2(W)$, where $\lambda_2(W) \in [0, 1)$ is the second-largest eigenvalue of $W$.

### Derivation

**Step 1: Matrix form.** The B1-DeGroot update is

$$x_i(t+1) = (1-\alpha)\sum_{j=1}^N w_{ij}\, x_j(t) + \alpha\, x_{\mathrm{UA}}.$$

Stacking into a column vector $\mathbf{x}(t) = (x_1(t), \ldots, x_N(t))^\top$ and writing $\mathbf{1} = (1, \ldots, 1)^\top$, this becomes

$$\mathbf{x}(t+1) = M\,\mathbf{x}(t) + \alpha\, x_{\mathrm{UA}}\,\mathbf{1}, \qquad M := (1-\alpha)W.$$

**Step 2: $M$ is a contraction.** The infinity norm of a matrix is $\|A\|_\infty = \max_i \sum_j |a_{ij}|$. Since $W$ is row-stochastic, $\|W\|_\infty = 1$. Therefore

$$\|M\|_\infty = (1-\alpha)\|W\|_\infty = (1-\alpha) < 1 \quad \text{for any } \alpha > 0.$$

By Banach's fixed-point theorem, the affine map $\mathbf{x} \mapsto M\mathbf{x} + \alpha x_{\mathrm{UA}}\mathbf{1}$ has a unique fixed point.

**Step 3: Identifying the fixed point.** Setting $\mathbf{x}^* = M\mathbf{x}^* + \alpha x_{\mathrm{UA}}\mathbf{1}$:

$$(I - M)\mathbf{x}^* = \alpha\, x_{\mathrm{UA}}\,\mathbf{1}.$$

Since $M = (1-\alpha)W$ and $W$ has the all-ones vector $\mathbf{1}$ as a right eigenvector with eigenvalue 1 (because $W$ is row-stochastic: $W\mathbf{1} = \mathbf{1}$), we can verify $\mathbf{x}^* = x_{\mathrm{UA}}\,\mathbf{1}$ is the solution:

$$(I - (1-\alpha)W)(x_{\mathrm{UA}}\,\mathbf{1}) = x_{\mathrm{UA}}\,\mathbf{1} - (1-\alpha)x_{\mathrm{UA}}\,\mathbf{1} = \alpha\, x_{\mathrm{UA}}\,\mathbf{1}. \quad \checkmark$$

Thus $x_i^* = x_{\mathrm{UA}}$ for all $i$.

**Step 4: Convergence rate.** Let $\mathbf{e}(t) = \mathbf{x}(t) - x_{\mathrm{UA}}\,\mathbf{1}$ denote the error vector. Subtracting the fixed-point equation from the update rule:

$$\mathbf{e}(t+1) = M\,\mathbf{e}(t) = (1-\alpha)W\,\mathbf{e}(t).$$

Applying recursively: $\mathbf{e}(t) = M^t\,\mathbf{e}(0) = (1-\alpha)^t W^t\,\mathbf{e}(0).$

Under the spectral decomposition of $W$, the dominant eigenvalue is 1 with eigenvector $\mathbf{1}$; all remaining eigenvalues satisfy $|\lambda_k(W)| \leq \lambda_2(W) < 1$ (by aperiodicity and strong connectivity). Because $\mathbf{e}(0) = \mathbf{x}(0) - x_{\mathrm{UA}}\mathbf{1}$ has zero component along $\mathbf{1}$ only if $\bar{x}(0) = x_{\mathrm{UA}}$ (otherwise it has a non-trivial component in the complementary subspace), the decay of $\|W^t \mathbf{e}(0)\|$ is governed by $\lambda_2(W)^t$. Combined with the $(1-\alpha)^t$ prefactor:

$$\|\mathbf{e}(t)\|_\infty \leq \left[(1-\alpha)\lambda_2(W)\right]^t \|\mathbf{e}(0)\|_\infty.$$

For the complete graph $w_{ij} = 1/N$, the matrix $W = \frac{1}{N}\mathbf{1}\mathbf{1}^\top$ has $\lambda_2(W) = 0$, so the peer-mixing term is already fully collapsed after one step, and the convergence is governed purely by the factor $(1-\alpha)^t$.

**Discussion.** The contraction argument via $\|M\|_\infty < 1$ is fully rigorous for any $\alpha > 0$. The spectral rate $(1-\alpha)\lambda_2(W)$ is the precise asymptotic decay rate of the error. Larger $\alpha$ simultaneously reduces both the contractivity factor and the influence of the second eigenvalue, yielding faster convergence on two fronts.

---

## Proposition 2: Discontinuous Phase Transition at $\alpha = 0$

**Statement.** The equilibrium of the B1-DeGroot system undergoes a discontinuous (first-order) transition at $\alpha = 0$: for $\alpha = 0$ the consensus is $\pi^\top \mathbf{x}(0)$ (a weighted average of initial opinions), while for any $\alpha > 0$, however small, the unique equilibrium is $x_{\mathrm{UA}}$, irrespective of initial conditions.

### Derivation

**Case $\alpha = 0$.** The system reduces to standard DeGroot: $\mathbf{x}(t+1) = W\mathbf{x}(t)$. By the classical DeGroot convergence theorem, for a strongly connected aperiodic $W$,

$$\lim_{t\to\infty}\mathbf{x}(t) = \mathbf{1}\,\pi^\top\mathbf{x}(0),$$

where $\pi$ is the stationary distribution of $W$ ($\pi^\top W = \pi^\top$, $\mathbf{1}^\top\pi = 1$). For the complete graph, $\pi = \frac{1}{N}\mathbf{1}$, so the consensus value is $\bar{x}(0) = \frac{1}{N}\sum_i x_i(0)$.

**Case $\alpha > 0$.** By Proposition 1, the unique fixed point is $x_{\mathrm{UA}}\,\mathbf{1}$, with geometric convergence. This holds for any $\alpha \in (0, 1]$, no matter how small.

**The discontinuity.** Define the asymptotic consensus value as a function $c(\alpha) = \lim_{t\to\infty} x_1(t)$ (all agents share the same limit). Then

$$c(0) = \pi^\top\mathbf{x}(0) = \bar{x}(0) \quad \text{and} \quad c(\alpha) = x_{\mathrm{UA}} \quad \text{for all } \alpha > 0.$$

The jump in $c(\alpha)$ at $\alpha = 0$ is $|x_{\mathrm{UA}} - \bar{x}(0)|$, which is generically nonzero. No matter how small $\alpha > 0$ is taken to be — even if $\alpha = 10^{-6}$ — the system will eventually converge to $x_{\mathrm{UA}}$; the only effect of small $\alpha$ is that convergence takes longer (proportional to $\log(1/\varepsilon_{\text{tol}}) / \log(1/(1-\alpha))$ steps to achieve tolerance $\varepsilon_{\text{tol}}$).

**Discussion.** This discontinuity is qualitatively important: it means that any nonzero coupling to the UA — even if agents assign it negligible weight — is sufficient to shift the consensus point entirely. The transition is not gradual but sharp. This is a direct consequence of the linearity of DeGroot dynamics: the fixed-point equation has a unique solution for $\alpha > 0$ that does not depend continuously on $\alpha$ through zero. In the bounded confidence case (Proposition 3), the situation is more nuanced because nonlinearity admits both qualitative changes in cluster structure and a genuinely smooth regime.

---

## Proposition 3: Critical Coupling in the HK-UA System

**Statement (Informal).** There exists a critical value $\alpha^*(\varepsilon) \in (0,1)$ such that for $\alpha > \alpha^*(\varepsilon)$ the B1-HK system converges to a single cluster centered near $x_{\mathrm{UA}}$, whereas for $\alpha < \alpha^*(\varepsilon)$, multiple opinion clusters persist.

### Derivation (Heuristic)

**Baseline fragmentation.** For $\alpha = 0$, the standard HK model on opinions drawn uniformly from $[0,1]$ produces approximately $K \approx \lfloor 1/(2\varepsilon) \rfloor$ clusters at stationarity, each spanning a region of width roughly $2\varepsilon$. This is a known heuristic for the uniform initial condition; the exact count depends on $N$ and the specific realization.

**Two-cluster reduction.** To make progress analytically, consider the simplest non-trivial fragmented configuration: two clusters at positions $\mu_L$ (left) and $\mu_R$ (right), with $\mu_R - \mu_L > \varepsilon$ so they do not interact under standard HK. With the UA term, the effective update for an agent in the left cluster becomes

$$x_i(t+1) = (1-\alpha)\,\mu_L + \alpha\, x_{\mathrm{UA}}.$$

After one step, this agent's position shifts to $\mu_L + \alpha(x_{\mathrm{UA}} - \mu_L)$. Similarly, right-cluster agents shift to $\mu_R + \alpha(x_{\mathrm{UA}} - \mu_R)$.

**Bridging condition.** After the UA-driven shift, the two cluster positions become

$$\mu_L' = \mu_L + \alpha(x_{\mathrm{UA}} - \mu_L) = (1-\alpha)\mu_L + \alpha x_{\mathrm{UA}},$$
$$\mu_R' = (1-\alpha)\mu_R + \alpha x_{\mathrm{UA}}.$$

Their separation is $|\mu_R' - \mu_L'| = (1-\alpha)|\mu_R - \mu_L|$. The clusters will enter each other's confidence range — and thus begin to merge — when this separation falls below $\varepsilon$:

$$(1-\alpha)|\mu_R - \mu_L| < \varepsilon.$$

Solving for $\alpha$:

$$\alpha > 1 - \frac{\varepsilon}{|\mu_R - \mu_L|} =: \alpha^*(\varepsilon, \mu_L, \mu_R).$$

For the generic case of $K$ initially well-separated clusters, the dominant barrier is the hardest pair to bridge — typically the two most extreme clusters. If the initial opinion distribution is $\mathcal{U}[0,1]$ and $x_{\mathrm{UA}}$ lies near the center, the two extreme clusters sit near positions roughly $\varepsilon$ and $1 - \varepsilon$, giving $|\mu_R - \mu_L| \approx 1 - 2\varepsilon$. The heuristic critical coupling is then

$$\boxed{\alpha^*(\varepsilon) \approx 1 - \frac{\varepsilon}{1 - 2\varepsilon} = \frac{1 - 3\varepsilon}{1 - 2\varepsilon}.}$$

For instance, at $\varepsilon = 0.3$ (the default parameter), this gives $\alpha^* \approx (0.1)/(0.4) = 0.25$.

**Remark on the derivation.** The formula above is a first-step heuristic: it assumes clusters move rigidly and ignores the dynamics within each cluster and the possibility of sequential merging events (where bridging one pair creates new interactions). The true $\alpha^*$ may be smaller in practice because a cascade of partial merges can bring distant clusters together at lower $\alpha$. The simulation results are expected to validate and refine this estimate.

**Discussion.** Crucially, $\alpha^*(\varepsilon) \to 1$ as $\varepsilon \to 0$ (for fixed cluster separation): extreme polarization requires near-complete UA dominance to resolve. Conversely, $\alpha^*(\varepsilon) \to 0$ as $\varepsilon \to |\mu_R - \mu_L|$ (clusters are nearly within range anyway). This formula provides the design principle: to merge opinion clusters of separation $\Delta = |\mu_R - \mu_L|$ using a UA, one needs $\alpha > 1 - \varepsilon/\Delta$.

---

## Proposition 4: Co-evolutionary Fixed Point of the Adaptive UA Model

**Statement.** In the B2-DeGroot system, the asymptotic consensus $x^*$ is a weighted combination of the initial UA opinion $x_{\mathrm{UA}}(0)$ and the initial population mean $\bar{x}(0)$. In the limit $\alpha/\beta \to \infty$, $x^* \to x_{\mathrm{UA}}(0)$; in the limit $\alpha/\beta \to 0$, $x^* \to \bar{x}(0)$.

### Derivation

**Step 1: Fixed-point structure.** At a co-evolutionary fixed point $(\mathbf{x}^*, x_{\mathrm{UA}}^*)$, the agent equation gives $x_i^* = x_{\mathrm{UA}}^*$ for all $i$ (by the argument in Proposition 1, since $\alpha > 0$), so $\bar{x}^* = x_{\mathrm{UA}}^*$. The UA equation becomes

$$x_{\mathrm{UA}}^* = (1-\beta)x_{\mathrm{UA}}^* + \beta\,\bar{x}^* = (1-\beta)x_{\mathrm{UA}}^* + \beta\, x_{\mathrm{UA}}^*,$$

which is satisfied for any $x_{\mathrm{UA}}^*$. Thus the system has a one-dimensional family of fixed points, and the reached equilibrium depends on initial conditions and the trajectory.

**Step 2: Deriving the attractor via a conserved-like quantity.** Consider the complete graph ($W = \frac{1}{N}\mathbf{1}\mathbf{1}^\top$). In this case all agents immediately share the same value after one step: $x_i(t) = \bar{x}(t)$ for $t \geq 1$. Define $s(t) = \bar{x}(t)$ and $u(t) = x_{\mathrm{UA}}(t)$. The coupled system for $t \geq 1$ reduces to

$$s(t+1) = (1-\alpha)\,s(t) + \alpha\, u(t),$$
$$u(t+1) = (1-\beta)\,u(t) + \beta\, s(t).$$

This is a linear two-dimensional system. Writing $\mathbf{z}(t) = (s(t), u(t))^\top$:

$$\mathbf{z}(t+1) = A\,\mathbf{z}(t), \qquad A = \begin{pmatrix} 1-\alpha & \alpha \\ \beta & 1-\beta \end{pmatrix}.$$

**Step 3: Spectral analysis.** The eigenvalues of $A$ are found from $\det(A - \lambda I) = 0$:

$$\lambda^2 - (2 - \alpha - \beta)\lambda + (1 - \alpha)(1-\beta) - \alpha\beta = 0.$$

Simplifying the constant term: $(1-\alpha-\beta+\alpha\beta) - \alpha\beta = 1-\alpha-\beta$. So

$$\lambda^2 - (2-\alpha-\beta)\lambda + (1-\alpha-\beta) = 0,$$

which factors as $(\lambda - 1)(\lambda - (1-\alpha-\beta)) = 0$. The two eigenvalues are

$$\lambda_1 = 1, \qquad \lambda_2 = 1 - \alpha - \beta.$$

The eigenvalue $\lambda_1 = 1$ corresponds to the conserved direction (the long-run attractor), and $\lambda_2 = 1 - \alpha - \beta$ governs the transient decay (stable when $\alpha + \beta < 2$, which holds for $\alpha, \beta \in (0,1)$; the decay is faster when $\alpha + \beta$ is larger).

**Step 4: Eigenvector for $\lambda_1 = 1$.** Solving $(A - I)\mathbf{v} = 0$:

$$\begin{pmatrix} -\alpha & \alpha \\ \beta & -\beta \end{pmatrix}\mathbf{v} = 0 \implies v_1 = v_2.$$

So the $\lambda_1 = 1$ eigenvector is $\mathbf{v}_1 = (1, 1)^\top$: when $s = u$, the system is at rest.

**Step 5: Identifying the conserved quantity.** The left eigenvector for $\lambda_1 = 1$ gives the quantity that is preserved along trajectories. Solving $\mathbf{w}^\top A = \mathbf{w}^\top$:

$$\begin{pmatrix} w_1 & w_2 \end{pmatrix} \begin{pmatrix} 1-\alpha & \alpha \\ \beta & 1-\beta \end{pmatrix} = \begin{pmatrix} w_1 & w_2 \end{pmatrix},$$

which gives $-\alpha w_1 + \beta w_2 = 0$, so $w_1/w_2 = \beta/\alpha$. Normalizing to $w_1 + w_2 = 1$:

$$\mathbf{w} = \left(\frac{\beta}{\alpha+\beta},\ \frac{\alpha}{\alpha+\beta}\right).$$

The conserved quantity along trajectories is

$$Q = \mathbf{w}^\top\mathbf{z}(t) = \frac{\beta\, s(t) + \alpha\, u(t)}{\alpha + \beta} = \text{constant for all } t.$$

**Step 6: Deriving $x^*$.** At the fixed point $s^* = u^* = x^*$, so $Q = x^*$. Evaluating $Q$ at $t = 1$ (using $s(1) = (1-\alpha)\bar{x}(0) + \alpha x_{\mathrm{UA}}(0)$ and $u(1) = (1-\beta)x_{\mathrm{UA}}(0) + \beta\bar{x}(0)$):

$$Q = \frac{\beta\,[(1-\alpha)\bar{x}(0) + \alpha x_{\mathrm{UA}}(0)] + \alpha\,[(1-\beta)x_{\mathrm{UA}}(0) + \beta\bar{x}(0)]}{\alpha+\beta}.$$

Expanding:

$$Q = \frac{(\beta - \alpha\beta + \alpha\beta)\bar{x}(0) + (\alpha\beta + \alpha - \alpha\beta)x_{\mathrm{UA}}(0)}{\alpha+\beta} = \frac{\beta\,\bar{x}(0) + \alpha\, x_{\mathrm{UA}}(0)}{\alpha+\beta}.$$

Equivalently, evaluating directly at $t = 0$: $Q = \frac{\beta\, \bar{x}(0) + \alpha\, x_{\mathrm{UA}}(0)}{\alpha+\beta}$, confirming the conserved quantity. Therefore the unique attractor is

$$\boxed{x^* = \frac{\beta\,\bar{x}(0) + \alpha\, x_{\mathrm{UA}}(0)}{\alpha + \beta}.}$$

**Step 7: Limiting cases.**
- When $\alpha \gg \beta$: $x^* \approx x_{\mathrm{UA}}(0)$. The UA influences agents rapidly before it can adapt, anchoring the consensus at the UA's initial position.
- When $\beta \gg \alpha$: $x^* \approx \bar{x}(0)$. The UA tracks the population so quickly that its influence is neutralized, and the system behaves like Stage A dynamics.
- The formula interpolates smoothly: $x^*$ is a weighted harmonic mean of $x_{\mathrm{UA}}(0)$ and $\bar{x}(0)$ with weights $\alpha$ and $\beta$ respectively.

**Discussion.** The derivation is exact for the complete graph. For general strongly connected aperiodic $W$, the same conserved quantity holds at the level of the mean $\bar{x}(t)$ (since $W\mathbf{1} = \mathbf{1}$), so the formula for $x^*$ applies in full generality. The transient dynamics toward this fixed point are governed by the rate $|\lambda_2|^t = (1-\alpha-\beta)^t$; systems with large $\alpha + \beta$ converge fastest.

---

## Proposition 5: Heterogeneous Coupling — Transient Stratification

**Statement.** Under B3 dynamics with heterogeneous couplings $\{\alpha_i\}$, the population exhibits transient stratification: high-$\alpha_i$ agents converge to $x_{\mathrm{UA}}$ on a time scale $\tau_i \sim 1/\alpha_i$, while low-$\alpha_i$ agents remain in peer-driven clusters for much longer. This creates a cascading convergence: early-converging agents join the UA cluster, which then acts as a larger attracting mass that pulls moderate-$\alpha$ agents into the UA cluster sooner than their own $\alpha_i$ would suggest.

### Derivation

**Step 1: Individual convergence time.** For agent $i$ with coupling $\alpha_i > 0$ in an otherwise fixed environment (treating peer opinions as fixed at their long-run values), the error decays as $|e_i(t)| \sim (1-\alpha_i)^t |e_i(0)|$. The number of steps to reduce the error to tolerance $\delta$ is

$$T_i^*(\alpha_i) = \left\lceil \frac{\log(|e_i(0)|/\delta)}{\log(1/(1-\alpha_i))} \right\rceil \approx \frac{\log(|e_i(0)|/\delta)}{\alpha_i},$$

where the approximation uses $\log(1/(1-\alpha_i)) \approx \alpha_i$ for small $\alpha_i$. Thus $T_i^* \propto 1/\alpha_i$: agents with small $\alpha_i$ take much longer to converge.

**Step 2: Stratification in DeGroot.** Even when all $\alpha_i > 0$, the heterogeneity in convergence times creates a multi-speed process. At any time $t$, the population partitions into a "converged core" $\mathcal{C}(t) = \{i : |x_i(t) - x_{\mathrm{UA}}| < \delta\}$ and a "peripheral" set $\mathcal{P}(t) = \{i : |x_i(t) - x_{\mathrm{UA}}| \geq \delta\}$. Agents in $\mathcal{C}(t)$ feed back into the peer term for agents in $\mathcal{P}(t)$, effectively increasing the latter agents' attraction toward $x_{\mathrm{UA}}$ beyond what their own $\alpha_i$ would produce. This cascade means that the effective convergence time of peripheral agents is shorter than $T_i^*(\alpha_i)$ once the core has formed.

**Step 3: Stratification in bounded confidence.** In the B3-HK model, there is a sharper qualitative distinction. Let $\alpha^*(\varepsilon)$ be the critical coupling derived in Proposition 3 for the homogeneous case. Then:

- Agents with $\alpha_i > \alpha^*(\varepsilon)$ experience sufficient UA influence to be pulled across confidence boundaries. They exit their local cluster and join the UA cluster within $O(1/\alpha_i)$ steps.
- Agents with $\alpha_i < \alpha^*(\varepsilon)$ remain in their local cluster in early time steps. However, as $\mathcal{C}(t)$ grows, these peripheral agents eventually interact with core agents (who have moved close to $x_{\mathrm{UA}}$) and may be recruited into the core by the combined influence of peers and UA.

The cascade is formally a monotone process: once an agent joins $\mathcal{C}(t)$, it does not leave (the dynamics are dissipative). The cascade terminates when either all agents are in $\mathcal{C}$ (full consensus near $x_{\mathrm{UA}}$) or the remaining peripheral agents are isolated from $\mathcal{C}$ in the confidence graph (persistent fragmentation).

**Step 4: Equilibrium for the heterogeneous DeGroot case.** Regardless of the distribution of $\{\alpha_i\}$, as long as all $\alpha_i > 0$ and $W$ is strongly connected and aperiodic, the unique fixed point remains $x_i^* = x_{\mathrm{UA}}$ for all $i$. This follows because the effective update operator $M$ with entries $m_{ij} = (1-\alpha_i)w_{ij}$ has row sums $\sum_j m_{ij} = 1 - \alpha_i < 1$ for all $i$, making the whole system contractive even without the common $\|M\|_\infty$ bound. The system is thus a (non-homogeneous) contraction and converges globally.

**Discussion.** The key empirical implication is that heterogeneous coupling produces a dynamic signature distinguishable from homogeneous coupling: the convergence profile will be multi-modal in time, with a fast-converging cohort and a slow-converging tail. Measuring convergence time distributions across agents provides a diagnostic for the heterogeneity of $P(\alpha)$.

---

## Open Questions

The following points require more careful analysis or simulation validation before they can be treated as established results.

1. **Finite-time vs. asymptotic behavior in Proposition 3.** The critical coupling $\alpha^*(\varepsilon)$ was derived under a one-step, two-cluster simplification. A rigorous treatment would require tracking the full HK dynamics over multiple steps, accounting for the sequential nature of cluster merges and the possibility that $\alpha^*$ varies with the number of clusters $K$ and the position of $x_{\mathrm{UA}}$ relative to the cluster array. Simulation is needed to validate the heuristic formula $\alpha^*(\varepsilon) \approx (1-3\varepsilon)/(1-2\varepsilon)$.

2. **Exact formula for $x^*$ under non-complete topologies (Proposition 4).** For general $W$, the mean $\bar{x}(t)$ follows the same conserved-quantity argument, so the formula $x^* = (\beta\bar{x}(0) + \alpha x_{\mathrm{UA}}(0))/(\alpha+\beta)$ holds at the level of the population mean. But individual agent opinions may not concentrate perfectly at $x^*$ if $W$ has asymmetric stationary distribution $\pi \neq \frac{1}{N}\mathbf{1}$. In that case, each agent's limit satisfies $x_i^* = x^*$ by Proposition 1 (since $\alpha > 0$), so the formula extends without modification — but this should be verified carefully in the adaptive case where $x_{\mathrm{UA}}$ is also evolving.

3. **Cascade dynamics in Proposition 5.** The cascade argument is qualitatively described but not formally characterized. A mean-field approximation (treating the growing core as a single "super-agent" with growing influence) could yield quantitative predictions for the cascade time scale, testable in simulation.

4. **Validity of the discontinuity in Proposition 2 under noise.** The discontinuity at $\alpha = 0$ is a statement about deterministic dynamics. If agents update with bounded noise (e.g., $x_i(t+1) = (1-\alpha)W\mathbf{x}(t) + \alpha x_{\mathrm{UA}} + \xi_i(t)$ with $\xi_i$ i.i.d. noise), the distinction between $\alpha = 0$ and small $\alpha > 0$ may be smoothed out; the transition becomes approximately discontinuous only for $\alpha$ larger than the noise scale.

5. **Multidimensional opinions ($d > 1$).** All derivations in this section assume $d = 1$. For $d > 1$, the DeGroot results extend dimension-by-dimension without modification (linearity decouples the coordinates). For bounded confidence with $d > 1$, the confidence neighborhood $\mathcal{N}_i(t)$ is defined by a Euclidean ball, and the cluster geometry is more complex; the heuristic formula for $\alpha^*$ does not directly generalize.

6. **Influence of $x_{\mathrm{UA}}$ position on $\alpha^*$.** Proposition 3 implicitly assumes $x_{\mathrm{UA}}$ lies between the extreme clusters (or near them). If $x_{\mathrm{UA}}$ lies far outside $[0,1]$ (the initial opinion support), the UA may pull all agents toward an external point more easily, potentially lowering $\alpha^*$. Conversely, if $x_{\mathrm{UA}}$ coincides with one cluster's position, it reinforces that cluster and may actually increase $\alpha^*$ for bridging the other. A full sensitivity analysis with respect to $x_{\mathrm{UA}}$ position is deferred to simulation.
