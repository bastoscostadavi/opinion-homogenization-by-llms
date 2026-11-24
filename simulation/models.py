"""
models.py — Core model classes for opinion dynamics simulation.

Implements:
  - DeGrootModel: standard linear consensus dynamics
  - DeGrootWithUA: DeGroot extended with a Universal Agent (UA)
  - BoundedConfidenceWithUA: Hegselmann-Krause model with optional UA
  - MultipleUAModel: k Universal Agents, each assigned to N//k agents
    (consistent partition); all UAs update on global population mean.

Helper functions:
  - compute_variance(x)
  - count_clusters(x, tol)
  - convergence_time(history, tol)
"""

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_variance(x: np.ndarray) -> float:
    """Compute population opinion variance.

    Var(t) = (1/N) * sum_i (x_i - x_bar)^2

    Parameters
    ----------
    x : array of shape (N,) or (N, d)
        Opinion vector at a single time step.

    Returns
    -------
    float
        Scalar variance value.
    """
    x = np.atleast_1d(x)
    mean = x.mean(axis=0)
    return float(np.mean(np.sum((x - mean) ** 2, axis=-1) if x.ndim > 1 else (x - mean) ** 2))


def count_clusters(x: np.ndarray, tol: float = 0.05) -> int:
    """Count distinct opinion clusters using single-linkage on sorted opinions.

    Two agents belong to the same cluster if |x_i - x_j| < tol for consecutive
    elements in the sorted sequence (single-linkage criterion).

    Parameters
    ----------
    x : array of shape (N,)
        Opinion values at a single time step.
    tol : float
        Tolerance for merging opinions into the same cluster.

    Returns
    -------
    int
        Number of distinct clusters.
    """
    x_sorted = np.sort(x.ravel())
    if len(x_sorted) == 0:
        return 0
    diffs = np.diff(x_sorted)
    # A new cluster starts wherever the gap exceeds tol
    n_clusters = 1 + int(np.sum(diffs >= tol))
    return n_clusters


def convergence_time(history: np.ndarray, tol: float = 1e-4) -> int:
    """Return the first time t at which max agent change falls below tol.

    T* = min{ t : max_i |x_i(t+1) - x_i(t)| < tol }

    Parameters
    ----------
    history : array of shape (T+1, N) or (T+1, N, d)
        Opinion trajectories returned by a model's run() method.
    tol : float
        Convergence tolerance delta.

    Returns
    -------
    int
        Convergence time, or len(history)-1 if never converged.
    """
    T = len(history) - 1
    for t in range(T):
        max_change = np.max(np.abs(history[t + 1] - history[t]))
        if max_change < tol:
            return t
    return T


# ---------------------------------------------------------------------------
# Weight matrix / network helpers
# ---------------------------------------------------------------------------

def _build_weight_matrix(N: int, topology: str, rng: np.random.Generator) -> np.ndarray:
    """Build a row-stochastic weight matrix for the given topology.

    Parameters
    ----------
    N : int
        Number of agents.
    topology : str
        One of 'complete', 'small_world', 'scale_free'.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    W : ndarray of shape (N, N)
        Row-stochastic weight matrix.
    """
    if topology == 'complete':
        W = np.full((N, N), 1.0 / N)

    elif topology == 'small_world':
        # Watts-Strogatz: k=4 nearest neighbours, rewiring probability p=0.1
        seed_int = int(rng.integers(0, 2**31))
        G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=seed_int)
        A = nx.to_numpy_array(G)
        # Add self-loops so every row is non-zero
        A = A + np.eye(N)
        row_sums = A.sum(axis=1, keepdims=True)
        W = A / row_sums

    elif topology == 'scale_free':
        # Barabasi-Albert: m=2 edges per new node
        seed_int = int(rng.integers(0, 2**31))
        G = nx.barabasi_albert_graph(N, m=2, seed=seed_int)
        A = nx.to_numpy_array(G)
        A = A + np.eye(N)
        row_sums = A.sum(axis=1, keepdims=True)
        W = A / row_sums

    else:
        raise ValueError(f"Unknown topology '{topology}'. Choose from "
                         "'complete', 'small_world', 'scale_free'.")
    return W


# ---------------------------------------------------------------------------
# DeGroot Model
# ---------------------------------------------------------------------------

class DeGrootModel:
    """Standard DeGroot linear consensus model.

    x_i(t+1) = sum_j w_ij * x_j(t)   [in matrix form: x(t+1) = W * x(t)]

    Parameters
    ----------
    N : int
        Number of agents.
    d : int
        Opinion dimensionality (default 1).
    topology : str
        Network structure: 'complete', 'small_world', or 'scale_free'.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, N: int, d: int = 1, topology: str = 'complete', seed=None):
        self.N = N
        self.d = d
        self.topology = topology
        self._rng = np.random.default_rng(seed)

        self.W = _build_weight_matrix(N, topology, self._rng)
        self.x: np.ndarray = np.zeros((N, d) if d > 1 else N)
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, x0=None):
        """Set initial opinions.

        Parameters
        ----------
        x0 : array-like or None
            If None, opinions are drawn uniformly from [0, 1].
        """
        if x0 is not None:
            self.x = np.array(x0, dtype=float)
        else:
            if self.d == 1:
                self.x = self._rng.uniform(0.0, 1.0, size=self.N)
            else:
                self.x = self._rng.uniform(0.0, 1.0, size=(self.N, self.d))
        self._initialized = True

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self):
        """Perform one DeGroot update step (in-place)."""
        if self.d == 1:
            self.x = self.W @ self.x
        else:
            self.x = self.W @ self.x  # (N,N) @ (N,d) -> (N,d)

    def run(self, T: int) -> np.ndarray:
        """Run the model for T steps.

        Parameters
        ----------
        T : int
            Number of time steps.

        Returns
        -------
        history : ndarray of shape (T+1, N) [or (T+1, N, d) if d>1]
            Opinion trajectories including the initial state.
        """
        if not self._initialized:
            self.initialize()

        if self.d == 1:
            history = np.empty((T + 1, self.N))
        else:
            history = np.empty((T + 1, self.N, self.d))

        history[0] = self.x.copy()
        for t in range(T):
            self.step()
            history[t + 1] = self.x.copy()
        return history


# ---------------------------------------------------------------------------
# DeGroot + Universal Agent
# ---------------------------------------------------------------------------

class DeGrootWithUA(DeGrootModel):
    """DeGroot model augmented with a Universal Agent (UA).

    Update rule:
        x_i(t+1) = (1 - alpha) * sum_j w_ij * x_j(t)  +  alpha * x_UA(t)

    UA variants:
        'fixed'        : x_UA stays constant throughout.
        'adaptive'     : x_UA(t+1) = (1-beta)*x_UA(t) + beta*mean(x(t))
        'heterogeneous': each agent has its own alpha_i (from alpha_dist)

    Parameters
    ----------
    N : int
    d : int
    alpha : float
        Uniform coupling weight in [0, 1].  Ignored for 'heterogeneous'.
    ua_type : str
        'fixed', 'adaptive', or 'heterogeneous'.
    ua_init : float
        Initial (or fixed) UA opinion.
    beta : float
        UA learning rate (used only for 'adaptive').
    topology : str
    alpha_dist : array-like or None
        Per-agent alpha values for 'heterogeneous' mode.
    seed : int or None
    """

    def __init__(
        self,
        N: int,
        d: int = 1,
        alpha: float = 0.3,
        ua_type: str = 'fixed',
        ua_init: float = 0.5,
        beta: float = 0.1,
        topology: str = 'complete',
        alpha_dist=None,
        seed=None,
    ):
        super().__init__(N=N, d=d, topology=topology, seed=seed)
        self.alpha = float(alpha)
        self.ua_type = ua_type
        self.ua_init = float(ua_init)
        self.beta = float(beta)

        if ua_type == 'heterogeneous':
            if alpha_dist is None:
                raise ValueError("alpha_dist must be provided for 'heterogeneous' ua_type.")
            self.alpha_arr = np.array(alpha_dist, dtype=float)
            if self.alpha_arr.shape != (N,):
                raise ValueError(f"alpha_dist must have shape ({N},), got {self.alpha_arr.shape}.")
        else:
            self.alpha_arr = None

        self.x_UA: float = ua_init
        self.x_UA_history: list = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, x0=None):
        """Initialize agent opinions and reset UA to ua_init."""
        super().initialize(x0=x0)
        self.x_UA = self.ua_init
        self.x_UA_history = [self.x_UA]

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self):
        """One DeGroot+UA update step."""
        # Peer term: W @ x
        peer = self.W @ self.x  # shape (N,) or (N, d)

        if self.ua_type == 'heterogeneous':
            # Vectorized: x_i(t+1) = (1-alpha_i)*peer_i + alpha_i*x_UA
            a = self.alpha_arr  # (N,)
            if self.d == 1:
                self.x = (1.0 - a) * peer + a * self.x_UA
            else:
                self.x = (1.0 - a[:, None]) * peer + a[:, None] * self.x_UA
        else:
            if self.d == 1:
                self.x = (1.0 - self.alpha) * peer + self.alpha * self.x_UA
            else:
                self.x = (1.0 - self.alpha) * peer + self.alpha * self.x_UA

        # Update UA if adaptive
        if self.ua_type == 'adaptive':
            pop_mean = float(self.x.mean())
            self.x_UA = (1.0 - self.beta) * self.x_UA + self.beta * pop_mean

        self.x_UA_history.append(self.x_UA)

    def run(self, T: int) -> np.ndarray:
        """Run for T steps and return opinion history.

        Also populates self.x_UA_history.
        """
        if not self._initialized:
            self.initialize()

        if self.d == 1:
            history = np.empty((T + 1, self.N))
        else:
            history = np.empty((T + 1, self.N, self.d))

        history[0] = self.x.copy()
        for t in range(T):
            self.step()
            history[t + 1] = self.x.copy()
        return history


# ---------------------------------------------------------------------------
# Bounded Confidence (HK) + Universal Agent
# ---------------------------------------------------------------------------

class BoundedConfidenceWithUA:
    """Hegselmann-Krause bounded confidence model with an optional Universal Agent.

    Synchronous update rule:
        N_i(t) = { j : |x_j(t) - x_i(t)| < epsilon }
        peer_i(t) = mean_{j in N_i(t)} x_j(t)
        x_i(t+1) = (1 - alpha) * peer_i(t) + alpha * x_UA(t)

    When alpha=0 this is the pure HK model.

    UA variants: 'fixed', 'adaptive', 'heterogeneous' — same semantics as
    DeGrootWithUA.

    Parameters
    ----------
    N : int
    epsilon : float
        Confidence bound.
    alpha : float
        Uniform coupling weight (ignored for 'heterogeneous').
    ua_type : str
        'fixed', 'adaptive', or 'heterogeneous'.
    ua_init : float
        Initial UA opinion.
    beta : float
        UA learning rate (adaptive only).
    topology : str
        Network topology (affects which peers are visible).  Currently the
        confidence-neighbourhood already restricts interactions; topology
        changes the *initial* set of reachable peers before confidence
        filtering.  For simplicity this parameter is accepted but the HK
        update uses the full N-agent confidence window regardless.
    seed : int or None
    """

    def __init__(
        self,
        N: int,
        epsilon: float = 0.3,
        alpha: float = 0.0,
        ua_type: str = 'fixed',
        ua_init: float = 0.5,
        beta: float = 0.1,
        topology: str = 'complete',
        seed=None,
    ):
        self.N = N
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.ua_type = ua_type
        self.ua_init = float(ua_init)
        self.beta = float(beta)
        self.topology = topology
        self._rng = np.random.default_rng(seed)

        self.x: np.ndarray = np.zeros(N)
        self.x_UA: float = ua_init
        self.x_UA_history: list = []
        self._initialized = False
        self.alpha_arr = None  # populated in initialize for heterogeneous

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, x0=None, alpha_dist=None):
        """Initialize agent opinions and UA state.

        Parameters
        ----------
        x0 : array-like or None
        alpha_dist : array-like or None
            Per-agent alphas for 'heterogeneous' mode.
        """
        if x0 is not None:
            self.x = np.array(x0, dtype=float)
        else:
            self.x = self._rng.uniform(0.0, 1.0, size=self.N)

        self.x_UA = self.ua_init
        self.x_UA_history = [self.x_UA]

        if self.ua_type == 'heterogeneous':
            if alpha_dist is not None:
                self.alpha_arr = np.array(alpha_dist, dtype=float)
            elif self.alpha_arr is None:
                raise ValueError("alpha_dist must be provided for 'heterogeneous' mode.")

        self._initialized = True

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self):
        """One synchronous HK update step with optional UA coupling."""
        x = self.x
        eps = self.epsilon

        # Compute pairwise absolute differences — shape (N, N)
        diff = np.abs(x[:, None] - x[None, :])  # x_i - x_j for all (i,j)
        in_conf = diff < eps  # boolean mask (N, N); diagonal always True

        # Peer average: row-mean over confidence neighbours
        counts = in_conf.sum(axis=1)  # (N,)
        peer = (in_conf * x[None, :]).sum(axis=1) / counts  # (N,)

        # Apply UA coupling
        if self.ua_type == 'heterogeneous':
            a = self.alpha_arr
            self.x = (1.0 - a) * peer + a * self.x_UA
        else:
            self.x = (1.0 - self.alpha) * peer + self.alpha * self.x_UA

        # Update UA if adaptive
        if self.ua_type == 'adaptive':
            pop_mean = float(self.x.mean())
            self.x_UA = (1.0 - self.beta) * self.x_UA + self.beta * pop_mean

        self.x_UA_history.append(self.x_UA)

    def run(self, T: int) -> np.ndarray:
        """Run for T steps and return opinion history of shape (T+1, N)."""
        if not self._initialized:
            self.initialize()

        history = np.empty((T + 1, self.N))
        history[0] = self.x.copy()
        for t in range(T):
            self.step()
            history[t + 1] = self.x.copy()
        return history


# ---------------------------------------------------------------------------
# Multiple Universal Agents (k UAs with consistent partition)
# ---------------------------------------------------------------------------

class MultipleUAModel:
    """k Universal Agents, each assigned to N//k agents (consistent partition).

    The population is split into k equal groups.  Agent i in group j interacts
    ONLY with UA_j.  All UAs update using the mean of ALL N agents (not just
    their partition), so under the adaptive rule the UAs eventually converge
    to the same value — but if β is small and α is large, agents lock onto
    their respective UA before the UAs merge, producing transient k-cluster
    formation.

    Agent update (B1-HK variant):
        x_i(t+1) = (1-α) * HK_peer_mean(within-ε neighbours) + α * x_UA_j(t)

    UA update:
        'fixed'   : x_UA_j unchanged
        'adaptive': x_UA_j(t+1) = (1-β)*x_UA_j(t) + β*target_j(t)

    For adaptive UAs, the adaptation target can be:
        'global': target_j(t) = global population mean for all j
        'local' : target_j(t) = mean of partition j

    Parameters
    ----------
    N : int
        Number of agents (should be divisible by k; if not, last group is
        smaller by at most k-1).
    k : int
        Number of Universal Agents / groups.
    epsilon : float
        HK bounded-confidence radius.
    alpha : float
        Coupling weight in [0, 1] of agents toward their assigned UA.
    beta : float
        UA learning rate (used only for 'adaptive').
    ua_type : str
        'fixed' or 'adaptive'.
    ua_init : str or array-like
        'spread' — UAs initialised at k evenly spaced positions in (0, 1).
        'same'   — all UAs at 0.8 (reduces to effective single-UA).
        array    — explicit initial values of length k.
    seed : int or None
    """

    def __init__(
        self,
        N: int = 100,
        k: int = 4,
        epsilon: float = 0.1,
        alpha: float = 0.3,
        beta: float = 0.05,
        ua_type: str = 'adaptive',
        ua_init='spread',
        adaptation_scope: str = 'global',
        seed=None,
    ):
        self.N = N
        self.k = k
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ua_type = ua_type
        self.ua_init = ua_init
        self.adaptation_scope = adaptation_scope
        self._rng = np.random.default_rng(seed)

        # Partition: group[j] = list of agent indices belonging to UA j
        self.groups = []
        base = N // k
        remainder = N % k
        start = 0
        for j in range(k):
            size = base + (1 if j < remainder else 0)
            self.groups.append(np.arange(start, start + size))
            start += size

        self.x: np.ndarray = np.zeros(N)
        self.x_UA: np.ndarray = np.zeros(k)
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, x0=None, ua0=None):
        """Set initial opinions and UA positions.

        Parameters
        ----------
        x0 : array-like or None
            Agent initial opinions.  If None, drawn uniformly from [0, 1].
        ua0 : array-like or None
            Initial UA opinions of length k.  If None, uses self.ua_init
            specification ('spread', 'same', or array).
        """
        if x0 is not None:
            self.x = np.array(x0, dtype=float)
        else:
            self.x = self._rng.uniform(0.0, 1.0, size=self.N)

        if ua0 is not None:
            self.x_UA = np.array(ua0, dtype=float)
        elif isinstance(self.ua_init, str):
            if self.ua_init == 'spread':
                # k evenly spaced positions in [0, 1], e.g. 0.1, 0.35, 0.65, 0.9
                self.x_UA = np.linspace(0.0, 1.0, self.k + 2)[1:-1]
            elif self.ua_init == 'same':
                self.x_UA = np.full(self.k, 0.8)
            else:
                raise ValueError(f"Unknown ua_init string '{self.ua_init}'. "
                                 "Use 'spread', 'same', or pass an array.")
        else:
            self.x_UA = np.array(self.ua_init, dtype=float)
            if self.x_UA.shape != (self.k,):
                raise ValueError(f"ua_init array must have length k={self.k}.")

        # Snapshot histories
        self._ua_history = [self.x_UA.copy()]
        self._initialized = True

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self):
        """One synchronous update step for all agents and all UAs."""
        x = self.x
        eps = self.epsilon
        x_UA = self.x_UA

        # --- Agent update ---
        new_x = np.empty(self.N)
        for j, group_idx in enumerate(self.groups):
            ua_j = x_UA[j]
            for i in group_idx:
                xi = x[i]
                # HK peer mean: within-ε neighbours (includes self)
                neighbours = x[np.abs(x - xi) < eps]
                peer_mean = neighbours.mean()
                new_x[i] = (1.0 - self.alpha) * peer_mean + self.alpha * ua_j

        self.x = new_x

        # --- UA update ---
        if self.ua_type == 'adaptive':
            if self.adaptation_scope == 'global':
                targets = np.full(self.k, float(self.x.mean()))
            elif self.adaptation_scope == 'local':
                targets = np.array([float(self.x[idx].mean()) for idx in self.groups])
            else:
                raise ValueError("adaptation_scope must be 'global' or 'local'.")
            self.x_UA = (1.0 - self.beta) * x_UA + self.beta * targets
        # 'fixed': x_UA unchanged

        self._ua_history.append(self.x_UA.copy())

    def run(self, T: int):
        """Run for T steps.

        Parameters
        ----------
        T : int
            Number of time steps.

        Returns
        -------
        history : ndarray of shape (T+1, N)
            Agent opinion trajectories.
        ua_history : ndarray of shape (T+1, k)
            UA opinion trajectories.
        """
        if not self._initialized:
            self.initialize()

        history = np.empty((T + 1, self.N))
        history[0] = self.x.copy()

        for t in range(T):
            self.step()
            history[t + 1] = self.x.copy()

        ua_history = np.array(self._ua_history)  # (T+1, k)
        return history, ua_history
