"""
experiments.py — All 6 experiments for the opinion homogenization project.

Each function returns a dict containing one or more pandas DataFrames (or
NumPy arrays) with the raw results, ready for plotting.

Experiments:
  1. Baseline convergence (no UA): vary N, topology, epsilon
  2. UA with varying alpha: fixed UA, log-scale alpha sweep, multiple epsilon
  3. Phase transition detection: 2-D sweep over alpha x epsilon (log alpha axis)
  4. Adaptive UA: vary beta, compare simulation vs analytical prediction (Prop 4)
  5. Heterogeneous coupling: trajectory cascade + cluster bar chart (Prop 5)
  6. Multiple Universal Agents: k-UA model (fixed vs. adaptive UAs, sweep k/beta)
"""

import numpy as np
import pandas as pd

from models import (
    DeGrootModel,
    DeGrootWithUA,
    BoundedConfidenceWithUA,
    MultipleUAModel,
    compute_variance,
    count_clusters,
    convergence_time,
)

# Number of independent realisations averaged per parameter combination.
N_SEEDS = 5
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Experiment 1 — Baseline convergence (no UA)
# ---------------------------------------------------------------------------

def experiment1_baseline(T: int = 300, n_seeds: int = N_SEEDS) -> dict:
    """Baseline convergence without a Universal Agent.

    Sweeps N, topology, and epsilon (for HK).  Runs both DeGroot and
    BoundedConfidence models with alpha=0 and records:
        - final cluster count
        - convergence time

    Parameters
    ----------
    T : int
        Simulation length.
    n_seeds : int
        Number of random seeds per parameter combination.

    Returns
    -------
    dict with key 'df': pandas DataFrame with columns
        [N, topology, epsilon, model, n_clusters, conv_time]
    """
    print("=== Experiment 1: Baseline convergence (no UA) ===")

    N_values = [50, 100, 200]
    topologies = ['complete', 'small_world', 'scale_free']
    epsilons = [0.1, 0.2, 0.3, 0.5]

    records = []
    total = len(N_values) * len(topologies) * len(epsilons) * 2 * n_seeds
    done = 0

    for N in N_values:
        for topo in topologies:
            for eps in epsilons:
                for model_name in ['degroot', 'hk']:
                    cluster_acc = []
                    conv_acc = []

                    for s in range(n_seeds):
                        seed = BASE_SEED + s * 1000

                        if model_name == 'degroot':
                            m = DeGrootModel(N=N, d=1, topology=topo, seed=seed)
                            m.initialize()
                            history = m.run(T)
                        else:
                            m = BoundedConfidenceWithUA(
                                N=N, epsilon=eps, alpha=0.0,
                                ua_type='fixed', topology=topo, seed=seed
                            )
                            m.initialize()
                            history = m.run(T)

                        final_x = history[-1]
                        cluster_acc.append(count_clusters(final_x, tol=0.05))
                        conv_acc.append(convergence_time(history, tol=1e-4))

                        done += 1
                        if done % 50 == 0:
                            print(f"  [{done}/{total}] N={N}, topo={topo}, eps={eps:.1f}, model={model_name}")

                    records.append({
                        'N': N,
                        'topology': topo,
                        'epsilon': eps,
                        'model': model_name,
                        'n_clusters': np.mean(cluster_acc),
                        'conv_time': np.mean(conv_acc),
                    })

    df = pd.DataFrame(records)
    print("  Done.\n")
    return {'df': df}


# ---------------------------------------------------------------------------
# Experiment 2 — Variance vs alpha (Proposition 3 validation)
# ---------------------------------------------------------------------------

def experiment2_alpha_sweep(T: int = 500, n_seeds: int = N_SEEDS) -> dict:
    """Effect of coupling strength alpha on opinion variance — Proposition 3.

    Uses ε ∈ {0.1, 0.15, 0.2}, all below the autonomous consensus threshold
    (so multiple clusters form without UA).  Sweeps α on a LOG scale from
    0.001 to 1.0.  Uses fixed UA only (cleaner story for Prop 3).

    The returned DataFrame includes:
        - normalized variance (variance / variance at alpha=0)
        - raw final variance
        - final cluster count

    Also stores:
        - 'alpha0_variance': dict {eps: baseline variance at alpha=0}
        - 'alpha_star_theory': dict {eps: theoretical α*(ε) = (1-3ε)/(1-2ε)}

    Returns
    -------
    dict with keys:
        'df'             : DataFrame [alpha, epsilon, ua_type, final_variance,
                           n_clusters, norm_variance]
        'alpha0_variance': dict mapping epsilon -> baseline variance (alpha=0)
        'alpha_star_theory': dict mapping epsilon -> theoretical alpha*
    """
    print("=== Experiment 2: Variance vs alpha (Prop 3 — log scale) ===")

    N = 100
    # Epsilon values below autonomous-consensus threshold (ε < 1/6 ≈ 0.167 for K>1 guaranteed)
    epsilon_values = [0.1, 0.15, 0.2]
    # Log-spaced alpha; include 0 separately for normalization baseline
    alphas_log = np.logspace(-3, 0, 40)   # 0.001 to 1.0
    x_UA = 0.8  # distinct from population mean ≈ 0.5

    records = []
    alpha0_variance = {}  # baseline variance per epsilon (at alpha=0)

    total = len(epsilon_values) * (len(alphas_log) + 1) * n_seeds
    done = 0

    for eps in epsilon_values:
        # --- baseline at alpha=0 ---
        var0_acc = []
        for s in range(n_seeds):
            seed = BASE_SEED + s * 1000
            m = BoundedConfidenceWithUA(
                N=N, epsilon=eps, alpha=0.0,
                ua_type='fixed', ua_init=x_UA,
                topology='complete', seed=seed
            )
            m.initialize()
            history = m.run(T)
            final_x = history[-1]
            var0_acc.append(compute_variance(final_x))
            done += 1

        var0_mean = float(np.mean(var0_acc))
        alpha0_variance[eps] = var0_mean

        clust0_acc = []
        for s in range(n_seeds):
            seed = BASE_SEED + s * 1000
            final_x0 = _run_single(N, eps, 0.0, x_UA, T, seed)
            clust0_acc.append(count_clusters(final_x0, tol=0.05))

        records.append({
            'alpha': 0.0,
            'epsilon': float(eps),
            'ua_type': 'fixed',
            'final_variance': var0_mean,
            'n_clusters': float(np.mean(clust0_acc)),
            'norm_variance': 1.0,  # by definition
        })

        # --- sweep log-alpha ---
        for alpha in alphas_log:
            var_acc = []
            clust_acc = []

            for s in range(n_seeds):
                seed = BASE_SEED + s * 1000
                m = BoundedConfidenceWithUA(
                    N=N, epsilon=eps, alpha=float(alpha),
                    ua_type='fixed', ua_init=x_UA,
                    topology='complete', seed=seed
                )
                m.initialize()
                history = m.run(T)
                final_x = history[-1]
                var_acc.append(compute_variance(final_x))
                clust_acc.append(count_clusters(final_x, tol=0.05))
                done += 1

            mean_var = float(np.mean(var_acc))
            norm_var = mean_var / var0_mean if var0_mean > 1e-10 else 0.0

            records.append({
                'alpha': float(alpha),
                'epsilon': float(eps),
                'ua_type': 'fixed',
                'final_variance': mean_var,
                'n_clusters': float(np.mean(clust_acc)),
                'norm_variance': norm_var,
            })

            if done % 50 == 0:
                print(f"  [{done}/{total}] eps={eps:.2f}, alpha={alpha:.4f}")

    df = pd.DataFrame(records)

    # Theoretical alpha* per epsilon: α*(ε) = (1-3ε)/(1-2ε)
    alpha_star_theory = {}
    for eps in epsilon_values:
        astar = (1.0 - 3.0 * eps) / (1.0 - 2.0 * eps)
        alpha_star_theory[eps] = max(0.0, astar)

    print("  Done.\n")
    return {
        'df': df,
        'alpha0_variance': alpha0_variance,
        'alpha_star_theory': alpha_star_theory,
    }


def _run_single(N, eps, alpha, ua_init, T, seed):
    """Helper: run one simulation and return final opinions."""
    m = BoundedConfidenceWithUA(
        N=N, epsilon=eps, alpha=float(alpha),
        ua_type='fixed', ua_init=ua_init,
        topology='complete', seed=seed
    )
    m.initialize()
    history = m.run(T)
    return history[-1]


# ---------------------------------------------------------------------------
# Experiment 3 — Phase transition heatmap (log alpha axis)
# ---------------------------------------------------------------------------

def experiment3_phase_transition(T: int = 500, n_seeds: int = N_SEEDS) -> dict:
    """2-D sweep over (alpha, epsilon) to map the phase transition boundary.

    Uses LOG-spaced alpha axis so that the transition (which occurs at small
    alpha for the ε values of interest) is visible.  Plots cluster count as
    the primary color, with the theoretical boundary overlaid.

    Returns
    -------
    dict with keys:
        'alphas'          : 1-D array of alpha values (log-spaced, length 35)
        'epsilons'        : 1-D array of epsilon values (length 8)
        'cluster_grid'    : 2-D array (len_alpha, len_eps) — mean cluster count
        'variance_grid'   : 2-D array (len_alpha, len_eps) — mean variance
        'alpha_star_theory': dict {eps: theoretical alpha*}
        'df'              : long-form DataFrame
    """
    print("=== Experiment 3: Phase transition heatmap (log alpha) ===")

    N = 100
    alphas = np.logspace(-3, 0, 35)    # 0.001 to 1.0, log-spaced
    epsilons = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    x_UA = 0.8

    variance_grid = np.zeros((len(alphas), len(epsilons)))
    cluster_grid = np.zeros((len(alphas), len(epsilons)))

    total = len(alphas) * len(epsilons) * n_seeds
    done = 0

    for j, eps in enumerate(epsilons):
        for i, alpha in enumerate(alphas):
            var_acc = []
            clust_acc = []

            for s in range(n_seeds):
                seed = BASE_SEED + s * 1000
                m = BoundedConfidenceWithUA(
                    N=N, epsilon=eps, alpha=float(alpha),
                    ua_type='fixed', ua_init=x_UA,
                    topology='complete', seed=seed
                )
                m.initialize()
                history = m.run(T)
                final_x = history[-1]
                var_acc.append(compute_variance(final_x))
                clust_acc.append(count_clusters(final_x, tol=0.05))

                done += 1
                if done % 100 == 0:
                    print(f"  [{done}/{total}] alpha={alpha:.4f}, eps={eps:.2f}")

            variance_grid[i, j] = np.mean(var_acc)
            cluster_grid[i, j] = np.mean(clust_acc)

    # Long-form DataFrame
    rows = []
    for j, eps in enumerate(epsilons):
        for i, alpha in enumerate(alphas):
            rows.append({
                'alpha': alphas[i],
                'epsilon': eps,
                'final_variance': variance_grid[i, j],
                'n_clusters': cluster_grid[i, j],
            })
    df = pd.DataFrame(rows)

    # Theoretical alpha* per epsilon: α*(ε) = (1-3ε)/(1-2ε), clipped to [0,1]
    alpha_star_theory = {}
    for eps in epsilons:
        astar = (1.0 - 3.0 * eps) / (1.0 - 2.0 * eps)
        alpha_star_theory[eps] = float(np.clip(astar, 0.0, 1.0))

    print("  Done.\n")
    return {
        'alphas': alphas,
        'epsilons': np.array(epsilons),
        'variance_grid': variance_grid,
        'cluster_grid': cluster_grid,
        'alpha_star_theory': alpha_star_theory,
        'df': df,
    }


# ---------------------------------------------------------------------------
# Experiment 4 — Adaptive UA fixed-point validation (Proposition 4)
# ---------------------------------------------------------------------------

def experiment4_adaptive_ua(T: int = 500, n_seeds: int = N_SEEDS) -> dict:
    """Validate Proposition 4: adaptive UA fixed point x* = (β·x̄(0) + α·x_UA(0))/(α+β).

    Uses ε=0.1, α=0.3, x_UA(0)=0.8 (distinct from population mean ≈0.5).
    Sweeps β ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}.

    For each β records:
        - final x_UA(T)   (UA opinion at end of run)
        - final pop mean x̄(T)
        - theoretical prediction x* = (β·x̄(0) + α·x_UA(0)) / (α+β)
        - final variance

    Returns
    -------
    dict with key 'df': DataFrame [beta, final_ua, final_pop_mean,
                                   theory_xstar, final_variance,
                                   xbar0, x_ua0]
    """
    print("=== Experiment 4: Adaptive UA fixed point vs theory (Prop 4) ===")

    N = 100
    alpha = 0.3
    eps = 0.1        # non-trivial regime — multi-cluster without UA
    x_UA_0 = 0.8    # UA starts clearly away from population
    betas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    records = []
    total = len(betas) * n_seeds
    done = 0

    for beta in betas:
        ua_acc = []
        mean_acc = []
        var_acc = []
        xbar0_acc = []

        for s in range(n_seeds):
            seed = BASE_SEED + s * 1000
            m = BoundedConfidenceWithUA(
                N=N, epsilon=eps, alpha=float(alpha),
                ua_type='adaptive', ua_init=x_UA_0, beta=float(beta),
                topology='complete', seed=seed
            )
            m.initialize()
            xbar0 = float(m.x.mean())  # initial population mean
            history = m.run(T)
            final_x = history[-1]
            final_ua = float(m.x_UA)
            final_mean = float(final_x.mean())

            ua_acc.append(final_ua)
            mean_acc.append(final_mean)
            var_acc.append(compute_variance(final_x))
            xbar0_acc.append(xbar0)

            done += 1
            if done % 5 == 0:
                print(f"  [{done}/{total}] beta={beta}")

        mean_xbar0 = float(np.mean(xbar0_acc))
        # Analytical prediction: x* = (β·x̄(0) + α·x_UA(0)) / (α + β)
        theory_xstar = (beta * mean_xbar0 + alpha * x_UA_0) / (alpha + beta)

        records.append({
            'beta': float(beta),
            'final_ua': float(np.mean(ua_acc)),
            'final_pop_mean': float(np.mean(mean_acc)),
            'theory_xstar': float(theory_xstar),
            'final_variance': float(np.mean(var_acc)),
            'xbar0': mean_xbar0,
            'x_ua0': x_UA_0,
            'alpha': alpha,
        })

    df = pd.DataFrame(records)
    print("  Done.\n")
    return {'df': df}


# ---------------------------------------------------------------------------
# Experiment 5 — Heterogeneous coupling + Cascade (Proposition 5)
# ---------------------------------------------------------------------------

def experiment5_heterogeneous(T: int = 100, n_seeds: int = N_SEEDS) -> dict:
    """Visualize Proposition 5 — cascade dynamics under heterogeneous coupling.

    Two sub-experiments:

    Panel A — Trajectory cascade (single run, N=50):
        Bimodal α_i: 25 agents with α_i=0.05 (low), 25 with α_i=0.6 (high).
        ε=0.1, x_UA=0.8.  Run T=100 steps.
        Store full trajectory history + alpha_arr for coloring.

    Panel B — Stratification bar chart (averaged over n_seeds):
        Compare final cluster count K across four α distributions,
        using ε=0.1, x_UA=0.8 (far from population center).
        Distributions:
            'uniform_low'  : all α_i = 0.05 (below α*(0.1)≈0.57)
            'bimodal_5_25' : 50% at 0.05, 50% at 0.25
            'bimodal_5_60' : 50% at 0.05, 50% at 0.60
            'heavy_tail'   : α_i ~ Beta(0.5, 2.0) (mean≈0.2, heavy low tail)

    Returns
    -------
    dict with keys:
        'traj_history'   : ndarray (T+1, N_traj) — Panel A trajectories
        'traj_alpha_arr' : ndarray (N_traj,) — per-agent alpha for coloring
        'traj_xUA'       : float — x_UA value used (0.8)
        'traj_T'         : int — number of time steps run
        'df'             : DataFrame [distribution, n_clusters, final_variance,
                           stratification] — Panel B data
    """
    print("=== Experiment 5: Heterogeneous coupling + cascade (Prop 5) ===")

    # --- Panel A: trajectory ---
    N_traj = 50
    eps = 0.1
    x_UA = 0.8
    T_traj = T  # 100 steps

    # Use alpha_i=0.0 for low group (pure HK, no UA coupling) and 0.6 for high
    # group.  This maximises the contrast and makes the cascade clearly visible.
    n_low, n_high = 25, 25
    alpha_arr_traj = np.concatenate([
        np.full(n_low, 0.0),    # low-α agents (blue) — no UA coupling
        np.full(n_high, 0.60),  # high-α agents (red)
    ])

    m_traj = BoundedConfidenceWithUA(
        N=N_traj, epsilon=eps, alpha=0.0,
        ua_type='heterogeneous', ua_init=x_UA,
        topology='complete', seed=BASE_SEED
    )
    m_traj.initialize(alpha_dist=alpha_arr_traj)
    traj_history = m_traj.run(T_traj)  # shape (T_traj+1, N_traj)

    print(f"  Panel A done: traj shape {traj_history.shape}")

    # --- Panel B: stratification bar chart ---
    N = 100
    T_bar = T  # same T for bar chart

    # Empirically, α*(ε=0.1) ≈ 0.003-0.005 (simulation) / 0.875 (theory upper bound).
    # The cascade effect means even α_i=0 agents get pulled in by high-α neighbours.
    # To show differentiated K across distributions we need:
    #   - Some agents fully decoupled (α_i=0) to stay in peer clusters
    #   - Some agents strongly coupled (α_i=0.6) to converge near x_UA
    #   - Heavy tail: most agents at very low α, a few at moderate α
    dist_configs = {
        # All agents with zero UA coupling — pure HK, gives K≈4 clusters
        'no_ua':        lambda rng: np.full(N, 0.0),
        # 50% no coupling, 50% strong coupling — cascade: high-α converge,
        # some low-α get recruited, some remain in peer clusters
        'bimodal_0_60': lambda rng: np.concatenate([
            np.full(N // 2, 0.0), np.full(N // 2, 0.60)
        ]),
        # 50% no coupling, 50% moderate coupling (below high-α threshold)
        'bimodal_0_30': lambda rng: np.concatenate([
            np.full(N // 2, 0.0), np.full(N // 2, 0.30)
        ]),
        # Heavy tail: most agents have very low α, mean≈0.2
        'heavy_tail':   lambda rng: np.clip(rng.beta(0.5, 2.0, size=N), 0, 1),
    }

    records = []
    total = len(dist_configs) * n_seeds
    done = 0

    for dist_name, alpha_sampler in dist_configs.items():
        clust_acc = []
        var_acc = []
        strat_acc = []

        for s in range(n_seeds):
            seed = BASE_SEED + s * 1000
            rng = np.random.default_rng(seed)
            alpha_i = alpha_sampler(rng)

            m = BoundedConfidenceWithUA(
                N=N, epsilon=eps, alpha=0.0,
                ua_type='heterogeneous', ua_init=x_UA,
                topology='complete', seed=seed
            )
            m.initialize(alpha_dist=alpha_i)
            history = m.run(T_bar)
            final_x = history[-1]

            n_c = count_clusters(final_x, tol=0.05)
            var = compute_variance(final_x)
            strat = _cluster_mean_std(final_x, tol=0.05)

            clust_acc.append(n_c)
            var_acc.append(var)
            strat_acc.append(strat)

            done += 1
            print(f"  [{done}/{total}] dist={dist_name}")

        records.append({
            'distribution': dist_name,
            'n_clusters': float(np.mean(clust_acc)),
            'final_variance': float(np.mean(var_acc)),
            'stratification': float(np.mean(strat_acc)),
        })

    df = pd.DataFrame(records)
    print("  Done.\n")

    return {
        'traj_history': traj_history,
        'traj_alpha_arr': alpha_arr_traj,
        'traj_xUA': x_UA,
        'traj_T': T_traj,
        'df': df,
    }


# ---------------------------------------------------------------------------
# Experiment 6 — Multiple Universal Agents (k-UA model)
# ---------------------------------------------------------------------------

def experiment6_multiple_ua(n_seeds: int = N_SEEDS) -> dict:
    """Two sub-experiments exploring k Universal Agents.

    Sub-experiment 6A — Fixed UAs, sweep k:
        Fix: N=100, ε=0.1, α=0.3, ua_type='fixed', T=100
        Sweep: k ∈ {1, 2, 4, 8}, UAs at evenly spaced positions in [0,1]
        Metric: cluster count K(t) and opinion variance Var(t) over time.
        Expected: k clusters form and persist (UAs are anchors).

    Sub-experiment 6B — Adaptive UAs, sweep β:
        Fix: N=100, ε=0.1, α=0.3, k=4, T=300
        UAs initially at {0.1, 0.35, 0.65, 0.9} (spread across opinion space)
        Sweep: β ∈ {0.001, 0.01, 0.1, 0.5}
        Metric: cluster count K(t) and UA spread (std of x_UA) over time.
        Expected: small β → k clusters persist long;
                  large β → UAs merge fast → 1 cluster.

    Returns
    -------
    dict with keys:
        '6A_histories'         : dict {k: ndarray (T+1, N)} — agent trajectories
        '6A_ua_histories'      : dict {k: ndarray (T+1, k)} — UA trajectories
        '6A_cluster_traj'      : dict {k: ndarray (T+1,)} — cluster count over time
        '6A_variance_traj'     : dict {k: ndarray (T+1,)} — variance over time
        '6A_params'            : dict of fixed parameters
        '6B_histories'         : dict {beta: ndarray (T+1, N)}
        '6B_ua_histories'      : dict {beta: ndarray (T+1, 4)}
        '6B_cluster_traj'      : dict {beta: ndarray (T+1,)} — cluster count
        '6B_ua_spread_traj'    : dict {beta: ndarray (T+1,)} — std of x_UA over time
        '6B_params'            : dict of fixed parameters
    """
    print("=== Experiment 6: Multiple Universal Agents ===")

    # ------------------------------------------------------------------
    # 6A — Fixed UAs, sweep k
    # ------------------------------------------------------------------
    print("  --- 6A: Fixed UAs, sweep k ---")
    N = 100
    eps = 0.1
    alpha = 0.3
    T_6A = 100
    k_values = [1, 2, 4, 8]

    results_6A_histories = {}
    results_6A_ua_histories = {}
    results_6A_cluster_traj = {}
    results_6A_variance_traj = {}

    for k in k_values:
        # Average cluster/variance trajectories over seeds; keep one trajectory
        # for the trajectory plot (use the first seed for k=4).
        cluster_traj_acc = np.zeros(T_6A + 1)
        variance_traj_acc = np.zeros(T_6A + 1)
        ref_history = None
        ref_ua_history = None

        for s in range(n_seeds):
            seed = BASE_SEED + s * 1000
            m = MultipleUAModel(
                N=N, k=k, epsilon=eps, alpha=alpha, beta=0.0,
                ua_type='fixed',
                ua_init='spread',
                seed=seed,
            )
            m.initialize()
            hist, ua_hist = m.run(T_6A)

            for t in range(T_6A + 1):
                cluster_traj_acc[t] += count_clusters(hist[t], tol=0.05)
                variance_traj_acc[t] += compute_variance(hist[t])

            if s == 0:
                ref_history = hist
                ref_ua_history = ua_hist

        results_6A_histories[k] = ref_history
        results_6A_ua_histories[k] = ref_ua_history
        results_6A_cluster_traj[k] = cluster_traj_acc / n_seeds
        results_6A_variance_traj[k] = variance_traj_acc / n_seeds
        print(f"    k={k}: final K={results_6A_cluster_traj[k][-1]:.2f}, "
              f"final Var={results_6A_variance_traj[k][-1]:.4f}")

    # ------------------------------------------------------------------
    # 6B — Adaptive UAs, sweep β
    # ------------------------------------------------------------------
    print("  --- 6B: Adaptive UAs, sweep beta ---")
    k4 = 4
    T_6B = 300
    betas = [0.001, 0.01, 0.1, 0.5]
    # Fixed UA starting positions spread across opinion space
    ua_init_6B = np.array([0.1, 0.35, 0.65, 0.9])

    results_6B_histories = {}
    results_6B_ua_histories = {}
    results_6B_cluster_traj = {}
    results_6B_ua_spread_traj = {}

    for beta in betas:
        cluster_traj_acc = np.zeros(T_6B + 1)
        ua_spread_acc = np.zeros(T_6B + 1)
        ref_history = None
        ref_ua_history = None

        for s in range(n_seeds):
            seed = BASE_SEED + s * 1000
            m = MultipleUAModel(
                N=N, k=k4, epsilon=eps, alpha=alpha, beta=beta,
                ua_type='adaptive',
                ua_init=ua_init_6B.copy(),
                seed=seed,
            )
            m.initialize()
            hist, ua_hist = m.run(T_6B)

            for t in range(T_6B + 1):
                cluster_traj_acc[t] += count_clusters(hist[t], tol=0.05)
                ua_spread_acc[t] += float(np.std(ua_hist[t]))

            if s == 0:
                ref_history = hist
                ref_ua_history = ua_hist

        results_6B_histories[beta] = ref_history
        results_6B_ua_histories[beta] = ref_ua_history
        results_6B_cluster_traj[beta] = cluster_traj_acc / n_seeds
        results_6B_ua_spread_traj[beta] = ua_spread_acc / n_seeds
        print(f"    beta={beta}: final K={results_6B_cluster_traj[beta][-1]:.2f}, "
              f"final UA spread={results_6B_ua_spread_traj[beta][-1]:.4f}")

    print("  Done.\n")
    return {
        '6A_histories': results_6A_histories,
        '6A_ua_histories': results_6A_ua_histories,
        '6A_cluster_traj': results_6A_cluster_traj,
        '6A_variance_traj': results_6A_variance_traj,
        '6A_params': {'N': N, 'epsilon': eps, 'alpha': alpha, 'T': T_6A,
                      'k_values': k_values, 'ua_type': 'fixed'},
        '6B_histories': results_6B_histories,
        '6B_ua_histories': results_6B_ua_histories,
        '6B_cluster_traj': results_6B_cluster_traj,
        '6B_ua_spread_traj': results_6B_ua_spread_traj,
        '6B_params': {'N': N, 'epsilon': eps, 'alpha': alpha, 'T': T_6B,
                      'k': k4, 'betas': betas, 'ua_init': ua_init_6B.tolist()},
    }


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _cluster_mean_std(x: np.ndarray, tol: float = 0.05) -> float:
    """Return the std deviation of cluster means (stratification measure).

    Uses the same single-linkage criterion as count_clusters().
    """
    x_sorted = np.sort(x)
    if len(x_sorted) == 0:
        return 0.0

    cluster_means = []
    current_cluster = [x_sorted[0]]

    for val in x_sorted[1:]:
        if val - current_cluster[-1] < tol:
            current_cluster.append(val)
        else:
            cluster_means.append(np.mean(current_cluster))
            current_cluster = [val]
    cluster_means.append(np.mean(current_cluster))

    if len(cluster_means) <= 1:
        return 0.0
    return float(np.std(cluster_means))
