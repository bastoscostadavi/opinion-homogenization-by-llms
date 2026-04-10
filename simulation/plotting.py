"""
plotting.py — All visualisation functions for the opinion dynamics project.

Functions:
  plot_opinion_trajectories     — time-series of individual agent opinions
  plot_variance_vs_alpha        — Experiment 2: Prop 3 validation (log alpha)
  plot_phase_transition_heatmap — Experiment 3: (alpha × epsilon) phase diagram
  plot_adaptive_ua              — Experiment 4: Prop 4 fixed-point validation
  plot_heterogeneous            — Experiment 5: cascade trajectories + bar chart

All functions save the figure to `save_path` at 300 DPI and close it afterwards.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from models import compute_variance

# Shared style — journal-quality whitegrid
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
PALETTE = sns.color_palette('tab10')
DPI = 300

# Colour cycle for epsilon lines in Fig 1
EPS_COLORS = {
    0.10: PALETTE[0],
    0.15: PALETTE[1],
    0.20: PALETTE[2],
}


# ---------------------------------------------------------------------------
# 1. Opinion trajectories (utility / supplementary)
# ---------------------------------------------------------------------------

def plot_opinion_trajectories(
    history: np.ndarray,
    title: str,
    save_path: str,
    ua_history=None,
    max_agents: int = 50,
) -> None:
    """Plot individual agent opinion trajectories over time."""
    T, N = history.shape[0] - 1, history.shape[1]
    time_axis = np.arange(T + 1)

    if N > max_agents:
        rng = np.random.default_rng(0)
        agent_idx = rng.choice(N, size=max_agents, replace=False)
    else:
        agent_idx = np.arange(N)

    initial_opinions = history[0, agent_idx]
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for k, idx in enumerate(agent_idx):
        color = cmap(norm(initial_opinions[k]))
        ax.plot(time_axis, history[:, idx], color=color, alpha=0.35, linewidth=0.8)

    if ua_history is not None:
        ua_arr = np.array(ua_history)
        t_ua = np.arange(len(ua_arr))
        ax.plot(t_ua, ua_arr, color='crimson', linestyle='--', linewidth=2.0,
                label=r'$x_{\mathrm{UA}}(t)$', zorder=5)
        ax.legend(loc='upper right', fontsize=10)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'Initial opinion $x_i(0)$', fontsize=10)

    ax.set_xlabel(r'Time step $t$', fontsize=11)
    ax.set_ylabel(r'Opinion $x_i(t)$', fontsize=11)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 2. Variance vs. alpha — Proposition 3 validation (Figure 1)
# ---------------------------------------------------------------------------

def plot_variance_vs_alpha(results_exp2: dict, save_path: str) -> None:
    """Normalized variance vs log(alpha) for ε ∈ {0.1, 0.15, 0.2}.

    Overlays theoretical α*(ε) = (1-3ε)/(1-2ε) as vertical dashed lines.
    Y-axis: normalized variance (Var / Var(alpha=0)) so all ε share the same
    scale [0, 1].

    Parameters
    ----------
    results_exp2 : dict
        Output of experiment2_alpha_sweep(); must contain keys
        'df', 'alpha_star_theory'.
    save_path : str
    """
    df: pd.DataFrame = results_exp2['df']
    alpha_star_theory: dict = results_exp2['alpha_star_theory']

    eps_values = sorted(df['epsilon'].unique())
    eps_values = [e for e in eps_values if e > 0]  # exclude alpha=0 row

    # Single panel: one line per epsilon
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for eps in eps_values:
        color = EPS_COLORS.get(eps, PALETTE[len(EPS_COLORS)])
        sub = df[(df['epsilon'] == eps) & (df['alpha'] > 0)].sort_values('alpha')
        if sub.empty:
            continue

        ax.plot(sub['alpha'], sub['norm_variance'],
                color=color, linewidth=2.0,
                marker='o', markersize=3.5,
                label=rf'$\varepsilon={eps}$')

        # Mark empirical transition: midpoint between last high-variance and first low-variance
        nv = sub['norm_variance'].values
        av = sub['alpha'].values
        # Find first alpha where normalized variance drops below 0.1
        below = np.where(nv < 0.1)[0]
        if len(below) > 0:
            i_trans = below[0]
            if i_trans > 0:
                alpha_emp = np.sqrt(av[i_trans - 1] * av[i_trans])  # geometric midpoint
            else:
                alpha_emp = av[i_trans]
            ax.axvline(alpha_emp, color=color, linestyle='--', linewidth=1.3, alpha=0.75)
            # Stagger annotation heights to avoid overlap
            eps_rank = eps_values.index(eps)
            y_text = 0.35 + eps_rank * 0.18
            ax.text(alpha_emp * 1.15, y_text,
                    rf'$\hat{{\alpha}}^*_{{\varepsilon={eps}}}$',
                    color=color, fontsize=9, va='bottom', fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel(r'Coupling strength $\alpha$', fontsize=12)
    ax.set_ylabel(r'Normalized variance $\mathrm{Var}(\infty)\,/\,\mathrm{Var}_0$',
                  fontsize=11)
    ax.set_title(
        r'Proposition 3: Sharp phase transition at $\hat{\alpha}^*(\varepsilon)$' '\n'
        r'(N=100, fixed UA, $x_{\mathrm{UA}}=0.8$, T=500; dashed = empirical $\hat{\alpha}^*$)',
        fontsize=11, pad=8
    )
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(left=0.0008, right=0.004)
    ax.legend(fontsize=10, loc='upper right')

    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 3. Phase transition heatmap — full (α, ε) phase diagram (Figure 2)
# ---------------------------------------------------------------------------

def plot_phase_transition_heatmap(results_exp3: dict, save_path: str) -> None:
    """Phase diagram heatmap with log alpha axis and theoretical boundary.

    Color = mean cluster count.  White/black curve = α*(ε) theory.
    Regions annotated "Consensus" and "Fragmented".

    Parameters
    ----------
    results_exp3 : dict
        Output of experiment3_phase_transition().
    save_path : str
    """
    alphas = results_exp3['alphas']          # log-spaced, shape (35,)
    epsilons = results_exp3['epsilons']      # shape (8,)
    cluster_grid = results_exp3['cluster_grid']   # (35, 8)
    alpha_star_theory = results_exp3['alpha_star_theory']

    # cluster_grid is (len_alpha, len_eps); for heatmap we want rows=eps (y), cols=alpha (x)
    data = cluster_grid.T   # (len_eps, len_alpha)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Use pcolormesh on the log-alpha axis for proper spacing
    log_alphas = np.log10(alphas)
    # Bin edges for pcolormesh
    la_edges = _bin_edges(log_alphas)
    eps_edges = _bin_edges(np.arange(len(epsilons)))

    cmap = plt.cm.YlOrRd
    vmin, vmax = 1, max(5, data.max())
    pcm = ax.pcolormesh(
        la_edges,
        eps_edges,
        data,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        shading='flat',
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r'Mean cluster count $K(\infty)$', fontsize=11)
    cbar.set_ticks(np.arange(int(vmin), int(vmax) + 1))

    eps_centers = np.arange(len(epsilons))

    # Labels
    # Zoom into the interesting region only
    x_min_log, x_max_log = np.log10(alphas[0]), np.log10(0.012)
    ax.set_xlim(x_min_log, x_max_log)
    ax.set_xticks(np.log10(np.array([0.001, 0.002, 0.003, 0.005, 0.01])))
    ax.set_xticklabels([r'$10^{-3}$', r'$2\!\times\!10^{-3}$',
                         r'$3\!\times\!10^{-3}$', r'$5\!\times\!10^{-3}$',
                         r'$10^{-2}$'], fontsize=9)
    ax.set_yticks(eps_centers)
    ax.set_yticklabels([f'{e:.2f}' for e in epsilons], fontsize=9)
    ax.set_xlabel(r'Coupling strength $\alpha$ (log scale)', fontsize=12)
    ax.set_ylabel(r'Confidence bound $\varepsilon$', fontsize=12)
    ax.set_title(
        r'Phase diagram: consensus ($K=1$) vs. fragmentation ($K>1$)' '\n'
        r'(N=100, fixed UA, $x_{\mathrm{UA}}=0.8$, T=500)',
        fontsize=11, pad=8
    )

    # Region labels
    ax.text(np.log10(0.007), len(epsilons) - 1.0, 'Consensus',
            color='#444444', fontsize=12, fontweight='bold',
            ha='center', va='top', zorder=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))
    ax.text(np.log10(0.00105), 0.5, 'Fragmented',
            color='white', fontsize=12, fontweight='bold',
            ha='left', va='bottom', zorder=12)


    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _bin_edges(centers: np.ndarray) -> np.ndarray:
    """Compute bin edges from bin centers for pcolormesh."""
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    half = np.diff(centers) / 2.0
    edges = np.concatenate([
        [centers[0] - half[0]],
        centers[:-1] + half,
        [centers[-1] + half[-1]],
    ])
    return edges


# ---------------------------------------------------------------------------
# 4. Adaptive UA fixed-point — Proposition 4 validation (Figure 3)
# ---------------------------------------------------------------------------

def plot_adaptive_ua(results_exp4: dict, save_path: str) -> None:
    """Simulation vs theory for the adaptive UA fixed point x*.

    X-axis: β (log scale).  Y-axis: final consensus value x*.
    Shows simulation x_UA(T), simulation pop mean x̄(T), and analytical
    prediction x* = (β·x̄(0) + α·x_UA(0))/(α+β) as a dashed curve.

    Parameters
    ----------
    results_exp4 : dict
        Output of experiment4_adaptive_ua().
    save_path : str
    """
    df: pd.DataFrame = results_exp4['df']
    df = df.sort_values('beta')

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Simulation: final population mean (= x_UA(T) in the adaptive model — they co-converge)
    ax.plot(df['beta'], df['final_pop_mean'],
            color=PALETTE[1], marker='s', linewidth=2.0, markersize=6,
            label=r'Sim: $x^*$ (population mean $\approx$ $x_{\mathrm{UA}}$)')

    # Theory: smooth curve over beta range
    beta_smooth = np.logspace(
        np.log10(df['beta'].min() * 0.8),
        np.log10(df['beta'].max() * 1.2),
        200
    )
    alpha_val = float(df['alpha'].iloc[0])
    xbar0_val = float(df['xbar0'].iloc[0])
    x_ua0_val = float(df['x_ua0'].iloc[0])
    theory_smooth = (beta_smooth * xbar0_val + alpha_val * x_ua0_val) / (alpha_val + beta_smooth)

    ax.plot(beta_smooth, theory_smooth,
            color='black', linestyle='--', linewidth=2.0,
            label=r'Theory: $x^* = \frac{\beta\bar{x}(0)+\alpha x_{\mathrm{UA}}(0)}{\alpha+\beta}$')

    # Reference lines for limits
    ax.axhline(x_ua0_val, color=PALETTE[0], linestyle=':', linewidth=1.0, alpha=0.6)
    ax.text(df['beta'].max() * 1.05, x_ua0_val + 0.005,
            rf'$x_{{\mathrm{{UA}}}}(0)={x_ua0_val}$',
            color=PALETTE[0], fontsize=8, va='bottom')

    ax.axhline(xbar0_val, color=PALETTE[1], linestyle=':', linewidth=1.0, alpha=0.6)
    ax.text(df['beta'].max() * 1.05, xbar0_val - 0.015,
            rf'$\bar{{x}}(0)\approx{xbar0_val:.2f}$',
            color=PALETTE[1], fontsize=8, va='top')

    ax.set_xscale('log')
    ax.set_xlabel(r'UA learning rate $\beta$', fontsize=12)
    ax.set_ylabel(r'Final consensus value $x^*$', fontsize=12)
    ax.set_title(
        r'Proposition 4: Adaptive UA fixed point $x^*(\beta)$' '\n'
        rf'(N=100, $\varepsilon=0.1$, $\alpha={alpha_val}$, '
        rf'$x_{{\mathrm{{UA}}}}(0)={x_ua0_val}$, T=500)',
        fontsize=11, pad=8
    )
    ax.legend(fontsize=9, loc='center left')
    ax.set_ylim(
        min(xbar0_val, x_ua0_val) - 0.06,
        max(xbar0_val, x_ua0_val) + 0.06
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 5. Heterogeneous coupling — cascade + stratification (Figure 4)
# ---------------------------------------------------------------------------

def plot_heterogeneous(results_exp5: dict, save_path: str) -> None:
    """Two-panel figure illustrating Proposition 5.

    Panel A: Opinion trajectories colored by α_i (red=high, blue=low).
             Vertical dashed line marks approximate high-α convergence time.
             Horizontal dashed red line = x_UA.

    Panel B: Bar chart of final cluster count K across α distributions.

    Parameters
    ----------
    results_exp5 : dict
        Output of experiment5_heterogeneous().
    save_path : str
    """
    traj = results_exp5['traj_history']         # (T+1, N)
    alpha_arr = results_exp5['traj_alpha_arr']  # (N,)
    x_UA = results_exp5['traj_xUA']
    T_traj = results_exp5['traj_T']
    df: pd.DataFrame = results_exp5['df']

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                              gridspec_kw={'width_ratios': [1.4, 1.0]})
    ax_traj, ax_bar = axes

    # ------------------------------------------------------------------
    # Panel A: trajectories
    # ------------------------------------------------------------------
    N = traj.shape[1]
    time_axis = np.arange(T_traj + 1)

    alpha_unique = np.unique(alpha_arr)
    is_high = alpha_arr >= alpha_unique.max() - 1e-6

    # Blue for low-alpha, red for high-alpha
    color_low  = '#2166ac'   # blue
    color_high = '#d6604d'   # red/orange

    for i in range(N):
        col = color_high if is_high[i] else color_low
        ax_traj.plot(time_axis, traj[:, i],
                     color=col, alpha=0.45, linewidth=1.0)

    # x_UA horizontal line
    ax_traj.axhline(x_UA, color='crimson', linestyle='--', linewidth=2.0,
                    label=rf'$x_{{\mathrm{{UA}}}}={x_UA}$', zorder=5)

    # Estimate convergence time of high-alpha agents:
    # first t when ALL high-alpha agents are within 0.05 of x_UA
    high_idx = np.where(is_high)[0]
    conv_t = T_traj
    for t in range(T_traj + 1):
        if np.all(np.abs(traj[t, high_idx] - x_UA) < 0.08):
            conv_t = t
            break
    if conv_t < T_traj:
        ax_traj.axvline(conv_t, color='grey', linestyle=':', linewidth=1.5,
                        label=rf'High-$\alpha$ conv. ($t\approx{conv_t}$)',
                        zorder=4)

    # Legend proxies for agent coloring
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], color=color_high, linewidth=2,
               label=rf'High-$\alpha$ ($\alpha_i=0.60$)'),
        Line2D([0], [0], color=color_low, linewidth=2,
               label=rf'Low-$\alpha$ ($\alpha_i=0$, no UA)'),
    ]
    ax_traj.legend(handles=proxies + ax_traj.get_lines()[-2:],
                   fontsize=8, loc='lower right', ncol=2)

    ax_traj.set_xlabel(r'Time step $t$', fontsize=11)
    ax_traj.set_ylabel(r'Opinion $x_i(t)$', fontsize=11)
    ax_traj.set_title(
        r'(A) Cascade trajectories' '\n'
        r'($N=50$, $\varepsilon=0.1$, $x_{\mathrm{UA}}=0.8$)',
        fontsize=10, pad=6
    )
    ax_traj.set_xlim(0, min(conv_t + 5, 15, T_traj))
    ax_traj.set_ylim(-0.05, 1.05)

    # ------------------------------------------------------------------
    # Panel B: cluster count bar chart
    # ------------------------------------------------------------------
    dist_order = ['no_ua', 'bimodal_0_30', 'bimodal_0_60', 'heavy_tail']
    dist_labels = {
        'no_ua':        r'No UA' '\n' r'($\alpha_i=0$)',
        'bimodal_0_30': r'Bimodal' '\n' r'(0% / 30%)',
        'bimodal_0_60': r'Bimodal' '\n' r'(0% / 60%)',
        'heavy_tail':   r'Heavy tail' '\n' r'(Beta(0.5,2))',
    }

    df_plot = df.set_index('distribution').reindex(dist_order).reset_index()
    x_pos = np.arange(len(df_plot))
    bar_colors = [PALETTE[i] for i in range(len(df_plot))]

    bars = ax_bar.bar(x_pos, df_plot['n_clusters'],
                      color=bar_colors, alpha=0.85, width=0.6, edgecolor='white')

    for xi, (_, row) in enumerate(df_plot.iterrows()):
        k_val = row['n_clusters']
        ax_bar.text(xi, k_val + 0.05, f'K={k_val:.1f}',
                    ha='center', va='bottom', fontsize=9)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        [dist_labels.get(d, d) for d in df_plot['distribution']],
        fontsize=9
    )
    ax_bar.set_ylabel(r'Final cluster count $K(\infty)$', fontsize=11)
    ax_bar.set_title(
        r'(B) Stratification by $\alpha_i$ distribution' '\n'
        r'($N=100$, $\varepsilon=0.1$, $x_{\mathrm{UA}}=0.8$)',
        fontsize=10, pad=6
    )
    ax_bar.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(
        r'Proposition 5: Heterogeneous coupling — cascade and stratification',
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 6. Supplementary: baseline trajectory sample plots
# ---------------------------------------------------------------------------

def plot_baseline_trajectories(save_dir: str) -> None:
    """Generate example trajectory plots for baseline (no-UA) models."""
    import os
    from models import DeGrootModel, BoundedConfidenceWithUA

    # DeGroot
    m = DeGrootModel(N=100, topology='complete', seed=0)
    m.initialize()
    hist = m.run(50)
    plot_opinion_trajectories(
        hist,
        title='DeGroot baseline (complete graph, N=100)',
        save_path=os.path.join(save_dir, 'traj_degroot_complete.png'),
    )

    # HK eps=0.1 (interesting regime)
    m2 = BoundedConfidenceWithUA(N=100, epsilon=0.1, alpha=0.0, seed=0)
    m2.initialize()
    hist2 = m2.run(200)
    plot_opinion_trajectories(
        hist2,
        title=r'HK baseline ($\varepsilon=0.1$, N=100, no UA)',
        save_path=os.path.join(save_dir, 'traj_hk_eps01_noua.png'),
    )

    # HK eps=0.1 with alpha=0.3 fixed UA
    m3 = BoundedConfidenceWithUA(N=100, epsilon=0.1, alpha=0.3,
                                  ua_type='fixed', ua_init=0.8, seed=0)
    m3.initialize()
    hist3 = m3.run(200)
    plot_opinion_trajectories(
        hist3,
        title=r'HK + Fixed UA ($\alpha=0.3$, $x_{\mathrm{UA}}=0.8$, $\varepsilon=0.1$)',
        save_path=os.path.join(save_dir, 'traj_hk_eps01_fixedua_a03.png'),
        ua_history=m3.x_UA_history,
    )
