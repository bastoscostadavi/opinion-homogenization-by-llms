"""
run_all.py — Entry point for the opinion dynamics simulation suite.

Usage:
    python run_all.py [--fast] [--figs-only 2 3 4 5]

The --fast flag reduces n_seeds to 2 and T to smaller values, for quick
testing.  Without it, the full suite runs with n_seeds=5.

The --figs-only flag (optional) takes a space-separated list of experiment
numbers to run (1–5).  If omitted, all experiments run.

Figures are saved to simulation/figures/.
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def fig(name: str) -> str:
    """Return absolute path for a figure file inside FIGURES_DIR."""
    return os.path.join(FIGURES_DIR, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run all opinion dynamics experiments.')
    parser.add_argument('--fast', action='store_true',
                        help='Use fewer seeds and shorter runs for quick testing.')
    parser.add_argument('--figs-only', nargs='+', type=int, metavar='N',
                        help='Only run experiments N (e.g. --figs-only 2 3 4 5).')
    args = parser.parse_args()

    n_seeds = 2 if args.fast else 5
    run_set = set(args.figs_only) if args.figs_only else {1, 2, 3, 4, 5}

    print("=" * 60)
    print("Opinion Dynamics Simulation — Universal Agent Project")
    print("=" * 60)
    print(f"Figures directory : {FIGURES_DIR}")
    print(f"Seeds per combo   : {n_seeds}")
    print(f"Running exps      : {sorted(run_set)}")
    print()

    from experiments import (
        experiment1_baseline,
        experiment2_alpha_sweep,
        experiment3_phase_transition,
        experiment4_adaptive_ua,
        experiment5_heterogeneous,
    )
    from plotting import (
        plot_opinion_trajectories,
        plot_variance_vs_alpha,
        plot_phase_transition_heatmap,
        plot_adaptive_ua,
        plot_heterogeneous,
        plot_baseline_trajectories,
    )

    wall_start = time.time()

    # ------------------------------------------------------------------
    # Supplementary trajectory plots (fast, illustrative)
    # ------------------------------------------------------------------
    if 0 in run_set or not args.figs_only:
        print("--- Generating illustrative trajectory plots ---")
        plot_baseline_trajectories(FIGURES_DIR)
        print()

    # ------------------------------------------------------------------
    # Experiment 1 — Baseline (no UA)
    # ------------------------------------------------------------------
    if 1 in run_set:
        t0 = time.time()
        res1 = experiment1_baseline(T=300, n_seeds=n_seeds)
        print(f"  Elapsed: {time.time()-t0:.1f}s\n")

        df1 = res1['df']
        print("-- Exp 1 summary (mean over seeds) --")
        summary1 = (df1.groupby(['model', 'topology', 'epsilon'])
                       [['n_clusters', 'conv_time']]
                       .mean()
                       .round(2))
        print(summary1.to_string())
        print()

        # Trajectory examples
        from models import BoundedConfidenceWithUA
        for eps in [0.1, 0.3]:
            m = BoundedConfidenceWithUA(N=50, epsilon=eps, alpha=0.0,
                                        ua_type='fixed', seed=0)
            m.initialize()
            hist = m.run(200)
            plot_opinion_trajectories(
                hist,
                title=rf'HK baseline $\varepsilon={eps}$ (no UA)',
                save_path=fig(f'exp1_traj_eps{int(eps*10):02d}.png'),
            )
    else:
        res1 = None
        df1 = None

    # ------------------------------------------------------------------
    # Experiment 2 — Variance vs alpha (Prop 3 validation)
    # ------------------------------------------------------------------
    if 2 in run_set:
        t0 = time.time()
        T_exp2 = 200 if args.fast else 500
        res2 = experiment2_alpha_sweep(T=T_exp2, n_seeds=n_seeds)
        print(f"  Elapsed: {time.time()-t0:.1f}s\n")

        plot_variance_vs_alpha(res2, save_path=fig('exp2_variance_vs_alpha.png'))

        df2 = res2['df']
        print("-- Exp 2 summary (fixed UA, eps=0.1, selected alphas) --")
        sub = df2[(df2['epsilon'] == 0.1) & (df2['alpha'] > 0)].sort_values('alpha')
        print(sub[['alpha', 'final_variance', 'norm_variance', 'n_clusters']].iloc[::5].to_string(index=False))
        print("Theoretical alpha*:")
        for eps, astar in res2['alpha_star_theory'].items():
            print(f"  eps={eps:.2f}  alpha*(theory)={astar:.4f}")
        print()

        # Sample trajectories below/above alpha*
        from models import BoundedConfidenceWithUA
        for alpha_plot, label in [(0.02, 'below_astar'), (0.6, 'above_astar')]:
            m = BoundedConfidenceWithUA(N=100, epsilon=0.1, alpha=alpha_plot,
                                        ua_type='fixed', ua_init=0.8, seed=0)
            m.initialize()
            hist = m.run(300)
            plot_opinion_trajectories(
                hist,
                title=rf'HK + Fixed UA, $\alpha={alpha_plot}$, $\varepsilon=0.1$',
                save_path=fig(f'exp2_traj_{label}.png'),
                ua_history=m.x_UA_history,
            )
    else:
        res2 = None
        df2 = None

    # ------------------------------------------------------------------
    # Experiment 3 — Phase transition heatmap
    # ------------------------------------------------------------------
    if 3 in run_set:
        t0 = time.time()
        T_exp3 = 200 if args.fast else 500
        res3 = experiment3_phase_transition(T=T_exp3, n_seeds=n_seeds)
        print(f"  Elapsed: {time.time()-t0:.1f}s\n")

        plot_phase_transition_heatmap(res3, save_path=fig('exp3_phase_transition_heatmap.png'))

        print("-- Exp 3 summary: theoretical alpha* vs epsilon --")
        for eps, astar in res3['alpha_star_theory'].items():
            print(f"  eps={eps:.2f}  alpha*(theory)={astar:.4f}")
        print()
    else:
        res3 = None

    # ------------------------------------------------------------------
    # Experiment 4 — Adaptive UA (Prop 4 validation)
    # ------------------------------------------------------------------
    if 4 in run_set:
        t0 = time.time()
        T_exp4 = 200 if args.fast else 500
        res4 = experiment4_adaptive_ua(T=T_exp4, n_seeds=n_seeds)
        print(f"  Elapsed: {time.time()-t0:.1f}s\n")

        plot_adaptive_ua(res4, save_path=fig('exp4_adaptive_ua.png'))

        df4 = res4['df']
        print("-- Exp 4 summary (simulation vs theory) --")
        print(df4[['beta', 'final_ua', 'final_pop_mean', 'theory_xstar',
                    'final_variance']].to_string(index=False))
        print()
    else:
        res4 = None
        df4 = None

    # ------------------------------------------------------------------
    # Experiment 5 — Heterogeneous coupling + cascade (Prop 5)
    # ------------------------------------------------------------------
    if 5 in run_set:
        t0 = time.time()
        T_exp5 = 60 if args.fast else 100
        res5 = experiment5_heterogeneous(T=T_exp5, n_seeds=n_seeds)
        print(f"  Elapsed: {time.time()-t0:.1f}s\n")

        plot_heterogeneous(res5, save_path=fig('exp5_heterogeneous.png'))

        df5 = res5['df']
        print("-- Exp 5 summary --")
        print(df5[['distribution', 'n_clusters', 'final_variance',
                    'stratification']].to_string(index=False))
        print()
    else:
        res5 = None
        df5 = None

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - wall_start
    print("=" * 60)
    print(f"All selected experiments complete.  Total time: {total_elapsed:.1f}s")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
