# Project Status — Opinion Homogenization via LLMs

## What this project is

A theory + simulation paper studying cultural homogenization induced by a **Universal Agent (UA)** — a single agent that interacts with every member of a population at each time step (formalizing how an LLM simultaneously influences millions of users). The paper extends DeGroot and Hegselmann–Krause (HK) opinion dynamics models.

**Target venue:** TAIGR @ ICML 2026 workshop (deadline: 24 April 2026)
**Format:** 8-page limit, double-blind, ICML 2026 template

---

## Repository layout

```
/
├── implementation_plan.md     original research plan
├── literature_review.md       background notes
├── model_specification.md     formal model writeup (pre-paper)
├── analytical_results.md      proposition sketches (pre-paper)
├── paper/
│   ├── main_icml.tex          PRIMARY PAPER FILE — submit this
│   ├── main_icml.pdf          compiled output (10 pages: 8 main + 2 appendix)
│   ├── main.tex               older draft, ignore
│   └── icml2026/              ICML 2026 style files
└── simulation/
    ├── models.py              model classes
    ├── experiments.py         6 experiments
    ├── plotting.py            all figure generation
    ├── run_all.py             entry point (runs everything)
    ├── results_exp5.pkl       cached exp5 results
    └── figures/               all output PNGs (referenced by paper)
```

---

## The model (three variants)

**A (baseline):** N agents with scalar opinions $x_i(t) \in [0,1]$, interacting via DeGroot (A1) or HK bounded-confidence (A2) rules.

**B1 — Fixed UA:** A single agent with fixed opinion $x_{UA}$ interacts with all N agents at every step. Each agent updates as:
$$x_i(t+1) = (1-\alpha)\cdot[\text{peer term}] + \alpha \cdot x_{UA}$$
where $\alpha \in [0,1]$ is the coupling weight.

**B2 — Adaptive UA:** The UA also updates:
$$x_{UA}(t+1) = (1-\beta)\,x_{UA}(t) + \beta\,\bar{x}(t)$$

**B3 — Heterogeneous coupling:** Each agent has its own $\alpha_i \sim P(\alpha)$.

**k-UA extension:** k UAs each interact with N/k assigned agents; all UAs update on the full population mean.

---

## Five analytical propositions (all proved)

| Prop | Result | Key formula |
|------|--------|-------------|
| 1 | Geometric convergence to $x_{UA}$ for any $\alpha>0$ | Rate $(1-\alpha)\lambda_2(W)$ |
| 2 | Discontinuous (first-order) phase transition at $\alpha=0$ | Jump $= \|x_{UA} - \bar{x}(0)\|$ |
| 3 | Critical coupling in HK-UA (heuristic) | $\alpha^*(\varepsilon)\approx(1-3\varepsilon)/(1-2\varepsilon)$ |
| 4 | Adaptive UA fixed point (exact) | $x^* = (\beta\bar{x}(0)+\alpha x_{UA}(0))/(\alpha+\beta)$ |
| 5 | Stratification under heterogeneous coupling | $T_i^* \propto 1/\alpha_i$, cascade mechanism |

**Important:** The heuristic formula for Prop 3 gives $\alpha^* \approx 0.125$–$0.167$, but simulated $\alpha^*$ is ~0.001–0.003. The two-order-of-magnitude gap is due to cascade effects (explained in the Remark after Prop 3).

---

## Simulations (6 experiments)

| Exp | File | Figure in paper | What it shows |
|-----|------|-----------------|---------------|
| 1a | experiments.py `experiment1_*` | (no figure) | Baseline HK cluster count; topology invariance |
| 1b | same | (no figure) | B1-HK on complete/ring/SW/BA topologies → identical K |
| 2 | `experiment2_*` | Fig 1 `exp2_variance_vs_alpha.png` | Variance vs α (log axis, x-max=0.004); sharp drop at α*≈0.001–0.003 |
| 3 | `experiment3_*` | Fig 2 `exp3_phase_transition_heatmap.png` | (α,ε) heatmap; empirical α*(ε) boundary |
| 4 | `experiment4_*` | Fig 3 `exp4_adaptive_ua.png` | Adaptive UA: sim vs theory x*(β); one orange curve + black dashed theory |
| 5 | `experiment5_*` | Figs 4A/4B `exp5_heterogeneous_{A,B}.png` | Cascade trajectories (A); K vs distribution (B) |
| 6 | `experiment6_*` | Figs 5A/5B `exp6_multiple_ua_{A,B}.png` | Fixed k=4 UAs (A); adaptive k=4, β sweep (B) |

**Regenerating figures:** Most experiments are fast. Exp 5 takes a few minutes; cached in `results_exp5.pkl`. Run individual functions from `plotting.py` or the full pipeline via `run_all.py`.

---

## Paper structure (`main_icml.tex`)

1. **Abstract** — correct α*≈0.001–0.003 (not 0.01–0.04)
2. **Introduction** — motivation, 5 contributions, paper roadmap
3. **Related Work** — 2 subsections: opinion dynamics; media models & AI homogenization
4. **Model** — Stage A (baseline), Stage B (B1/B2/B3), parameter table, hypotheses H1–H4
5. **Analytical Results** — Props 1–5 with proofs (Prop 1 inline; Props 2–5 reference Appendix)
6. **Simulations** — setup, Experiments 1a–1b, 2–3, 4, 5, 6
7. **Discussion** — AI design implications, policy threshold, adaptive AI, stratification, limitations
8. **Conclusion**
9. **Impact Statement**
10. **References** (~20 citations)
11. **Appendix** — proofs for Props 2–5

---

## Known issues / confirmed bugs (fixed)

- `\usepackage{microtype}` was briefly removed due to math errors but is fine with icml2026 style
- `[Author Names]` in author field caused hyperref issues → replaced with plain text `Anonymous Author(s)`
- Figures are PNGs in `simulation/figures/`; paths in LaTeX are `../simulation/figures/filename.png`
- The `[draft]` option was removed from graphicx so figures render

---

## What was done across sessions

### Session 1 (orchestrated via subagents, ~21 min wall-clock)
- Literature review → model specification → analytical derivations → Python simulation codebase → full paper draft

### Session 2 (iterative improvements)
- Fixed uninformative figures (wrong ε regime for heatmap; switched to log α axis)
- Fixed Fig 2: switched from theoretical to empirical α*(ε) boundary
- Fixed Fig 3 (adaptive UA): removed duplicate blue curve (x_UA and x̄ co-converge identically)
- Fixed Fig 4A left panel x-axis (clipped to conv_t+5)
- Reformatted paper to ICML 2026 two-column template (`icml2026.sty`)
- Downloaded `icml2026.zip` from official ICML 2026 styles page
- Added k-UA model extension (Experiment 6, Figs 5A/5B) with new framing around LLM ecosystem diversity
- Retitled paper: "Opinion Homogenization via Large Language Models"
- Split wide figures into A/B panels for two-column layout (Figs 4A/4B, 5A/5B)
- Trimmed to 8 main pages + 2 appendix by moving proofs 2–5 to appendix

### Session 3 (paper review and corrections)
- Fixed 10 issues found during careful read-through:
  1. α* values corrected from 0.01–0.04 → 0.001–0.003 throughout (abstract, contributions, sim text, figure captions)
  2. Abstract: merged two redundant governance sentences
  3. Proof of Prop 1: fixed equation index error (`e(t+1)=…=(1-α)^t` → split into recurrence + inductive solution)
  4. Fig 3 caption: fixed "solid curves / several pairs" → single curve; corrected axis reference from "α/β" to "β"
  5. Experiment 5 text: rewrote to match actual figure (4 distributions: No UA, Bimodal 0%/30%, 0%/60%, Heavy-tail Beta(0.5,2)); fixed wrong α values (was 0.55/0.05, now 0.60/0.00)
  6. Fig 4B caption: fixed "K=3" → "K≈3.4–3.8"; described all four conditions
  7. Section title: "Experiments 1 & 1ext" → "Experiments 1a–1b"
  8. Contributions: "confirm H1–H4" → "H1, H3, H4 fully; H2 partially"
  9. Remark after Prop 3: replaced forward reference "cascade effects discussed above" with brief explanation + reference to Prop 5
  10. Fig 1 caption: updated to mention ε=0.15 curve and empirical α* dashed lines

---

## Pending / future work
- Empirical calibration of α from LLM interaction data
- Multi-dimensional opinion dynamics (d>1)
- Network effects on P(α) (AI usage correlated with social position)
- H2 (transition sharpness fixed vs adaptive) only partially confirmed; needs wider parameter sweep
