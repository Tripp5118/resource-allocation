"""
paper_figures.py
----------------
Generates all publication figures for the LLM-guided MOBO paper.

Usage:
    source venv/bin/activate
    python paper_figures.py

All figures are saved to ./paper_figures/ in the project root.

Figure index:
    fig3_baseline_hypervolume.pdf   - HV convergence: K-CTE-MAIN + MP-D-MAIN
    fig4_agent_allocation.pdf       - Agent n_exp dynamics: K-CTE-MAIN + MP-D-MAIN
    fig5_event_allocation.pdf       - Agent n_exp shift at events: BUDGET events
    fig6_event_hypervolume.pdf      - HV convergence under all 6 event experiments (2×3)
    stats_table.txt                 - Paired t-test results for all experiments

Supplementary:
    suppA_kcte_events_allocation.pdf  - n_exp for K-CTE TIME + DUAL events
    suppB_mpd_events_allocation.pdf   - n_exp for MP-D TIME + DUAL events
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional

# ===========================================================================
# PATHS
# ===========================================================================

BASE = Path("final-results")
RERUNS = BASE / "reruns" / "results"
RESULTS = BASE / "results"

OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mapping: paper name -> (directory, event_iters, event_labels)
EXPERIMENTS = {
    "K-CTE-MAIN":   (RERUNS / "k-cte_3-step_3-batch",              [],          []),
    "K-CTE-BUDGET": (RERUNS / "k-cte_3-step_budget-event_3-batch", [3],         ["Budget cut"]),
    "K-CTE-TIME":   (RERUNS / "k-cte_3-step_time-event_3-batch",   [5],         ["Time cut"]),
    "K-CTE-DUAL":   (RERUNS / "k-cte_3-step_dual-events_3-batch",  [5, 7],      ["Time cut", "Budget cut"]),
    "MP-D-MAIN":    (RERUNS / "MP-D_MAIN",                          [],          []),
    "MP-D-BUDGET":  (RESULTS / "MP-D_BUDGET",                       [15],        ["Budget cut"]),
    "MP-D-TIME":    (RESULTS / "MP-D_TIME",                         [15],        ["Time cut"]),
    "MP-D-DUAL":    (RESULTS / "MP-D_DUAL",                         [5, 7],      ["Time cut", "Budget cut"]),
}

STRATEGIES = ["Agent_MultiStage", "qEHVI", "qUCB"]

LABELS = {
    "Agent_MultiStage": "3-Stage Agent",
    "qEHVI":            "qEHVI",
    "qUCB":             "qUCB",
}

# Okabe-Ito color-blind palette
COLORS = {
    "Agent_MultiStage": "#0072B2",   # blue
    "qEHVI":            "#D55E00",   # vermilion
    "qUCB":             "#009E73",   # bluish green
}

LINESTYLES = {
    "Agent_MultiStage": "-",
    "qEHVI":            "--",
    "qUCB":             ":",
}

MARKERS = {
    "Agent_MultiStage": "o",
    "qEHVI":            "s",
    "qUCB":             "^",
}

# Publication sizing (inches): 7" ≈ double column for ACS/RSC
FIG_W_DOUBLE = 7.0
FIG_W_SINGLE = 3.4
FIG_H_STANDARD = 3.2

# Font sizes
LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 8
TITLE_SIZE = 10

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_experiment(
    exp_dir: Path,
    metric: str = "total_hypervolume",
) -> Dict[str, List[pd.DataFrame]]:
    """Load convergence data for all strategies in an experiment."""
    data: Dict[str, List[pd.DataFrame]] = {s: [] for s in STRATEGIES}
    seeds = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    for seed in seeds:
        for strat in STRATEGIES:
            csv = seed / strat / f"{strat}_convergence.csv"
            if csv.exists():
                data[strat].append(pd.read_csv(csv))
    return data


def load_paired_final_hv(exp_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load final hypervolume values in seed-sorted order for paired t-tests.
    Returns {strategy: array of final HV values, one per seed}.
    """
    seeds = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    out: Dict[str, List[float]] = {s: [] for s in STRATEGIES}
    for seed in seeds:
        for strat in STRATEGIES:
            csv = seed / strat / f"{strat}_convergence.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                if "total_hypervolume" in df.columns:
                    out[strat].append(df["total_hypervolume"].iloc[-1])
    return {s: np.array(v) for s, v in out.items()}


# ===========================================================================
# STATISTICS
# ===========================================================================

def mean_ci(dfs: List[pd.DataFrame], metric: str, confidence: float = 0.95):
    """
    Compute mean and 95% CI across seeds for each iteration.
    Returns: (iterations, means, ci_lower, ci_upper)
    """
    if not dfs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    valid = [df for df in dfs if metric in df.columns]
    if not valid:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_iter = int(max(df["iteration"].max() for df in valid))
    iters, means, lo, hi = [], [], [], []

    for i in range(max_iter + 1):
        vals = [
            df.loc[df["iteration"] == i, metric].values[0]
            for df in valid
            if i in df["iteration"].values
        ]
        if vals:
            n = len(vals)
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            t_crit = stats.t.ppf((1 + confidence) / 2, max(n - 1, 1)) if n > 1 else 0.0
            iters.append(i)
            means.append(m)
            lo.append(m - t_crit * se)
            hi.append(m + t_crit * se)

    return np.array(iters), np.array(means), np.array(lo), np.array(hi)


def paired_ttest_all(exp_dir: Path) -> Dict[str, Tuple[float, float]]:
    """
    Run paired t-tests: Agent vs qEHVI and Agent vs qUCB.
    Returns {'vs_qEHVI': (t, p), 'vs_qUCB': (t, p)}.
    """
    hv = load_paired_final_hv(exp_dir)
    agent = hv["Agent_MultiStage"]
    results = {}
    for baseline in ["qEHVI", "qUCB"]:
        b = hv[baseline]
        n = min(len(agent), len(b))
        t, p = stats.ttest_rel(agent[:n], b[:n])
        results[f"vs_{baseline}"] = (t, p)
    return results


def format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


# ===========================================================================
# PLOTTING HELPERS
# ===========================================================================

def _add_event_markers(ax, event_iters: List[int], event_labels: List[str], ymax_frac=0.97):
    """Add vertical dashed lines and text labels for resource events."""
    colors_event = ["#CC79A7", "#F0E442"]   # reddish-purple, yellow (Okabe-Ito)
    for i, (it, lbl) in enumerate(zip(event_iters, event_labels)):
        ec = colors_event[i % len(colors_event)]
        ax.axvline(it, color=ec, linewidth=1.2, linestyle="--", zorder=3)
        ylim = ax.get_ylim()
        y_text = ylim[0] + ymax_frac * (ylim[1] - ylim[0])
        ax.text(it + 0.15, y_text, lbl, fontsize=7, color=ec,
                va="top", ha="left", rotation=90)


def _format_axis(ax, max_iter: int, step: int = 5):
    """Integer x-ticks, no grid."""
    ax.grid(False)
    if max_iter <= 12:
        ticks = np.arange(1, max_iter + 1)
    elif max_iter <= 30:
        ticks = np.arange(0, max_iter + 1, 5)
    else:
        ticks = np.arange(0, max_iter + 1, 10)
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0] - 0.5, max_iter + 0.5)


def _plot_hv_panel(
    ax,
    data: Dict[str, List[pd.DataFrame]],
    event_iters: List[int] = (),
    event_labels: List[str] = (),
    ylabel: bool = True,
    legend: bool = True,
):
    """Draw hypervolume mean ± 95% CI for all strategies onto ax."""
    max_iter = 0
    for strat in STRATEGIES:
        iters, means, lo, hi = mean_ci(data[strat], "total_hypervolume")
        if len(iters) == 0:
            continue
        max_iter = max(max_iter, int(iters.max()))
        c = COLORS[strat]
        ax.plot(iters, means, color=c, linewidth=1.5,
                linestyle=LINESTYLES[strat], label=LABELS[strat],
                marker=MARKERS[strat], markersize=3, markevery=max(1, len(iters)//10))
        ax.fill_between(iters, lo, hi, color=c, alpha=0.15)

    _format_axis(ax, max_iter)
    ax.set_xlabel("BO Iteration")
    if ylabel:
        ax.set_ylabel("Hypervolume")
    if legend:
        ax.legend(loc="lower right", framealpha=0.9)
    if event_iters:
        _add_event_markers(ax, event_iters, event_labels)


def _plot_alloc_panel(
    ax,
    data: Dict[str, List[pd.DataFrame]],
    batch_size: int,
    event_iters: List[int] = (),
    event_labels: List[str] = (),
    ylabel: bool = True,
):
    """Draw Agent n_exp mean ± 95% CI onto ax, with balanced reference line."""
    iters, means, lo, hi = mean_ci(data["Agent_MultiStage"], "selected_n_exp")
    if len(iters) == 0:
        return
    max_iter = int(iters.max())

    c = COLORS["Agent_MultiStage"]
    ax.plot(iters, means, color=c, linewidth=1.5, linestyle="-",
            marker="o", markersize=3)
    ax.fill_between(iters, lo, hi, color=c, alpha=0.2)

    # Balanced reference line
    ax.axhline(batch_size / 2, color="gray", linewidth=0.8, linestyle=":", zorder=1,
               label=f"Balanced (n={batch_size/2:.1f})")

    _format_axis(ax, max_iter)
    ax.set_xlabel("BO Iteration")
    ax.set_ylim(-0.2, batch_size + 0.4)
    ax.set_yticks(range(0, batch_size + 1))
    if ylabel:
        ax.set_ylabel("Exploration points selected ($n_{\\mathrm{exp}}$)")

    if event_iters:
        _add_event_markers(ax, event_iters, event_labels)


# ===========================================================================
# FIGURE 3: Baseline hypervolume convergence (K-CTE-MAIN + MP-D-MAIN)
# ===========================================================================

def make_fig3():
    kcte_dir, _, _ = EXPERIMENTS["K-CTE-MAIN"]
    mpd_dir,  _, _ = EXPERIMENTS["MP-D-MAIN"]

    kcte_data = load_experiment(kcte_dir)
    mpd_data  = load_experiment(mpd_dir)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H_STANDARD))

    _plot_hv_panel(axes[0], kcte_data, ylabel=True, legend=True)
    axes[0].set_title("(a) K-CTE-MAIN (Fe–Co–Ni–Cr–V)", fontsize=LABEL_SIZE)

    _plot_hv_panel(axes[1], mpd_data, ylabel=False, legend=False)
    axes[1].set_title("(b) MP-D-MAIN (Ti–V–Nb–Mo–Hf–Ta–W)", fontsize=LABEL_SIZE)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_baseline_hypervolume.pdf")
    fig.savefig(OUTPUT_DIR / "fig3_baseline_hypervolume.png")
    plt.close(fig)
    print("Saved fig3_baseline_hypervolume")


# ===========================================================================
# FIGURE 4: Agent allocation dynamics (K-CTE-MAIN + MP-D-MAIN)
# ===========================================================================

def make_fig4():
    kcte_dir, _, _ = EXPERIMENTS["K-CTE-MAIN"]
    mpd_dir,  _, _ = EXPERIMENTS["MP-D-MAIN"]

    kcte_data = load_experiment(kcte_dir)
    mpd_data  = load_experiment(mpd_dir)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H_STANDARD))

    _plot_alloc_panel(axes[0], kcte_data, batch_size=3, ylabel=True)
    axes[0].set_title("(a) K-CTE-MAIN ($q = 3$)", fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=LEGEND_SIZE, loc="lower left")

    _plot_alloc_panel(axes[1], mpd_data, batch_size=5, ylabel=False)
    axes[1].set_title("(b) MP-D-MAIN ($q = 5$)", fontsize=LABEL_SIZE)
    axes[1].legend(fontsize=LEGEND_SIZE, loc="lower left")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_agent_allocation.pdf")
    fig.savefig(OUTPUT_DIR / "fig4_agent_allocation.png")
    plt.close(fig)
    print("Saved fig4_agent_allocation")


# ===========================================================================
# FIGURE 5: Agent allocation at budget events (K-CTE-BUDGET + MP-D-BUDGET)
# ===========================================================================

def make_fig5():
    kcte_dir, kcte_ev, kcte_lb = EXPERIMENTS["K-CTE-BUDGET"]
    mpd_dir,  mpd_ev,  mpd_lb  = EXPERIMENTS["MP-D-BUDGET"]

    kcte_data = load_experiment(kcte_dir)
    mpd_data  = load_experiment(mpd_dir)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H_STANDARD))

    _plot_alloc_panel(axes[0], kcte_data, batch_size=3,
                      event_iters=kcte_ev, event_labels=kcte_lb, ylabel=True)
    axes[0].set_title("(a) K-CTE-BUDGET ($q = 3$, budget cut iter. 3)",
                      fontsize=LABEL_SIZE)

    _plot_alloc_panel(axes[1], mpd_data, batch_size=5,
                      event_iters=mpd_ev, event_labels=mpd_lb, ylabel=False)
    axes[1].set_title("(b) MP-D-BUDGET ($q = 5$, budget cut iter. 15)",
                      fontsize=LABEL_SIZE)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_event_allocation.pdf")
    fig.savefig(OUTPUT_DIR / "fig5_event_allocation.png")
    plt.close(fig)
    print("Saved fig5_event_allocation")


# ===========================================================================
# FIGURE 6: HV under all 6 event experiments (2 rows × 3 cols)
# ===========================================================================

def make_fig6():
    kcte_events = ["K-CTE-BUDGET", "K-CTE-TIME", "K-CTE-DUAL"]
    mpd_events  = ["MP-D-BUDGET",  "MP-D-TIME",  "MP-D-DUAL"]

    titles_kcte = [
        "(a) K-CTE-BUDGET\n(budget cut iter. 3)",
        "(b) K-CTE-TIME\n(time cut iter. 5)",
        "(c) K-CTE-DUAL\n(time cut iter. 5, budget cut iter. 7)",
    ]
    titles_mpd = [
        "(d) MP-D-BUDGET\n(budget cut iter. 15)",
        "(e) MP-D-TIME\n(time cut iter. 15)",
        "(f) MP-D-DUAL\n(time cut iter. 5, budget cut iter. 7)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W_DOUBLE * 1.5, FIG_H_STANDARD * 2 + 0.4))

    for col, exp_name in enumerate(kcte_events):
        exp_dir, ev_iters, ev_labels = EXPERIMENTS[exp_name]
        data = load_experiment(exp_dir)
        _plot_hv_panel(axes[0, col], data,
                       event_iters=ev_iters, event_labels=ev_labels,
                       ylabel=(col == 0), legend=(col == 0))
        axes[0, col].set_title(titles_kcte[col], fontsize=LABEL_SIZE - 1)

    for col, exp_name in enumerate(mpd_events):
        exp_dir, ev_iters, ev_labels = EXPERIMENTS[exp_name]
        data = load_experiment(exp_dir)
        _plot_hv_panel(axes[1, col], data,
                       event_iters=ev_iters, event_labels=ev_labels,
                       ylabel=(col == 0), legend=False)
        axes[1, col].set_title(titles_mpd[col], fontsize=LABEL_SIZE - 1)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_event_hypervolume.pdf")
    fig.savefig(OUTPUT_DIR / "fig6_event_hypervolume.png")
    plt.close(fig)
    print("Saved fig6_event_hypervolume")


# ===========================================================================
# SUPPLEMENTARY A: Agent n_exp for K-CTE TIME + DUAL events
# ===========================================================================

def make_suppA():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H_STANDARD))

    for col, exp_name, title in [
        (0, "K-CTE-TIME", "(a) K-CTE-TIME (time cut iter. 5)"),
        (1, "K-CTE-DUAL", "(b) K-CTE-DUAL (time cut iter. 5, budget cut iter. 7)"),
    ]:
        exp_dir, ev_iters, ev_labels = EXPERIMENTS[exp_name]
        data = load_experiment(exp_dir)
        _plot_alloc_panel(axes[col], data, batch_size=3,
                          event_iters=ev_iters, event_labels=ev_labels,
                          ylabel=(col == 0))
        axes[col].set_title(title, fontsize=LABEL_SIZE)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "suppA_kcte_events_allocation.pdf")
    fig.savefig(OUTPUT_DIR / "suppA_kcte_events_allocation.png")
    plt.close(fig)
    print("Saved suppA_kcte_events_allocation")


# ===========================================================================
# SUPPLEMENTARY B: Agent n_exp for MP-D TIME + DUAL events
# ===========================================================================

def make_suppB():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H_STANDARD))

    for col, exp_name, title in [
        (0, "MP-D-TIME", "(a) MP-D-TIME (time cut iter. 15)"),
        (1, "MP-D-DUAL", "(b) MP-D-DUAL (time cut iter. 5, budget cut iter. 7)"),
    ]:
        exp_dir, ev_iters, ev_labels = EXPERIMENTS[exp_name]
        data = load_experiment(exp_dir)
        _plot_alloc_panel(axes[col], data, batch_size=5,
                          event_iters=ev_iters, event_labels=ev_labels,
                          ylabel=(col == 0))
        axes[col].set_title(title, fontsize=LABEL_SIZE)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "suppB_mpd_events_allocation.pdf")
    fig.savefig(OUTPUT_DIR / "suppB_mpd_events_allocation.png")
    plt.close(fig)
    print("Saved suppB_mpd_events_allocation")


# ===========================================================================
# STATISTICS TABLE
# ===========================================================================

def make_stats_table():
    lines = []
    lines.append("Paired t-test results (Agent_MultiStage vs baselines, final hypervolume)")
    lines.append("=" * 75)
    lines.append(f"{'Experiment':<18} {'n':>4}  {'Agent mean±std':>18}  "
                 f"{'vs qEHVI':>20}  {'vs qUCB':>20}")
    lines.append("-" * 75)

    for exp_name, (exp_dir, _, _) in EXPERIMENTS.items():
        hv = load_paired_final_hv(exp_dir)
        agent = hv["Agent_MultiStage"]
        n = len(agent)

        tp1, p1 = stats.ttest_rel(agent, hv["qEHVI"][:n])
        tp2, p2 = stats.ttest_rel(agent, hv["qUCB"][:n])

        agent_str = f"{np.mean(agent):.2f}±{np.std(agent):.2f}"
        ehvi_str  = f"{np.mean(hv['qEHVI']):.2f}±{np.std(hv['qEHVI']):.2f}"
        ucb_str   = f"{np.mean(hv['qUCB']):.2f}±{np.std(hv['qUCB']):.2f}"

        sig1 = "***" if p1 < 0.001 else ("**" if p1 < 0.01 else ("*" if p1 < 0.05 else "ns"))
        sig2 = "***" if p2 < 0.001 else ("**" if p2 < 0.01 else ("*" if p2 < 0.05 else "ns"))

        lines.append(f"\n{exp_name}")
        lines.append(f"  n={n}  Agent: {agent_str}")
        lines.append(f"  qEHVI: {ehvi_str}  Agent vs qEHVI: t={tp1:.3f}, {format_p(p1)} {sig1}")
        lines.append(f"  qUCB:  {ucb_str}  Agent vs qUCB:  t={tp2:.3f}, {format_p(p2)} {sig2}")

    lines.append("\n" + "=" * 75)
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    lines.append("All tests: two-sided paired t-test (paired by shared random seed)")

    table_text = "\n".join(lines)
    out_path = OUTPUT_DIR / "stats_table.txt"
    out_path.write_text(table_text)
    print("Saved stats_table.txt")
    print(table_text)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_suppA()
    make_suppB()
    make_stats_table()

    print("\nDone.")
