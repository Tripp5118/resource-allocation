"""Figures and tables for the LLM-guided resource allocation paper.

Usage:
    python analysis/paper_figures.py [--results-dir DIR] [--out-dir DIR] [TARGET ...]

The results directory must contain results_mpd_v3/ and results_magnets_v3/ as
written by replicate.sh. It defaults to campaign_results/, the published runs.
With no targets, everything is generated.

Hypervolume is normalized by the oracle hypervolume of each design space's
Pareto front. Error bands are 95% t-intervals on the mean across seeds.
Pairwise comparisons are two-sided paired Wilcoxon signed-rank tests, paired by
seed.
"""

import argparse
import json
import math
import warnings
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
GROUND_TRUTH = REPO / "ground_truth_models"
MPD_GT_DIR = GROUND_TRUTH / "TiVNbMoHfTaW-MeltingVsDensity"
MAG_GT_DIR = GROUND_TRUTH / "FeCoNiVMoCrCuMnCWTaNbAlTiSi-Magnets"
MPD_TRAINING_DATA = GROUND_TRUTH / "data" / "refractory-meltingvsdensity.xlsx"
MAG_TRAINING_DATA = GROUND_TRUTH / "data" / "FeNiCo_comp-prop_imp.csv"

MPD_ELEMENTS = ["Ti", "V", "Nb", "Mo", "Hf", "Ta", "W"]
MAG_ELEMENTS = ["Fe", "Co", "Ni", "V", "Mo", "C", "Nb", "Ti", "Si"]

EXPERIMENTS = {
    "MP-D-MAIN":      ("results_mpd_v3/MP-D_MAIN",          [],       []),
    "MP-D-BUDGET":    ("results_mpd_v3/MP-D_BUDGET",        [15],     ["Budget cut"]),
    "MP-D-TIME":      ("results_mpd_v3/MP-D_TIME",          [15],     ["Time cut"]),
    "MP-D-DUAL":      ("results_mpd_v3/MP-D_DUAL",          [5, 7],   ["Time cut", "Budget cut"]),
    "Magnets-MAIN":   ("results_magnets_v3/Magnets-MAIN",   [],       []),
    "Magnets-BUDGET": ("results_magnets_v3/Magnets-BUDGET", [18],     ["Budget cut"]),
    "Magnets-TIME":   ("results_magnets_v3/Magnets-TIME",   [13],     ["Time cut"]),
    "Magnets-DUAL":   ("results_magnets_v3/Magnets-DUAL",   [13, 18], ["Time cut", "Budget cut"]),
}
EVENTS = ["MAIN", "BUDGET", "TIME", "DUAL"]
EVENT_TITLES = {
    "MAIN":   "Baseline",
    "BUDGET": "Budget event",
    "TIME":   "Time event",
    "DUAL":   "Dual event",
}
SHORT_EVENT_LABELS = {"MAIN": "Baseline", "BUDGET": "Budget cut", "TIME": "Time cut", "DUAL": "Dual"}

STRATEGIES = ["Adaptive-Beta", "Rule-Swap", "qEHVI", "Fixed-Mixed"]
AGENT_STRATEGIES = ["Adaptive-Beta", "Rule-Swap"]
MIXED_STRATEGIES = ["Adaptive-Beta", "Rule-Swap", "Fixed-Mixed"]
TABLE_STRATEGIES = ["Adaptive-Beta", "Rule-Swap", "Fixed-Mixed", "qEHVI"]
TABLE_PAIRS = [
    ("Adaptive-Beta", "Rule-Swap"),
    ("Adaptive-Beta", "Fixed-Mixed"),
    ("Adaptive-Beta", "qEHVI"),
    ("Rule-Swap", "Fixed-Mixed"),
    ("Rule-Swap", "qEHVI"),
    ("qEHVI", "Fixed-Mixed"),
]

LABELS = {
    "Adaptive-Beta": "Adaptive-β",
    "Rule-Swap":     "Rule-Swap",
    "qEHVI":         "qEHVI",
    "Fixed-Mixed":   "Fixed Mixed",
}
TEX_LABELS = {s: label.replace("β", r"$\beta$") for s, label in LABELS.items()}

COLORS = {
    "Adaptive-Beta": "#0072B2",
    "Rule-Swap":     "#CC79A7",
    "qEHVI":         "#D55E00",
    "Fixed-Mixed":   "#009E73",
}
LINESTYLES = {
    "Adaptive-Beta": "-",
    "Rule-Swap":     "-.",
    "qEHVI":         "--",
    "Fixed-Mixed":   ":",
}
MARKERS = {
    "Adaptive-Beta": "o",
    "Rule-Swap":     "D",
    "qEHVI":         "s",
    "Fixed-Mixed":   "^",
}
EVENT_COLORS = ["#CC79A7", "#E69F00"]

BETA_LOGS = {
    "Adaptive-Beta": ("adaptive_scout_iteration_", "beta"),
    "Rule-Swap":     ("qehvi_first_iteration_", "call1_beta"),
}

BATCH_SIZE = 5

FIG_W_DOUBLE = 7.0
FIG_H_STANDARD = 3.2
TITLE_SIZE = 11
LABEL_SIZE = 11
TICK_SIZE = 9.5
LEGEND_SIZE = 8.5

RHEA_LOW_COLOR = np.array([0.32, 0.58, 0.84])
RHEA_HIGH_COLOR = np.array([0.84, 0.18, 0.18])
SQRT3_2 = np.sqrt(3) / 2.0

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


@contextmanager
def _style(scale=1.0):
    """Text sizes for one figure, scaled so wide figures stay legible at page width."""
    with plt.rc_context({
        "font.size": TICK_SIZE * scale,
        "axes.titlesize": TITLE_SIZE * scale,
        "axes.labelsize": LABEL_SIZE * scale,
        "xtick.labelsize": TICK_SIZE * scale,
        "ytick.labelsize": TICK_SIZE * scale,
        "legend.fontsize": LEGEND_SIZE * scale,
    }):
        yield


def _panel_title(letter, text):
    return rf"$\mathbf{{({letter})}}$ {text}"


def system_of(name):
    return "Magnets" if name.startswith("Magnets") else "MP-D"


def experiment_name(system, event):
    return f"{system}-{event}"


# ===========================================================================
# Data loading
# ===========================================================================

class Results:
    """Campaign results under a results directory, loaded lazily and cached."""

    def __init__(self, root):
        self.root = Path(root)
        self._convergence = {}
        self._beta = {}

    def seed_dirs(self, name):
        exp_dir = self.root / EXPERIMENTS[name][0]
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"{exp_dir} does not exist; check --results-dir")
        return sorted(d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed"))

    def convergence(self, name):
        """{strategy: [per-seed convergence DataFrame]}."""
        if name not in self._convergence:
            data = {s: [] for s in STRATEGIES}
            for seed_dir in self.seed_dirs(name):
                for strat in STRATEGIES:
                    csv = seed_dir / strat / f"{strat}_convergence.csv"
                    if csv.exists():
                        data[strat].append(pd.read_csv(csv))
            self._convergence[name] = data
        return self._convergence[name]

    def beta(self, name):
        """{agent strategy: [per-seed {iteration: beta}]}, read from the agent logs."""
        if name not in self._beta:
            data = {s: [] for s in AGENT_STRATEGIES}
            for seed_dir in self.seed_dirs(name):
                for strat in AGENT_STRATEGIES:
                    prefix, key = BETA_LOGS[strat]
                    trajectory = {}
                    for log in (seed_dir / strat / "agent_logs").glob(f"{prefix}*.json"):
                        entry = json.loads(log.read_text())
                        iteration = int(entry.get("iteration", -1))
                        value = entry.get(key)
                        if iteration >= 0 and value is not None:
                            trajectory[iteration] = float(value)
                    data[strat].append(trajectory)
            self._beta[name] = data
        return self._beta[name]


def _pareto_mask(Y):
    return is_non_dominated(torch.tensor(Y, dtype=torch.float64)).numpy()


@cache
def oracle_hv(system):
    """Hypervolume of the ground-truth Pareto front of the full design space."""
    if system == "MP-D":
        preds = pd.read_csv(MPD_GT_DIR / "design_space_predictions.csv").values
        Y = np.column_stack([preds[:, 0], -np.abs(preds[:, 1])])
    else:
        preds = pd.read_csv(MAG_GT_DIR / "design_space_predictions.csv").values
        Y = np.column_stack([
            preds[:, 0],
            -np.log(np.abs(preds[:, 1]) + 1e-6),
            np.log(preds[:, 5] + 1e-6),
        ])
    ref = np.min(Y, axis=0) - 0.1
    front = Y[_pareto_mask(Y)]
    hv = Hypervolume(torch.tensor(ref, dtype=torch.float64))
    return float(hv.compute(torch.tensor(front, dtype=torch.float64)))


# ===========================================================================
# Statistics
# ===========================================================================

def _t_interval(values, scale=1.0, confidence=0.95):
    n = len(values)
    mean = np.mean(values) / scale
    if n < 2:
        return mean, mean, mean
    half = stats.t.ppf((1 + confidence) / 2, n - 1) * np.std(values, ddof=1) / np.sqrt(n) / scale
    return mean, mean - half, mean + half


def _as_arrays(rows):
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return tuple(np.array(column) for column in zip(*rows))


def mean_ci(dfs, column, scale=1.0):
    """Per-iteration (iterations, means, lower, upper) across seeds."""
    dfs = [df for df in dfs if column in df.columns]
    if not dfs:
        return _as_arrays([])
    rows = []
    for i in range(int(max(df["iteration"].max() for df in dfs)) + 1):
        values = [df.loc[df["iteration"] == i, column].values[0]
                  for df in dfs if i in df["iteration"].values]
        if values:
            rows.append((i, *_t_interval(values, scale)))
    return _as_arrays(rows)


def mean_ci_by_iteration(trajectories):
    """mean_ci() for per-seed {iteration: value} dictionaries."""
    rows = []
    for i in sorted({k for t in trajectories for k in t}):
        values = [t[i] for t in trajectories if i in t]
        rows.append((i, *_t_interval(values)))
    return _as_arrays(rows)


def with_cumulative_mi(dfs):
    out = []
    for df in dfs:
        if "information_gain" in df.columns:
            df = df.copy()
            df["cum_mi"] = df["information_gain"].cumsum()
            out.append(df)
    return out


def final_hv(dfs, norm=1.0):
    return np.array([df["total_hypervolume"].iloc[-1] / norm
                     for df in dfs if "total_hypervolume" in df.columns])


def auc_hv(dfs, norm=1.0):
    """Trapezoid integral of the normalized HV curve over iteration index."""
    return np.array([np.trapezoid(df["total_hypervolume"].values / norm)
                     for df in dfs if "total_hypervolume" in df.columns])


def cumulative_mi(dfs):
    return np.array([df["information_gain"].sum()
                     for df in dfs if "information_gain" in df.columns])


def strategy_metrics(results, name):
    """{strategy: {"hv", "auc", "mi": per-seed arrays}} with HV oracle-normalized."""
    norm = oracle_hv(system_of(name))
    metrics = {}
    for strat, dfs in results.convergence(name).items():
        if dfs:
            metrics[strat] = {
                "hv": final_hv(dfs, norm),
                "auc": auc_hv(dfs, norm),
                "mi": cumulative_mi(dfs),
            }
    return metrics


def paired_wilcoxon(a, b):
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    if not np.any(diff != 0):
        return 1.0
    return float(stats.wilcoxon(diff, alternative="two-sided").pvalue)


def dominance_rate(a, b):
    """Fraction of seed-matched pairs where a > b."""
    n = min(len(a), len(b))
    return float(np.mean(a[:n] > b[:n]))


def sem(values):
    return np.std(values, ddof=1) / np.sqrt(len(values))


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def first_crossing(df, norm, threshold):
    """First iteration at which normalized HV reaches threshold, or None."""
    hits = np.where(df["total_hypervolume"].values / norm >= threshold)[0]
    return int(df["iteration"].values[hits[0]]) if len(hits) else None


def spread(values):
    """(mean, SD, SEM, 95% CI half-width) of a sample."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n == 0:
        return (np.nan,) * 4
    mean = float(np.mean(a))
    if n == 1:
        return mean, 0.0, 0.0, 0.0
    sd = float(np.std(a, ddof=1))
    se = sd / np.sqrt(n)
    return mean, sd, se, float(stats.t.ppf(0.975, n - 1)) * se


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _event_title(base, event_iters, wrap=False):
    if not event_iters:
        return base
    separator = "\n" if wrap else " "
    return f"{base}{separator}(iter. {', '.join(str(i) for i in event_iters)})"


def _add_event_markers(ax, event_iters, event_labels, text_top=False):
    for i, (iteration, label) in enumerate(zip(event_iters, event_labels)):
        color = EVENT_COLORS[i % len(EVENT_COLORS)]
        ax.axvline(iteration, color=color, linewidth=1.2, linestyle="--", zorder=3)
        lo, hi = ax.get_ylim()
        y = hi - 0.04 * (hi - lo) if text_top else lo + 0.04 * (hi - lo)
        ax.text(iteration + 0.15, y, label, fontsize="small", color=color,
                va="top" if text_top else "bottom", ha="left", rotation=90)


def _format_axis(ax, max_iter):
    ax.grid(False)
    if max_iter <= 12:
        ticks = np.arange(1, max_iter + 1)
    elif max_iter <= 30:
        ticks = np.arange(0, max_iter + 1, 5)
    else:
        ticks = np.arange(0, max_iter + 1, 10)
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0] - 0.5, max_iter + 0.5)


def _draw_band(ax, strat, iters, means, lo, hi, fill_alpha, sparse_markers=False, zorder=None):
    z = {} if zorder is None else {"zorder": zorder}
    ax.plot(iters, means, color=COLORS[strat], linewidth=1.5,
            linestyle=LINESTYLES[strat], label=LABELS[strat],
            marker=MARKERS[strat], markersize=3,
            markevery=max(1, len(iters) // 10) if sparse_markers else None, **z)
    ax.fill_between(iters, lo, hi, color=COLORS[strat], alpha=fill_alpha, **z)


def _plot_hv_panel(ax, data, hv_norm, event_iters=(), event_labels=(), ylabel=True, legend=True):
    max_iter = 0
    for strat in STRATEGIES:
        iters, means, lo, hi = mean_ci(data[strat], "total_hypervolume", scale=hv_norm)
        if len(iters):
            max_iter = max(max_iter, int(iters.max()))
            _draw_band(ax, strat, iters, means, lo, hi, 0.15, sparse_markers=True)
    _format_axis(ax, max_iter)
    ax.set_xlabel("BO Iteration")
    if ylabel:
        ax.set_ylabel("Normalized Hypervolume")
    if legend:
        ax.legend(loc="lower right", framealpha=0.9)
    if event_iters:
        _add_event_markers(ax, event_iters, event_labels)


def _plot_mi_panel(ax, data, ylabel=True):
    max_iter = 0
    for strat in STRATEGIES:
        iters, means, lo, hi = mean_ci(data[strat], "information_gain")
        if len(iters):
            max_iter = max(max_iter, int(iters.max()))
            _draw_band(ax, strat, iters, means, lo, hi, 0.15, sparse_markers=True)
    _format_axis(ax, max_iter)
    ax.set_xlabel("BO Iteration")
    if ylabel:
        ax.set_ylabel("Mutual Information")


def _plot_beta_panel(ax, beta, event_iters=(), event_labels=(), ylabel=True):
    max_iter = 0
    for strat in AGENT_STRATEGIES:
        iters, means, lo, hi = mean_ci_by_iteration(beta[strat])
        if len(iters):
            max_iter = max(max_iter, int(iters.max()))
            _draw_band(ax, strat, iters, means, lo, hi, 0.2, zorder=2)
    _format_axis(ax, max_iter)
    ax.set_xlabel("BO Iteration")
    if ylabel:
        ax.set_ylabel(r"$\beta$ (UCB coefficient)")
    if event_iters:
        _add_event_markers(ax, event_iters, event_labels, text_top=True)


def _plot_allocation_panel(ax, data, event_iters=(), event_labels=(), ylabel=True):
    max_iter = 0
    for strat in AGENT_STRATEGIES:
        iters, means, lo, hi = mean_ci(data[strat], "selected_n_exp")
        if len(iters):
            max_iter = max(max_iter, int(iters.max()))
            _draw_band(ax, strat, iters, means, lo, hi, 0.2)
    ax.axhline(BATCH_SIZE / 2, color="gray", linewidth=0.8, linestyle=":", zorder=1,
               label=f"Balanced (n={BATCH_SIZE / 2:.1f})")
    _format_axis(ax, max_iter)
    ax.set_xlabel("BO Iteration")
    ax.set_ylim(-0.2, BATCH_SIZE + 0.4)
    ax.set_yticks(range(0, BATCH_SIZE + 1))
    if ylabel:
        ax.set_ylabel("Exploration points\nselected ($n_{\\mathrm{exp}}$)")
    if event_iters:
        _add_event_markers(ax, event_iters, event_labels, text_top=True)


def _shared_legend(fig, ax, extra_handles=()):
    handles, _ = ax.get_legend_handles_labels()
    handles = [*handles, *extra_handles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), frameon=False)


def _save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _write(path, text):
    path.write_text(text)
    print(f"wrote {path}")


# ===========================================================================
# Figure 3: design and objective spaces
# ===========================================================================

def _ngon_vertices(n):
    angles = np.pi / 2 + np.arange(n) * 2 * np.pi / n
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _ngon_limits(X, vertices, margin=0.12):
    """Equal-aspect bounds around the projected data, widened to fit vertex labels."""
    X2d = X @ vertices
    xmin, xmax = X2d[:, 0].min(), X2d[:, 0].max()
    ymin, ymax = X2d[:, 1].min(), X2d[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin
    if dx > dy:
        ymin, ymax = ymin - (dx - dy) / 2, ymax + (dx - dy) / 2
    else:
        xmin, xmax = xmin - (dy - dx) / 2, xmax + (dy - dx) / 2
    m = margin * (xmax - xmin)
    labels = vertices * 1.14
    xmin = min(xmin - m, labels[:, 0].min())
    xmax = max(xmax + m, labels[:, 0].max())
    ymin = min(ymin - m, labels[:, 1].min())
    ymax = max(ymax + m, labels[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin
    if dx > dy:
        ymin, ymax = ymin - (dx - dy) / 2, ymax + (dx - dy) / 2
    else:
        xmin, xmax = xmin - (dy - dx) / 2, xmax + (dy - dx) / 2
    return (xmin, xmax), (ymin, ymax)


def _draw_ngon_frame(ax, vertices, labels, xlim, ylim):
    ax.add_patch(Polygon(vertices, closed=True, facecolor="#f0f0f0", edgecolor="#aaaaaa",
                         lw=0.8, zorder=0))
    for (vx, vy), label in zip(vertices, labels):
        lx, ly = vx * 1.12, vy * 1.12
        if xlim[0] <= lx <= xlim[1] and ylim[0] <= ly <= ylim[1]:
            ax.text(lx, ly, label, ha="center", va="center",
                    fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _draw_ternary_frame(ax):
    triangle = np.array([[0.5, SQRT3_2], [0.0, 0.0], [1.0, 0.0]])
    ax.add_patch(Polygon(triangle, closed=True, facecolor="#f0f0f0", edgecolor="#aaaaaa",
                         lw=0.8, zorder=0))
    offsets = [(0, 0.07), (-0.08, -0.05), (0.08, -0.05)]
    for (vx, vy), label, (ox, oy) in zip(triangle, ["Fe", "Co", "Ni"], offsets):
        ax.text(vx + ox, vy + oy, label, ha="center", va="center",
                fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, SQRT3_2 + 0.12)


def _ternary_xy(fe, co, ni):
    """Fe-Co-Ni barycentric coordinates, renormalized to their sum, to 2D."""
    total = fe + co + ni
    total = np.where(total > 0, total, 1.0)
    a, c = fe / total, ni / total
    return 0.5 * a + c, SQRT3_2 * a


def _luminance_order(colors):
    return np.argsort(0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2])


def _rhea_colors(tm, rho, tm_range, rho_range):
    """Blue (low Tm, high density) to red (high Tm, low density)."""
    t_tm = np.clip((tm - tm_range[0]) / (tm_range[1] - tm_range[0]), 0.0, 1.0)
    t_rho = np.clip(1.0 - (rho - rho_range[0]) / (rho_range[1] - rho_range[0]), 0.0, 1.0)
    quality = (0.6 * t_tm + 0.4 * t_rho).reshape(-1, 1)
    return np.clip(RHEA_LOW_COLOR + quality * (RHEA_HIGH_COLOR - RHEA_LOW_COLOR), 0.0, 1.0)


def _feconi_colors(ms, loghc, loghv, ms_range, hc_range, hv_range):
    """R = Ms, G = log HV, B = low log Hc, each normalized over the design space."""
    r = np.clip((ms - ms_range[0]) / (ms_range[1] - ms_range[0]), 0.0, 1.0)
    g = np.clip((loghv - hv_range[0]) / (hv_range[1] - hv_range[0]), 0.0, 1.0)
    b = np.clip(1.0 - (loghc - hc_range[0]) / (hc_range[1] - hc_range[0]), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


@_style(scale=1.0)
def make_fig3_design_space(results, out_dir):
    mpd_X = pd.read_csv(MPD_GT_DIR / "design_space_inputs.csv")[MPD_ELEMENTS].values
    mpd_preds = pd.read_csv(MPD_GT_DIR / "design_space_predictions.csv")
    mpd_tm = mpd_preds["Melting Point (K)"].values
    mpd_rho = mpd_preds["Density (g/cm3)"].values

    mag_X = pd.read_csv(MAG_GT_DIR / "design_space_inputs.csv")[MAG_ELEMENTS].values
    mag_preds = pd.read_csv(MAG_GT_DIR / "design_space_predictions.csv")
    mag_ms = mag_preds["Ms"].values
    mag_loghc = mag_preds["LogHC"].values
    mag_loghv = mag_preds["LogHV"].values

    mpd_train = pd.read_excel(MPD_TRAINING_DATA)
    mpd_train_X = mpd_train[MPD_ELEMENTS].values
    mpd_train_tm = mpd_train["Melting Point (K)"].values
    mpd_train_rho = mpd_train["PROP RT Density (g/cm3)"].values

    mag_train = pd.read_csv(MAG_TRAINING_DATA)
    mag_train_X = mag_train[MAG_ELEMENTS].values / 100.0
    row_sum = mag_train_X.sum(axis=1, keepdims=True)
    mag_train_X = mag_train_X / np.where(row_sum == 0, 1.0, row_sum)

    tm_range = (mpd_tm.min(), mpd_tm.max())
    rho_range = (mpd_rho.min(), mpd_rho.max())
    mag_ranges = [(v.min(), v.max()) for v in (mag_ms, mag_loghc, mag_loghv)]

    mpd_colors = _rhea_colors(mpd_tm, mpd_rho, tm_range, rho_range)
    mpd_train_colors = _rhea_colors(mpd_train_tm, mpd_train_rho, tm_range, rho_range)
    mag_colors = _feconi_colors(mag_ms, mag_loghc, mag_loghv, *mag_ranges)
    mag_train_colors = _feconi_colors(mag_train["Ms"].values, mag_train["LogHC"].values,
                                      mag_train["LogHV"].values, *mag_ranges)

    mpd_pareto = _pareto_mask(np.column_stack([mpd_tm, -mpd_rho]))
    mag_pareto = _pareto_mask(np.column_stack([mag_ms, -mag_loghc, mag_loghv]))

    vertices = _ngon_vertices(len(MPD_ELEMENTS))
    xlim, ylim = _ngon_limits(mpd_X, vertices)
    i_fe, i_co, i_ni = (MAG_ELEMENTS.index(e) for e in ("Fe", "Co", "Ni"))

    fig = plt.figure(figsize=(FIG_W_DOUBLE, FIG_H_STANDARD * 3 + 0.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    layer_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markeredgewidth=0, markersize=3, alpha=0.5, label="Design space"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444",
               markeredgewidth=0, markersize=5, label="Training data"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888888",
               markeredgecolor="k", markeredgewidth=0.4, markersize=7, label="Pareto front"),
    ]

    mpd_2d = mpd_X @ vertices
    mpd_train_2d = mpd_train_X @ vertices
    order = _luminance_order(mpd_colors)
    _draw_ngon_frame(ax_a, vertices, MPD_ELEMENTS, xlim, ylim)
    ax_a.scatter(mpd_2d[order, 0], mpd_2d[order, 1], c=mpd_colors[order],
                 s=0.5, alpha=0.18, linewidths=0, zorder=1)
    ax_a.set_xlim(*xlim)
    ax_a.set_ylim(*ylim)
    ax_a.scatter(mpd_train_2d[:, 0], mpd_train_2d[:, 1], c=mpd_train_colors,
                 s=6, alpha=0.75, linewidths=0, zorder=2)
    ax_a.scatter(mpd_2d[mpd_pareto, 0], mpd_2d[mpd_pareto, 1], c=mpd_colors[mpd_pareto],
                 s=40, marker="*", edgecolors="k", linewidths=0.3, zorder=4)
    ax_a.set_title(_panel_title("a", "RHEA — design space\n"
                   f"($n_{{ds}}={len(mpd_X):,}$, $n_{{tr}}={len(mpd_train_X):,}$)"))
    ax_a.legend(handles=layer_handles, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 0.0), bbox_transform=ax_a.transAxes,
                fontsize="xx-small", framealpha=0.9, borderpad=0.4, handlelength=1.2)

    ax_b.scatter(mpd_tm[order], mpd_rho[order], c=mpd_colors[order],
                 s=2, alpha=0.5, linewidths=0)
    ax_b.scatter(mpd_tm[mpd_pareto], mpd_rho[mpd_pareto], c=mpd_colors[mpd_pareto],
                 s=60, marker="*", edgecolors="k", linewidths=0.3, zorder=5)
    ax_b.set_xlabel(r"$T_m$ (K)")
    ax_b.set_ylabel(r"$\rho$ (g cm$^{-3}$)")
    ax_b.set_title(_panel_title("b", "RHEA — $T_m$ vs. $\\rho$\n"
                                f"($n_{{pf}}={int(mpd_pareto.sum()):,}$ Pareto points)"))

    cmap = LinearSegmentedColormap.from_list("rhea", [RHEA_LOW_COLOR, RHEA_HIGH_COLOR])
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=[ax_a, ax_b], orientation="vertical",
                        fraction=0.03, pad=0.01, aspect=25, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([r"$\rho$↑, $T_m$↓", r"$T_m$↑, $\rho$↓"], fontsize="small")

    tx, ty = _ternary_xy(mag_X[:, i_fe], mag_X[:, i_co], mag_X[:, i_ni])
    train_tx, train_ty = _ternary_xy(mag_train_X[:, i_fe], mag_train_X[:, i_co],
                                     mag_train_X[:, i_ni])
    order = _luminance_order(mag_colors)
    _draw_ternary_frame(ax_c)
    ax_c.scatter(tx[order], ty[order], c=mag_colors[order],
                 s=0.3, alpha=0.15, linewidths=0, zorder=1)
    ax_c.scatter(train_tx, train_ty, c=mag_train_colors, s=5, alpha=0.75, linewidths=0, zorder=2)
    ax_c.scatter(tx[mag_pareto], ty[mag_pareto], c=mag_colors[mag_pareto],
                 s=30, marker="*", edgecolors="k", linewidths=0.3, zorder=4)
    ax_c.set_title(_panel_title("c", "FeCoNi — design space\n"
                                f"($n_{{ds}}={len(mag_X):,}$, $n_{{tr}}={len(mag_train_X):,}$)"))
    ax_c.legend(handles=layer_handles, loc="upper right", fontsize="xx-small",
                framealpha=0.9, borderpad=0.4, handlelength=1.2)

    n_pf = int(mag_pareto.sum())
    for ax, x, y, xlabel, ylabel, letter in [
        (ax_d, mag_ms, -mag_loghc, r"$M_s$", r"$-\log H_c$", "d"),
        (ax_e, mag_ms, mag_loghv, r"$M_s$", r"$\log H_V$", "e"),
        (ax_f, -mag_loghc, mag_loghv, r"$-\log H_c$", r"$\log H_V$", "f"),
    ]:
        ax.scatter(x[order], y[order], c=mag_colors[order], s=1, alpha=0.25, linewidths=0)
        ax.scatter(x[mag_pareto], y[mag_pareto], c=mag_colors[mag_pareto],
                   s=30, marker="*", edgecolors="k", linewidths=0.2, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(_panel_title(letter, f"FeCoNi — {xlabel} vs. {ylabel}\n"
                                          f"($n_{{pf}}={n_pf:,}$ Pareto points)"))

    color_key = [
        Patch(facecolor=(0.8, 0.1, 0.1), label=r"R: $M_s$ ↑"),
        Patch(facecolor=(0.1, 0.65, 0.1), label=r"G: $\log H_V$ ↑"),
        Patch(facecolor=(0.15, 0.15, 0.80), label=r"B: $\log H_c$ ↓"),
    ]
    ax_d.legend(handles=color_key, loc="upper right", fontsize="xx-small", framealpha=0.9,
                borderpad=0.4, handlelength=1.0, title="Color key", title_fontsize="xx-small")

    _save(fig, out_dir / "fig3_design_space.png", tight=False)


# ===========================================================================
# Figure 4: baseline hypervolume and mutual information
# ===========================================================================

@_style(scale=1.0)
def make_fig4_baseline(results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_DOUBLE, FIG_H_STANDARD * 2 + 0.4))
    for col, (system, title) in enumerate([("MP-D", "RHEA"), ("Magnets", "FeCoNi")]):
        data = results.convergence(experiment_name(system, "MAIN"))
        _plot_hv_panel(axes[0, col], data, oracle_hv(system), ylabel=(col == 0), legend=(col == 0))
        axes[0, col].set_title(_panel_title("ab"[col], title))
        _plot_mi_panel(axes[1, col], data, ylabel=(col == 0))
        axes[1, col].set_title(_panel_title("cd"[col], title))
    _save(fig, out_dir / "fig4_baseline_combined.png")


# ===========================================================================
# Figure 5: information-optimization tradeoff
# ===========================================================================

MI_TRADEOFF_EXPERIMENTS = [experiment_name(s, e) for s in ("MP-D", "Magnets") for e in EVENTS]
MI_TRADEOFF_ALPHA = {"MP-D": 1.0, "Magnets": 0.55}
MI_TRADEOFF_REFERENCE = {
    ("MP-D-MAIN", "Adaptive-Beta"):    (51.5923, 19.539),
    ("MP-D-MAIN", "qEHVI"):            (52.0163, 14.232),
    ("MP-D-DUAL", "Fixed-Mixed"):      (7.1045, 5.316),
    ("Magnets-MAIN", "Adaptive-Beta"): (38.2267, 19.629),
    ("Magnets-DUAL", "qEHVI"):         (18.3905, 9.970),
}


def mi_tradeoff_points(results):
    """AUC-HV and cumulative MI of each mixed strategy as % change vs qEHVI."""
    means = {}
    for name in MI_TRADEOFF_EXPERIMENTS:
        for strat, m in strategy_metrics(results, name).items():
            means[name, strat] = (float(np.mean(m["auc"])), float(np.mean(m["mi"])))

    mismatches = [
        f"{name} {strat}: AUC-HV {means[name, strat][0]:.4f} (expected {auc:.4f}), "
        f"cumulative MI {means[name, strat][1]:.4f} (expected {mi:.4f})"
        for (name, strat), (auc, mi) in MI_TRADEOFF_REFERENCE.items()
        if abs(means[name, strat][0] - auc) > 5e-3 or abs(means[name, strat][1] - mi) > 5e-3
    ]
    if mismatches:
        warnings.warn("Results differ from the published values:\n  " + "\n  ".join(mismatches))

    points = []
    for name in MI_TRADEOFF_EXPERIMENTS:
        auc_q, mi_q = means[name, "qEHVI"]
        for strat in MIXED_STRATEGIES:
            auc, mi = means[name, strat]
            points.append({
                "experiment": name,
                "strategy": strat,
                "dx": 100.0 * (auc - auc_q) / auc_q,
                "dy": 100.0 * (mi - mi_q) / mi_q,
            })
    return points


def mi_tradeoff_summary(points):
    lines = [
        "Figure 5(c): % change vs qEHVI of the same experiment",
        f"{'experiment':16s} {'strategy':14s} {'dAUC-HV%':>9s} {'dMI%':>8s} {'ratio':>7s}",
    ]
    for p in points:
        ratio = abs(p["dy"] / p["dx"]) if p["dx"] != 0 else float("inf")
        lines.append(f"{p['experiment']:16s} {p['strategy']:14s} "
                     f"{p['dx']:9.2f} {p['dy']:8.2f} {ratio:7.1f}")
    above = sum(1 for p in points if p["dy"] > -p["dx"])
    ratios = [abs(p["dy"] / p["dx"]) for p in points if p["dx"] != 0]
    lines += [
        "",
        f"points above the equal-trade line: {above}/{len(points)}",
        f"gain/cost ratio: median {np.median(ratios):.1f}x, "
        f"range {min(ratios):.1f}-{max(ratios):.1f}x",
    ]
    return "\n".join(lines) + "\n"


@_style(scale=0.9)
def make_fig5_mi_tradeoff(results, out_dir):
    points = mi_tradeoff_points(results)
    _write(out_dir / "fig5_mi_tradeoff_summary.txt", mi_tradeoff_summary(points))

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE, 2.7))
    ax_a, ax_b, ax_c = axes

    for ax, name, title in [(ax_a, "MP-D-MAIN", "RHEA"), (ax_b, "Magnets-MAIN", "Fe-Co-Ni")]:
        max_iter = 0
        ends = []
        for strat in STRATEGIES:
            iters, means, lo, hi = mean_ci(with_cumulative_mi(results.convergence(name)[strat]),
                                           "cum_mi")
            if len(iters):
                max_iter = max(max_iter, int(iters.max()))
                _draw_band(ax, strat, iters, means, lo, hi, 0.15, sparse_markers=True)
                ends.append((float(iters[-1]), float(means[-1]), COLORS[strat]))
        _format_axis(ax, max_iter)
        ax.set_xlabel("BO Iteration")
        ax.set_title(title)
        ax.set_xlim(right=max_iter * 1.19)
        for x, y, color in ends:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(4, 0),
                        va="center", ha="left", fontsize="x-small", color=color,
                        fontweight="bold")
    ax_a.set_ylabel("Cum. MI (nats)")

    for p in points:
        ax_c.scatter(p["dx"], p["dy"], s=34, marker="o",
                     facecolor=COLORS[p["strategy"]], edgecolor="black", linewidth=0.4,
                     alpha=MI_TRADEOFF_ALPHA[system_of(p["experiment"])], zorder=3)

    xlo = min(p["dx"] for p in points)
    xhi = max(p["dx"] for p in points)
    pad = 0.08 * (xhi - xlo)
    xs = np.linspace(min(xlo - pad, 0.0), 0.0, 50)
    ax_c.plot(xs, -xs, linestyle=":", color="0.35", linewidth=1.2, zorder=1)
    ax_c.axhline(0, color="0.8", linewidth=0.8, zorder=0)
    ax_c.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    ax_c.scatter([0], [0], s=34, marker="*", color=COLORS["qEHVI"],
                 edgecolor="black", linewidth=0.4, zorder=4)
    ax_c.annotate("qEHVI", (0, 0), textcoords="offset points", xytext=(4, -9),
                  fontsize="small", color=COLORS["qEHVI"])
    ax_c.set_xlabel("AUC-HV change\nvs qEHVI (%)")
    ax_c.set_ylabel("Cum. MI change\nvs qEHVI (%)")
    ax_c.set_xlim(xlo - pad, max(pad, xhi + pad))
    ax_c.set_ylim(-6.0, max(p["dy"] for p in points) * 1.08)
    ax_c.set_title("All experiments")

    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", loc="left", x=-0.3, fontsize="x-large", fontweight="bold")

    fig.tight_layout()
    equal_trade = Line2D([], [], linestyle=":", color="0.35", linewidth=1.2, label="Equal trade")
    _shared_legend(fig, ax_a, extra_handles=[equal_trade])
    _save(fig, out_dir / "fig5_mi_tradeoff.png", tight=False)


# ===========================================================================
# Figure 6: hypervolume under resource events
# ===========================================================================

@_style(scale=1.15)
def make_fig6_event_hypervolume(results, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W_DOUBLE * 1.5, FIG_H_STANDARD * 2 + 0.4))
    for col, event in enumerate(EVENTS[1:]):
        for row, (system, title) in enumerate([("MP-D", "RHEA"), ("Magnets", "FeCoNi")]):
            name = experiment_name(system, event)
            _, event_iters, event_labels = EXPERIMENTS[name]
            ax = axes[row, col]
            _plot_hv_panel(ax, results.convergence(name), oracle_hv(system),
                           event_iters=event_iters, event_labels=event_labels,
                           ylabel=(col == 0), legend=False)
            ax.set_title(_panel_title(["abc", "def"][row][col],
                                      f"{title} — {_event_title(EVENT_TITLES[event], event_iters, wrap=True)}"))
    fig.tight_layout()
    _shared_legend(fig, axes[0, 0])
    _save(fig, out_dir / "fig6_event_hypervolume.png", tight=False)


# ===========================================================================
# Figure 7: beta trajectories
# ===========================================================================

@_style(scale=1.15)
def make_fig7_beta_trajectories(results, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_DOUBLE * 2.0, FIG_H_STANDARD * 2 + 0.4),
                             sharey="row")
    for col, event in enumerate(EVENTS):
        for row, (system, title) in enumerate([("MP-D", "RHEA"), ("Magnets", "FeCoNi")]):
            name = experiment_name(system, event)
            _, event_iters, event_labels = EXPERIMENTS[name]
            ax = axes[row, col]
            _plot_beta_panel(ax, results.beta(name), event_iters=event_iters,
                             event_labels=event_labels, ylabel=(col == 0))
            ax.set_title(_panel_title(["abcd", "efgh"][row][col],
                                      f"{title} — {_event_title(EVENT_TITLES[event], event_iters, wrap=True)}"))
    fig.tight_layout()
    _shared_legend(fig, axes[0, 0])
    _save(fig, out_dir / "fig7_beta_trajectories.png", tight=False)


# ===========================================================================
# Figure 8: exploration allocation under resource events
# ===========================================================================

@_style(scale=1.15)
def make_fig8_event_allocation(results, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W_DOUBLE * 1.5, FIG_H_STANDARD * 2 + 0.4),
                             sharey="row")
    for col, event in enumerate(EVENTS[1:]):
        for row, (system, title) in enumerate([("MP-D", "RHEA"), ("Magnets", "FeCoNi")]):
            name = experiment_name(system, event)
            _, event_iters, event_labels = EXPERIMENTS[name]
            ax = axes[row, col]
            _plot_allocation_panel(ax, results.convergence(name), event_iters=event_iters,
                                   event_labels=event_labels, ylabel=(col == 0))
            ax.set_title(_panel_title(["abc", "def"][row][col],
                                      f"{title} — {_event_title(EVENT_TITLES[event], event_iters, wrap=True)}"))
    fig.tight_layout()
    _shared_legend(fig, axes[0, 0])
    _save(fig, out_dir / "fig8_event_allocation.png", tight=False)


# ===========================================================================
# Tables
# ===========================================================================

TABLE_SYSTEMS = [("MP-D", "RHEA"), ("Magnets", "Fe-Co-Ni")]
METRIC_LABELS = {"hv": "Final HV (norm)", "auc": "AUC-HV (norm)", "mi": "Cumulative MI"}


def _format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _tex_p(p):
    body = "<0.001" if p < 0.001 else f"{p:.3f}"
    stars = significance_stars(p)
    return f"${body}$" + (f"\\textsuperscript{{{stars}}}" if stars else "")


def _tabular(columns, header, rows):
    return "\n".join([
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"


def make_stats_table_txt(results, out_dir):
    lines = [
        "Pairwise Wilcoxon signed-rank tests — final hypervolume, AUC-HV, cumulative MI",
        "=" * 90,
        "All tests: two-sided paired Wilcoxon signed-rank (paired by shared random seed)",
        "Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
        "Dom. rate: fraction of seed-matched pairs where strategy A > strategy B",
        "HV normalized by oracle HV (Pareto HV of full design space):",
        f"  MP-D oracle HV = {oracle_hv('MP-D'):.6f},  "
        f"Magnets oracle HV = {oracle_hv('Magnets'):.6f}",
        "=" * 90,
    ]
    for name in EXPERIMENTS:
        metrics = strategy_metrics(results, name)
        present = [s for s in STRATEGIES if s in metrics]
        if not present:
            lines.append(f"\n{name}  — no data")
            continue
        lines += [f"\n{'─' * 90}", f"  {name}", f"{'─' * 90}"]
        lines.append(f"  {'Strategy':<20} {'n':>4}  {'Final HV mean±SEM':>22}  "
                     f"{'AUC-HV mean±SEM':>22}  {'Cum. MI mean±SEM':>22}")
        for strat in present:
            m = metrics[strat]
            lines.append(
                f"  {LABELS[strat]:<20} {len(m['hv']):>4}  "
                f"{np.mean(m['hv']):>10.4f}±{sem(m['hv']):>9.5f}  "
                f"{np.mean(m['auc']):>10.4f}±{sem(m['auc']):>9.5f}  "
                f"{np.mean(m['mi']):>10.3f}±{sem(m['mi']):>8.4f}"
            )
        lines.append(f"\n  {'Pair':<38}  {'Metric':<15}  {'p-value':>10}  {'Sig':>4}  {'Dom. rate':>10}")
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                for key, label in METRIC_LABELS.items():
                    p = paired_wilcoxon(metrics[a][key], metrics[b][key])
                    lines.append(
                        f"  {LABELS[a] + ' vs ' + LABELS[b]:<38}  {label:<15}  "
                        f"{_format_p(p):>10}  {significance_stars(p) or 'ns':>4}  "
                        f"{dominance_rate(metrics[a][key], metrics[b][key]):>10.2f}"
                    )
    _write(out_dir / "stats_table.txt", "\n".join(lines))


def make_stats_table_tex(results, out_dir):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Summary statistics for all experiments. "
        r"HV normalized by oracle (full design space Pareto front). "
        r"AUC-HV: trapezoid integral of normalized HV curve. "
        r"Cum.~MI: cumulative mutual information. "
        r"Mean $\pm$ SEM across $n{=}25$ seeds. "
        r"Superscripts: $^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$ "
        r"(paired Wilcoxon vs.\ qEHVI).}",
        r"\label{tab:summary_stats}",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"System & Exp. & Strategy & $n$ & Final HV & AUC-HV & Cum.~MI \\",
        r"\midrule",
    ]
    previous_system = previous_event = None
    for name in EXPERIMENTS:
        metrics = strategy_metrics(results, name)
        system = "Ms/$H_c$/$H_V$" if system_of(name) == "Magnets" else r"$T_\mathrm{m}/\rho$"
        event = name.split("-")[-1]
        for strat in [s for s in STRATEGIES if s in metrics]:
            m = metrics[strat]
            stars = {key: "" for key in METRIC_LABELS}
            if strat != "qEHVI" and "qEHVI" in metrics:
                stars = {key: significance_stars(paired_wilcoxon(m[key], metrics["qEHVI"][key]))
                         for key in METRIC_LABELS}
            if previous_system is not None and system != previous_system:
                lines.append(r"\midrule")
            system_cell = system if system != previous_system else ""
            event_cell = event if event != previous_event else ""
            previous_system, previous_event = system, event
            cells = [
                f"${np.mean(m['hv']):.4f} \\pm {sem(m['hv']):.5f}$",
                f"${np.mean(m['auc']):.3f} \\pm {sem(m['auc']):.4f}$",
                f"${np.mean(m['mi']):.2f} \\pm {sem(m['mi']):.3f}$",
            ]
            cells = [c + (f"$^{{{stars[k]}}}$" if stars[k] else "")
                     for c, k in zip(cells, ["hv", "auc", "mi"])]
            lines.append(f"  {system_cell} & {event_cell} & {TEX_LABELS[strat]} & "
                         f"{len(m['hv'])} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(out_dir / "stats_table.tex", "\n".join(lines))


def make_tab_cumulative_mi(results, out_dir):
    rows = []
    for system, system_label in TABLE_SYSTEMS:
        for event in EVENTS:
            if rows:
                rows.append(r"\midrule")
            metrics = strategy_metrics(results, experiment_name(system, event))
            for j, strat in enumerate(TABLE_STRATEGIES):
                m = metrics[strat]
                rows.append(
                    f"{system_label if j == 0 else ''} & "
                    f"{SHORT_EVENT_LABELS[event] if j == 0 else ''} & {TEX_LABELS[strat]} & "
                    f"${np.mean(m['mi']):.2f} \\pm {sem(m['mi']):.2f}$ & "
                    f"${np.mean(m['hv']):.4f} \\pm {sem(m['hv']):.5f}$ & "
                    f"${np.mean(m['auc']):.2f} \\pm {sem(m['auc']):.3f}$ \\\\"
                )
    header = r"System & Experiment & Strategy & Cumulative MI (nats) & Final HV & AUC-HV \\"
    _write(out_dir / "tab_cumulative_mi_full.tex", _tabular("lllccc", header, rows))


def make_tab_significance(results, out_dir):
    rows = []
    for system, system_label in TABLE_SYSTEMS:
        for event in EVENTS:
            if rows:
                rows.append(r"\midrule")
            metrics = strategy_metrics(results, experiment_name(system, event))
            for j, (a, b) in enumerate(TABLE_PAIRS):
                p_values = [_tex_p(paired_wilcoxon(metrics[a][k], metrics[b][k]))
                            for k in METRIC_LABELS]
                rows.append(
                    f"{system_label if j == 0 else ''} & "
                    f"{SHORT_EVENT_LABELS[event] if j == 0 else ''} & "
                    f"{TEX_LABELS[a]} vs.\\ {TEX_LABELS[b]} & {' & '.join(p_values)} \\\\"
                )
    header = (r"System & Experiment & Strategy pair & $p$ (final HV) & "
              r"$p$ (AUC-HV) & $p$ (cum.\ MI) \\")
    _write(out_dir / "tab_significance.tex", _tabular("lllccc", header, rows))


CONVERGENCE_SYSTEMS = [("Magnets", "Fe-Co-Ni", 0.80), ("MP-D", "RHEA", 0.90)]
CONVERGENCE_EVENT_LABELS = {
    "MAIN": "Baseline",
    "BUDGET": "Budget event",
    "TIME": "Time event",
    "DUAL": "Dual event",
}


def convergence_milestones(results):
    """{(experiment, strategy): spread of batches to 50% and the upper HV milestone}."""
    milestones = {}
    for system, _, upper in CONVERGENCE_SYSTEMS:
        norm = oracle_hv(system)
        for event in EVENTS:
            name = experiment_name(system, event)
            for strat, dfs in results.convergence(name).items():
                dfs = [df for df in dfs if "total_hypervolume" in df.columns]
                half = [first_crossing(df, norm, 0.50) for df in dfs]
                top = [first_crossing(df, norm, upper) for df in dfs]
                half = [c for c in half if c is not None]
                top = [c for c in top if c is not None]
                milestones[name, strat] = {
                    "n50": len(half),
                    "half": spread(half),
                    "n_upper": len(top),
                    "upper": spread(top),
                    "reached": 100.0 * len(top) / len(dfs) if dfs else float("nan"),
                }
    return milestones


def make_convergence_summary_txt(results, out_dir):
    milestones = convergence_milestones(results)
    lines = [
        f"{'experiment':<15} {'strategy':<12} "
        f"{'n50':>4} {'mean50':>7} {'sd':>6} {'sem':>6} {'ci95':>6}   "
        f"{'nUP':>4} {'meanUP':>7} {'sd':>6} {'sem':>6} {'ci95':>6} {'%reach':>7}",
        "-" * 116,
    ]
    for (name, strat), m in milestones.items():
        lines.append(
            f"{name:<15} {LABELS[strat]:<12} "
            f"{m['n50']:>4} {m['half'][0]:>7.1f} {m['half'][1]:>6.1f} "
            f"{m['half'][2]:>6.1f} {m['half'][3]:>6.1f}   "
            f"{m['n_upper']:>4} {m['upper'][0]:>7.1f} {m['upper'][1]:>6.1f} "
            f"{m['upper'][2]:>6.1f} {m['upper'][3]:>6.1f} {m['reached']:>6.0f}%"
        )
    _write(out_dir / "convergence_milestones.txt", "\n".join(lines) + "\n")


def make_tab_convergence_speed(results, out_dir):
    milestones = convergence_milestones(results)
    lines = [
        r"\begin{tabular}{@{}lllccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{System} & \multirow{2}{*}{Experiment} & \multirow{2}{*}{Strategy}",
        r"  & \multicolumn{1}{c}{Batches to 50\% HV}",
        f"  & \\multicolumn{{2}}{{c}}{{Batches to {CONVERGENCE_SYSTEMS[0][2] * 100:.0f}\\% HV}} \\\\",
        r"\cmidrule(l){5-6}",
        r" & & & Mean $\pm$ 95\% CI & Mean $\pm$ 95\% CI & \% reached \\",
        r"\midrule",
    ]
    for k, (system, system_label, upper) in enumerate(CONVERGENCE_SYSTEMS):
        if k:
            lines += [
                r"\midrule",
                f"\\multicolumn{{4}}{{@{{}}l}}{{}} & \\multicolumn{{2}}{{c}}"
                f"{{Batches to {upper * 100:.0f}\\% HV}} \\\\[-4pt]",
                r"\cmidrule(l){5-6}",
            ]
        for i, event in enumerate(EVENTS):
            if i:
                lines.append(r"  \addlinespace[2pt]")
            for j, strat in enumerate(STRATEGIES):
                m = milestones[experiment_name(system, event), strat]
                system_cell = (f"\\multirow{{{len(EVENTS) * len(STRATEGIES)}}}{{*}}{{{system_label}}}"
                               if i == 0 and j == 0 else "")
                event_cell = (f"\\multirow{{{len(STRATEGIES)}}}{{*}}"
                              f"{{{CONVERGENCE_EVENT_LABELS[event]}}}" if j == 0 else "")
                lines.append(
                    f"  {system_cell} & {event_cell} & {TEX_LABELS[strat]} & "
                    f"${m['half'][0]:.1f} \\pm {m['half'][3]:.1f}$ & "
                    f"${m['upper'][0]:.1f} \\pm {m['upper'][3]:.1f}$ & {m['reached']:.0f}\\% \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_dir / "tab_convergence_speed.tex", "\n".join(lines) + "\n")


# ===========================================================================
# Entry point
# ===========================================================================

TARGETS = {
    "fig3": [make_fig3_design_space],
    "fig4": [make_fig4_baseline],
    "fig5": [make_fig5_mi_tradeoff],
    "fig6": [make_fig6_event_hypervolume],
    "fig7": [make_fig7_beta_trajectories],
    "fig8": [make_fig8_event_allocation],
    "tables": [
        make_stats_table_txt,
        make_stats_table_tex,
        make_tab_cumulative_mi,
        make_tab_significance,
        make_convergence_summary_txt,
        make_tab_convergence_speed,
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", metavar="TARGET",
                        help=f"any of: {', '.join(TARGETS)} (default: all)")
    parser.add_argument("--results-dir", type=Path, default=REPO / "campaign_results",
                        help="directory containing results_mpd_v3/ and results_magnets_v3/ "
                             "(default: campaign_results/)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures",
                        help="output directory (default: figures/)")
    args = parser.parse_args()

    unknown = sorted(set(args.targets) - set(TARGETS))
    if unknown:
        parser.error(f"unknown target(s): {', '.join(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = Results(args.results_dir)
    for target in args.targets or TARGETS:
        for make in TARGETS[target]:
            make(results, args.out_dir)


if __name__ == "__main__":
    main()
