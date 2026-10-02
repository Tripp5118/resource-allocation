"""
export_paper_data.py
--------------------
Exports all paper data to a single Excel workbook: paper_figures/paper_data.xlsx

Usage (run from project root):
    source .venv/bin/activate
    python _data_analysis/export_paper_data.py

Sheet layout:
    MP-D_MAIN, MP-D_BUDGET, MP-D_TIME, MP-D_DUAL,
    Magnets-MAIN, Magnets-BUDGET, Magnets-TIME, Magnets-DUAL
        → per-strategy per-iteration summary (mean ± SEM, percentile bands)
    Summary_Stats
        → Table A: per-strategy aggregate metrics (final HV, AUC-HV, cumulative MI)
        → Table B: all pairwise Wilcoxon signed-rank tests on those metrics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

RESULTS_MPD     = Path("campaign_results/results_mpd_v3")
RESULTS_MAGNETS = Path("campaign_results/results_magnets_v3")
OUTPUT_DIR = Path("paper_figures_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

STRATEGIES = ["Adaptive-Beta", "Rule-Swap", "qEHVI", "Fixed-Mixed"]

EXPERIMENTS = {
    "MP-D_MAIN":      RESULTS_MPD     / "MP-D_MAIN",
    "MP-D_BUDGET":    RESULTS_MPD     / "MP-D_BUDGET",
    "MP-D_TIME":      RESULTS_MPD     / "MP-D_TIME",
    "MP-D_DUAL":      RESULTS_MPD     / "MP-D_DUAL",
    "Magnets-MAIN":   RESULTS_MAGNETS / "Magnets-MAIN",
    "Magnets-BUDGET": RESULTS_MAGNETS / "Magnets-BUDGET",
    "Magnets-TIME":   RESULTS_MAGNETS / "Magnets-TIME",
    "Magnets-DUAL":   RESULTS_MAGNETS / "Magnets-DUAL",
}


BETA_LOG_PATTERNS = {
    "Adaptive-Beta": ("adaptive_scout_iteration_{}.json", "beta"),
    "Rule-Swap":     ("qehvi_first_iteration_{}.json", "call1_beta"),
}


def load_beta_trajectory(seed_dir: Path, strat: str) -> dict:
    """Load beta value per iteration from agent logs. Returns {iteration: beta}."""
    if strat not in BETA_LOG_PATTERNS:
        return {}
    pattern, key = BETA_LOG_PATTERNS[strat]
    log_dir = seed_dir / strat / "agent_logs"
    if not log_dir.exists():
        return {}
    result = {}
    for f in log_dir.glob("*.json"):
        if f.name.startswith(pattern.split("{")[0]):
            try:
                d = json.loads(f.read_text())
                it = int(d.get("iteration", -1))
                val = d.get(key)
                if it >= 0 and val is not None:
                    result[it] = float(val)
            except Exception:
                pass
    return result


def load_convergence(exp_dir: Path) -> dict:
    """Load all convergence CSVs. Returns {strategy: [DataFrame per seed]}."""
    data = {s: [] for s in STRATEGIES}
    seeds = sorted(d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed"))
    for seed in seeds:
        for strat in STRATEGIES:
            csv = seed / strat / f"{strat}_convergence.csv"
            if csv.exists():
                data[strat].append(pd.read_csv(csv))
    return data


def _iter_summary(dfs: list, metric: str) -> tuple:
    """Per-iteration mean, SEM, p25, p75 across seeds. Returns (iters, mean, sem, p25, p75)."""
    if not dfs:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    valid = [df for df in dfs if metric in df.columns]
    if not valid:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    max_iter = int(max(df["iteration"].max() for df in valid))
    iters, means, sems, p25s, p75s = [], [], [], [], []
    for i in range(max_iter + 1):
        vals = [
            df.loc[df["iteration"] == i, metric].values[0]
            for df in valid if i in df["iteration"].values
        ]
        if vals:
            a = np.array(vals)
            iters.append(i)
            means.append(np.mean(a))
            sems.append(np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
            p25s.append(np.percentile(a, 25))
            p75s.append(np.percentile(a, 75))
    return (np.array(iters), np.array(means), np.array(sems),
            np.array(p25s), np.array(p75s))


def build_experiment_sheet(exp_name: str, exp_dir: Path) -> pd.DataFrame:
    """Build per-strategy per-iteration summary DataFrame for one experiment."""
    data = load_convergence(exp_dir)
    rows = []
    for strat in STRATEGIES:
        dfs = data[strat]
        if not dfs:
            continue
        n_seeds = len(dfs)

        # n_evaluated map from first available df
        n_eval_map = {}
        for df in dfs:
            if "total_evaluated" in df.columns:
                for _, row in df.iterrows():
                    it = int(row["iteration"])
                    if it not in n_eval_map:
                        n_eval_map[it] = int(row["total_evaluated"])
                break

        # Beta trajectories from agent logs (agent strategies only)
        seeds = sorted(d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed"))
        beta_by_iter = {}  # {iter: [beta values across seeds]}
        if strat in BETA_LOG_PATTERNS:
            for seed_dir in seeds:
                traj = load_beta_trajectory(seed_dir, strat)
                for it, val in traj.items():
                    beta_by_iter.setdefault(it, []).append(val)

        iters_hv, hv_mean, hv_sem, hv_p25, hv_p75 = _iter_summary(dfs, "total_hypervolume")
        _, par_mean, par_sem, par_p25, par_p75 = _iter_summary(dfs, "pareto_size")
        _, mi_mean, mi_sem, mi_p25, mi_p75 = _iter_summary(dfs, "information_gain")
        _, nexp_mean, nexp_sem, _, _ = _iter_summary(dfs, "selected_n_exp")

        for idx, it in enumerate(iters_hv.astype(int)):
            beta_vals = beta_by_iter.get(it, [])
            row = {
                "experiment": exp_name,
                "strategy": strat,
                "n_seeds": n_seeds,
                "iteration": it,
                "n_evaluated": n_eval_map.get(it, np.nan),
                "hv_mean": hv_mean[idx] if idx < len(hv_mean) else np.nan,
                "hv_sem": hv_sem[idx] if idx < len(hv_sem) else np.nan,
                "hv_p25": hv_p25[idx] if idx < len(hv_p25) else np.nan,
                "hv_p75": hv_p75[idx] if idx < len(hv_p75) else np.nan,
                "pareto_mean": par_mean[idx] if idx < len(par_mean) else np.nan,
                "pareto_sem": par_sem[idx] if idx < len(par_sem) else np.nan,
                "mi_mean": mi_mean[idx] if idx < len(mi_mean) else np.nan,
                "mi_sem": mi_sem[idx] if idx < len(mi_sem) else np.nan,
                "nexp_mean": nexp_mean[idx] if idx < len(nexp_mean) else np.nan,
                "nexp_sem": nexp_sem[idx] if idx < len(nexp_sem) else np.nan,
                "beta_mean": np.mean(beta_vals) if beta_vals else np.nan,
                "beta_sem": (np.std(beta_vals, ddof=1) / np.sqrt(len(beta_vals))
                             if len(beta_vals) > 1 else np.nan),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _per_seed_metrics(dfs: list) -> tuple:
    """Returns (final_hv, auc_hv, cumulative_mi) arrays — one value per seed."""
    final_hv, auc_hv, cum_mi = [], [], []
    for df in dfs:
        if "total_hypervolume" in df.columns:
            hv = df["total_hypervolume"].values
            final_hv.append(hv[-1])
            auc_hv.append(np.trapezoid(hv))
        else:
            final_hv.append(np.nan)
            auc_hv.append(np.nan)
        if "information_gain" in df.columns:
            cum_mi.append(df["information_gain"].sum())
        else:
            cum_mi.append(np.nan)
    return np.array(final_hv), np.array(auc_hv), np.array(cum_mi)


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Wilcoxon signed-rank p-value; returns 1.0 if all differences are zero."""
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or not np.any(diff != 0):
        return 1.0
    _, p = stats.wilcoxon(diff, alternative="two-sided")
    return float(p)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def build_summary_sheet(all_data: dict) -> tuple:
    """
    Returns (table_a_df, table_b_df) for the Summary_Stats sheet.
    all_data: {exp_name: {strategy: [DataFrame per seed]}}
    """
    # Table A: per-experiment × strategy aggregate metrics
    rows_a = []
    # Store per-seed arrays for Table B tests
    seed_metrics = {}  # (exp_name, strategy) -> (final_hv, auc_hv, cum_mi)

    for exp_name, data in all_data.items():
        for strat in STRATEGIES:
            dfs = data[strat]
            if not dfs:
                continue
            fhv, ahv, cmi = _per_seed_metrics(dfs)
            seed_metrics[(exp_name, strat)] = (fhv, ahv, cmi)
            rows_a.append({
                "experiment": exp_name,
                "strategy": strat,
                "n_seeds": len(dfs),
                "final_hv_mean": np.nanmean(fhv),
                "final_hv_sem": np.nanstd(fhv, ddof=1) / np.sqrt(np.sum(~np.isnan(fhv))),
                "auc_hv_mean": np.nanmean(ahv),
                "auc_hv_sem": np.nanstd(ahv, ddof=1) / np.sqrt(np.sum(~np.isnan(ahv))),
                "cumulative_mi_mean": np.nanmean(cmi),
                "cumulative_mi_sem": np.nanstd(cmi, ddof=1) / np.sqrt(np.sum(~np.isnan(cmi))),
            })

    table_a = pd.DataFrame(rows_a)

    # Table B: pairwise Wilcoxon tests
    rows_b = []
    metrics_info = [
        ("final_hv", "Final HV"),
        ("auc_hv", "AUC-HV"),
        ("cumulative_mi", "Cumulative MI"),
    ]

    for exp_name in all_data:
        available = [s for s in STRATEGIES if (exp_name, s) in seed_metrics]
        for s_a, s_b in combinations(available, 2):
            for metric_key, metric_label in metrics_info:
                idx = {"final_hv": 0, "auc_hv": 1, "cumulative_mi": 2}[metric_key]
                a_vals = seed_metrics[(exp_name, s_a)][idx]
                b_vals = seed_metrics[(exp_name, s_b)][idx]
                n = min(len(a_vals), len(b_vals))
                p = _wilcoxon_p(a_vals, b_vals)
                dom_rate = float(np.mean(a_vals[:n] > b_vals[:n]))
                mean_a = np.nanmean(a_vals)
                mean_b = np.nanmean(b_vals)
                rows_b.append({
                    "experiment": exp_name,
                    "strategy_a": s_a,
                    "strategy_b": s_b,
                    "metric": metric_label,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "wilcoxon_p": p,
                    "significant": _sig_stars(p),
                    "dominance_rate_a_over_b": dom_rate,
                })

    table_b = pd.DataFrame(rows_b)
    return table_a, table_b


def main():
    print("Loading experiment data...")
    all_data = {}
    for exp_name, exp_dir in EXPERIMENTS.items():
        print(f"  {exp_name}...", end=" ", flush=True)
        all_data[exp_name] = load_convergence(exp_dir)
        strat_counts = {s: len(dfs) for s, dfs in all_data[exp_name].items() if dfs}
        print(" | ".join(f"{s}: {n}" for s, n in strat_counts.items()))

    out_path = OUTPUT_DIR / "paper_data.xlsx"
    print(f"\nWriting {out_path} ...")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Per-experiment sheets
        for exp_name, exp_dir in EXPERIMENTS.items():
            print(f"  Sheet: {exp_name}")
            df = build_experiment_sheet(exp_name, exp_dir)
            df.to_excel(writer, sheet_name=exp_name, index=False)

        # Summary stats sheet
        print("  Sheet: Summary_Stats")
        table_a, table_b = build_summary_sheet(all_data)

        # Write both tables to same sheet with a gap row
        startrow = 0
        table_a.to_excel(writer, sheet_name="Summary_Stats", index=False, startrow=startrow)
        startrow += len(table_a) + 3  # gap
        table_b.to_excel(writer, sheet_name="Summary_Stats", index=False, startrow=startrow)

    print(f"\nDone. {out_path.resolve()}")
    print(f"  Experiment sheets: {len(EXPERIMENTS)}")
    print(f"  Summary rows (Table A): {len(table_a)}")
    print(f"  Wilcoxon test rows (Table B): {len(table_b)}")


if __name__ == "__main__":
    main()
