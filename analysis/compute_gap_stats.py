"""
compute_gap_stats.py
Computes the three stat gaps not yet in stats_table.txt:
  1. Pre-event AUC-HV comparisons (MWU, all 6 event experiments, all strategy pairs)
  2. n_exp behavioral stats (Δn_exp pre/post event, paired Wilcoxon, AdaptiveScout + QEHVIFirst)
  3. Late-campaign per-iteration MI rates (MAIN experiments, by phase)

Run from the project root:
  source .venv/bin/activate && python3 _data_analysis/compute_gap_stats.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS_MPD     = REPO / "campaign_results" / "results_mpd_v3"
RESULTS_MAGNETS = REPO / "campaign_results" / "results_magnets_v3"

def _results_dir(exp_name: str) -> Path:
    return RESULTS_MAGNETS if exp_name.startswith("Magnets") else RESULTS_MPD

ORACLE_HV = {"MP-D": 444.219358, "Magnets": 0.968626}

STRATEGIES = ["Adaptive-Beta", "Rule-Swap", "qEHVI", "Fixed-Mixed"]
LABELS = {
    "Adaptive-Beta": "Adaptive-β",
    "Rule-Swap":     "Rule-Swap",
    "qEHVI":         "qEHVI",
    "Fixed-Mixed":   "Fixed Mixed",
}

# (first_event_iter, last_event_iter)
EVENT_ITERS = {
    "MP-D_BUDGET":    (15, 15),
    "MP-D_TIME":      (15, 15),
    "MP-D_DUAL":      (5,  7),
    "Magnets-BUDGET": (18, 18),
    "Magnets-TIME":   (13, 13),
    "Magnets-DUAL":   (13, 18),
}

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def load_convergence(exp_dir: Path) -> dict:
    """Load convergence CSVs for all seeds × strategies. Returns {strat: [df, ...]}."""
    data = {s: [] for s in STRATEGIES}
    for seed_dir in sorted(exp_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        for strat in STRATEGIES:
            csv = seed_dir / strat / f"{strat}_convergence.csv"
            if csv.exists():
                data[strat].append(pd.read_csv(csv))
    return data


# ===========================================================================
# 1. Pre-event AUC-HV comparisons
# ===========================================================================

def compute_pre_event_auc(exp_name: str):
    """
    For each seed × strategy, compute AUC-HV from iter 0 up to (but not including)
    the first event iteration. Then run all 4C2 pairwise MWU tests.
    AUC = sum of oracle-normalized HV values across those iterations (trapezoidal ≈ sum
    since step size = 1 iteration everywhere).
    Returns: list of result dicts.
    """
    system = "Magnets" if exp_name.startswith("Magnets") else "MP-D"
    norm = ORACLE_HV[system]
    exp_dir = _results_dir(exp_name) / exp_name
    first_event, _ = EVENT_ITERS[exp_name]

    data = load_convergence(exp_dir)

    # Per-seed AUC (sum of normalized HV from iter 0 to first_event-1 inclusive)
    auc_by_strat = {}
    for strat in STRATEGIES:
        aucs = []
        for df in data[strat]:
            pre = df[df["iteration"] < first_event]["total_hypervolume"] / norm
            aucs.append(pre.sum())
        auc_by_strat[strat] = np.array(aucs)

    pairs = [(STRATEGIES[i], STRATEGIES[j])
             for i in range(len(STRATEGIES)) for j in range(i+1, len(STRATEGIES))]

    results = []
    for a, b in pairs:
        arr_a, arr_b = auc_by_strat[a], auc_by_strat[b]
        _, p = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        mean_a = arr_a.mean()
        mean_b = arr_b.mean()
        sem_a = arr_a.std(ddof=1) / np.sqrt(len(arr_a))
        sem_b = arr_b.std(ddof=1) / np.sqrt(len(arr_b))
        dom = np.mean(arr_a > arr_b)
        results.append({
            "exp": exp_name,
            "a": a, "b": b,
            "mean_a": mean_a, "sem_a": sem_a,
            "mean_b": mean_b, "sem_b": sem_b,
            "p": p, "sig": stars(p), "dom_rate": dom,
        })
    return results, auc_by_strat


# ===========================================================================
# 2. n_exp behavioral stats
# ===========================================================================

def compute_nexp_stats(exp_name: str):
    """
    Pre/post event n_exp (selected_n_exp column) for AdaptiveScout and QEHVIFirst.
    pre  = iterations 1 to first_event_iter - 1 inclusive
    post = iterations last_event_iter + 1 to end inclusive
    Paired Wilcoxon on (post_mean - pre_mean) per seed.
    Also reports MWU comparison of post-event n_exp vs MAIN at same iterations.
    """
    system = "Magnets" if exp_name.startswith("Magnets") else "MP-D"
    first_ev, last_ev = EVENT_ITERS[exp_name]
    exp_dir = _results_dir(exp_name) / exp_name
    main_name = "MP-D_MAIN" if "MP-D" in system else "Magnets-MAIN"

    results = []
    for strat in ["Adaptive-Beta", "Rule-Swap"]:
        pre_means, post_means = [], []
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue
            csv = seed_dir / strat / f"{strat}_convergence.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if "selected_n_exp" not in df.columns:
                continue
            pre = df[(df["iteration"] >= 1) & (df["iteration"] < first_ev)]["selected_n_exp"]
            post = df[df["iteration"] > last_ev]["selected_n_exp"]
            if len(pre) == 0 or len(post) == 0:
                continue
            pre_means.append(pre.mean())
            post_means.append(post.mean())

        if not pre_means:
            continue

        pre_arr = np.array(pre_means)
        post_arr = np.array(post_means)
        diff = post_arr - pre_arr
        n = len(diff)
        _, p = stats.wilcoxon(diff, alternative="two-sided")

        results.append({
            "exp": exp_name, "strat": strat,
            "pre_mean": pre_arr.mean(), "pre_sem": pre_arr.std(ddof=1)/np.sqrt(n),
            "post_mean": post_arr.mean(), "post_sem": post_arr.std(ddof=1)/np.sqrt(n),
            "delta_mean": diff.mean(), "delta_sem": diff.std(ddof=1)/np.sqrt(n),
            "p": p, "sig": stars(p), "n": n,
        })

        # MWU: post-event n_exp vs MAIN n_exp at same iterations
        main_dir = _results_dir(main_name) / main_name
        main_post_means = []
        for seed_dir in sorted(main_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue
            csv = seed_dir / strat / f"{strat}_convergence.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if "selected_n_exp" not in df.columns:
                continue
            same_iters = df[df["iteration"] > last_ev]["selected_n_exp"]
            if len(same_iters) > 0:
                main_post_means.append(same_iters.mean())

        if main_post_means:
            main_arr = np.array(main_post_means)
            _, p_mwu = stats.mannwhitneyu(post_arr, main_arr, alternative="two-sided")
            results[-1]["main_mean"] = main_arr.mean()
            results[-1]["main_sem"] = main_arr.std(ddof=1)/np.sqrt(len(main_arr))
            results[-1]["p_vs_main"] = p_mwu
            results[-1]["sig_vs_main"] = stars(p_mwu)

    return results


# ===========================================================================
# 3. Late-campaign per-iteration MI by phase
# ===========================================================================

def compute_mi_phases(exp_name: str):
    """
    Per-iteration information_gain (MI) by phase for a MAIN experiment.
    Returns per-strategy mean ± SEM for early/mid/late phases.
    Also computes late-mid difference (decline).
    """
    system = "Magnets" if exp_name.startswith("Magnets") else "MP-D"
    exp_dir = _results_dir(exp_name) / exp_name

    if system == "MP-D":
        phases = {"early": (1, 10), "mid": (11, 30), "late": (31, 55)}
    else:
        phases = {"early": (1, 10), "mid": (11, 30), "late": (31, 50)}

    data = load_convergence(exp_dir)
    results = {}
    for strat in STRATEGIES:
        strat_result = {}
        for phase_name, (lo, hi) in phases.items():
            phase_means = []
            for df in data[strat]:
                if "information_gain" not in df.columns:
                    continue
                vals = df[(df["iteration"] >= lo) & (df["iteration"] <= hi)]["information_gain"]
                if len(vals) > 0:
                    phase_means.append(vals.mean())
            if phase_means:
                arr = np.array(phase_means)
                n = len(arr)
                strat_result[phase_name] = {
                    "mean": arr.mean(),
                    "sem": arr.std(ddof=1)/np.sqrt(n),
                    "n": n,
                }
        if "late" in strat_result and "mid" in strat_result:
            strat_result["late_minus_mid"] = strat_result["late"]["mean"] - strat_result["mid"]["mean"]
        results[strat] = strat_result
    return results


# ===========================================================================
# MAIN
# ===========================================================================

def fmt(val, decimals=3):
    return f"{val:.{decimals}f}"

def run():
    lines = []
    sep = "─" * 90

    # -----------------------------------------------------------------------
    # SECTION 1: Pre-event AUC-HV comparisons
    # -----------------------------------------------------------------------
    lines += [
        "",
        sep,
        "  PRE-EVENT AUC-HV COMPARISONS (Mann-Whitney U, unpaired)",
        "  AUC computed as sum of oracle-normalized HV from iter 0 to first_event_iter-1",
        "  Tests whether strategies differ in convergence BEFORE any event is applied.",
        sep,
    ]

    for exp_name in ["MP-D_BUDGET", "MP-D_TIME", "MP-D_DUAL",
                     "Magnets-BUDGET", "Magnets-TIME", "Magnets-DUAL"]:
        first_ev, _ = EVENT_ITERS[exp_name]
        system = "Magnets" if "Magnets" in exp_name else "MP-D"
        norm = ORACLE_HV[system]
        res, auc_by_strat = compute_pre_event_auc(exp_name)

        lines.append(f"\n  {exp_name}  (pre-event = iters 0–{first_ev-1})")
        lines.append(f"  {'Strategy':<20}  {'Pre-event AUC mean±SEM':<30}")
        for strat in STRATEGIES:
            arr = auc_by_strat[strat]
            n = len(arr)
            mean = arr.mean()
            sem = arr.std(ddof=1)/np.sqrt(n)
            lines.append(f"  {LABELS[strat]:<20}  {mean:.4f} ± {sem:.5f}  (n={n})")

        lines.append(f"\n  {'Pair':<45}  {'p-value':<12}  {'Sig':<6}  {'Dom. rate'}")
        for r in res:
            pair = f"{LABELS[r['a']]} vs {LABELS[r['b']]}"
            lines.append(f"  {pair:<45}  p = {r['p']:.3f}      {r['sig']:<6}  {r['dom_rate']:.2f}")

    # -----------------------------------------------------------------------
    # SECTION 2: n_exp behavioral stats
    # -----------------------------------------------------------------------
    lines += [
        "",
        sep,
        "  N_EXP BEHAVIORAL STATISTICS — Pre/Post Event (Paired Wilcoxon, within-experiment)",
        "  pre  = iterations 1 to first_event_iter-1  (all iterations before any event)",
        "  post = iterations last_event_iter+1 to end (all iterations after last event)",
        "  Strategies: Adaptive-β and Rule-Swap only (qEHVI and Fixed Mixed have no n_exp decisions)",
        sep,
    ]

    for exp_name in ["MP-D_BUDGET", "MP-D_TIME", "MP-D_DUAL",
                     "Magnets-BUDGET", "Magnets-TIME", "Magnets-DUAL"]:
        first_ev, last_ev = EVENT_ITERS[exp_name]
        res = compute_nexp_stats(exp_name)
        lines.append(f"\n  {exp_name}  (event iters: {first_ev}–{last_ev})")
        lines.append(f"  {'Strategy':<16}  {'Pre mean±SEM':<22}  {'Post mean±SEM':<22}  {'Δn_exp mean±SEM':<22}  {'Wilcoxon p':<12}  {'Sig'}")
        for r in res:
            lines.append(
                f"  {LABELS[r['strat']]:<16}  "
                f"{r['pre_mean']:.3f}±{r['pre_sem']:.3f}          "
                f"{r['post_mean']:.3f}±{r['post_sem']:.3f}          "
                f"{r['delta_mean']:+.3f}±{r['delta_sem']:.3f}          "
                f"p = {r['p']:.3f}       {r['sig']}"
            )

    lines += [
        "",
        "  N_EXP POST-EVENT vs MAIN AT SAME ITERATIONS (Mann-Whitney U, unpaired)",
        f"  {'Exp / Strategy':<35}  {'Event-exp post mean±SEM':<28}  {'MAIN same-iters mean±SEM':<28}  {'Δn_exp':<10}  {'MWU p':<12}  Sig",
    ]
    for exp_name in ["MP-D_BUDGET", "MP-D_TIME", "MP-D_DUAL",
                     "Magnets-BUDGET", "Magnets-TIME", "Magnets-DUAL"]:
        res = compute_nexp_stats(exp_name)
        for r in res:
            if "main_mean" not in r:
                continue
            label = f"{exp_name} / {LABELS[r['strat']]}"
            delta = r["post_mean"] - r["main_mean"]
            lines.append(
                f"  {label:<35}  "
                f"{r['post_mean']:.3f}±{r['post_sem']:.3f}                    "
                f"{r['main_mean']:.3f}±{r['main_sem']:.3f}                    "
                f"{delta:+.3f}        "
                f"p = {r['p_vs_main']:.3f}       {r['sig_vs_main']}"
            )

    # -----------------------------------------------------------------------
    # SECTION 3: Late-campaign per-iteration MI
    # -----------------------------------------------------------------------
    lines += [
        "",
        sep,
        "  PER-ITERATION MI BY PHASE — MAIN EXPERIMENTS",
        "  information_gain column; values are mean per-iteration MI ± SEM across seeds",
        "  MP-D:    early=iters 1-10, mid=11-30, late=31-55",
        "  Magnets: early=iters 1-10, mid=11-30, late=31-50",
        sep,
    ]

    for exp_name in ["MP-D_MAIN", "Magnets-MAIN"]:
        res = compute_mi_phases(exp_name)
        lines.append(f"\n  {exp_name}")
        lines.append(f"  {'Strategy':<16}  {'Early mean±SEM':<22}  {'Mid mean±SEM':<22}  {'Late mean±SEM':<22}  {'Late−Mid'}")
        for strat in STRATEGIES:
            r = res[strat]
            e = r.get("early", {})
            m = r.get("mid", {})
            l = r.get("late", {})
            lm = r.get("late_minus_mid", float("nan"))
            lines.append(
                f"  {LABELS[strat]:<16}  "
                f"{e.get('mean',float('nan')):.4f}±{e.get('sem',float('nan')):.4f}            "
                f"{m.get('mean',float('nan')):.4f}±{m.get('sem',float('nan')):.4f}            "
                f"{l.get('mean',float('nan')):.4f}±{l.get('sem',float('nan')):.4f}            "
                f"{lm:+.4f}"
            )

    # -----------------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------------
    output = "\n".join(lines)
    print(output)
    return output


if __name__ == "__main__":
    output = run()
    out_path = Path(__file__).resolve().parent / "gap_stats_output_v3.txt"
    out_path.write_text(output)
    print(f"\n[Saved to {out_path}]")
