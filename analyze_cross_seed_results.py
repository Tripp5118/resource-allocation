# analyze_cross_seed_results.py
"""
Analyze cross-seed results and generate convergence plots with confidence intervals.
This script can handle two types of experiment directories:
1. Beta grid search: results/beta_grid_search/beta{value}/seed{seed}/
2. High statistics: results/high_statistics_beta2/seed{seed}/
Usage:
    python analyze_cross_seed_results.py --exp_dir results/beta_grid_search --mode grid
    python analyze_cross_seed_results.py --exp_dir results/high_statistics_beta2 --mode single
    python analyze_cross_seed_results.py --exp_dir results/high_statistics_beta2 --mode single --show_constraints
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats
# Hard-coded strategy configuration
STRATEGIES = ["qEHVI", "qUCB", "Agent"]

# Aliases for legacy directory/file naming conventions.
# Files/directories whose names start with any of these prefixes are loaded
# under the corresponding canonical strategy key.
# "convergence" is also listed under qEHVI because old runs sometimes used
# a bare "convergence" prefix before strategy names were standardised.
STRATEGY_ALIASES: Dict[str, List[str]] = {
    "qEHVI": ["PureExploit"],
    "qUCB":  ["PureExplore"],
    "Agent": [],
}

# Build a flat prefix → canonical-strategy lookup for fast resolution
_ALIAS_TO_STRATEGY: Dict[str, str] = {}
for _strategy, _prefixes in STRATEGY_ALIASES.items():
    for _prefix in _prefixes:
        _ALIAS_TO_STRATEGY[_prefix] = _strategy


def _resolve_strategy_path(
    seed_dir: Path, strategy: str, suffix: str
) -> Optional[Path]:
    """
    Return the first existing path for a given strategy file, trying both the
    canonical name and any registered aliases.

    The suffix should include the file-name template *without* the leading
    strategy name, e.g. ``"_convergence.csv"`` or ``"_evaluations.csv"``.

    Search order:
      1. Canonical:  seed_dir / strategy / f"{strategy}{suffix}"
      2. Each alias: seed_dir / alias   / f"{alias}{suffix}"
    """
    # Canonical path
    canonical = seed_dir / strategy / f"{strategy}{suffix}"
    if canonical.exists():
        return canonical
    # Alias paths
    for alias in STRATEGY_ALIASES.get(strategy, []):
        alias_path = seed_dir / alias / f"{alias}{suffix}"
        if alias_path.exists():
            return alias_path
    return None

LABELS = {
    "qEHVI": "qEHVI",
    "qUCB":  "qUCB",   # beta value appended at plot time
    "Agent": "Agent"
}
COLORS = {
    "qEHVI": "#3498DB",
    "qUCB":  "#E74C3C",
    "Agent": "#2ECC71"
}
# Colors for constraint lines
CONSTRAINT_COLORS = {
    "budget_remaining": "#9B59B6",  # Purple
    "time_remaining": "#F39C12",    # Orange
}
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_convergence_data(
    exp_dir: str,
    beta_value: float = None,
    mode: str = "grid"
) -> Dict[str, List[pd.DataFrame]]:
    """
    Load convergence data for all strategies across seeds.
    Args:
        exp_dir: Base experiment directory
        beta_value: Beta value (only for grid mode)
        mode: "grid" for beta grid search, "single" for high statistics
    Returns:
        Dictionary mapping strategy names to lists of DataFrames (one per seed)
    """
    data = {strategy: [] for strategy in STRATEGIES}
    if mode == "grid":
        if beta_value is None:
            raise ValueError("beta_value must be specified for grid mode")
        search_dir = Path(exp_dir) / f"beta{beta_value}"
    else:
        search_dir = Path(exp_dir)
    if not search_dir.exists():
        raise ValueError(f"Directory does not exist: {search_dir}")
    seed_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    if len(seed_dirs) == 0:
        raise ValueError(f"No seed directories found in {search_dir}")
    print(f"Found {len(seed_dirs)} seed directories in {search_dir}")
    for seed_dir in seed_dirs:
        seed = seed_dir.name
        for strategy in STRATEGIES:
            conv_path = _resolve_strategy_path(seed_dir, strategy, "_convergence.csv")
            if conv_path is not None:
                df = pd.read_csv(conv_path)
                data[strategy].append(df)
            else:
                print(f"Warning: Missing convergence file for {seed}/{strategy}")
    for strategy in STRATEGIES:
        print(f"Loaded {len(data[strategy])} runs for {strategy}")
    return data
def load_evaluations_data(
    exp_dir: str,
    beta_value: float = None,
    mode: str = "grid"
) -> Tuple[List[str], Dict[str, List[pd.DataFrame]]]:
    """
    Load evaluations data (full Y_history) for all strategies across seeds.
    Each seed directory contains {strategy}/{strategy}_evaluations.csv with columns:
        iteration, point_idx, CTE, K, score, x_0..x_n, source
    Args:
        exp_dir: Base experiment directory
        beta_value: Beta value (only for grid mode)
        mode: "grid" for beta grid search, "single" for high statistics
    Returns:
        Tuple of:
          - seed_names: ordered list of seed directory names
          - data: dict mapping strategy -> list of DataFrames (one per seed, in seed order)
    """
    if mode == "grid":
        if beta_value is None:
            raise ValueError("beta_value must be specified for grid mode")
        search_dir = Path(exp_dir) / f"beta{beta_value}"
    else:
        search_dir = Path(exp_dir)
    if not search_dir.exists():
        raise ValueError(f"Directory does not exist: {search_dir}")
    seed_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    if len(seed_dirs) == 0:
        raise ValueError(f"No seed directories found in {search_dir}")
    seed_names = [d.name for d in seed_dirs]
    data = {strategy: [] for strategy in STRATEGIES}
    for seed_dir in seed_dirs:
        seed = seed_dir.name
        for strategy in STRATEGIES:
            eval_path = _resolve_strategy_path(seed_dir, strategy, "_evaluations.csv")
            if eval_path is not None:
                df = pd.read_csv(eval_path)
                data[strategy].append(df)
            else:
                print(f"Warning: Missing evaluations file for {seed}/{strategy}")
                data[strategy].append(None)  # placeholder to keep indices aligned
    return seed_names, data
# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------
def compute_pareto_mask(points: np.ndarray) -> np.ndarray:
    """
    Compute a boolean mask of non-dominated points (maximisation of both objectives).
    Uses a pure-numpy implementation so botorch is not required at post-processing time,
    though the logic is identical to botorch's is_non_dominated.
    Args:
        points: (N, 2) array of objective values
    Returns:
        Boolean mask of length N; True = Pareto-optimal
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # A point is dominated if there exists another point that is >= in all
        # objectives and strictly > in at least one
        dominated_by_i = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        dominated_by_i[i] = False
        # If i is dominated by any remaining candidate, mark it out
        if np.any(np.all(points[is_pareto] >= points[i], axis=1) & np.any(points[is_pareto] > points[i], axis=1)):
            is_pareto[i] = False
    return is_pareto
def get_pareto_points(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split evaluated points into Pareto-optimal and dominated sets.
    Args:
        Y: (N, 2) array; column 0 = obj1 (-|CTE| by default), column 1 = obj2 (K)
    Returns:
        pareto_points: (M, 2) subset that are non-dominated
        non_pareto_points: (N-M, 2) dominated points
    """
    if len(Y) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    mask = compute_pareto_mask(Y)
    return Y[mask], Y[~mask]
# ---------------------------------------------------------------------------
# Per-seed Pareto plot and data saving
# ---------------------------------------------------------------------------
def plot_per_seed_pareto(
    seed_name: str,
    eval_data: Dict[str, Optional[pd.DataFrame]],
    beta_value: float,
    obj1_label: str,
    obj2_label: str,
    save_path: str,
):
    """
    Plot Pareto front for all three strategies for a single seed.
    Non-Pareto points are plotted in the strategy colour but transparent.
    Pareto points are solid with star markers and connected by a sorted line.
    The beta value appears only in the qUCB legend entry.
    Args:
        seed_name: Seed identifier string (for plot title)
        eval_data: Dict mapping strategy name -> evaluations DataFrame (or None if missing)
        beta_value: Beta value shown in the qUCB legend entry
        obj1_label: Display label for objective 1 (x-axis)
        obj2_label: Display label for objective 2 (y-axis)
        save_path: File path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    for strategy in STRATEGIES:
        df = eval_data.get(strategy)
        if df is None or len(df) == 0:
            print(f"  Skipping {strategy} — no evaluation data")
            continue
        # Build Y array: obj1 = -|CTE| = -(abs(CTE)), obj2 = K
        # The evaluations CSV stores raw CTE and K values.
        # We negate CTE to turn minimisation into maximisation (matching runtime convention).
        Y = np.column_stack([-np.abs(df["CTE"].values), df["K"].values])
        pareto_pts, non_pareto_pts = get_pareto_points(Y)
        color = COLORS.get(strategy, "gray")
        legend_label = f"qUCB (β={beta_value})" if strategy == "qUCB" else LABELS.get(strategy, strategy)
        # Non-Pareto: same colour, very transparent, no legend entry
        if len(non_pareto_pts) > 0:
            ax.scatter(
                non_pareto_pts[:, 0], non_pareto_pts[:, 1],
                color=color, s=60, alpha=0.20, zorder=1, linewidths=0,
            )
        # Pareto: solid colour, star markers, connecting line
        if len(pareto_pts) > 0:
            sorted_idx = np.argsort(pareto_pts[:, 0])
            pareto_sorted = pareto_pts[sorted_idx]
            ax.plot(
                pareto_sorted[:, 0], pareto_sorted[:, 1],
                color=color, linewidth=2, alpha=0.7, zorder=2,
            )
            ax.scatter(
                pareto_pts[:, 0], pareto_pts[:, 1],
                color=color, s=140, marker="*",
                edgecolors="white", linewidth=0.8,
                label=f"{legend_label} (Pareto: {len(pareto_pts)})",
                zorder=3,
            )
    ax.set_xlabel(obj1_label, fontsize=13, fontweight="bold")
    ax.set_ylabel(obj2_label, fontsize=13, fontweight="bold")
    ax.set_title(
        f"Pareto Front Comparison — {seed_name}",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Pareto plot → {save_path}")
def save_pareto_data(
    seed_name: str,
    eval_data: Dict[str, Optional[pd.DataFrame]],
    save_path: str,
):
    """
    Save all Pareto-optimal points (across all three strategies) for one seed to CSV.
    The saved file has columns:
        strategy, obj1, obj2, pareto_rank (always 1 here — non-dominated within strategy),
        CTE_raw, K_raw
    This is intended as the input for future cross-seed Pareto averaging.
    Args:
        seed_name: Seed identifier (written into the 'seed' column)
        eval_data: Dict mapping strategy name -> evaluations DataFrame (or None)
        save_path: File path for the output CSV
    """
    rows = []
    for strategy in STRATEGIES:
        df = eval_data.get(strategy)
        if df is None or len(df) == 0:
            continue
        Y = np.column_stack([-np.abs(df["CTE"].values), df["K"].values])
        pareto_pts, _= get_pareto_points(Y)
        for pt in pareto_pts:
            rows.append({
                "seed": seed_name,
                "strategy": strategy,
                "obj1": pt[0],   # -|CTE|
                "obj2": pt[1],   # K
            })
    if rows:
        out_df = pd.DataFrame(rows, columns=["seed", "strategy", "obj1", "obj2"])
        out_df.to_csv(save_path, index=False)
        print(f"  Saved Pareto data  → {save_path}")
    else:
        print(f"  No Pareto data to save for {seed_name}")
def generate_pareto_plots_and_data(
    search_dir: Path,
    output_dir: Path,
    beta_value: float,
    obj1_label: str,
    obj2_label: str,
    mode: str,
):
    """
    Iterate over all seed directories under search_dir, produce one Pareto plot
    per seed, and save per-seed Pareto CSVs for later cross-seed averaging.
    Args:
        search_dir: Directory containing seed{N} subdirectories
        output_dir: Where to write output files
        beta_value: Beta value for qUCB legend entry
        obj1_label: X-axis label
        obj2_label: Y-axis label
        mode: "grid" or "single" (controls subdirectory naming only for logging)
    """
    seed_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    if not seed_dirs:
        print("  No seed directories found — skipping Pareto analysis.")
        return
    pareto_output_dir = output_dir / "pareto_data"
    pareto_output_dir.mkdir(exist_ok=True)
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f"  Processing Pareto for {seed_name} ...")
        # Load evaluations for each strategy
        eval_data = {}
        for strategy in STRATEGIES:
            eval_path = _resolve_strategy_path(seed_dir, strategy, "_evaluations.csv")
            if eval_path is not None:
                eval_data[strategy] = pd.read_csv(eval_path)
            else:
                print(f"    Warning: Missing evaluations file for {seed_name}/{strategy}")
                eval_data[strategy] = None
        # Plot
        plot_path = output_dir / f"pareto_{seed_name}.png"
        plot_per_seed_pareto(
            seed_name=seed_name,
            eval_data=eval_data,
            beta_value=beta_value,
            obj1_label=obj1_label,
            obj2_label=obj2_label,
            save_path=str(plot_path),
        )
        # Save Pareto CSV for this seed
        csv_path = pareto_output_dir / f"pareto_{seed_name}.csv"
        save_pareto_data(
            seed_name=seed_name,
            eval_data=eval_data,
            save_path=str(csv_path),
        )
# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def compute_statistics(
    data: Dict[str, List[pd.DataFrame]],
    metric: str = "best_score",
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mean and confidence intervals across seeds.
    Args:
        data: Dictionary mapping strategy names to lists of DataFrames
        metric: Column name to analyze (e.g., "best_score", "total_hypervolume")
        confidence_level: Confidence level for intervals (default 0.95)
    Returns:
        Dictionary with statistics for each strategy
    """
    results = {}
    for strategy, dfs in data.items():
        if len(dfs) == 0:
            print(f"Warning: No data for {strategy}")
            continue
        max_iters = max(df['iteration'].max() for df in dfs)
        values_by_iter = []
        for iter_num in range(int(max_iters) + 1):
            iter_values = []
            for df in dfs:
                if iter_num in df['iteration'].values:
                    value = df[df['iteration'] == iter_num][metric].values[0]
                    iter_values.append(value)
            values_by_iter.append(iter_values)
        iterations = np.arange(len(values_by_iter))
        means = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
        stds = np.array([np.std(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
        n_samples = np.array([len(vals) for vals in values_by_iter])
        t_critical = np.array([
            stats.t.ppf((1 + confidence_level) / 2, max(n - 1, 1)) if n > 0 else np.nan
            for n in n_samples
        ])
        ci_half_width = t_critical * stds / np.sqrt(n_samples)
        results[strategy] = {
            'iterations': iterations,
            'mean': means,
            'std': stds,
            'ci_lower': means - ci_half_width,
            'ci_upper': means + ci_half_width,
            'n_samples': n_samples,
        }
    return results
def compute_combined_uncertainty_statistics(
    data: Dict[str, List[pd.DataFrame]],
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mean and confidence intervals for combined uncertainty (obj1 + obj2) across seeds.
    For each row in each seed's convergence CSV, sums total_uncertainty_obj1 and
    total_uncertainty_obj2, then aggregates those combined values across seeds per iteration.
    Args:
        data: Dictionary mapping strategy names to lists of DataFrames
        confidence_level: Confidence level for intervals (default 0.95)
    Returns:
        Dictionary with statistics for each strategy, or empty dict if columns are absent.
    """
    results = {}
    for strategy, dfs in data.items():
        if len(dfs) == 0:
            print(f"Warning: No data for {strategy}")
            continue
        required_cols = {'total_uncertainty_obj1', 'total_uncertainty_obj2'}
        valid_dfs = [df for df in dfs if required_cols.issubset(df.columns)]
        if not valid_dfs:
            print(f"Warning: No runs for {strategy} contain uncertainty columns — skipping.")
            continue
        if len(valid_dfs) < len(dfs):
            print(
                f"Warning: {len(dfs) - len(valid_dfs)} run(s) for {strategy} "
                f"are missing uncertainty columns and will be excluded."
            )
        max_iters = max(df['iteration'].max() for df in valid_dfs)
        values_by_iter = []
        for iter_num in range(int(max_iters) + 1):
            iter_values = []
            for df in valid_dfs:
                if iter_num in df['iteration'].values:
                    row = df[df['iteration'] == iter_num].iloc[0]
                    combined = row['total_uncertainty_obj1'] + row['total_uncertainty_obj2']
                    iter_values.append(combined)
            values_by_iter.append(iter_values)
        iterations = np.arange(len(values_by_iter))
        means = np.array([np.mean(vals) if vals else np.nan for vals in values_by_iter])
        stds = np.array([np.std(vals) if vals else np.nan for vals in values_by_iter])
        n_samples = np.array([len(vals) for vals in values_by_iter])
        t_critical = np.array([
            stats.t.ppf((1 + confidence_level) / 2, max(n - 1, 1)) if n > 0 else np.nan
            for n in n_samples
        ])
        safe_n = np.where(n_samples > 0, n_samples, np.nan)
        ci_half_width = t_critical * stds / np.sqrt(safe_n)
        results[strategy] = {
            'iterations': iterations,
            'mean': means,
            'std': stds,
            'ci_lower': means - ci_half_width,
            'ci_upper': means + ci_half_width,
            'n_samples': n_samples,
        }
    return results
def compute_constraint_statistics(
    data: Dict[str, List[pd.DataFrame]],
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute mean and confidence intervals for constraint columns (budget_remaining, time_remaining).
    
    Args:
        data: Dictionary mapping strategy names to lists of DataFrames
        confidence_level: Confidence level for intervals (default 0.95)
    
    Returns:
        Dictionary mapping strategy -> constraint_name -> statistics dict
    """
    constraint_cols = ['budget_remaining', 'time_remaining']
    results = {}
    
    for strategy, dfs in data.items():
        if len(dfs) == 0:
            continue
        
        results[strategy] = {}
        
        for constraint in constraint_cols:
            # Check if constraint column exists in any of the dataframes
            valid_dfs = [df for df in dfs if constraint in df.columns]
            if not valid_dfs:
                continue
            
            max_iters = max(df['iteration'].max() for df in valid_dfs)
            values_by_iter = []
            
            for iter_num in range(int(max_iters) + 1):
                iter_values = []
                for df in valid_dfs:
                    if iter_num in df['iteration'].values:
                        value = df[df['iteration'] == iter_num][constraint].values[0]
                        iter_values.append(value)
                values_by_iter.append(iter_values)
            
            iterations = np.arange(len(values_by_iter))
            means = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
            stds = np.array([np.std(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
            n_samples = np.array([len(vals) for vals in values_by_iter])
            
            t_critical = np.array([
                stats.t.ppf((1 + confidence_level) / 2, max(n - 1, 1)) if n > 0 else np.nan
                for n in n_samples
            ])
            ci_half_width = t_critical * stds / np.sqrt(np.where(n_samples > 0, n_samples, np.nan))
            
            results[strategy][constraint] = {
                'iterations': iterations,
                'mean': means,
                'std': stds,
                'ci_lower': means - ci_half_width,
                'ci_upper': means + ci_half_width,
                'n_samples': n_samples,
            }
    
    return results
# ---------------------------------------------------------------------------
# Convergence plotting
# ---------------------------------------------------------------------------
def plot_convergence_with_ci(
    plot_stats: Dict[str, Dict[str, np.ndarray]],
    metric_name: str,
    beta_value: float,
    n_seeds: int = None,
    save_path: str = None,
    confidence_level: float = 0.95,
    constraint_stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = None,
    show_constraints: bool = False,
):
    """
    Plot convergence curves with confidence intervals.
    Args:
        plot_stats: Statistics dictionary from compute_statistics()
        metric_name: Name of metric for y-axis label
        beta_value: Beta value — shown next to qUCB in the legend
        n_seeds: Number of seeds (for title)
        save_path: Path to save figure
        confidence_level: Confidence level for CI
        constraint_stats: Optional constraint statistics from compute_constraint_statistics()
        show_constraints: Whether to show constraint lines (budget_remaining, time_remaining)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ci_pct = int(confidence_level * 100)
    for strategy, stat in plot_stats.items():
        color = COLORS.get(strategy, 'gray')
        # Beta value lives only in the qUCB legend entry; CI bands have no label
        if strategy == "qUCB":
            line_label = f"qUCB (β={beta_value})"
        else:
            line_label = LABELS.get(strategy, strategy)
        iters = stat['iterations']
        mean = stat['mean']
        ax.plot(iters, mean, color=color, linewidth=2.5, label=line_label, marker='o', markersize=5)
        # CI shading — intentionally unlabelled
        ax.fill_between(
            iters,
            stat['ci_lower'],
            stat['ci_upper'],
            color=color,
            alpha=0.2,
        )
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    
    # Add constraint axes if requested (only from Agent strategy)
    ax2 = None  # time_remaining axis
    ax3 = None  # budget_remaining axis
    
    if show_constraints and constraint_stats and 'Agent' in constraint_stats:
        agent_constraints = constraint_stats['Agent']
        
        has_time = 'time_remaining' in agent_constraints
        has_budget = 'budget_remaining' in agent_constraints
        
        if has_time:
            ax2 = ax.twinx()
            ax2.set_ylabel('Time Remaining', fontsize=12, fontweight='bold', 
                          color=CONSTRAINT_COLORS['time_remaining'])
            ax2.tick_params(axis='y', labelcolor=CONSTRAINT_COLORS['time_remaining'])
            
            time_stat = agent_constraints['time_remaining']
            iters = time_stat['iterations']
            mean = time_stat['mean']
            color = CONSTRAINT_COLORS['time_remaining']
            
            ax2.plot(iters, mean, color=color, linewidth=1.5, alpha=0.6,
                    linestyle='--', label='Time Remaining', zorder=1)
        
        if has_budget:
            ax3 = ax.twinx()
            # Offset the third axis to the right
            if has_time:
                ax3.spines['right'].set_position(('outward', 60))
            ax3.set_ylabel('Budget Remaining', fontsize=12, fontweight='bold',
                          color=CONSTRAINT_COLORS['budget_remaining'])
            ax3.tick_params(axis='y', labelcolor=CONSTRAINT_COLORS['budget_remaining'])
            
            budget_stat = agent_constraints['budget_remaining']
            iters = budget_stat['iterations']
            mean = budget_stat['mean']
            color = CONSTRAINT_COLORS['budget_remaining']
            
            ax3.plot(iters, mean, color=color, linewidth=1.5, alpha=0.6,
                    linestyle='--', label='Budget Remaining', zorder=1)
    
    if n_seeds is not None:
        title = f'Strategy Comparison: {metric_name}\n({n_seeds} seeds, {ci_pct}% CI)'
    else:
        title = f'Strategy Comparison: {metric_name}\n({ci_pct}% CI)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends from all axes
    lines, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    if ax3 is not None:
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines += lines3
        labels += labels3
    
    legend = ax.legend(lines, labels, fontsize=11, loc='best', framealpha=0.9)
    legend.set_zorder(100)  # Draw legend above all plot elements
    
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()
# ---------------------------------------------------------------------------
# Top-level analysis functions
# ---------------------------------------------------------------------------
def analyze_beta_grid(
    exp_dir: str,
    beta_values: List[float],
    confidence_level: float = 0.95,
    obj1_label: str = "-|CTE|",
    obj2_label: str = "K",
    show_constraints: bool = False,
):
    """
    Analyze beta grid search results and generate plots for each beta.
    Args:
        exp_dir: Base experiment directory
        beta_values: List of beta values to analyze
        confidence_level: Confidence level for intervals
        obj1_label: Display label for objective 1 (Pareto x-axis)
        obj2_label: Display label for objective 2 (Pareto y-axis)
        show_constraints: Whether to show constraint lines on best_score plot
    """
    print(f"\n{'='*80}")
    print("ANALYZING BETA GRID SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Beta values: {beta_values}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Confidence level: {confidence_level}")
    print(f"Show constraints: {show_constraints}")
    print(f"{'='*80}\n")
    output_dir = Path(exp_dir) / "cross_seed_analysis"
    output_dir.mkdir(exist_ok=True)
    for beta in beta_values:
        print(f"\n{'-'*80}")
        print(f"Processing β_explore = {beta}")
        print(f"{'-'*80}")
        try:
            data = load_convergence_data(exp_dir, beta_value=beta, mode="grid")
            n_seeds = 0
            for strategy in STRATEGIES:
                if len(data[strategy]) > 0:
                    n_seeds = len(data[strategy])
                    break
            
            # Compute constraint statistics if needed
            constraint_stats = None
            if show_constraints:
                constraint_stats = compute_constraint_statistics(data, confidence_level=confidence_level)
            
            # best_score
            stats_score = compute_statistics(data, metric="best_score", confidence_level=confidence_level)
            save_path = output_dir / f"convergence_beta{beta}_ci.png"
            plot_convergence_with_ci(
                stats_score,
                metric_name="K / |CTE|",
                beta_value=beta,
                n_seeds=n_seeds,
                save_path=str(save_path),
                confidence_level=confidence_level,
                constraint_stats=constraint_stats,
                show_constraints=show_constraints,
            )
            # hypervolume (optional)
            has_hypervolume = any(
                len(data[s]) > 0 and 'total_hypervolume' in data[s][0].columns
                for s in STRATEGIES
            )
            if has_hypervolume:
                stats_hv = compute_statistics(data, metric="total_hypervolume", confidence_level=confidence_level)
                save_path_hv = output_dir / f"hypervolume_beta{beta}_ci.png"
                plot_convergence_with_ci(
                    stats_hv,
                    metric_name="Hypervolume",
                    beta_value=beta,
                    n_seeds=n_seeds,
                    save_path=str(save_path_hv),
                    confidence_level=confidence_level,
                )
            # combined uncertainty (optional)
            stats_unc = compute_combined_uncertainty_statistics(data, confidence_level=confidence_level)
            if stats_unc:
                save_path_unc = output_dir / f"uncertainty_beta{beta}_ci.png"
                plot_convergence_with_ci(
                    stats_unc,
                    metric_name="Total Uncertainty (obj1 + obj2)",
                    beta_value=beta,
                    n_seeds=n_seeds,
                    save_path=str(save_path_unc),
                    confidence_level=confidence_level,
                )
            # Per-seed Pareto plots and data
            print(f"\n  Generating per-seed Pareto plots for β = {beta} ...")
            search_dir = Path(exp_dir) / f"beta{beta}"
            beta_output_dir = output_dir / f"beta{beta}"
            beta_output_dir.mkdir(exist_ok=True)
            generate_pareto_plots_and_data(
                search_dir=search_dir,
                output_dir=beta_output_dir,
                beta_value=beta,
                obj1_label=obj1_label,
                obj2_label=obj2_label,
                mode="grid",
            )
            print(f"✓ Completed β = {beta}")
        except Exception as e:
            print(f"✗ Error processing β = {beta}: {e}")
            continue
    print(f"\n{'='*80}")
    print("BETA GRID ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*80}\n")
def analyze_high_statistics(
    exp_dir: str,
    beta_value: float = 2.0,
    confidence_level: float = 0.95,
    metric_name: str = "K / |CTE|",
    obj1_label: str = "-|CTE|",
    obj2_label: str = "K",
    show_constraints: bool = False,
):
    """
    Analyze high statistics results and generate plot.
    Args:
        exp_dir: Base experiment directory
        beta_value: Beta value (shown in qUCB legend entry)
        confidence_level: Confidence level for intervals
        metric_name: Name of the metric being analyzed
        obj1_label: Display label for objective 1 (Pareto x-axis)
        obj2_label: Display label for objective 2 (Pareto y-axis)
        show_constraints: Whether to show constraint lines on best_score plot
    """
    print(f"\n{'='*80}")
    print("ANALYZING HIGH STATISTICS RESULTS")
    print(f"{'='*80}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Beta value: {beta_value}")
    print(f"Confidence level: {confidence_level}")
    print(f"Show constraints: {show_constraints}")
    print(f"{'='*80}\n")
    output_dir = Path(exp_dir) / "cross_seed_analysis"
    output_dir.mkdir(exist_ok=True)
    try:
        data = load_convergence_data(exp_dir, mode="single")
        n_seeds = 0
        reference_strategy = None
        for strategy in STRATEGIES:
            if len(data[strategy]) > 0:
                n_seeds = len(data[strategy])
                reference_strategy = strategy
                break
        
        # Compute constraint statistics if needed
        constraint_stats = None
        if show_constraints:
            constraint_stats = compute_constraint_statistics(data, confidence_level=confidence_level)
        
        # best_score
        stats_score = compute_statistics(data, metric="best_score", confidence_level=confidence_level)
        save_path = output_dir / f"convergence_beta{beta_value}_n{n_seeds}_ci.png"
        plot_convergence_with_ci(
            stats_score,
            metric_name=metric_name,
            beta_value=beta_value,
            n_seeds=n_seeds,
            save_path=str(save_path),
            confidence_level=confidence_level,
            constraint_stats=constraint_stats,
            show_constraints=show_constraints,
        )
        # hypervolume (optional)
        has_hypervolume = (
            reference_strategy is not None and
            len(data[reference_strategy]) > 0 and
            'total_hypervolume' in data[reference_strategy][0].columns
        )
        if has_hypervolume:
            stats_hv = compute_statistics(data, metric="total_hypervolume", confidence_level=confidence_level)
            save_path_hv = output_dir / f"hypervolume_beta{beta_value}_n{n_seeds}_ci.png"
            plot_convergence_with_ci(
                stats_hv,
                metric_name="Hypervolume",
                beta_value=beta_value,
                n_seeds=n_seeds,
                save_path=str(save_path_hv),
                confidence_level=confidence_level,
            )
        # combined uncertainty (optional)
        stats_unc = compute_combined_uncertainty_statistics(data, confidence_level=confidence_level)
        if stats_unc:
            save_path_unc = output_dir / f"uncertainty_beta{beta_value}_n{n_seeds}_ci.png"
            plot_convergence_with_ci(
                stats_unc,
                metric_name="Total Uncertainty (obj1 + obj2)",
                beta_value=beta_value,
                n_seeds=n_seeds,
                save_path=str(save_path_unc),
                confidence_level=confidence_level,
            )
        # Summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        for strategy, stat in stats_score.items():
            final_mean = stat['mean'][-1]
            final_std = stat['std'][-1]
            display_label = LABELS.get(strategy, strategy)
            print(f"{display_label} ({strategy}):")
            print(f"  Final {metric_name}: {final_mean:.4f} ± {final_std:.4f} (std)")
            print(f"  {int(confidence_level*100)}% CI: [{stat['ci_lower'][-1]:.4f}, {stat['ci_upper'][-1]:.4f}]")
        print(f"{'='*80}\n")
        print(f"✓ Analysis complete")
        # Per-seed Pareto plots and data
        print(f"\nGenerating per-seed Pareto plots ...")
        generate_pareto_plots_and_data(
            search_dir=Path(exp_dir),
            output_dir=output_dir,
            beta_value=beta_value,
            obj1_label=obj1_label,
            obj2_label=obj2_label,
            mode="single",
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    print(f"\n{'='*80}")
    print("HIGH STATISTICS ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*80}\n")
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-seed BO experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # High statistics mode
    python analyze_cross_seed_results.py --exp_dir results/high_statistics --mode single
    # Beta grid search mode
    python analyze_cross_seed_results.py --exp_dir results/beta_grid_search --mode grid
    # Custom objective axis labels for Pareto plots
    python analyze_cross_seed_results.py --exp_dir results/high_statistics --mode single \\
        --obj1_label "-|CTE| (deg)" --obj2_label "K (N/m)"
    # Show constraint lines on best_score plot
    python analyze_cross_seed_results.py --exp_dir results/high_statistics --mode single --show_constraints
        """
    )
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory")
    parser.add_argument("--mode", type=str, choices=["grid", "single"], required=True,
                        help="Analysis mode: 'grid' for beta grid search, 'single' for high statistics")
    parser.add_argument("--beta_values", type=float, nargs="+", default=[2, 4, 6, 8, 10],
                        help="Beta values to analyze (grid mode only)")
    parser.add_argument("--beta_value", type=float, default=2.0,
                        help="Beta value shown in qUCB legend entry (single mode only)")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for intervals (default: 0.95)")
    parser.add_argument("--metric_name", type=str, default="K / |CTE|",
                        help="Display name for the convergence metric (default: 'K / |CTE|')")
    parser.add_argument("--obj1_label", type=str, default="-|CTE|",
                        help="Pareto plot x-axis label — objective 1 (default: '-|CTE|')")
    parser.add_argument("--obj2_label", type=str, default="K",
                        help="Pareto plot y-axis label — objective 2 (default: 'K')")
    parser.add_argument("--show_constraints", action="store_true", default=False,
                        help="Show budget_remaining and time_remaining on best_score plot (default: False)")
    args = parser.parse_args()
    if args.mode == "grid":
        analyze_beta_grid(
            args.exp_dir,
            args.beta_values,
            args.confidence,
            args.obj1_label,
            args.obj2_label,
            args.show_constraints,
        )
    else:
        analyze_high_statistics(
            args.exp_dir,
            args.beta_value,
            args.confidence,
            args.metric_name,
            args.obj1_label,
            args.obj2_label,
            args.show_constraints,
        )
if __name__ == "__main__":
    main()