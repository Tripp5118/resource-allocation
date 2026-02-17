# analyze_cross_seed_results.py
"""
Analyze cross-seed results and generate convergence plots with confidence intervals.

This script can handle two types of experiment directories:
1. Beta grid search: results/beta_grid_search/beta{value}/seed{seed}/
2. High statistics: results/high_statistics_beta2/seed{seed}/

Usage:
    python analyze_cross_seed_results.py --exp_dir results/beta_grid_search --mode grid
    python analyze_cross_seed_results.py --exp_dir results/high_statistics_beta2 --mode single
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
        Dictionary mapping strategy names to lists of DataFrames
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
            conv_path = seed_dir / strategy / f"{strategy}_convergence.csv"
            if conv_path.exists():
                df = pd.read_csv(conv_path)
                data[strategy].append(df)
            else:
                print(f"Warning: Missing convergence file for {seed}/{strategy}")
    
    for strategy in STRATEGIES:
        print(f"Loaded {len(data[strategy])} runs for {strategy}")
    
    return data


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


def plot_convergence_with_ci(
    plot_stats: Dict[str, Dict[str, np.ndarray]],
    metric_name: str,
    beta_value: float,
    n_seeds: int = None,
    save_path: str = None,
    confidence_level: float = 0.95,
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

    if n_seeds is not None:
        title = f'Strategy Comparison: {metric_name}\n({n_seeds} seeds, {ci_pct}% CI)'
    else:
        title = f'Strategy Comparison: {metric_name}\n({ci_pct}% CI)'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_beta_grid(
    exp_dir: str,
    beta_values: List[float],
    confidence_level: float = 0.95
):
    """
    Analyze beta grid search results and generate plots for each beta.
    
    Args:
        exp_dir: Base experiment directory
        beta_values: List of beta values to analyze
        confidence_level: Confidence level for intervals
    """
    print(f"\n{'='*80}")
    print("ANALYZING BETA GRID SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Beta values: {beta_values}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Confidence level: {confidence_level}")
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
    metric_name: str = "K / |CTE|"
):
    """
    Analyze high statistics results and generate plot.
    
    Args:
        exp_dir: Base experiment directory
        beta_value: Beta value (shown in qUCB legend entry)
        confidence_level: Confidence level for intervals
        metric_name: Name of the metric being analyzed
    """
    print(f"\n{'='*80}")
    print("ANALYZING HIGH STATISTICS RESULTS")
    print(f"{'='*80}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Beta value: {beta_value}")
    print(f"Confidence level: {confidence_level}")
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
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    
    print(f"\n{'='*80}")
    print("HIGH STATISTICS ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*80}\n")


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
                       help="Display name for the metric (default: 'K / |CTE|')")
    
    args = parser.parse_args()
    
    if args.mode == "grid":
        analyze_beta_grid(
            args.exp_dir,
            args.beta_values,
            args.confidence
        )
    else:
        analyze_high_statistics(
            args.exp_dir,
            args.beta_value,
            args.confidence,
            args.metric_name
        )


if __name__ == "__main__":
    main()