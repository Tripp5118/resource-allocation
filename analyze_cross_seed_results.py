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
from typing import Dict, List, Tuple
import scipy.stats as stats


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
    strategies = ["PureExploit", "PureExplore", "Agent"]
    data = {strategy: [] for strategy in strategies}
    
    if mode == "grid":
        if beta_value is None:
            raise ValueError("beta_value must be specified for grid mode")
        search_dir = Path(exp_dir) / f"beta{beta_value}"
    else:  # mode == "single"
        search_dir = Path(exp_dir)
    
    if not search_dir.exists():
        raise ValueError(f"Directory does not exist: {search_dir}")
    
    # Find all seed directories
    seed_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    
    if len(seed_dirs) == 0:
        raise ValueError(f"No seed directories found in {search_dir}")
    
    print(f"Found {len(seed_dirs)} seed directories in {search_dir}")
    
    # Load data for each strategy
    for seed_dir in seed_dirs:
        seed = seed_dir.name
        for strategy in strategies:
            conv_path = seed_dir / strategy / f"{strategy}_convergence.csv"
            if conv_path.exists():
                df = pd.read_csv(conv_path)
                data[strategy].append(df)
            else:
                print(f"Warning: Missing convergence file for {seed}/{strategy}")
    
    # Report what was loaded
    for strategy in strategies:
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
        
        # Find maximum number of iterations across all runs
        max_iters = max(df['iteration'].max() for df in dfs)
        
        # Collect metric values for each iteration
        values_by_iter = []
        for iter_num in range(int(max_iters) + 1):
            iter_values = []
            for df in dfs:
                if iter_num in df['iteration'].values:
                    value = df[df['iteration'] == iter_num][metric].values[0]
                    iter_values.append(value)
            values_by_iter.append(iter_values)
        
        # Compute statistics
        iterations = np.arange(len(values_by_iter))
        means = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
        stds = np.array([np.std(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
        
        # Compute confidence intervals using t-distribution
        n_samples = np.array([len(vals) for vals in values_by_iter])
        # Use t-distribution for small samples
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


def plot_convergence_with_ci(
    stats: Dict[str, Dict[str, np.ndarray]],
    metric_name: str,
    beta_value: float = None,
    n_seeds: int = None,
    save_path: str = None,
    show_std: bool = True,
    show_ci: bool = True,
    confidence_level: float = 0.95,
):
    """
    Plot convergence curves with confidence intervals.
    
    Args:
        stats: Statistics dictionary from compute_statistics()
        metric_name: Name of metric for y-axis label
        beta_value: Beta value (for title)
        n_seeds: Number of seeds (for title)
        save_path: Path to save figure
        show_std: Whether to show ±1 std shading
        show_ci: Whether to show confidence interval shading
        confidence_level: Confidence level for CI
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'PureExploit': '#3498DB', 'PureExplore': '#E74C3C', 'Agent': '#2ECC71'}
    labels = {'PureExploit': 'Pure Exploitation', 'PureExplore': 'Pure Exploration', 'Agent': 'LLM Agent'}
    
    for strategy, stat in stats.items():
        color = colors.get(strategy, 'gray')
        label = labels.get(strategy, strategy)
        
        iters = stat['iterations']
        mean = stat['mean']
        
        # Plot mean line
        ax.plot(iters, mean, color=color, linewidth=2.5, label=label, marker='o', markersize=5)
        
        # Plot confidence interval
        if show_ci:
            ax.fill_between(
                iters,
                stat['ci_lower'],
                stat['ci_upper'],
                color=color,
                alpha=0.2,
                label=f'{label} {int(confidence_level*100)}% CI'
            )
        
        # Plot standard deviation
        if show_std and not show_ci:  # Only show one or the other to avoid clutter
            ax.fill_between(
                iters,
                mean - stat['std'],
                mean + stat['std'],
                color=color,
                alpha=0.2,
                label=f'{label} ±1 std'
            )
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric_name}', fontsize=14, fontweight='bold')
    
    # Build title
    if beta_value is not None and n_seeds is not None:
        title = f'Strategy Comparison: {metric_name}\n(β_explore={beta_value}, {n_seeds} seeds, {int(confidence_level*100)}% confidence intervals)'
    elif n_seeds is not None:
        title = f'Strategy Comparison: {metric_name}\n({n_seeds} seeds, {int(confidence_level*100)}% confidence intervals)'
    else:
        title = f'Strategy Comparison: {metric_name}\n({int(confidence_level*100)}% confidence intervals)'
    
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


def analyze_beta_grid(exp_dir: str, beta_values: List[float], confidence_level: float = 0.95):
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
    print(f"Confidence level: {confidence_level}")
    print(f"{'='*80}\n")
    
    output_dir = Path(exp_dir) / "cross_seed_analysis"
    output_dir.mkdir(exist_ok=True)
    
    for beta in beta_values:
        print(f"\n{'-'*80}")
        print(f"Processing β_explore = {beta}")
        print(f"{'-'*80}")
        
        try:
            # Load data
            data = load_convergence_data(exp_dir, beta_value=beta, mode="grid")
            n_seeds = len(data['PureExploit'])
            
            # Compute statistics for best_score
            stats_score = compute_statistics(data, metric="best_score", confidence_level=confidence_level)
            
            # Plot convergence with CI
            save_path = output_dir / f"convergence_beta{beta}_ci.png"
            plot_convergence_with_ci(
                stats_score,
                metric_name="K / |CTE|",
                beta_value=beta,
                n_seeds=n_seeds,
                save_path=str(save_path),
                show_std=False,
                show_ci=True,
                confidence_level=confidence_level,
            )
            
            # Compute statistics for hypervolume
            if 'total_hypervolume' in data['PureExploit'][0].columns:
                stats_hv = compute_statistics(data, metric="total_hypervolume", confidence_level=confidence_level)
                save_path_hv = output_dir / f"hypervolume_beta{beta}_ci.png"
                plot_convergence_with_ci(
                    stats_hv,
                    metric_name="Hypervolume",
                    beta_value=beta,
                    n_seeds=n_seeds,
                    save_path=str(save_path_hv),
                    show_std=False,
                    show_ci=True,
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


def analyze_high_statistics(exp_dir: str, beta_value: float = 2.0, confidence_level: float = 0.95, metric_name: str = "K / |CTE|"):
    """
    Analyze high statistics results and generate plot.
    
    Args:
        exp_dir: Base experiment directory
        beta_value: Beta value (for plot title)
        confidence_level: Confidence level for intervals
    """
    print(f"\n{'='*80}")
    print("ANALYZING HIGH STATISTICS RESULTS")
    print(f"{'='*80}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Beta value: {beta_value}")
    print(f"Confidence level: {confidence_level}")
    print(f"{'='*80}\n")
    
    output_dir = Path(exp_dir) / "cross_seed_analysis"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        data = load_convergence_data(exp_dir, mode="single")
        n_seeds = len(data['PureExploit'])
        
        # Compute statistics for best_score
        stats_score = compute_statistics(data, metric="best_score", confidence_level=confidence_level)
        
        # Plot convergence with CI
        save_path = output_dir / f"convergence_beta{beta_value}_n{n_seeds}_ci.png"
        plot_convergence_with_ci(
            stats_score,
            metric_name="K / |CTE|",
            beta_value=beta_value,
            n_seeds=n_seeds,
            save_path=str(save_path),
            show_std=False,
            show_ci=True,
            confidence_level=confidence_level,
        )
        
        # Compute statistics for hypervolume
        if 'total_hypervolume' in data['PureExploit'][0].columns:
            stats_hv = compute_statistics(data, metric="total_hypervolume", confidence_level=confidence_level)
            save_path_hv = output_dir / f"hypervolume_beta{beta_value}_n{n_seeds}_ci.png"
            plot_convergence_with_ci(
                stats_hv,
                metric_name="Hypervolume",
                beta_value=beta_value,
                n_seeds=n_seeds,
                save_path=str(save_path_hv),
                show_std=False,
                show_ci=True,
                confidence_level=confidence_level,
            )
        
        # Generate summary statistics table
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        for strategy, stat in stats_score.items():
            final_mean = stat['mean'][-1]
            final_std = stat['std'][-1]
            final_ci = (stat['ci_upper'][-1] - stat['ci_lower'][-1]) / 2
            print(f"{strategy}:")
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
    parser = argparse.ArgumentParser(description="Analyze cross-seed BO experiment results")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--mode", type=str, choices=["grid", "single"], required=True,
                       help="Analysis mode: 'grid' for beta grid search, 'single' for high statistics")
    parser.add_argument("--beta_values", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0],
                       help="Beta values to analyze (grid mode only)")
    parser.add_argument("--beta_value", type=float, default=2.0,
                       help="Beta value for plot title (single mode only)")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for intervals (default: 0.95)")
    
    args = parser.parse_args()
    
    if args.mode == "grid":
        analyze_beta_grid(args.exp_dir, args.beta_values, args.confidence)
    else:  # mode == "single"
        analyze_high_statistics(args.exp_dir, args.beta_value, args.confidence)


if __name__ == "__main__":
    main()