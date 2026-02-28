"""
Simple analysis script for prompt_comparison_3styles experiments.
Generates 6 types of plots from cross-seed results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import scipy.stats as stats

# ===========================================================================
# HARDCODED CONFIGURATION
# ===========================================================================

# Base experiment directory
EXP_DIR = Path("results/k-H_simple_no_unc")

# Output directory for plots
OUTPUT_DIR = EXP_DIR / "analysis_plots"

# Strategy names (folder names within each seed directory)
STRATEGIES = [
    "Agent_Default_NoUncertainty",
    "Agent_Default_WithUncertainty",
    "Agent_Minimal_NoUncertainty",
    "Agent_Minimal_WithUncertainty",
    "Agent_SimpleGuideline_NoUncertainty",
    "Agent_SimpleGuideline_WithUncertainty",
    "qEHVI",
    "qUCB"
]

# Display labels for plots (shorter names)
LABELS = {
    "Agent_Default_NoUncertainty": "Default (No Unc)",
    "Agent_Default_WithUncertainty": "Default (With Unc)",
    "Agent_Minimal_NoUncertainty": "Minimal (No Unc)",
    "Agent_Minimal_WithUncertainty": "Minimal (With Unc)",
    "Agent_SimpleGuideline_NoUncertainty": "SimpleGuideline (No Unc)",
    "Agent_SimpleGuideline_WithUncertainty": "SimpleGuideline (With Unc)",
    "qEHVI": "qEHVI",
    "qUCB": "qUCB"
}

# Colors for each strategy
COLORS = {
    "Agent_Default_NoUncertainty": "#3498DB",        # Blue
    "Agent_Default_WithUncertainty": "#1A5276",      # Dark Blue
    "Agent_Minimal_NoUncertainty": "#E74C3C",        # Red
    "Agent_Minimal_WithUncertainty": "#922B21",      # Dark Red
    "Agent_SimpleGuideline_NoUncertainty": "#2ECC71", # Green
    "Agent_SimpleGuideline_WithUncertainty": "#1D8348", # Dark Green
    "qEHVI": "#3498DB",
    "qUCB":  "#E74C3C",
}

# Confidence level for intervals
CONFIDENCE_LEVEL = 0.95

# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_convergence_data() -> Dict[str, List[pd.DataFrame]]:
    """
    Load convergence data for all strategies across all seeds.
    
    Returns:
        Dictionary mapping strategy names to lists of DataFrames (one per seed)
    """
    data = {strategy: [] for strategy in STRATEGIES}
    
    # Find all seed directories
    seed_dirs = sorted([
        d for d in EXP_DIR.iterdir() 
        if d.is_dir() and d.name.startswith("seed")
    ])
    
    if len(seed_dirs) == 0:
        raise ValueError(f"No seed directories found in {EXP_DIR}")
    
    print(f"Found {len(seed_dirs)} seed directories")
    
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        
        for strategy in STRATEGIES:
            # Path: seed_dir / strategy / {strategy}_convergence.csv
            conv_path = seed_dir / strategy / f"{strategy}_convergence.csv"
            
            if conv_path.exists():
                df = pd.read_csv(conv_path)
                data[strategy].append(df)
            else:
                print(f"Warning: Missing {conv_path}")
    
    # Print summary
    for strategy in STRATEGIES:
        print(f"  Loaded {len(data[strategy])} runs for {LABELS[strategy]}")
    
    return data

# ===========================================================================
# STATISTICS FUNCTIONS
# ===========================================================================

def compute_mean_ci_statistics(
    data: Dict[str, List[pd.DataFrame]],
    metric: str,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mean and confidence intervals across seeds for a given metric.
    """
    results = {}
    
    for strategy, dfs in data.items():
        if len(dfs) == 0:
            print(f"Warning: No data for {strategy}")
            continue
        
        # Filter to DataFrames that have the required column
        valid_dfs = [df for df in dfs if metric in df.columns]
        if not valid_dfs:
            print(f"Warning: No valid data for {strategy} with metric '{metric}'")
            continue
        
        max_iters = max(df['iteration'].max() for df in valid_dfs)
        values_by_iter = []
        
        for iter_num in range(int(max_iters) + 1):
            iter_values = []
            for df in valid_dfs:
                if iter_num in df['iteration'].values:
                    value = df[df['iteration'] == iter_num][metric].values[0]
                    iter_values.append(value)
            values_by_iter.append(iter_values)
        
        iterations = np.arange(len(values_by_iter))
        means = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
        stds = np.array([np.std(vals, ddof=1) if len(vals) > 1 else 0.0 for vals in values_by_iter])
        n_samples = np.array([len(vals) for vals in values_by_iter])
        
        # t-critical values for CI
        t_critical = np.array([
            stats.t.ppf((1 + confidence_level) / 2, max(n - 1, 1)) if n > 1 else 0.0
            for n in n_samples
        ])
        
        ci_half_width = t_critical * stds / np.sqrt(np.maximum(n_samples, 1))
        
        results[strategy] = {
            'iterations': iterations,
            'mean': means,
            'std': stds,
            'ci_lower': means - ci_half_width,
            'ci_upper': means + ci_half_width,
            'n_samples': n_samples,
        }
    
    return results


def compute_median_quantile_statistics(
    data: Dict[str, List[pd.DataFrame]],
    metric: str,
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute median and quantiles across seeds for a given metric.
    """
    results = {}
    
    for strategy, dfs in data.items():
        if len(dfs) == 0:
            continue
        
        valid_dfs = [df for df in dfs if metric in df.columns]
        if not valid_dfs:
            continue
        
        max_iters = max(df['iteration'].max() for df in valid_dfs)
        values_by_iter = []
        
        for iter_num in range(int(max_iters) + 1):
            iter_values = []
            for df in valid_dfs:
                if iter_num in df['iteration'].values:
                    value = df[df['iteration'] == iter_num][metric].values[0]
                    iter_values.append(value)
            values_by_iter.append(iter_values)
        
        iterations = np.arange(len(values_by_iter))
        medians = np.array([np.median(vals) if len(vals) > 0 else np.nan for vals in values_by_iter])
        q_lower = np.array([np.percentile(vals, lower_quantile * 100) if len(vals) > 0 else np.nan for vals in values_by_iter])
        q_upper = np.array([np.percentile(vals, upper_quantile * 100) if len(vals) > 0 else np.nan for vals in values_by_iter])
        n_samples = np.array([len(vals) for vals in values_by_iter])
        
        results[strategy] = {
            'iterations': iterations,
            'median': medians,
            'q_lower': q_lower,
            'q_upper': q_upper,
            'n_samples': n_samples,
        }
    
    return results


def compute_combined_uncertainty_statistics(
    data: Dict[str, List[pd.DataFrame]],
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mean and CI for combined uncertainty (obj1 + obj2) across seeds.
    """
    results = {}
    
    for strategy, dfs in data.items():
        if len(dfs) == 0:
            continue
        
        required_cols = {'total_uncertainty_obj1', 'total_uncertainty_obj2'}
        valid_dfs = [df for df in dfs if required_cols.issubset(df.columns)]
        
        if not valid_dfs:
            print(f"Warning: No uncertainty columns for {strategy}")
            continue
        
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
        stds = np.array([np.std(vals, ddof=1) if len(vals) > 1 else 0.0 for vals in values_by_iter])
        n_samples = np.array([len(vals) for vals in values_by_iter])
        
        t_critical = np.array([
            stats.t.ppf((1 + confidence_level) / 2, max(n - 1, 1)) if n > 1 else 0.0
            for n in n_samples
        ])
        
        ci_half_width = t_critical * stds / np.sqrt(np.maximum(n_samples, 1))
        
        results[strategy] = {
            'iterations': iterations,
            'mean': means,
            'std': stds,
            'ci_lower': means - ci_half_width,
            'ci_upper': means + ci_half_width,
            'n_samples': n_samples,
        }
    
    return results


def compute_selection_statistics(
    data: Dict[str, List[pd.DataFrame]],
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute mean and CI for selected_n_opt and selected_n_exp across seeds.
    Only processes strategies with "agent" in the name (case-insensitive).
    
    Returns:
        Dict mapping strategy -> {'n_opt': stats_dict, 'n_exp': stats_dict}
    """
    results = {}
    
    for strategy, dfs in data.items():
        # Only process strategies with "agent" in the name
        if "agent" not in strategy.lower():
            continue
            
        if len(dfs) == 0:
            continue
        
        required_cols = {'selected_n_opt', 'selected_n_exp'}
        valid_dfs = [df for df in dfs if required_cols.issubset(df.columns)]
        
        if not valid_dfs:
            print(f"Warning: No selection columns for {strategy}")
            continue
        
        results[strategy] = {}
        
        for col_name, key in [('selected_n_opt', 'n_opt'), ('selected_n_exp', 'n_exp')]:
            max_iters = max(df['iteration'].max() for df in valid_dfs)
            values_by_iter = []
            
            for iter_num in range(int(max_iters) + 1):
                iter_values = []
                for df in valid_dfs:
                    if iter_num in df['iteration'].values:
                        value = df[df['iteration'] == iter_num][col_name].values[0]
                        iter_values.append(value)
                values_by_iter.append(iter_values)
            
            iterations = np.arange(len(values_by_iter))
            means = np.array([np.mean(vals) if vals else np.nan for vals in values_by_iter])
            stds = np.array([np.std(vals, ddof=1) if len(vals) > 1 else 0.0 for vals in values_by_iter])
            n_samples = np.array([len(vals) for vals in values_by_iter])
            
            t_critical = np.array([
                stats.t.ppf((1 + confidence_level) / 2, max(n - 1, 1)) if n > 1 else 0.0
                for n in n_samples
            ])
            
            ci_half_width = t_critical * stds / np.sqrt(np.maximum(n_samples, 1))
            
            results[strategy][key] = {
                'iterations': iterations,
                'mean': means,
                'std': stds,
                'ci_lower': means - ci_half_width,
                'ci_upper': means + ci_half_width,
                'n_samples': n_samples,
            }
    
    return results

# ===========================================================================
# HELPER FUNCTION FOR AXIS FORMATTING
# ===========================================================================

def format_axis(ax, max_iter: int):
    """
    Apply consistent formatting to axes:
    - Remove grid
    - Set x-axis ticks to integers starting from 1
    """
    ax.grid(False)
    # Set x-axis ticks to 1, 2, 3, 4, 5, ...
    tick_positions = np.arange(1, max_iter + 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(x)) for x in tick_positions])
    ax.set_xlim(0.5, max_iter + 0.5)

# ===========================================================================
# PLOTTING FUNCTIONS
# ===========================================================================

def plot_convergence_mean_ci(
    stats: Dict[str, Dict[str, np.ndarray]],
    metric_name: str,
    save_path: str,
    n_seeds: int = None
):
    """
    Plot 1: Convergence with mean and 95% CI.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    max_iter = 0
    for strategy, stat in stats.items():
        color = COLORS.get(strategy, 'gray')
        label = LABELS.get(strategy, strategy)
        iters = stat['iterations']
        mean = stat['mean']
        max_iter = max(max_iter, int(iters.max()))
        
        ax.plot(iters, mean, color=color, linewidth=2, label=label, marker='o', markersize=4)
        ax.fill_between(iters, stat['ci_lower'], stat['ci_upper'], color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    
    title = f'Convergence: {metric_name} (Mean ± 95% CI)'
    if n_seeds:
        title += f' - {n_seeds} seeds'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    format_axis(ax, max_iter)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence_median_quantiles(
    stats: Dict[str, Dict[str, np.ndarray]],
    metric_name: str,
    save_path: str,
    n_seeds: int = None
):
    """
    Plot 2: Convergence with median and 2.5-97.5 quantiles.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    max_iter = 0
    for strategy, stat in stats.items():
        color = COLORS.get(strategy, 'gray')
        label = LABELS.get(strategy, strategy)
        iters = stat['iterations']
        median = stat['median']
        max_iter = max(max_iter, int(iters.max()))
        
        ax.plot(iters, median, color=color, linewidth=2, label=label, marker='o', markersize=4)
        ax.fill_between(iters, stat['q_lower'], stat['q_upper'], color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    
    title = f'Convergence: {metric_name} (Median, 2.5-97.5 Quantiles)'
    if n_seeds:
        title += f' - {n_seeds} seeds'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    format_axis(ax, max_iter)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_hypervolume(
    stats: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
    n_seeds: int = None
):
    """
    Plot 3: Averaged hypervolume per iteration.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    max_iter = 0
    for strategy, stat in stats.items():
        color = COLORS.get(strategy, 'gray')
        label = LABELS.get(strategy, strategy)
        iters = stat['iterations']
        mean = stat['mean']
        max_iter = max(max_iter, int(iters.max()))
        
        ax.plot(iters, mean, color=color, linewidth=2, label=label, marker='o', markersize=4)
        ax.fill_between(iters, stat['ci_lower'], stat['ci_upper'], color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize=14, fontweight='bold')
    
    title = 'Hypervolume per Iteration (Mean ± 95% CI)'
    if n_seeds:
        title += f' - {n_seeds} seeds'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    format_axis(ax, max_iter)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_total_uncertainty(
    stats: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
    n_seeds: int = None
):
    """
    Plot 4: Averaged total uncertainty per iteration.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    max_iter = 0
    for strategy, stat in stats.items():
        color = COLORS.get(strategy, 'gray')
        label = LABELS.get(strategy, strategy)
        iters = stat['iterations']
        mean = stat['mean']
        max_iter = max(max_iter, int(iters.max()))
        
        ax.plot(iters, mean, color=color, linewidth=2, label=label, marker='o', markersize=4)
        ax.fill_between(iters, stat['ci_lower'], stat['ci_upper'], color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Uncertainty (obj1 + obj2)', fontsize=14, fontweight='bold')
    
    title = 'Total Uncertainty per Iteration (Mean ± 95% CI)'
    if n_seeds:
        title += f' - {n_seeds} seeds'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    format_axis(ax, max_iter)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_selection_counts(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    save_dir: Path,
    n_seeds: int = None
):
    """
    Plot 5: For each strategy with "agent" in the name, plot n_opt and n_exp 
    selection counts with 95% CI. Creates one plot per strategy.
    """
    for strategy, selection_stats in stats.items():
        # Only process strategies with "agent" in the name
        if "agent" not in strategy.lower():
            continue
            
        if 'n_opt' not in selection_stats or 'n_exp' not in selection_stats:
            print(f"Warning: Missing selection data for {strategy}, skipping.")
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot n_opt (optimization points)
        opt_stats = selection_stats['n_opt']
        iters = opt_stats['iterations']
        max_iter = int(iters.max())
        
        ax.plot(iters, opt_stats['mean'], color='#E74C3C', linewidth=2, 
                label='Optimization (n_opt)', marker='o', markersize=4)
        ax.fill_between(iters, opt_stats['ci_lower'], opt_stats['ci_upper'], 
                        color='#E74C3C', alpha=0.2)
        
        # Plot n_exp (exploration points)
        exp_stats = selection_stats['n_exp']
        ax.plot(iters, exp_stats['mean'], color='#3498DB', linewidth=2, 
                label='Exploration (n_exp)', marker='s', markersize=4)
        ax.fill_between(iters, exp_stats['ci_lower'], exp_stats['ci_upper'], 
                        color='#3498DB', alpha=0.2)
        
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Points Selected', fontsize=14, fontweight='bold')
        
        display_label = LABELS.get(strategy, strategy)
        title = f'Selection Counts: {display_label} (Mean ± 95% CI)'
        if n_seeds:
            title += f' - {n_seeds} seeds'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        format_axis(ax, max_iter)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        save_path = save_dir / f"selection_counts_{strategy}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def plot_exploration_all_strategies(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    save_path: str,
    n_seeds: int = None
):
    """
    Plot 6: Exploration points (selected_n_exp) for all 6 agent strategies on one plot.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    max_iter = 0
    for strategy, selection_stats in stats.items():
        if 'n_exp' not in selection_stats:
            print(f"Warning: Missing n_exp data for {strategy}, skipping.")
            continue
        
        exp_stats = selection_stats['n_exp']
        color = COLORS.get(strategy, 'gray')
        label = LABELS.get(strategy, strategy)
        iters = exp_stats['iterations']
        max_iter = max(max_iter, int(iters.max()))
        
        ax.plot(iters, exp_stats['mean'], color=color, linewidth=2, 
                label=label, marker='o', markersize=4)
        ax.fill_between(iters, exp_stats['ci_lower'], exp_stats['ci_upper'], 
                        color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Exploration Points Selected', fontsize=14, fontweight='bold')
    
    title = 'Exploration Points (n_exp) per Iteration (Mean ± 95% CI)'
    if n_seeds:
        title += f' - {n_seeds} seeds'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    format_axis(ax, max_iter)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ===========================================================================
# MAIN ANALYSIS FUNCTION
# ===========================================================================

def run_analysis():
    """
    Main function to run all analyses and generate all 6 plots.
    """
    print("=" * 80)
    print("PROMPT COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Experiment directory: {EXP_DIR}")
    print(f"Strategies: {STRATEGIES}")
    print("=" * 80 + "\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Load all data
    print("Loading convergence data...")
    data = load_convergence_data()
    
    # Count seeds
    n_seeds = 0
    for strategy in STRATEGIES:
        if len(data[strategy]) > 0:
            n_seeds = len(data[strategy])
            break
    print(f"Number of seeds: {n_seeds}\n")
    
    # =========================================================================
    # PLOT 1: Convergence with Mean and 95% CI
    # =========================================================================
    print("-" * 80)
    print("PLOT 1: Convergence (Mean ± 95% CI)")
    print("-" * 80)
    
    stats_mean_ci = compute_mean_ci_statistics(data, metric="best_score", confidence_level=CONFIDENCE_LEVEL)
    plot_convergence_mean_ci(
        stats_mean_ci,
        metric_name="Best Score",
        save_path=str(OUTPUT_DIR / "plot1_convergence_mean_ci.png"),
        n_seeds=n_seeds
    )
    
    # =========================================================================
    # PLOT 2: Convergence with Median and 2.5-97.5 Quantiles
    # =========================================================================
    print("-" * 80)
    print("PLOT 2: Convergence (Median, 2.5-97.5 Quantiles)")
    print("-" * 80)
    
    stats_median_quantile = compute_median_quantile_statistics(
        data, metric="best_score", lower_quantile=0.025, upper_quantile=0.975
    )
    plot_convergence_median_quantiles(
        stats_median_quantile,
        metric_name="Best Score",
        save_path=str(OUTPUT_DIR / "plot2_convergence_median_quantiles.png"),
        n_seeds=n_seeds
    )
    
    # =========================================================================
    # PLOT 3: Hypervolume
    # =========================================================================
    print("-" * 80)
    print("PLOT 3: Hypervolume per Iteration")
    print("-" * 80)
    
    # Check if hypervolume column exists
    has_hypervolume = False
    for strategy in STRATEGIES:
        if len(data[strategy]) > 0 and 'total_hypervolume' in data[strategy][0].columns:
            has_hypervolume = True
            break
    
    if has_hypervolume:
        stats_hv = compute_mean_ci_statistics(data, metric="total_hypervolume", confidence_level=CONFIDENCE_LEVEL)
        plot_hypervolume(
            stats_hv,
            save_path=str(OUTPUT_DIR / "plot3_hypervolume.png"),
            n_seeds=n_seeds
        )
    else:
        print("Warning: No hypervolume data found in convergence files. Skipping Plot 3.")
    
    # =========================================================================
    # PLOT 4: Total Uncertainty
    # =========================================================================
    print("-" * 80)
    print("PLOT 4: Total Uncertainty per Iteration")
    print("-" * 80)
    
    stats_unc = compute_combined_uncertainty_statistics(data, confidence_level=CONFIDENCE_LEVEL)
    if stats_unc:
        plot_total_uncertainty(
            stats_unc,
            save_path=str(OUTPUT_DIR / "plot4_total_uncertainty.png"),
            n_seeds=n_seeds
        )
    else:
        print("Warning: No uncertainty data found. Skipping Plot 4.")
    
    # =========================================================================
    # PLOT 5: Selection Counts (n_opt and n_exp) per Strategy (Agent only)
    # =========================================================================
    print("-" * 80)
    print("PLOT 5: Selection Counts per Strategy (Agent strategies only)")
    print("-" * 80)
    
    stats_selection = compute_selection_statistics(data, confidence_level=CONFIDENCE_LEVEL)
    if stats_selection:
        plot_selection_counts(
            stats_selection,
            save_dir=OUTPUT_DIR,
            n_seeds=n_seeds
        )
    else:
        print("Warning: No selection count data found. Skipping Plot 5.")
    
    # =========================================================================
    # PLOT 6: Exploration Points (n_exp) for All Strategies
    # =========================================================================
    print("-" * 80)
    print("PLOT 6: Exploration Points (n_exp) - All Strategies")
    print("-" * 80)
    
    if stats_selection:
        plot_exploration_all_strategies(
            stats_selection,
            save_path=str(OUTPUT_DIR / "plot6_exploration_all_strategies.png"),
            n_seeds=n_seeds
        )
    else:
        print("Warning: No selection count data found. Skipping Plot 6.")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_analysis()