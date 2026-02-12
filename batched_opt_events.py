import os
import gc
import time
import shutil
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import sys
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

from core.design_space import DesignSpace
from core.truth_interface import TruthModelEvaluator
from core.gp_models import GPModelManager
from core.acq_ucb import AcquisitionFunctionManager
from core.agent_manager import BOAgent, ResourceEvent
from core.fixed_policy import PureExploitation, PureExploration
from core.logging_utils import LoggingManager
from core.visualization import VisualizationManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

INIT_N = 5
ITERS = 20
MC_SAMPLES = 256
POOL_SUBSAMPLE = 5000
TOTAL_BATCH_SIZE = 5

TOTAL_BUDGET = 10000.0
TOTAL_TIME = 20.0
COST_PER_POINT = 100.0
TIME_PER_ITERATION = 1.0

# Budget reduction event configuration
BUDGET_EVENT_ITERATION = 4
BUDGET_EVENT_REMAINING_ITERS = 3  # Only 2 more iterations after event

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.7
MAX_REASONING_STEPS = 5

CREATE_VIS = True
CREATE_GIF = True

SETUP_SEED = 42
OUTPUT_DIR = "./test"

COMPONENTS = [
    ("Fe", 0.10, 0.40),
    ("Co", 0.10, 0.40),
    ("Ni", 0.10, 0.40),
    ("Cr", 0.10, 0.40),
    ("V",  0.10, 0.40),
]
STEP = 0.025

MODEL_PATH = "ground_truth_models/FeCoNiCrV_Min_CTE_Max_K/models/RFR_best_model.pkl"
X_SCALER_PATH = "ground_truth_models/FeCoNiCrV_Min_CTE_Max_K/models/x_scaler.pkl"
Y_SCALER_PATH = "ground_truth_models/FeCoNiCrV_Min_CTE_Max_K/models/y_scaler.pkl"

OBJ1_NAME = "CTE"
OBJ2_NAME = "K"
SCORE_NAME = "K / |CTE|"

PROBLEM_DESCRIPTION = """
High Entropy Alloy (HEA) optimization in the Fe-Co-Ni-Cr-V composition space.

Objectives:
- Minimize CTE (Coefficient of Thermal Expansion)
- Maximize K (Thermal Conductivity)

Goal: Maximize K / |CTE| ratio

Input space: 5D compositions (Fe, Co, Ni, Cr, V), each in [0.1, 0.4], summing to 1.0.

You should balance exploration and exploitation given the qEHVI and Mutual Information
acquisition values for each option to best reach the goal within budget and time constraints.
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def postprocess_outputs(preds_raw: np.ndarray) -> np.ndarray:
    cte = -1.0 * preds_raw[:, 0]
    k = preds_raw[:, 1]
    return np.column_stack([cte, k])


def score_fn(Y: np.ndarray) -> np.ndarray:
    cte = Y[:, 0]
    k = Y[:, 1]
    return k / (np.abs(cte) + 1e-12)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# RESOURCE EVENT MANAGER (for non-agent strategies)
# ============================================================================

class SimpleResourceEventManager:
    """Manages resource events for non-agent strategies"""
    
    def __init__(self):
        self.events: List[ResourceEvent] = []
    
    def add_event(self, event: ResourceEvent):
        self.events.append(event)
    
    def process_iteration_events(
        self,
        iteration: int,
        budget_remaining: float,
        time_remaining: float,
        cost_per_point: float
    ) -> Tuple[float, float, float, List[str]]:
        """
        Process any events scheduled for this iteration.
        Returns: (new_budget, new_time, new_cost, event_messages)
        """
        messages = []
        
        for event in self.events:
            if event.iteration == iteration:
                if event.event_type == "budget_change":
                    old_budget = budget_remaining
                    budget_remaining = event.modifier(budget_remaining)
                    messages.append(
                        f"[EVENT] {event.description}\n"
                        f"        Budget: ${old_budget:.2f} → ${budget_remaining:.2f}"
                    )
                elif event.event_type == "time_change":
                    old_time = time_remaining
                    time_remaining = event.modifier(time_remaining)
                    messages.append(
                        f"[EVENT] {event.description}\n"
                        f"        Time: {old_time:.1f} → {time_remaining:.1f}"
                    )
                elif event.event_type == "cost_change":
                    old_cost = cost_per_point
                    cost_per_point = event.modifier(cost_per_point)
                    messages.append(
                        f"[EVENT] {event.description}\n"
                        f"        Cost/point: ${old_cost:.2f} → ${cost_per_point:.2f}"
                    )
        
        return budget_remaining, time_remaining, cost_per_point, messages


# ============================================================================
# ONE-TIME SETUP (SHARED ACROSS ALL EXPERIMENTS)
# ============================================================================

print(f"\n{'='*80}")
print("SHARED SETUP - DESIGN SPACE AND NORMALIZATION")
print(f"{'='*80}")

np.random.seed(SETUP_SEED)
torch.manual_seed(SETUP_SEED)

design_space = DesignSpace(components=COMPONENTS, step=STEP, seed=SETUP_SEED)
bounds_np = design_space.get_bounds()
bounds_t = torch.tensor(bounds_np, dtype=DTYPE, device=DEVICE)

print(f"\nDesign space created with {len(design_space.space)} valid compositions")

print("\nComputing output normalization parameters...")
sample_X = design_space.sample(n=100, method='sobol')

temp_evaluator = TruthModelEvaluator(
    model_path=MODEL_PATH,
    x_scaler_path=X_SCALER_PATH,
    y_scaler_path=Y_SCALER_PATH,
    postprocess_outputs=postprocess_outputs,
    normalize_outputs=False,
)

sample_result = temp_evaluator.evaluate(sample_X)
sample_Y = sample_result.y

y_mean = np.mean(sample_Y, axis=0)
y_std = np.std(sample_Y, axis=0)

print(f"\nNormalization parameters:")
print(f"  {OBJ1_NAME}: mean={y_mean[0]:.4f}, std={y_std[0]:.4f}")
print(f"  {OBJ2_NAME}: mean={y_mean[1]:.4f}, std={y_std[1]:.4f}")

normalization_params = {'mean': y_mean, 'std': y_std}

temp_evaluator.cleanup()
del temp_evaluator
cleanup_memory()

print(f"\n{'='*80}")
print("SHARED SETUP COMPLETE")
print(f"{'='*80}\n")


# ============================================================================
# BO EXPERIMENT RUNNER (MODIFIED)
# ============================================================================

def run_bo_experiment(
    experiment_name: str,
    strategy,
    X0_init: np.ndarray,
    Y0_init_raw: np.ndarray,
    Y0_init_normalized: np.ndarray,
    normalization_params: dict,
    beta_explore: float,
    beta_exploit: float,
    seed: int,
    output_dir: str,
    create_visualization: bool = CREATE_VIS,
    use_budget_event: bool = False,
) -> Tuple[np.ndarray, np.ndarray, LoggingManager]:
    
    print(f"\n{'='*80}")
    print(f"[Experiment] {experiment_name}")
    print(f"[Strategy] {type(strategy).__name__}")
    if use_budget_event:
        print(f"[Budget Event] Active - will trigger at iteration {BUDGET_EVENT_ITERATION}")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    evaluator = TruthModelEvaluator(
        model_path=MODEL_PATH,
        x_scaler_path=X_SCALER_PATH,
        y_scaler_path=Y_SCALER_PATH,
        postprocess_outputs=postprocess_outputs,
        normalize_outputs=True,
        normalization_params=normalization_params,
    )
    
    gp_manager = GPModelManager(bounds=bounds_t)
    acq_manager = AcquisitionFunctionManager(
        bounds=bounds_t,
        num_restarts=3,
        raw_samples=256,
        exploitation_beta=beta_exploit,
        exploration_beta=beta_explore,
    )
    
    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = LoggingManager(exp_dir, experiment_name)
    
    vis_manager = None
    if create_visualization:
        print(f"[Viz] Initializing visualization manager for {experiment_name}...")
        vis_manager = VisualizationManager(
            design_space=design_space,
            log_dir=exp_dir,
            experiment_name=experiment_name,
            normalization_params=normalization_params,
            obj1_name=OBJ1_NAME,
            obj2_name=OBJ2_NAME,
        )
        print(f"[Viz] Visualization manager initialized successfully")
    
    # Setup event manager
    event_manager = None
    if use_budget_event:
        if isinstance(strategy, BOAgent):
            # BOAgent has built-in event handling
            event_manager = strategy
        else:
            # Use simple event manager for other strategies
            event_manager = SimpleResourceEventManager()
        
        # Calculate budget for exactly BUDGET_EVENT_REMAINING_ITERS more iterations
        budget_for_remaining = BUDGET_EVENT_REMAINING_ITERS * TOTAL_BATCH_SIZE * COST_PER_POINT
        
        event = ResourceEvent(
            iteration=BUDGET_EVENT_ITERATION,
            event_type="budget_change",
            description=f"Budget reduced to afford only {BUDGET_EVENT_REMAINING_ITERS} more iterations "
                       f"({BUDGET_EVENT_REMAINING_ITERS} × {TOTAL_BATCH_SIZE} × ${COST_PER_POINT:.2f})",
            modifier=lambda current_budget, b=budget_for_remaining: b
        )
        event_manager.add_event(event)
        print(f"[Event] Configured budget reduction at iteration {BUDGET_EVENT_ITERATION}")
        print(f"        Post-event budget: ${budget_for_remaining:.2f}")
    
    print(f"\n[Init] Using {len(X0_init)} initial samples")
    
    X_torch = torch.tensor(X0_init.copy(), dtype=DTYPE, device=DEVICE)
    Y_torch_normalized = torch.tensor(Y0_init_normalized.copy(), dtype=DTYPE, device=DEVICE)
    
    X_history_np = X0_init.copy()
    Y_history_raw_np = Y0_init_raw.copy()
    Y_history_normalized_np = Y0_init_normalized.copy()
    
    scores = score_fn(Y_history_raw_np)
    best_score = float(np.max(scores))
    print(f"[Init] Best {SCORE_NAME}: {best_score:.4f}")
    
    logger.log_iteration(
        iteration=0,
        X_history=X_history_np,
        Y_history=Y_history_raw_np,
        n_new_points=len(X0_init),
        strategy=strategy,
        timing=0.0,
        extra_info={"experiment": experiment_name, "shared_init": True, "budget_event": use_budget_event},
        acquisition_data=None,
        score_fn=score_fn,
        obj1_name=OBJ1_NAME,
        obj2_name=OBJ2_NAME,
    )
    logger.log_evaluations(
        iteration=0,
        X_new=X0_init,
        Y_new=Y0_init_raw,
        score_fn=score_fn,
        obj1_name=OBJ1_NAME,
        obj2_name=OBJ2_NAME,
    )
    
    budget_remaining = TOTAL_BUDGET - (INIT_N * COST_PER_POINT)
    time_remaining = TOTAL_TIME - TIME_PER_ITERATION
    cost_per_point = COST_PER_POINT
    
    # Modified loop with budget checking
    iteration = 0
    while iteration < ITERS:
        # Check if we have enough resources to continue
        if budget_remaining < cost_per_point * TOTAL_BATCH_SIZE or time_remaining < TIME_PER_ITERATION:
            print(f"\n[Stop] Insufficient resources at iteration {iteration}")
            print(f"       Budget: ${budget_remaining:.2f} (need ${cost_per_point * TOTAL_BATCH_SIZE:.2f})")
            print(f"       Time: {time_remaining:.1f} (need {TIME_PER_ITERATION:.1f})")
            break
        
        iteration += 1
        iter_start = time.perf_counter()
        
        # Process events for this iteration
        event_msgs = []
        if use_budget_event and event_manager is not None:
            budget_remaining, time_remaining, cost_per_point, event_msgs = \
                event_manager.process_iteration_events(
                    iteration, budget_remaining, time_remaining, cost_per_point
                )
            
            if event_msgs:
                print("\n" + "="*80)
                for msg in event_msgs:
                    print(msg)
                print("="*80)
        
        print(f"\n{'='*80}")
        print(f"[Iter {iteration}/{ITERS}] Starting...")
        print(f"[Resources] Budget: ${budget_remaining:.2f} | Time: {time_remaining:.1f}")
        print(f"{'='*80}")
        
        print("[GP] Fitting model on normalized outputs...")
        model = gp_manager.fit_model(X_torch, Y_torch_normalized)
        
        print("[Acq] Computing Pareto front on normalized outputs...")
        pareto_Y, ref_point = acq_manager.compute_pareto_front(Y_torch_normalized)
        
        if POOL_SUBSAMPLE is not None:
            X_pool_np = design_space.sample(POOL_SUBSAMPLE, method="sobol")
        else:
            X_pool_np = design_space.space.copy()
        X_pool = torch.tensor(X_pool_np, dtype=DTYPE, device=DEVICE)
        
        strategy_name = type(strategy).__name__
        
        if strategy_name == "BOAgent":
            print("[Strategy] LLM Agent - Computing all 6 allocation options")
            allocation_results = acq_manager.compute_all_allocations(
                model=model,
                mc_samples=MC_SAMPLES,
                pareto_front=pareto_Y,
                reference_point=ref_point
            )
        elif "Exploit" in strategy_name or strategy_name == "PureExploitation":
            print("[Strategy] Pure Exploitation - Computing option 0 only")
            allocation_results = acq_manager.compute_single_allocation(
                model=model,
                num_exploitation=TOTAL_BATCH_SIZE,
                num_exploration=0,
                mc_samples=MC_SAMPLES,
                pareto_front=pareto_Y,
                reference_point=ref_point
            )
        elif "Explor" in strategy_name or strategy_name == "PureExploration":
            print("[Strategy] Pure Exploration - Computing option 5 only")
            allocation_results = acq_manager.compute_single_allocation(
                model=model,
                num_exploitation=0,
                num_exploration=TOTAL_BATCH_SIZE,
                mc_samples=MC_SAMPLES,
                pareto_front=pareto_Y,
                reference_point=ref_point
            )
        else:
            print(f"[Strategy] Unknown strategy {strategy_name} - Computing all options")
            allocation_results = acq_manager.compute_all_allocations(
                model=model,
                mc_samples=MC_SAMPLES,
                pareto_front=pareto_Y,
                reference_point=ref_point
            )
        
        selected_idx = None
        
        if isinstance(strategy, BOAgent):
            selected_idx, reasoning, selected_point_arrays = strategy.select_resource_allocation(
                iteration=iteration,
                budget_remaining=budget_remaining,
                time_remaining=time_remaining,
                cost_per_point=cost_per_point,
                time_per_point=TIME_PER_ITERATION,
                X_history=X_torch.cpu().numpy(),
                Y_history=Y_torch_normalized.cpu().numpy(),
                allocation_results=allocation_results,
                max_batch_size=TOTAL_BATCH_SIZE,
                score_fn=score_fn,
            )
            if selected_point_arrays:
                X_new_np = np.vstack(selected_point_arrays)
            else:
                print("[Warning] Agent selected no points; using balanced fallback")
                selected_idx = 2
                balanced_option = allocation_results.options[2]
                parts = []
                if balanced_option.exploitation_points.size > 0:
                    parts.append(balanced_option.exploitation_points)
                if balanced_option.exploration_points.size > 0:
                    parts.append(balanced_option.exploration_points)
                X_new_np = np.vstack(parts) if parts else design_space.sample(1, method="random")
            extra_info = {
                "agent_selected_option": selected_idx,
                "agent_reasoning": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
                "budget_remaining": budget_remaining,
                "time_remaining": time_remaining,
            }
        else:
            X_new_np, reasoning = strategy.select_points(allocation_results)
            strategy_name = type(strategy).__name__
            if "Exploit" in strategy_name or strategy_name == "PureExploitation":
                selected_idx = 0
            elif "Explor" in strategy_name or strategy_name == "PureExploration":
                selected_idx = 5 if len(allocation_results.options) > 1 else 0
            else:
                selected_idx = 0
            extra_info = {
                "policy": type(strategy).__name__,
                "reasoning": reasoning,
                "budget_remaining": budget_remaining,
                "time_remaining": time_remaining,
            }
        
        if selected_idx is not None and 0 <= selected_idx < len(allocation_results.options):
            selected_option = allocation_results.options[selected_idx]
            extra_info["hypervolume_improvement"] = float(selected_option.hypervolume_improvement)
            extra_info["information_gain"] = float(selected_option.information_gain)
            extra_info["n_optimization"] = selected_option.num_exploitation
            extra_info["n_exploration"] = selected_option.num_exploration
        else:
            extra_info["hypervolume_improvement"] = None
            extra_info["information_gain"] = None
            extra_info["n_optimization"] = None
            extra_info["n_exploration"] = None
        
        # Add event information to extra_info
        if event_msgs:
            extra_info["events_triggered"] = "; ".join(event_msgs)
        
        result_new = evaluator.evaluate(X_new_np)
        Y_new_normalized = result_new.y
        Y_new_raw = evaluator.denormalize(Y_new_normalized)
        
        X_history_np = np.vstack([X_history_np, X_new_np])
        Y_history_normalized_np = np.vstack([Y_history_normalized_np, Y_new_normalized])
        Y_history_raw_np = np.vstack([Y_history_raw_np, Y_new_raw])
        X_torch = torch.tensor(X_history_np, dtype=DTYPE, device=DEVICE)
        Y_torch_normalized = torch.tensor(Y_history_normalized_np, dtype=DTYPE, device=DEVICE)
        
        points_used = len(X_new_np)
        budget_spent = points_used * cost_per_point
        time_spent = TIME_PER_ITERATION
        budget_remaining -= budget_spent
        time_remaining -= time_spent
        iter_time = time.perf_counter() - iter_start
        
        logger.log_iteration(
            iteration=iteration,
            X_history=X_history_np,
            Y_history=Y_history_raw_np,
            n_new_points=points_used,
            strategy=strategy,
            timing=iter_time,
            extra_info=extra_info,
            acquisition_data=allocation_results,
            score_fn=score_fn,
            obj1_name=OBJ1_NAME,
            obj2_name=OBJ2_NAME,
        )
        logger.log_evaluations(
            iteration=iteration,
            X_new=X_new_np,
            Y_new=Y_new_raw,
            score_fn=score_fn,
            obj1_name=OBJ1_NAME,
            obj2_name=OBJ2_NAME,
        )
        
        if vis_manager is not None:
            vis_manager.create_iteration_plot(
                iteration=iteration,
                X_history=X_history_np,
                Y_history=Y_history_raw_np,
                X_new=X_new_np,
                model=model,
                bounds=bounds_t,
                selection_info=extra_info,
                score_fn=score_fn,
                score_name=SCORE_NAME,
                obj1_name=OBJ1_NAME,
                obj2_name=OBJ2_NAME,
            )
        
        scores = score_fn(Y_history_raw_np)
        best_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
        print(f"[Iter {iteration}] Complete in {iter_time:.2f}s")
        print(f"  Best {SCORE_NAME}: {best_score:.4f}")
        print(f"  Mean {SCORE_NAME}: {mean_score:.4f}")
        print(f"  Points evaluated: {points_used}")
        print(f"  Budget remaining: ${budget_remaining:.2f}")
        
        cleanup_memory()
    
    logger.finalize()
    
    if vis_manager is not None and CREATE_GIF:
        vis_manager.create_gif(duration=2.0, gif_name=f"{experiment_name}.gif")
    
    if isinstance(strategy, BOAgent):
        strategy.save_global_memory()
    
    evaluator.cleanup()
    del evaluator
    cleanup_memory()
    
    X_final = X_history_np
    Y_final = Y_history_raw_np
    scores_final = score_fn(Y_final)
    print(f"\n[{experiment_name}] Experiment complete!")
    print(f"  Total evaluations: {len(X_final)}")
    print(f"  Best {SCORE_NAME}: {np.max(scores_final):.4f}")
    print(f"  Mean {SCORE_NAME}: {np.mean(scores_final):.4f}")
    
    return X_final, Y_final, logger


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_strategy_decisions(logger, seed: int, beta_exploit: float, beta_explore: float, budget_event: bool = False):
    df = pd.read_csv(logger.convergence_path)
    df = df[df['iteration'] > 0].copy()
    df = df.sort_values('iteration')
    opt_color = '#2ECC71'
    exp_color = '#6C5CE7'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['iteration'], df['selected_n_opt'],
            color=opt_color, linewidth=2, alpha=0.85)
    ax.scatter(df['iteration'], df['selected_n_opt'],
               color=opt_color, s=100, marker='o',
               label='Optimization Points',
               alpha=0.9, edgecolors='black', linewidth=1)
    ax.plot(df['iteration'], df['selected_n_exp'],
            color=exp_color, linewidth=2, alpha=0.85)
    ax.scatter(df['iteration'], df['selected_n_exp'],
               color=exp_color, s=100, marker='s',
               label='Exploration Points',
               alpha=0.9, edgecolors='black', linewidth=1)
    
    # Add vertical line at budget event if applicable
    if budget_event:
        ax.axvline(x=BUDGET_EVENT_ITERATION, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Budget Event (Iter {BUDGET_EVENT_ITERATION})')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Points Selected', fontsize=12, fontweight='bold')
    
    title = f'Strategy Decisions: {logger.experiment_name}\n(Seed: {seed}, β_exploit: {beta_exploit}, β_explore: {beta_explore}'
    if budget_event:
        title += f', Budget Event: Iter {BUDGET_EVENT_ITERATION})'
    else:
        title += ')'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_yticks(range(0, 6))
    ax.set_ylim(-0.5, 5.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(logger.log_dir,
                             f"{logger.experiment_name}_strategy_decisions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved strategy decisions plot to {save_path}")


def plot_experiments_comparison(loggers, seed: int, beta_exploit: float, beta_explore: float, budget_event: bool = False):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(loggers)))
    for logger, color in zip(loggers, colors):
        df = pd.read_csv(logger.convergence_path)
        df = df[df['iteration'] > 0].copy()
        df['cumulative_hvi'] = df['hypervolume_improvement'].cumsum()
        ax1.plot(df['iteration'], df['cumulative_hvi'],
                marker='o', linewidth=2, label=logger.experiment_name,
                color=color, alpha=0.8)
        ax2.plot(df['iteration'], df['information_gain'],
                marker='s', linewidth=2, label=logger.experiment_name,
                color=color, alpha=0.8)
    
    # Add vertical line at budget event if applicable
    if budget_event:
        ax1.axvline(x=BUDGET_EVENT_ITERATION, color='red', linestyle='--', 
                    linewidth=2, alpha=0.5, label=f'Budget Event')
        ax2.axvline(x=BUDGET_EVENT_ITERATION, color='red', linestyle='--', 
                    linewidth=2, alpha=0.5)
    
    title = f'Acquisition Metrics Comparison\n(Seed: {seed}, β_exploit: {beta_exploit}, β_explore: {beta_explore}'
    if budget_event:
        title += f', Budget Event: Iter {BUDGET_EVENT_ITERATION})'
    else:
        title += ')'
    
    ax1.set_ylabel('Cumulative Hypervolume Improvement', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mutual Information per Iteration', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    plt.tight_layout()
    parent_dir = os.path.dirname(loggers[0].log_dir)
    save_path = os.path.join(parent_dir, "acquisition_metrics_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved comparison plot to {save_path}")


def plot_convergence_comparison(loggers, seed: int, beta_exploit: float, beta_explore: float, budget_event: bool = False):
    logger_exploit = loggers[0]
    logger_explore = loggers[1]
    logger_agent = loggers[2]
    df_exploit = pd.read_csv(logger_exploit.convergence_path)
    df_explore = pd.read_csv(logger_explore.convergence_path)
    df_agent = pd.read_csv(logger_agent.convergence_path)
    exploit_max_iter = df_exploit.loc[df_exploit['best_score'].idxmax(), 'iteration']
    explore_max_iter = df_explore.loc[df_explore['best_score'].idxmax(), 'iteration']
    agent_max_iter = df_agent.loc[df_agent['best_score'].idxmax(), 'iteration']
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_exploit['iteration'], df_exploit['best_score'],
            marker='o', linewidth=2.5, label='Pure Exploitation', markersize=6)
    ax.plot(df_explore['iteration'], df_explore['best_score'],
            marker='s', linewidth=2.5, label='Pure Exploration', markersize=6)
    ax.plot(df_agent['iteration'], df_agent['best_score'],
            marker='^', linewidth=2.5, label='LLM Agent', markersize=7)
    ax.axvline(x=exploit_max_iter, color='C0', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(x=explore_max_iter, color='C1', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(x=agent_max_iter, color='C2', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Add vertical line at budget event if applicable
    if budget_event:
        ax.axvline(x=BUDGET_EVENT_ITERATION, color='red', linestyle='--', 
                   linewidth=2.5, alpha=0.7, label=f'Budget Event (Iter {BUDGET_EVENT_ITERATION})')
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Best {SCORE_NAME}', fontsize=14, fontweight='bold')
    
    title = f'FeCoNiCrV K/CTE Optimization: Strategy Comparison\n(Seed: {seed}, β_exploit: {beta_exploit}, β_explore: {beta_explore}'
    if budget_event:
        title += f', Budget Event: Iter {BUDGET_EVENT_ITERATION})'
    else:
        title += ')'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    parent_dir = os.path.dirname(loggers[0].log_dir)
    save_path = os.path.join(parent_dir, "convergence_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved convergence comparison to {save_path}")


# ============================================================================
# BATCH EXPERIMENT EXECUTION (MODIFIED)
# ============================================================================

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

np.random.seed(int(time.time()))
seeds = np.random.randint(1, 100000, size=5).tolist()
betas = [(0.01, 100), (0.1, 10), (0.05, 5)]

print(f"\n{'='*80}")
print("BATCH EXPERIMENT CONFIGURATION")
print(f"{'='*80}")
print(f"Random seeds: {seeds}")
print(f"Beta configurations: {betas}")
print(f"Budget event: Iteration {BUDGET_EVENT_ITERATION}, {BUDGET_EVENT_REMAINING_ITERS} iterations remaining")
print(f"Total experiments: {len(seeds) * len(betas) * 3} = {len(seeds)} seeds × {len(betas)} betas × 3 strategies")
print(f"NOTE: All runs include budget event at iteration {BUDGET_EVENT_ITERATION}")
print(f"{'='*80}\n")

for seed_idx, seed in enumerate(seeds):
    print(f"\n{'#'*80}")
    print(f"SEED {seed_idx+1}/{len(seeds)}: {seed}")
    print(f"{'#'*80}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X0_shared = design_space.sample(INIT_N, method="random")
    
    temp_evaluator_raw = TruthModelEvaluator(
        model_path=MODEL_PATH,
        x_scaler_path=X_SCALER_PATH,
        y_scaler_path=Y_SCALER_PATH,
        postprocess_outputs=postprocess_outputs,
        normalize_outputs=False,
    )
    
    temp_evaluator_norm = TruthModelEvaluator(
        model_path=MODEL_PATH,
        x_scaler_path=X_SCALER_PATH,
        y_scaler_path=Y_SCALER_PATH,
        postprocess_outputs=postprocess_outputs,
        normalize_outputs=True,
        normalization_params=normalization_params,
    )
    
    result0_raw = temp_evaluator_raw.evaluate(X0_shared)
    Y0_shared_raw = result0_raw.y
    
    result0_norm = temp_evaluator_norm.evaluate(X0_shared)
    Y0_shared_normalized = result0_norm.y
    
    temp_evaluator_raw.cleanup()
    temp_evaluator_norm.cleanup()
    del temp_evaluator_raw, temp_evaluator_norm
    cleanup_memory()
    
    print(f"\nGenerated {INIT_N} initial samples for seed {seed}")
    
    for beta_idx, (beta_exploit, beta_explore) in enumerate(betas):
        print(f"\n{'-'*80}")
        print(f"BETA CONFIG {beta_idx+1}/{len(betas)}: exploit={beta_exploit}, explore={beta_explore}")
        print(f"{'-'*80}")
        
        # All runs use budget event
        use_budget_event = True
        budget_suffix = "_budget_gutted"
        
        exp_group_dir = os.path.join(OUTPUT_DIR, f"seed{seed}_beta{beta_exploit}_{beta_explore}{budget_suffix}")
        os.makedirs(exp_group_dir, exist_ok=True)
            
        print(f"\n[1/3] Running Pure Exploitation (with budget event)...")
        
        X_exploit, Y_exploit, logger_exploit = run_bo_experiment(
            experiment_name="PureExploit",
            strategy=PureExploitation(),
            X0_init=X0_shared,
            Y0_init_raw=Y0_shared_raw,
            Y0_init_normalized=Y0_shared_normalized,
            normalization_params=normalization_params,
            beta_exploit=beta_exploit,
            beta_explore=beta_explore,
            seed=seed,
            output_dir=exp_group_dir,
            create_visualization=CREATE_VIS,
            use_budget_event=use_budget_event,
        )
        
        print(f"\n[2/3] Running Pure Exploration (with budget event)...")
        X_explore, Y_explore, logger_explore = run_bo_experiment(
            experiment_name="PureExplore",
            strategy=PureExploration(),
            X0_init=X0_shared,
            Y0_init_raw=Y0_shared_raw,
            Y0_init_normalized=Y0_shared_normalized,
            normalization_params=normalization_params,
            beta_exploit=beta_exploit,
            beta_explore=beta_explore,
            seed=seed,
            output_dir=exp_group_dir,
            create_visualization=CREATE_VIS,
            use_budget_event=use_budget_event,
        )
        
        print(f"\n[3/3] Running LLM Agent (with budget event)...")
        agent_log_dir = os.path.join(exp_group_dir, "Agent", "agent_logs")
        os.makedirs(agent_log_dir, exist_ok=True)
        
        strategy_agent = BOAgent(
            model=AGENT_MODEL,
            temperature=AGENT_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            log_dir=agent_log_dir,
            problem_description=PROBLEM_DESCRIPTION,
            obj1_name=OBJ1_NAME,
            obj2_name=OBJ2_NAME,
        )
        
        X_agent, Y_agent, logger_agent = run_bo_experiment(
            experiment_name="Agent",
            strategy=strategy_agent,
            X0_init=X0_shared,
            Y0_init_raw=Y0_shared_raw,
            Y0_init_normalized=Y0_shared_normalized,
            normalization_params=normalization_params,
            beta_exploit=beta_exploit,
            beta_explore=beta_explore,
            seed=seed,
            output_dir=exp_group_dir,
            create_visualization=CREATE_VIS,
            use_budget_event=use_budget_event,
        )
        
        print(f"\n[Plotting] Generating comparison plots for seed{seed}_beta{beta_exploit}_{beta_explore}{budget_suffix}...")
        plot_strategy_decisions(logger_agent, seed, beta_exploit, beta_explore, budget_event=use_budget_event)
        plot_experiments_comparison([logger_exploit, logger_explore, logger_agent], seed, beta_exploit, beta_explore, budget_event=use_budget_event)
        plot_convergence_comparison([logger_exploit, logger_explore, logger_agent], seed, beta_exploit, beta_explore, budget_event=use_budget_event)
        
        del strategy_agent
        cleanup_memory()
        
        print(f"\n✓ Completed seed{seed}_beta{beta_exploit}_{beta_explore}{budget_suffix}")

print(f"\n{'='*80}")
print("ALL BATCH EXPERIMENTS COMPLETE!")
print(f"{'='*80}")
print(f"Total experiments run: {len(seeds) * len(betas) * 3}")
print(f"All experiments included budget event at iteration {BUDGET_EVENT_ITERATION}")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"{'='*80}\n")