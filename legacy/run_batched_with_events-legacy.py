# run_batched_with_events.py
"""Clean runner for batched BO experiments with resource events."""

import os
import time
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from core.design_space import DesignSpace
from core.batched_runs import (
    run_bo_experiment,
    compute_normalization_params,
    generate_shared_initialization,
    cleanup_memory,
)
from core.agent_manager import BOAgent, ResourceEvent
from core.fixed_policy import PureExploitation, PureExploration
from core.visualization import (
    plot_strategy_decisions,
    plot_acquisition_metrics_comparison,
    plot_convergence_comparison,
)

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Data paths
MODEL_PATH = "ground_truth_models/FeCoNiCrV_Min_CTE_Max_K/models/RFR_best_model.pkl"
X_SCALER_PATH = "ground_truth_models/FeCoNiCrV_Min_CTE_Max_K/models/x_scaler.pkl"
Y_SCALER_PATH = "ground_truth_models/FeCoNiCrV_Min_CTE_Max_K/models/y_scaler.pkl"

# Design space
COMPONENTS = [
    ("Fe", 0.10, 0.40),
    ("Co", 0.10, 0.40),
    ("Ni", 0.10, 0.40),
    ("Cr", 0.10, 0.40),
    ("V",  0.10, 0.40),
]
STEP = 0.025

# BO parameters
INIT_N = 5
ITERS = 20
MC_SAMPLES = 256
POOL_SUBSAMPLE = 5000
TOTAL_BATCH_SIZE = 5

# Resource parameters
TOTAL_BUDGET = 10000.0
TOTAL_TIME = 20.0
COST_PER_POINT = 100.0
TIME_PER_ITERATION = 1.0

# Event configuration
BUDGET_EVENT_ITERATION = 4
BUDGET_EVENT_REMAINING_ITERS = 3  # Only 3 more iterations after event

# Agent parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.7

# Objective names
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

# Visualization
CREATE_VIS = True
CREATE_GIF = True

# Output
SETUP_SEED = 42
OUTPUT_DIR = "./results_with_events"

# Batch experiment configuration
NUM_SEEDS = 5
BETA_EXPLORE_VALUES = [100, 10, 5]  # Exploration beta values to test

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def postprocess_outputs(preds_raw: np.ndarray) -> np.ndarray:
    """Postprocess model outputs: negate CTE, keep K as is."""
    cte = -1.0 * preds_raw[:, 0]
    k = preds_raw[:, 1]
    return np.column_stack([cte, k])


def score_fn(Y: np.ndarray) -> np.ndarray:
    """Compute K / |CTE| score."""
    cte = Y[:, 0]
    k = Y[:, 1]
    return k / (np.abs(cte) + 1e-12)


def create_budget_event() -> ResourceEvent:
    """Create a budget reduction event."""
    budget_for_remaining = BUDGET_EVENT_REMAINING_ITERS * TOTAL_BATCH_SIZE * COST_PER_POINT
    
    event = ResourceEvent(
        iteration=BUDGET_EVENT_ITERATION,
        event_type="budget_change",
        description=f"Budget reduced to afford only {BUDGET_EVENT_REMAINING_ITERS} more iterations "
                   f"({BUDGET_EVENT_REMAINING_ITERS} × {TOTAL_BATCH_SIZE} × ${COST_PER_POINT:.2f})",
        modifier=lambda current_budget, b=budget_for_remaining: b
    )
    
    return event


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

if __name__ == "__main__":
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set!")
    
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Setup design space
    print(f"\n{'='*80}")
    print("SHARED SETUP - DESIGN SPACE AND NORMALIZATION")
    print(f"{'='*80}")
    
    np.random.seed(SETUP_SEED)
    torch.manual_seed(SETUP_SEED)
    
    design_space = DesignSpace(components=COMPONENTS, step=STEP, seed=SETUP_SEED)
    bounds_np = design_space.get_bounds()
    bounds_t = torch.tensor(bounds_np, dtype=torch.double)
    
    print(f"\nDesign space created with {len(design_space.space)} valid compositions")
    
    # Compute normalization parameters
    normalization_params = compute_normalization_params(
        design_space=design_space,
        model_path=MODEL_PATH,
        x_scaler_path=X_SCALER_PATH,
        y_scaler_path=Y_SCALER_PATH,
        postprocess_fn=postprocess_outputs,
        n_samples=100,
        obj1_name=OBJ1_NAME,
        obj2_name=OBJ2_NAME,
    )
    
    print(f"\n{'='*80}")
    print("SHARED SETUP COMPLETE")
    print(f"{'='*80}\n")
    
    # Generate random seeds
    np.random.seed(int(time.time()))
    seeds = np.random.randint(1, 100000, size=NUM_SEEDS).tolist()
    
    print(f"\n{'='*80}")
    print("BATCH EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Random seeds: {seeds}")
    print(f"Exploration beta values: {BETA_EXPLORE_VALUES}")
    print(f"Budget event: Iteration {BUDGET_EVENT_ITERATION}, {BUDGET_EVENT_REMAINING_ITERS} iterations remaining")
    print(f"Total experiments: {NUM_SEEDS * len(BETA_EXPLORE_VALUES) * 3} = "
          f"{NUM_SEEDS} seeds × {len(BETA_EXPLORE_VALUES)} betas × 3 strategies")
    print(f"NOTE: All runs include budget event at iteration {BUDGET_EVENT_ITERATION}")
    print(f"{'='*80}\n")
    
    # Run experiments
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#'*80}")
        print(f"SEED {seed_idx+1}/{NUM_SEEDS}: {seed}")
        print(f"{'#'*80}")
        
        # Generate shared initialization for this seed
        X0_shared, Y0_shared_raw, Y0_shared_normalized = generate_shared_initialization(
            design_space=design_space,
            model_path=MODEL_PATH,
            x_scaler_path=X_SCALER_PATH,
            y_scaler_path=Y_SCALER_PATH,
            postprocess_fn=postprocess_outputs,
            normalization_params=normalization_params,
            n_init=INIT_N,
            seed=seed,
        )
        
        print(f"\nGenerated {INIT_N} initial samples for seed {seed}")
        
        for beta_idx, beta_explore in enumerate(BETA_EXPLORE_VALUES):
            print(f"\n{'-'*80}")
            print(f"BETA CONFIG {beta_idx+1}/{len(BETA_EXPLORE_VALUES)}: β_explore={beta_explore}")
            print(f"{'-'*80}")
            
            exp_group_dir = os.path.join(OUTPUT_DIR, f"seed{seed}_beta{beta_explore}_event")
            os.makedirs(exp_group_dir, exist_ok=True)
            
            # Create budget event
            budget_event = create_budget_event()
            
            # Run Pure Exploitation (with event)
            print(f"\n[1/3] Running Pure Exploitation (with budget event)...")
            X_exploit, Y_exploit, logger_exploit = run_bo_experiment(
                experiment_name="PureExploit",
                strategy=PureExploitation(),
                output_dir=exp_group_dir,
                model_path=MODEL_PATH,
                x_scaler_path=X_SCALER_PATH,
                y_scaler_path=Y_SCALER_PATH,
                postprocess_fn=postprocess_outputs,
                score_fn=score_fn,
                design_space=design_space,
                bounds=bounds_t,
                X0_init=X0_shared,
                Y0_init_raw=Y0_shared_raw,
                Y0_init_normalized=Y0_shared_normalized,
                normalization_params=normalization_params,
                exploration_beta=beta_explore,
                n_iterations=ITERS,
                mc_samples=MC_SAMPLES,
                total_batch_size=TOTAL_BATCH_SIZE,
                pool_subsample=POOL_SUBSAMPLE,
                total_budget=TOTAL_BUDGET,
                total_time=TOTAL_TIME,
                cost_per_point=COST_PER_POINT,
                time_per_iteration=TIME_PER_ITERATION,
                obj1_name=OBJ1_NAME,
                obj2_name=OBJ2_NAME,
                score_name=SCORE_NAME,
                seed=seed,
                create_visualization=CREATE_VIS,
                create_gif=CREATE_GIF,
                events=[budget_event],
            )
            
            # Run Pure Exploration (with event)
            print(f"\n[2/3] Running Pure Exploration (with budget event)...")
            budget_event = create_budget_event()  # Recreate event
            X_explore, Y_explore, logger_explore = run_bo_experiment(
                experiment_name="PureExplore",
                strategy=PureExploration(),
                output_dir=exp_group_dir,
                model_path=MODEL_PATH,
                x_scaler_path=X_SCALER_PATH,
                y_scaler_path=Y_SCALER_PATH,
                postprocess_fn=postprocess_outputs,
                score_fn=score_fn,
                design_space=design_space,
                bounds=bounds_t,
                X0_init=X0_shared,
                Y0_init_raw=Y0_shared_raw,
                Y0_init_normalized=Y0_shared_normalized,
                normalization_params=normalization_params,
                exploration_beta=beta_explore,
                n_iterations=ITERS,
                mc_samples=MC_SAMPLES,
                total_batch_size=TOTAL_BATCH_SIZE,
                pool_subsample=POOL_SUBSAMPLE,
                total_budget=TOTAL_BUDGET,
                total_time=TOTAL_TIME,
                cost_per_point=COST_PER_POINT,
                time_per_iteration=TIME_PER_ITERATION,
                obj1_name=OBJ1_NAME,
                obj2_name=OBJ2_NAME,
                score_name=SCORE_NAME,
                seed=seed,
                create_visualization=CREATE_VIS,
                create_gif=CREATE_GIF,
                events=[budget_event],
            )
            
            # Run LLM Agent (with event)
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
            
            budget_event = create_budget_event()  # Recreate event
            X_agent, Y_agent, logger_agent = run_bo_experiment(
                experiment_name="Agent",
                strategy=strategy_agent,
                output_dir=exp_group_dir,
                model_path=MODEL_PATH,
                x_scaler_path=X_SCALER_PATH,
                y_scaler_path=Y_SCALER_PATH,
                postprocess_fn=postprocess_outputs,
                score_fn=score_fn,
                design_space=design_space,
                bounds=bounds_t,
                X0_init=X0_shared,
                Y0_init_raw=Y0_shared_raw,
                Y0_init_normalized=Y0_shared_normalized,
                normalization_params=normalization_params,
                exploration_beta=beta_explore,
                n_iterations=ITERS,
                mc_samples=MC_SAMPLES,
                total_batch_size=TOTAL_BATCH_SIZE,
                pool_subsample=POOL_SUBSAMPLE,
                total_budget=TOTAL_BUDGET,
                total_time=TOTAL_TIME,
                cost_per_point=COST_PER_POINT,
                time_per_iteration=TIME_PER_ITERATION,
                obj1_name=OBJ1_NAME,
                obj2_name=OBJ2_NAME,
                score_name=SCORE_NAME,
                seed=seed,
                create_visualization=CREATE_VIS,
                create_gif=CREATE_GIF,
                events=[budget_event],
            )
            
            # Generate comparison plots (with event marker)
            print(f"\n[Plotting] Generating comparison plots...")
            plot_strategy_decisions(
                logger_agent, seed, beta_explore,
                event_iteration=BUDGET_EVENT_ITERATION,
                save_dir=exp_group_dir
            )
            plot_acquisition_metrics_comparison(
                [logger_exploit, logger_explore, logger_agent],
                seed, beta_explore,
                event_iteration=BUDGET_EVENT_ITERATION,
                save_dir=exp_group_dir
            )
            plot_convergence_comparison(
                [logger_exploit, logger_explore, logger_agent],
                seed, beta_explore, SCORE_NAME,
                event_iteration=BUDGET_EVENT_ITERATION,
                save_dir=exp_group_dir
            )
            
            del strategy_agent
            cleanup_memory()
            
            print(f"\n✓ Completed seed{seed}_beta{beta_explore}_event")
    
    print(f"\n{'='*80}")
    print("ALL BATCH EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total experiments run: {NUM_SEEDS * len(BETA_EXPLORE_VALUES) * 3}")
    print(f"All experiments included budget event at iteration {BUDGET_EVENT_ITERATION}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")