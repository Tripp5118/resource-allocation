# MP-D_DUAL.py
"""
Runner for Mp / D optimization comparing three strategies:
- 3-Step Agent
- qEHVI (Pure Exploitation)
- qUCB (Pure Exploration)

25 seeds, 20 iterations each.

DUAL Events
TODO: NEED TO SEE WHEN TO INITIATE EVENTS SO WE ARE CONSTRAINTED WHEN OPTIMIZING TO GLOBAL MAX
"""

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
from core.llm_decision_maker import MultiStageLLMDecisionMaker
from core.visualization import (
    plot_strategy_decisions,
    plot_convergence_comparison,
)

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Batch experiment name
BATCH_EXPERIMENT_NAME = "MP-D_DUAL"

# Data paths
MODEL_PATH = "ground_truth_models/TiVNbMoHfTaW-MeltingVsDensity/models/RFR_best_model.pkl"
X_SCALER_PATH = "ground_truth_models/TiVNbMoHfTaW-MeltingVsDensity/models/x_scaler.pkl"
Y_SCALER_PATH = "ground_truth_models/TiVNbMoHfTaW-MeltingVsDensity/models/y_scaler.pkl"

# Design space (7D simplex with per-element bounds)
COMPONENTS = [
    ("Ti", 0.0,   0.35),
    ("V",  0.0,   0.50),
    ("Nb", 0.225, 0.675),
    ("Mo", 0.0,   0.125),
    ("Hf", 0.0,   0.125),
    ("Ta", 0.0,   0.45),
    ("W",  0.0,   0.125),
]
STEP = 0.05
USE_DISCRETE = True

# BO parameters
INIT_N = 5
ITERS = 55
MC_SAMPLES = 256
POOL_SUBSAMPLE = 5000
TOTAL_BATCH_SIZE = 5

# Resource parameters
TOTAL_BUDGET = 28000.0
TOTAL_TIME = 56.0
COST_PER_POINT = 100.0
TIME_PER_ITERATION = 1.0

# Agent parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.2

# Objective names

OBJ1_NAME = "Melting Point (K)"
OBJ2_NAME = "Density (g/cm3)"
OBJ1_DISPLAY = "Melting Point (K)"
OBJ2_DISPLAY = "Density (g/cm3)"


SCORE_NAME = "Melting Point / Density (K cm3 g-1)"
PROBLEM_DESCRIPTION = """
Refractory High Entropy Alloy (RHEA) optimization for high melting point and low density.

Objectives:
- Minimize Density
- Maximize Melting Point

Goal: Discover a large pareto front of optimal alloys as quickly as possible.

Input space: 7D compositions, [Ti, V, Nb, Mo, Hf, Ta, W]

You should balance exploration and optimization given the qEHVI and Mutual Information
acquisition values for each option to best reach the goal within budget and time constraints.
"""

# Visualization
CREATE_VIS = True
CREATE_GIF = True

# Output
SETUP_SEED = 42
OUTPUT_BASE_DIR = "./results"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, BATCH_EXPERIMENT_NAME)

# Experiment configuration
NUM_SEEDS = 25
BETA_EXPLORE = 2.0  # Fixed beta value

# Event configuration
TIME_EVENT_ITERATION = 5
TIME_AFTER_EVENT = 15.0  # Reduced to 15 weeks total

BUDGET_EVENT_ITERATION = 7
BUDGET_REMAINING_ITERS = 3  # 3 more iterations after event (iteration 10 is last)

USE_FIXED_REFERENCE_POINT = True
REF_POINT_MARGIN = 0.1

# Strategy configurations
STRATEGY_CONFIGS = [
    {"type": "agent", "name": "Agent_MultiStage"},
    {"type": "qEHVI", "name": "qEHVI"},
    {"type": "qUCB",  "name": "qUCB"},
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def postprocess_outputs(preds_raw: np.ndarray) -> np.ndarray:
    """Postprocess model outputs: negate CTE, keep K as is."""
    mp = preds_raw[:, 0]
    d = -1 * np.abs(preds_raw[:, 1])
    return np.column_stack([mp, d])


def score_fn(Y: np.ndarray) -> np.ndarray:
    """Compute K / |CTE| score."""
    mp = Y[:, 0]
    d = Y[:, 1]
    return mp / np.abs(d + 1e-12) # Abs so score is printed positive

def create_events() -> list:
    """Create both time and budget events."""
    events = []
    
    # Time event at iteration 5: reduce to 15 weeks total
    events.append(
        ResourceEvent(
            iteration=TIME_EVENT_ITERATION,
            event_type="time_change",
            description=f"Time reduced at iteration {TIME_EVENT_ITERATION} to {TIME_AFTER_EVENT} weeks total "
                        f"(reduced from {TOTAL_TIME} weeks)",
            modifier=lambda current_time, t=TIME_AFTER_EVENT: t
        )
    )
    
    # Budget event at iteration 7: only 3 more iterations possible
    budget_for_remaining = BUDGET_REMAINING_ITERS * TOTAL_BATCH_SIZE * COST_PER_POINT
    events.append(
        ResourceEvent(
            iteration=BUDGET_EVENT_ITERATION,
            event_type="budget_change",
            description=f"Budget reduced at iteration {BUDGET_EVENT_ITERATION} to allow only "
                        f"{BUDGET_REMAINING_ITERS} more iterations "
                        f"({BUDGET_REMAINING_ITERS} × {TOTAL_BATCH_SIZE} × ${COST_PER_POINT:.2f} = ${budget_for_remaining:.2f})",
            modifier=lambda current_budget, b=budget_for_remaining: b
        )
    )
    
    return events

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
    
    # Compute fixed reference point by evaluating the entire design space once.
    # Because the design space is discrete and small this is cheap, and the result
    # is seed-independent so hypervolume is comparable across all runs.
    if USE_FIXED_REFERENCE_POINT:
        print("\n[RefPoint] Evaluating full design space to compute fixed reference point...")
        from core.truth_interface import TruthModelEvaluator
        _ref_evaluator = TruthModelEvaluator(
            model_path=MODEL_PATH,
            x_scaler_path=X_SCALER_PATH,
            y_scaler_path=Y_SCALER_PATH,
            postprocess_outputs=postprocess_outputs,
            normalize_outputs=False,
        )
        X_full = design_space.space  # Every valid composition
        Y_full = _ref_evaluator.evaluate(X_full).y
        _ref_evaluator.cleanup()
        del _ref_evaluator
        fixed_reference_point = np.min(Y_full, axis=0) - REF_POINT_MARGIN
        print(f"[RefPoint] Design space size: {len(X_full)} points")
        print(f"[RefPoint] Objective minima: {np.min(Y_full, axis=0)}")
        print(f"[RefPoint] Fixed reference point: {fixed_reference_point}")
    else:
        fixed_reference_point = None
        print("\n[RefPoint] Using dynamic reference point (min(Y) - 0.1 per iteration)")

    print(f"\n{'='*80}")
    print("SHARED SETUP COMPLETE")
    print(f"{'='*80}\n")
    
    # Generate random seeds
    np.random.seed(int(time.time()))
    seeds = np.random.randint(1, 100000, size=NUM_SEEDS).tolist()
    
    print(f"\n{'='*80}")
    print("K/CTE EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Batch name: {BATCH_EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seeds: {NUM_SEEDS} seeds")
    print(f"Fixed exploration beta: {BETA_EXPLORE}")
    print(f"Reference point: {'fixed (full design space min - ' + str(REF_POINT_MARGIN) + ')' if USE_FIXED_REFERENCE_POINT else 'dynamic (min(Y) - 0.1)'}")
    print(f"Iterations per run: {ITERS}")
    print(f"Total experiments: {NUM_SEEDS * len(STRATEGY_CONFIGS)} = {NUM_SEEDS} seeds × {len(STRATEGY_CONFIGS)} strategies")
    print(f"Strategies: {[c['name'] for c in STRATEGY_CONFIGS]}")
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
        
        exp_group_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(exp_group_dir, exist_ok=True)
        
        # Run each strategy
        for strategy_config in STRATEGY_CONFIGS:
            strategy_name = strategy_config["name"]
            strategy_type = strategy_config["type"]
            
            print(f"\n[Strategy] Running {strategy_name}...")
            
            if strategy_type == "agent":
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=AGENT_MODEL,
                    temperature=AGENT_TEMPERATURE,
                    api_key=OPENAI_API_KEY,
                )

                agent_log_dir = os.path.join(exp_group_dir, strategy_name, "agent_logs")
                os.makedirs(agent_log_dir, exist_ok=True)

                decision_maker = MultiStageLLMDecisionMaker(
                    llm=llm,
                    problem_description=PROBLEM_DESCRIPTION,
                    obj1_name=OBJ1_NAME,
                    obj2_name=OBJ2_NAME,
                    log_dir=agent_log_dir,
                    stage1_temperature=0.2,
                    stage2_temperature=0.2,
                    stage3_temperature=0.4,
                )

                strategy = BOAgent(
                    decision_maker=decision_maker,
                    log_dir=agent_log_dir,
                    problem_description=PROBLEM_DESCRIPTION,
                    obj1_name=OBJ1_NAME,
                    obj2_name=OBJ2_NAME,
                )
                
            elif strategy_type == "qEHVI":
                strategy = PureExploitation()

            elif strategy_type == "qUCB":
                strategy = PureExploration()
            
            # Run experiment
            X_result, Y_result, logger = run_bo_experiment(
                experiment_name=strategy_name,
                strategy=strategy,
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
                exploration_beta=BETA_EXPLORE,
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
                obj1_display=OBJ1_DISPLAY,
                obj2_display=OBJ2_DISPLAY,
                score_name=SCORE_NAME,
                seed=seed,
                use_discrete=USE_DISCRETE,
                create_visualization=CREATE_VIS,
                create_gif=CREATE_GIF,
                events=create_events(),
                pass_uncertainty_to_agent=False,
                fixed_reference_point=fixed_reference_point
            )
            
            # Generate decision plot for agent
            if strategy_type == "agent":
                print(f"\n[Plotting] Generating decision plot for {strategy_name}...")
                plot_strategy_decisions(logger, seed, BETA_EXPLORE, save_dir=os.path.join(exp_group_dir, strategy_name))
            
            # Cleanup
            if strategy_type == "agent":
                del strategy
            cleanup_memory()
            
            print(f"✓ Completed {strategy_name}")
        
        print(f"\n✓ Completed seed_{seed} ({seed_idx+1}/{NUM_SEEDS})")
    
    print(f"\n{'='*80}")
    print("K/CTE EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Total experiments run: {NUM_SEEDS * len(STRATEGY_CONFIGS)}")
    print(f"Beta value: {BETA_EXPLORE}")
    print(f"Reference point: {'fixed' if USE_FIXED_REFERENCE_POINT else 'dynamic'}")
    print(f"Iterations: {ITERS}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nDirectory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── seed_<seed1>/")
    for cfg in STRATEGY_CONFIGS:
        print(f"    │   ├── {cfg['name']}/")
    print(f"    ├── seed_<seed2>/")
    print(f"    └── ...")
    print(f"\nStrategies tested:")
    for i, cfg in enumerate(STRATEGY_CONFIGS, 1):
        print(f"  {i}. {cfg['name']} (type={cfg['type']})")
    print(f"{'='*80}\n")