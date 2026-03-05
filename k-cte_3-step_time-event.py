# k-cte_3-step_time_event.py
"""
Runner for K/CTE optimization comparing three strategies with a time resource event:
- Agent_MultiStage
- qEHVI (Pure Exploitation)
- qUCB (Pure Exploration)

25 seeds, 8 iterations each, with time event at iteration 4.
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
BATCH_EXPERIMENT_NAME = "k-cte_3-step_time-event"

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
USE_DISCRETE = True

# BO parameters
INIT_N = 5
ITERS = 9  # End after iteration 8 (no iteration 9)
MC_SAMPLES = 256
POOL_SUBSAMPLE = 5000
TOTAL_BATCH_SIZE = 5

# Resource parameters
TOTAL_BUDGET = 10500.0
TOTAL_TIME = 21.0
COST_PER_POINT = 100.0
TIME_PER_ITERATION = 1.0

# Event configuration
TIME_EVENT_ITERATION = 4  # Event happens at iteration 4
TIME_AFTER_EVENT = 4.0    # Time remaining after event (allows iterations 5-8)

# Agent parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.2

# Objective names
OBJ1_NAME = "CTE"
OBJ2_NAME = "K"
OBJ1_DISPLAY = "-|CTE|"
OBJ2_DISPLAY = "K"
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
OUTPUT_BASE_DIR = "./test"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, BATCH_EXPERIMENT_NAME)

# Experiment configuration
NUM_SEEDS = 25  # 25 seeds as requested
BETA_EXPLORE = 2.0  # Fixed beta value

# Reference point configuration
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
    cte = -1.0 * np.abs(preds_raw[:, 0])
    k = preds_raw[:, 1]
    return np.column_stack([cte, k])


def score_fn(Y: np.ndarray) -> np.ndarray:
    """Compute K / |CTE| score."""
    cte = Y[:, 0]
    k = Y[:, 1]
    return k / (np.abs(cte) + 1e-12)


def create_time_event() -> ResourceEvent:
    """Create a time reduction event at iteration 4."""
    return ResourceEvent(
        iteration=TIME_EVENT_ITERATION,
        event_type="time_change",
        description=f"Time reduced at iteration {TIME_EVENT_ITERATION} to {TIME_AFTER_EVENT} weeks total "
                    f"(reduced from {TOTAL_TIME} weeks)",
        modifier=lambda current_time, t=TIME_AFTER_EVENT: t
    )


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
    
    # Compute fixed reference point
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
        X_full = design_space.space
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
    print("K/CTE EXPERIMENT WITH TIME EVENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Batch name: {BATCH_EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seeds: {NUM_SEEDS} seeds")
    print(f"Fixed exploration beta: {BETA_EXPLORE}")
    print(f"Reference point: {'fixed (full design space min - ' + str(REF_POINT_MARGIN) + ')' if USE_FIXED_REFERENCE_POINT else 'dynamic (min(Y) - 0.1)'}")
    print(f"Iterations per run: {ITERS}")
    print(f"Time event: At iteration {TIME_EVENT_ITERATION}, time reduced to {TIME_AFTER_EVENT} weeks")
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
            
            # Run experiment with time event
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
                events=[create_time_event()],  # Time event added here
                pass_uncertainty_to_agent=False,
                fixed_reference_point=fixed_reference_point,
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
    print("K/CTE EXPERIMENT WITH TIME EVENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Total experiments run: {NUM_SEEDS * len(STRATEGY_CONFIGS)}")
    print(f"Beta value: {BETA_EXPLORE}")
    print(f"Reference point: {'fixed' if USE_FIXED_REFERENCE_POINT else 'dynamic'}")
    print(f"Iterations: {ITERS}")
    print(f"Time event: Iteration {TIME_EVENT_ITERATION} → {TIME_AFTER_EVENT} weeks")
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