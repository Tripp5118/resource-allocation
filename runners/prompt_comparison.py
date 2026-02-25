# runners/prompt_comparison/run_prompt_comparison.py
"""Runner for comparing three prompt styles: minimal, default, and simple_guideline."""

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
from core.agent_manager import BOAgent
from core.fixed_policy import PureExploitation, PureExploration
from core.prompt_builder import create_prompt_builder
from core.visualization import (
    plot_strategy_decisions,
    plot_convergence_comparison,
)

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Batch experiment name
BATCH_EXPERIMENT_NAME = "prompt_comparison_3styles"

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
ITERS = 20
MC_SAMPLES = 256
POOL_SUBSAMPLE = 5000
TOTAL_BATCH_SIZE = 5

# Resource parameters
TOTAL_BUDGET = 10000.0
TOTAL_TIME = 20.0
COST_PER_POINT = 100.0
TIME_PER_ITERATION = 1.0

# Agent parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.7
ITER_HISTORY = 3  # Only 3 previous iterations

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
OUTPUT_BASE_DIR = "./results"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, BATCH_EXPERIMENT_NAME)

# Experiment configuration
NUM_SEEDS = 25
BETA_EXPLORE = 2.0  # Fixed beta value

# Prompt configurations to test (2 runs per style: with and without uncertainty/HV)
PROMPT_CONFIGS = [
    {"style": "minimal",          "name": "Agent_Minimal_NoUncertainty",  "pass_uncertainty": False},
    {"style": "minimal",          "name": "Agent_Minimal_WithUncertainty","pass_uncertainty": True},
    {"style": "default",          "name": "Agent_Default_NoUncertainty",  "pass_uncertainty": False},
    {"style": "default",          "name": "Agent_Default_WithUncertainty","pass_uncertainty": True},
    {"style": "simple_guideline", "name": "Agent_SimpleGuideline_NoUncertainty",  "pass_uncertainty": False},
    {"style": "simple_guideline", "name": "Agent_SimpleGuideline_WithUncertainty","pass_uncertainty": True},
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
    print("PROMPT COMPARISON EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Batch name: {BATCH_EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seeds: {NUM_SEEDS} seeds")
    print(f"Fixed exploration beta: {BETA_EXPLORE}")
    print(f"Iterations per run: {ITERS}")
    print(f"Context window: {ITER_HISTORY} previous iterations")
    print(f"Total experiments: {NUM_SEEDS * len(PROMPT_CONFIGS)} = {NUM_SEEDS} seeds × {len(PROMPT_CONFIGS)} configs")
    print(f"Prompt styles: {[c['name'] for c in PROMPT_CONFIGS]}")
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
        
        exp_group_dir = os.path.join(OUTPUT_DIR, f"seed{seed}")
        os.makedirs(exp_group_dir, exist_ok=True)
        
        # Run each prompt configuration
        for prompt_config in PROMPT_CONFIGS:
            prompt_name = prompt_config["name"]
            prompt_style = prompt_config["style"]
            
            print(f"\n[Prompt] Running {prompt_name} (style: {prompt_style})...")
            
            # Create prompt builder
            prompt_builder = create_prompt_builder(
                style=prompt_style,
                include_uncertainty=prompt_config["pass_uncertainty"],
                include_hypervolume=prompt_config["pass_uncertainty"],
            )
            
            # Create agent log directory
            agent_log_dir = os.path.join(exp_group_dir, prompt_name, "agent_logs")
            os.makedirs(agent_log_dir, exist_ok=True)
            
            # Create agent with custom prompt builder
            strategy_agent = BOAgent(
                model=AGENT_MODEL,
                temperature=AGENT_TEMPERATURE,
                api_key=OPENAI_API_KEY,
                log_dir=agent_log_dir,
                problem_description=PROBLEM_DESCRIPTION,
                obj1_name=OBJ1_NAME,
                obj2_name=OBJ2_NAME,
                iter_history=ITER_HISTORY,
                prompt_builder=prompt_builder,
            )
            
            # Run experiment
            X_agent, Y_agent, logger_agent = run_bo_experiment(
                experiment_name=prompt_name,
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
                events=None,  # No events
                pass_uncertainty_to_agent=prompt_config["pass_uncertainty"],
            )
            
            # Generate decision plot
            print(f"\n[Plotting] Generating decision plot for {prompt_name}...")
            plot_strategy_decisions(logger_agent, seed, BETA_EXPLORE, save_dir=os.path.join(exp_group_dir, prompt_name))
            
            del strategy_agent
            cleanup_memory()
            
            print(f"✓ Completed {prompt_name}")
        
        print(f"\n✓ Completed seed{seed} ({seed_idx+1}/{NUM_SEEDS})")
    
    print(f"\n{'='*80}")
    print("PROMPT COMPARISON EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Total experiments run: {NUM_SEEDS * len(PROMPT_CONFIGS)}")
    print(f"Beta value: {BETA_EXPLORE}")
    print(f"Iterations: {ITERS}")
    print(f"Context window: {ITER_HISTORY} iterations")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nDirectory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── seed<seed1>/")
    for cfg in PROMPT_CONFIGS:
        print(f"    │   ├── {cfg['name']}/")
    print(f"    ├── seed<seed2>/")
    print(f"    └── ...")
    print(f"\nConfigurations tested:")
    for i, cfg in enumerate(PROMPT_CONFIGS, 1):
        unc = "with uncertainty+HV" if cfg["pass_uncertainty"] else "no uncertainty/HV"
        print(f"  {i}. {cfg['name']} (style={cfg['style']}, {unc})")
    print(f"{'='*80}\n")