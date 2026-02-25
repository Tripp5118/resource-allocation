# k-H_simple_no_unc.py
"""
Runner for Ti-V-Nb-Mo-Hf-Ta-W refractory alloy optimization comparing three strategies:
- Agent_SimpleGuideline_NoUncertainty
- qEHVI (Pure Exploitation)
- qUCB (Pure Exploration)

Objectives: Maximize Configurational Entropy and Thermal Conductivity
50 seeds, 20 iterations each.
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
BATCH_EXPERIMENT_NAME = "k-H_simple_no_unc"

# Data paths
MODEL_PATH = "ground_truth_models/TiVNbMoHfTaW_Max_S_Max_TCond/models/RFR_best_model.pkl"
X_SCALER_PATH = "ground_truth_models/TiVNbMoHfTaW_Max_S_Max_TCond/models/x_scaler.pkl"
Y_SCALER_PATH = "ground_truth_models/TiVNbMoHfTaW_Max_S_Max_TCond/models/y_scaler.pkl"

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

# BO parameters (matching prompt_comparison.py)
INIT_N = 5
ITERS = 20
MC_SAMPLES = 256
POOL_SUBSAMPLE = 5000
TOTAL_BATCH_SIZE = 5

# Resource parameters
TOTAL_BUDGET = 10500.0
TOTAL_TIME = 20.0
COST_PER_POINT = 100.0
TIME_PER_ITERATION = 1.0

# Agent parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.7
ITER_HISTORY = 3

# Objective names
# Note: RF outputs are [ThermCond, SConf], but we reorder in postprocess to [SConf, ThermCond]
OBJ1_NAME = "SConf"
OBJ2_NAME = "ThermCond"
OBJ1_DISPLAY = "Configurational Entropy"
OBJ2_DISPLAY = "Thermal Conductivity"

# Fixed global min/max from original RF training data (for score normalization)
SCONF_MIN = 1.104704674
SCONF_MAX = 1.592615027
TC_MIN = 7.279877639
TC_MAX = 36.56773905

SCORE_NAME = "ThermCond×Entropy (global-normalized)"

PROBLEM_DESCRIPTION = """
Ti-V-Nb-Mo-Hf-Ta-W refractory alloy optimization.

Objectives:
- Maximize configurational entropy.
- Maximize room-temperature thermal conductivity (W/(mK)).

Input space: 7D compositions (Ti, V, Nb, Mo, Hf, Ta, W) summing to 1.0,
with element-specific bounds and grid step 0.05.

Take into account the options for the next candidate batch along with the optimization progress. 
You should balance exploration and optimization given the qEHVI and Entropy for each option 
to best reach your goal within the time limit and budget constraints. 
Think critically about what action should be taken given each constraint.
"""

# Visualization
CREATE_VIS = True
CREATE_GIF = True

# Output
SETUP_SEED = 42
OUTPUT_BASE_DIR = "./results"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, BATCH_EXPERIMENT_NAME)

# Experiment configuration
NUM_SEEDS = 50
BETA_EXPLORE = 2.0  # Fixed beta value

# Strategy configurations
STRATEGY_CONFIGS = [
    {"type": "agent", "name": "Agent_SimpleGuideline_NoUncertainty", "style": "simple_guideline", "pass_uncertainty": False},
    {"type": "qEHVI", "name": "qEHVI"},
    {"type": "qUCB",  "name": "qUCB"},
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def postprocess_outputs(preds_raw: np.ndarray) -> np.ndarray:
    """
    Postprocess model outputs.
    RF outputs: [ThermCond, SConf]
    We reorder to: [SConf, ThermCond] to match OBJ1=SConf, OBJ2=ThermCond
    """
    therm_cond = preds_raw[:, 0]
    sconf = preds_raw[:, 1]
    return np.column_stack([sconf, therm_cond])


def score_fn(Y: np.ndarray) -> np.ndarray:
    """
    Compute ThermCond × Entropy score with global normalization.
    Y[:, 0] = SConf (configurational entropy)
    Y[:, 1] = ThermCond (thermal conductivity)
    """
    sconf = Y[:, 0]
    tcond = Y[:, 1]
    
    sconf_norm = (sconf - SCONF_MIN) / (SCONF_MAX - SCONF_MIN + 1e-12)
    tcond_norm = (tcond - TC_MIN) / (TC_MAX - TC_MIN + 1e-12)
    
    return sconf_norm * tcond_norm


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
    print("K-H (ThermCond × Entropy) EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Batch name: {BATCH_EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seeds: {NUM_SEEDS} seeds")
    print(f"Fixed exploration beta: {BETA_EXPLORE}")
    print(f"Iterations per run: {ITERS}")
    print(f"Context window: {ITER_HISTORY} previous iterations")
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
                # Create prompt builder for agent
                prompt_builder = create_prompt_builder(
                    style=strategy_config["style"],
                    include_uncertainty=strategy_config["pass_uncertainty"],
                    include_hypervolume=strategy_config["pass_uncertainty"],
                )
                
                # Create agent log directory
                agent_log_dir = os.path.join(exp_group_dir, strategy_name, "agent_logs")
                os.makedirs(agent_log_dir, exist_ok=True)
                
                # Create agent with custom prompt builder
                strategy = BOAgent(
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
                pass_uncertainty = strategy_config["pass_uncertainty"]
                
            elif strategy_type == "qEHVI":
                strategy = PureExploitation()
                pass_uncertainty = False
                
            elif strategy_type == "qUCB":
                strategy = PureExploration()
                pass_uncertainty = False
            
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
                events=None,
                pass_uncertainty_to_agent=pass_uncertainty,
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
    print("K-H (ThermCond × Entropy) EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Total experiments run: {NUM_SEEDS * len(STRATEGY_CONFIGS)}")
    print(f"Beta value: {BETA_EXPLORE}")
    print(f"Iterations: {ITERS}")
    print(f"Context window: {ITER_HISTORY} iterations")
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