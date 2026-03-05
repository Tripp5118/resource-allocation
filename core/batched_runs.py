# core/batched_runs.py
"""Core library for running batched BO experiments with multiple strategies."""

import os
import gc
import time
import numpy as np
import torch
from typing import Tuple, Optional, Callable, Dict, Any, List

from core.design_space import DesignSpace
from core.truth_interface import TruthModelEvaluator
from core.gp_models import GPModelManager
from core.acq_ucb import AcquisitionFunctionManager
from core.agent_manager import BOAgent, ResourceEvent
from core.fixed_policy import PureExploitation, PureExploration
from core.logging_utils import LoggingManager
from core.visualization import VisualizationManager

# Import for hypervolume calculation
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


# ============================================================================
# RESOURCE EVENT MANAGER (for non-agent strategies)
# ============================================================================

class SimpleResourceEventManager:
    """Manages resource events for non-agent strategies"""
    
    def __init__(self):
        self.events: List[ResourceEvent] = []
    
    def add_event(self, event: ResourceEvent):
        """Add a resource event to be triggered at a specific iteration."""
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
        
        Returns:
            (new_budget, new_time, new_cost, event_messages)
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
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_memory():
    """Clean up memory and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_hypervolume(
    Y_objectives: np.ndarray,
    ref_point: np.ndarray
) -> float:
    """
    Compute hypervolume of Pareto front.
    
    Args:
        Y_objectives: (n, 2) array of objective values (both maximized)
        ref_point: (2,) reference point for hypervolume
    
    Returns:
        Hypervolume value
    """
    if Y_objectives.shape[0] == 0:
        return 0.0
    
    Y_tensor = torch.tensor(Y_objectives, dtype=DTYPE, device=DEVICE)
    ref_tensor = torch.tensor(ref_point, dtype=DTYPE, device=DEVICE)
    
    # Get Pareto front
    pareto_mask = is_non_dominated(Y_tensor)
    pareto_Y = Y_tensor[pareto_mask]
    
    if len(pareto_Y) == 0:
        return 0.0
    
    # Compute hypervolume
    hv = Hypervolume(ref_point=ref_tensor)
    volume = hv.compute(pareto_Y)
    
    return volume


def compute_normalization_params(
    design_space: DesignSpace,
    model_path: str,
    x_scaler_path: str,
    y_scaler_path: str,
    postprocess_fn: Callable,
    n_samples: int = 100,
    obj1_name: str = "Obj1",
    obj2_name: str = "Obj2"
) -> Dict[str, np.ndarray]:
    """
    Compute output normalization parameters from design space samples.
    
    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    print("\nComputing output normalization parameters...")
    sample_X = design_space.sample(n=n_samples, method='sobol')
    
    temp_evaluator = TruthModelEvaluator(
        model_path=model_path,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        postprocess_outputs=postprocess_fn,
        normalize_outputs=False,
    )
    
    sample_result = temp_evaluator.evaluate(sample_X)
    sample_Y = sample_result.y
    
    y_mean = np.mean(sample_Y, axis=0)
    y_std = np.std(sample_Y, axis=0)
    
    print(f"\nNormalization parameters:")
    print(f"  {obj1_name}: mean={y_mean[0]:.4f}, std={y_std[0]:.4f}")
    print(f"  {obj2_name}: mean={y_mean[1]:.4f}, std={y_std[1]:.4f}")
    
    temp_evaluator.cleanup()
    del temp_evaluator
    cleanup_memory()
    
    return {'mean': y_mean, 'std': y_std}


def generate_shared_initialization(
    design_space: DesignSpace,
    model_path: str,
    x_scaler_path: str,
    y_scaler_path: str,
    postprocess_fn: Callable,
    normalization_params: Dict[str, np.ndarray],
    n_init: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate shared initialization data for all strategies in a seed.
    
    Returns:
        (X0_shared, Y0_shared_raw, Y0_shared_normalized)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X0_shared = design_space.sample(n_init, method="random")
    
    # Raw evaluator
    temp_evaluator_raw = TruthModelEvaluator(
        model_path=model_path,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        postprocess_outputs=postprocess_fn,
        normalize_outputs=False,
    )
    
    # Normalized evaluator
    temp_evaluator_norm = TruthModelEvaluator(
        model_path=model_path,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        postprocess_outputs=postprocess_fn,
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
    
    return X0_shared, Y0_shared_raw, Y0_shared_normalized


# ============================================================================
# MAIN BO EXPERIMENT RUNNER
# ============================================================================

def run_bo_experiment(
    # Experiment identification
    experiment_name: str,
    strategy: Any,  # BOAgent, PureExploitation, or PureExploration
    output_dir: str,
    
    # Data paths and functions
    model_path: str,
    x_scaler_path: str,
    y_scaler_path: str,
    postprocess_fn: Callable,
    score_fn: Callable,
    
    # Design space and bounds
    design_space: DesignSpace,
    bounds: torch.Tensor,
    
    # Initialization data
    X0_init: np.ndarray,
    Y0_init_raw: np.ndarray,
    Y0_init_normalized: np.ndarray,
    normalization_params: Dict[str, np.ndarray],
    
    # Acquisition function parameters
    exploration_beta: float,
    
    # Experiment parameters
    n_iterations: int,
    mc_samples: int,
    total_batch_size: int,
    pool_subsample: Optional[int],
    
    # Resource parameters
    total_budget: float,
    total_time: float,
    cost_per_point: float,
    time_per_iteration: float,
    
    # Objective names
    obj1_name: str,
    obj2_name: str,
    obj1_display: str,
    obj2_display: str,
    score_name: str,
    
    # Optional parameters
    use_discrete: bool = True,
    seed: int = 42,
    create_visualization: bool = True,
    create_gif: bool = True,
    events: Optional[List[ResourceEvent]] = None,
    pass_uncertainty_to_agent: bool = False,
    fixed_reference_point: Optional[np.ndarray] = None,
    
) -> Tuple[np.ndarray, np.ndarray, LoggingManager]:
    """
    Run a single BO experiment with a given strategy.
    
    Args:
        experiment_name: Name of the experiment (e.g., "PureExploit")
        strategy: Strategy object (BOAgent, PureExploitation, PureExploration)
        output_dir: Directory for outputs
        model_path: Path to ground truth model
        x_scaler_path: Path to X scaler
        y_scaler_path: Path to Y scaler
        postprocess_fn: Function to postprocess outputs
        score_fn: Function to compute scores from objectives
        design_space: DesignSpace object
        bounds: Tensor of bounds (2, d)
        X0_init: Initial X data
        Y0_init_raw: Initial Y data (raw)
        Y0_init_normalized: Initial Y data (normalized)
        normalization_params: Dict with 'mean' and 'std'
        exploration_beta: Beta for exploration (qUCB)
        n_iterations: Number of BO iterations
        mc_samples: MC samples for acquisition
        total_batch_size: Batch size per iteration
        pool_subsample: Size of pool subsample (None = use all)
        total_budget: Total budget
        total_time: Total time
        cost_per_point: Cost per evaluation
        time_per_iteration: Time per iteration
        obj1_name: Name of first objective
        obj2_name: Name of second objective
        score_name: Name of score metric
        seed: Random seed
        create_visualization: Whether to create iteration plots
        create_gif: Whether to create GIF
        events: List of ResourceEvent objects to trigger
        fixed_reference_point: Optional fixed (2,) reference point for hypervolume. 
            If None, computed dynamically as min(Y) - 0.1 each iteration.
    
    Returns:
        (X_final, Y_final, logger)
    """
    print(f"\n{'='*80}")
    print(f"[Experiment] {experiment_name}")
    print(f"[Strategy] {type(strategy).__name__}")
    if events:
        print(f"[Events] {len(events)} resource events configured")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup evaluator
    evaluator = TruthModelEvaluator(
        model_path=model_path,
        x_scaler_path=x_scaler_path,
        y_scaler_path=y_scaler_path,
        postprocess_outputs=postprocess_fn,
        normalize_outputs=True,
        normalization_params=normalization_params,
    )
    
    # Setup GP and acquisition managers
    gp_manager = GPModelManager(bounds=bounds)
    acq_manager = AcquisitionFunctionManager(
        bounds=bounds,
        num_restarts=3,
        raw_samples=256,
        exploration_beta=exploration_beta,
        design_space=design_space,
        use_discrete=use_discrete
    )
    
    # Setup logging
    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = LoggingManager(exp_dir, experiment_name)
    
    # Setup visualization
    vis_manager = None
    if create_visualization:
        print(f"[Viz] Initializing visualization manager...")
        vis_manager = VisualizationManager(
            design_space=design_space,
            log_dir=exp_dir,
            experiment_name=experiment_name,
            normalization_params=normalization_params,
            obj1_name=obj1_name,
            obj2_name=obj2_name,
        )
    
    # Setup event manager
    event_manager = None
    if events:
        if isinstance(strategy, BOAgent):
            # BOAgent has built-in event handling
            event_manager = strategy
            for event in events:
                event_manager.add_event(event)
        else:
            # Use simple event manager for other strategies
            event_manager = SimpleResourceEventManager()
            for event in events:
                event_manager.add_event(event)
        print(f"[Events] Configured {len(events)} resource events")
    
    # Initialize data
    print(f"\n[Init] Using {len(X0_init)} initial samples")

    X_torch = torch.tensor(X0_init.copy(), dtype=DTYPE, device=DEVICE)
    Y_torch_normalized = torch.tensor(Y0_init_normalized.copy(), dtype=DTYPE, device=DEVICE)

    X_history_np = X0_init.copy()
    Y_history_raw_np = Y0_init_raw.copy()
    Y_history_normalized_np = Y0_init_normalized.copy()

    scores = score_fn(Y_history_raw_np)
    best_score = float(np.max(scores))
    print(f"[Init] Best {score_name}: {best_score:.4f}")

    # Reference point: use fixed if provided, otherwise compute dynamically
    if fixed_reference_point is not None:
        ref_point_raw = np.array(fixed_reference_point, dtype=float)
        print(f"[RefPoint] Using fixed reference point: {ref_point_raw}")
    else:
        ref_point_raw = Y_history_raw_np.min(axis=0) - 0.1 * np.ones(2)
        print(f"[RefPoint] Using dynamic reference point: {ref_point_raw}")

    # Deduct init cost up front
    budget_remaining = total_budget - (len(X0_init) * cost_per_point)
    time_remaining = total_time - time_per_iteration
    current_cost_per_point = cost_per_point

    # Log iteration 0: init points + post-deduction resources.
    # HV and uncertainty are NOT logged here — they will be logged at the start
    # of iteration 1 as the values the agent actually sees before its first decision.
    logger.log_iteration(
        iteration=0,
        X_history=X_history_np,
        Y_history=Y_history_raw_np,
        n_new_points=len(X0_init),
        strategy=strategy,
        timing=0.0,
        extra_info={
            "experiment": experiment_name,
            "shared_init": True,
            "has_events": events is not None,
            "budget_remaining": budget_remaining,
            "time_remaining": time_remaining,
        },
        acquisition_data=None,
        score_fn=score_fn,
        obj1_name=obj1_name,
        obj2_name=obj2_name,
        hypervolume=None,
        ref_point=ref_point_raw,
    )
    logger.log_evaluations(
        iteration=0,
        X_new=X0_init,
        Y_new=Y0_init_raw,
        score_fn=score_fn,
        obj1_name=obj1_name,
        obj2_name=obj2_name,
    )

    print(f"[Init] Budget after init: ${budget_remaining:.2f} | Time after init: {time_remaining:.1f}")

    # Main BO loop
    iteration = 0
    while iteration < n_iterations:
        iteration += 1

        # Check resources before starting this iteration
        if budget_remaining < current_cost_per_point * total_batch_size or time_remaining < time_per_iteration:
            print(f"\n[Stop] Insufficient resources at iteration {iteration}")
            print(f"       Budget: ${budget_remaining:.2f} (need ${current_cost_per_point * total_batch_size:.2f})")
            print(f"       Time: {time_remaining:.1f} (need {time_per_iteration:.1f})")
            break

        iter_start = time.perf_counter()

        # Deduct resources at the top of the iteration, before any work is done.
        # This means budget_remaining and time_remaining logged this iteration
        # reflect what's left AFTER committing to this batch.
        points_used = total_batch_size
        budget_spent = points_used * current_cost_per_point
        budget_remaining -= budget_spent
        time_remaining -= time_per_iteration

        # Process events for this iteration
        event_msgs = []
        if event_manager is not None:
            budget_remaining, time_remaining, current_cost_per_point, event_msgs = \
                event_manager.process_iteration_events(
                    iteration, budget_remaining, time_remaining, current_cost_per_point
                )
            if event_msgs:
                print("\n" + "="*80)
                for msg in event_msgs:
                    print(msg)
                print("="*80)

        print(f"\n{'='*80}")
        print(f"[Iter {iteration}/{n_iterations}] Starting...")
        print(f"[Resources] Budget remaining: ${budget_remaining:.2f} | Time remaining: {time_remaining:.1f}")
        print(f"{'='*80}")

        # Fit GP on all data collected so far (does not include this iteration's points yet)
        print("[GP] Fitting model on normalized outputs...")
        model = gp_manager.fit_model(X_torch, Y_torch_normalized)

        # Compute uncertainty and HV — this is exactly what the agent sees this iteration
        print("[GP] Computing total uncertainty across design space...")
        total_uncertainty_obj1, total_uncertainty_obj2 = compute_total_uncertainty(
            model=model,
            design_space=design_space,
        )
        print(f"[GP] Total uncertainty - {obj1_name}: {total_uncertainty_obj1:.4f}, {obj2_name}: {total_uncertainty_obj2:.4f}")

        current_hv = compute_hypervolume(Y_history_raw_np, ref_point_raw)
        print(f"[GP] Hypervolume: {current_hv:.4f}")

        # Compute Pareto front
        print("[Acq] Computing Pareto front on normalized outputs...")
        pareto_Y, ref_point = acq_manager.compute_pareto_front(Y_torch_normalized)

        # Build pareto_front_info for agent: captures state BEFORE this iteration's evaluation.
        # This is paired with record_iteration_outcome() (called after evaluation) so the agent
        # can reconstruct the before→after delta for the decision it is about to make.
        if isinstance(strategy, BOAgent):
            pareto_mask_current = is_non_dominated(
                torch.tensor(Y_history_raw_np, dtype=DTYPE, device=DEVICE)
            )
            current_pareto_count = int(pareto_mask_current.sum().item())
            last_pareto_count = getattr(strategy, "_last_pareto_count", current_pareto_count)
            points_added_since_last = current_pareto_count - last_pareto_count

            pareto_front_info = {
                "num_points": current_pareto_count,
                "hypervolume": current_hv,
                "points_added": points_added_since_last,
                "points_improved": 0,  # Not tracking individual improvements
            }
            # Persist for next iteration's delta calculation
            strategy._last_pareto_count = current_pareto_count
            strategy._last_hv = current_hv
        else:
            pareto_front_info = None

        # Pool for entropy ranking
        if pool_subsample is not None:
            X_pool_np = design_space.sample(pool_subsample, method="sobol")
        else:
            X_pool_np = design_space.space.copy()
        X_pool = torch.tensor(X_pool_np, dtype=DTYPE, device=DEVICE)

        # Compute allocation options
        strategy_name = type(strategy).__name__

        if strategy_name == "BOAgent":
            print("[Strategy] LLM Agent - Computing all 6 allocation options")
            allocation_results = acq_manager.compute_all_allocations(
                model=model,
                mc_samples=mc_samples,
                pareto_front=pareto_Y,
                reference_point=ref_point,
                total_batch_size=total_batch_size
            )
        elif "Exploit" in strategy_name or strategy_name == "PureExploitation":
            print("[Strategy] Pure Exploitation - Computing option 0 only")
            allocation_results = acq_manager.compute_single_allocation(
                model=model,
                num_exploitation=total_batch_size,
                num_exploration=0,
                mc_samples=mc_samples,
                pareto_front=pareto_Y,
                reference_point=ref_point
            )
        elif "Explor" in strategy_name or strategy_name == "PureExploration":
            print("[Strategy] Pure Exploration - Computing option 5 only")
            allocation_results = acq_manager.compute_single_allocation(
                model=model,
                num_exploitation=0,
                num_exploration=total_batch_size,
                mc_samples=mc_samples,
                pareto_front=pareto_Y,
                reference_point=ref_point
            )
        else:
            print(f"[Strategy] Unknown strategy {strategy_name} - Computing all options")
            allocation_results = acq_manager.compute_all_allocations(
                model=model,
                mc_samples=mc_samples,
                pareto_front=pareto_Y,
                reference_point=ref_point,
                total_batch_size=total_batch_size
            )

        # Strategy selection
        selected_idx = None

        if isinstance(strategy, BOAgent):
            selected_idx, reasoning, selected_point_arrays = strategy.select_resource_allocation(
                iteration=iteration,
                budget_remaining=budget_remaining,
                time_remaining=time_remaining,
                cost_per_point=current_cost_per_point,
                time_per_point=time_per_iteration,
                X_history=X_torch.cpu().numpy(),
                Y_history=Y_history_raw_np,
                allocation_results=allocation_results,
                max_batch_size=total_batch_size,
                score_fn=score_fn,
                pareto_front_info=pareto_front_info,
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
            }

        # Extract metrics from selected option
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

        # All values below are exactly what the agent saw when making its decision
        extra_info["total_uncertainty_obj1"] = total_uncertainty_obj1
        extra_info["total_uncertainty_obj2"] = total_uncertainty_obj2
        extra_info["hypervolume"] = current_hv
        extra_info["budget_remaining"] = budget_remaining
        extra_info["time_remaining"] = time_remaining

        if event_msgs:
            extra_info["events_triggered"] = "; ".join(event_msgs)

        # Evaluate new points
        result_new = evaluator.evaluate(X_new_np)
        Y_new_normalized = result_new.y
        Y_new_raw = evaluator.denormalize(Y_new_normalized)

        # Update history
        X_history_np = np.vstack([X_history_np, X_new_np])
        Y_history_normalized_np = np.vstack([Y_history_normalized_np, Y_new_normalized])
        Y_history_raw_np = np.vstack([Y_history_raw_np, Y_new_raw])
        X_torch = torch.tensor(X_history_np, dtype=DTYPE, device=DEVICE)
        Y_torch_normalized = torch.tensor(Y_history_normalized_np, dtype=DTYPE, device=DEVICE)

        # Complete the outcome record for this iteration so the agent can learn from it.
        # We compute the Pareto state AFTER evaluation and pass it alongside the new best score.
        if isinstance(strategy, BOAgent):
            scores_after = score_fn(Y_history_raw_np)
            best_score_after = float(np.max(scores_after))

            pareto_mask_after = is_non_dominated(
                torch.tensor(Y_history_raw_np, dtype=DTYPE, device=DEVICE)
            )
            pareto_count_after = int(pareto_mask_after.sum().item())
            hv_after = compute_hypervolume(Y_history_raw_np, ref_point_raw)

            pareto_after = {
                "num_points": pareto_count_after,
                "hypervolume": hv_after,
                "points_added": pareto_count_after - pareto_front_info["num_points"],
                "points_improved": 0,
            }
            strategy.record_iteration_outcome(best_score_after, pareto_after)

        iter_time = time.perf_counter() - iter_start

        # Log this iteration — hypervolume and uncertainty are exactly what the agent saw
        logger.log_iteration(
            iteration=iteration,
            X_history=X_history_np,
            Y_history=Y_history_raw_np,
            n_new_points=len(X_new_np),
            strategy=strategy,
            timing=iter_time,
            extra_info=extra_info,
            acquisition_data=allocation_results,
            score_fn=score_fn,
            obj1_name=obj1_name,
            obj2_name=obj2_name,
            hypervolume=current_hv,
            ref_point=ref_point_raw,
        )
        logger.log_evaluations(
            iteration=iteration,
            X_new=X_new_np,
            Y_new=Y_new_raw,
            score_fn=score_fn,
            obj1_name=obj1_name,
            obj2_name=obj2_name,
        )

        # Visualization
        if vis_manager is not None:
            vis_manager.create_iteration_plot(
                iteration=iteration,
                X_history=X_history_np,
                Y_history=Y_history_raw_np,
                X_new=X_new_np,
                model=model,
                bounds=bounds,
                selection_info=extra_info,
                score_fn=score_fn,
                score_name=score_name,
                obj1_name=obj1_name,
                obj2_name=obj2_name,
            )

        # Print iteration summary
        scores = score_fn(Y_history_raw_np)
        best_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
        print(f"[Iter {iteration}] Complete in {iter_time:.2f}s")
        print(f"  Best {score_name}: {best_score:.4f}")
        print(f"  Mean {score_name}: {mean_score:.4f}")
        print(f"  Hypervolume (agent saw): {current_hv:.4f}")
        print(f"  Points evaluated: {len(X_new_np)}")
        print(f"  Budget remaining: ${budget_remaining:.2f}")

        cleanup_memory()

    # Finalize logger
    logger.finalize()

    # Create final visualizations
    if vis_manager is not None:
        vis_manager.create_hypervolume_plot(logger, score_name)
        vis_manager.create_pareto_front_plot(
            Y_history=Y_history_raw_np,
            obj1_name=obj1_name,
            obj2_name=obj2_name,
            obj1_display=obj1_display,
            obj2_display=obj2_display
        )
        if create_gif:
            vis_manager.create_gif(duration=2.0, gif_name=f"{experiment_name}.gif")

    if isinstance(strategy, BOAgent):
        strategy.save_global_memory()

    evaluator.cleanup()
    del evaluator
    cleanup_memory()

    # Final summary
    X_final = X_history_np
    Y_final = Y_history_raw_np
    scores_final = score_fn(Y_final)
    final_hv = compute_hypervolume(Y_history_raw_np, ref_point_raw)

    print(f"\n[{experiment_name}] Experiment complete!")
    print(f"  Total evaluations: {len(X_final)}")
    print(f"  Best {score_name}: {np.max(scores_final):.4f}")
    print(f"  Mean {score_name}: {np.mean(scores_final):.4f}")
    print(f"  Final hypervolume: {final_hv:.4f}")

    return X_final, Y_final, logger

def compute_total_uncertainty(
    model,
    design_space: DesignSpace,
) -> Tuple[float, float]:
    """
    Compute total uncertainty (sum of posterior std) across the entire design space.
    
    Args:
        model: Fitted GP model
        design_space: DesignSpace object containing the full space
    
    Returns:
        (total_uncertainty_obj1, total_uncertainty_obj2)
    """
    # Use the entire design space
    X_full = design_space.space  # Full discrete space
    X_full_torch = torch.tensor(X_full, dtype=DTYPE, device=DEVICE)
    
    with torch.no_grad():
        # Get posterior distribution
        posterior = model.posterior(X_full_torch)
        
        # Get standard deviation (uncertainty) for each objective
        # posterior.variance has shape (n_points, n_objectives)
        variance = posterior.variance
        std = torch.sqrt(variance)
        
        # Sum uncertainty across all points for each objective
        total_std_obj1 = float(std[:, 0].sum().cpu())
        total_std_obj2 = float(std[:, 1].sum().cpu())
    
    return total_std_obj1, total_std_obj2