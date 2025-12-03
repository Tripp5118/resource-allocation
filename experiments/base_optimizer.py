'''Base Bayesian Optimizer orchestrating the BO loop with LLM Agent.'''
import os
import time
import numpy as np
import torch
from typing import Optional, List
import gc
from botorch.utils.transforms import normalize

from core.truth_interface import TruthModelEvaluator
from core.design_space import DesignSpace
from core.gp_models import GPModelManager
from core.acquisition_functions import AcquisitionFunctionManager
from core.visualization import VisualizationManager
from core.logging_utils import LoggingManager
from core.agent_manager import BOAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


class AgentDrivenBayesianOptimizer:
    """Bayesian Optimizer with LLM Agent decision-making."""
    
    def __init__(self, 
                 model_dir="models/", 
                 seed: int = 42,
                 min_comp: float = 0.10, 
                 max_comp: float = 0.40, 
                 step: float = 0.005,
                 use_priors: bool = False,
                 agent_api_key: Optional[str] = None,
                 agent_model: str = "gpt-4o",
                 max_reasoning_steps: int = 10):
        """Initialize the agent-driven BO system.
        
        Args:
            model_dir: Directory containing truth models
            seed: Random seed
            min_comp: Minimum composition fraction
            max_comp: Maximum composition fraction
            step: Discretization step
            use_priors: Whether to use prior models
            agent_api_key: OpenAI API key for agent
            agent_model: LLM model to use for agent
            max_reasoning_steps: Maximum reasoning iterations for agent decisions
        """
        self.seed = seed
        torch.manual_seed(seed)
        
        self.design_space = DesignSpace(min_comp, max_comp, step, seed)
        bounds_np = np.array([[min_comp]*5, [max_comp]*5])
        self.bounds = torch.tensor(bounds_np, dtype=DTYPE, device=DEVICE)
        
        self.use_priors = use_priors
        self.gp_manager = GPModelManager(self.bounds, use_priors=use_priors)
        self.acq_manager = AcquisitionFunctionManager(self.bounds)
        self.evaluator = TruthModelEvaluator()
        self.model_dir = model_dir
        
        # Agent will be initialized in run()
        self.agent: Optional[BOAgent] = None
        self.agent_api_key = agent_api_key
        self.agent_model = agent_model
        self.max_reasoning_steps = max_reasoning_steps
        
    def __enter__(self):
        print(f"[BO] Using Random Forest surrogate models")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.evaluator is not None:
            self.evaluator.cleanup()
        if self.agent is not None:
            self.agent.save_global_memory()
        print("[BO] Cleaned up surrogate evaluator and agent")
        
    def run(self, 
            init_n: int = 10, 
            iters: int = 10, 
            total_budget: float = 10000.0,
            total_time: float = 20.0,
            cost_per_point: float = 100.0,
            time_per_iteration: float = 1.0,
            mc_samples: int = 128, 
            pool_subsample: Optional[int] = None,
            log_dir: str = "results", 
            experiment_name: str = "experiment", 
            create_visualizations: bool = True, 
            create_gif: bool = True,
            events: Optional[List] = None):
        """Run agent-driven Bayesian Optimization.
        
        Args:
            init_n: Number of initial random samples
            iters: Maximum number of BO iterations
            total_budget: Total budget available ($)
            total_time: Total time available (weeks)
            cost_per_point: Cost per experimental point ($)
            time_per_iteration: Time per iteration (weeks) - each iteration takes 1 week
            mc_samples: MC samples for acquisition functions
            pool_subsample: Size of candidate pool (None = use full space)
            log_dir: Directory for logs
            experiment_name: Name of experiment
            create_visualizations: Whether to create plots
            create_gif: Whether to create animated GIF
            
        Note:
            Each iteration takes 1 week to complete regardless of batch size, as all points
            in a batch are synthesized and characterized in parallel.
        """
        # Setup logging and visualization
        logger = None
        vis_manager = None
        exp_dir = None
        
        if log_dir is not None:
            exp_dir = os.path.join(log_dir, experiment_name)
            os.makedirs(exp_dir, exist_ok=True)
            logger = LoggingManager(exp_dir, experiment_name)
            if create_visualizations:
                vis_manager = VisualizationManager(exp_dir, self.design_space, experiment_name)
        
        # Initialize agent
        print("\n" + "="*80)
        print("[BO] Initializing LLM Agent")
        print(f"[BO] Max reasoning steps per decision: {self.max_reasoning_steps}")
        print("="*80)
        agent_log_dir = os.path.join(exp_dir, "agent_logs") if exp_dir else None
        self.agent = BOAgent(
            api_key=self.agent_api_key,
            model=self.agent_model,
            log_dir=agent_log_dir,
            max_reasoning_steps=self.max_reasoning_steps
        )

        if events:
            print(f"\n[BO] Scheduling {len(events)} resource events")
            for event in events:
                self.agent.add_event(event)
                print(f"  - Iteration {event.iteration}: {event.description}")

        # Agent configures GP kernel
        print("\n[BO] Agent configuring GP kernel...")
        kernel_code, kernel_reasoning = self.agent.configure_gp_kernel(self.use_priors)
        
        if kernel_code:
            print(f"[BO] Agent requested custom kernel")
            print(f"[BO] Custom kernel not yet implemented, using default")
        else:
            print(f"[BO] Agent selected default kernel")
        
        # Initial random sampling (counts as 1 week of time)
        print(f"\n[BO] Initializing with {init_n} random samples")
        X0 = self.design_space.sample(init_n, method="random")
        result0 = self.evaluator.evaluate(X0)
        X = torch.tensor(X0, dtype=DTYPE, device=DEVICE)
        Y = torch.tensor(np.column_stack([result0.cte, result0.k]), dtype=DTYPE, device=DEVICE)
        
        # Track budget and time
        initial_budget_spent = init_n * cost_per_point
        initial_time_spent = time_per_iteration  # 1 week for initial sampling
        budget_remaining = total_budget - initial_budget_spent
        time_remaining = total_time - initial_time_spent
        
        print(f"[BO] Initial sampling used: ${initial_budget_spent:.2f} / {initial_time_spent:.1f} weeks")
        print(f"[BO] Remaining: ${budget_remaining:.2f} / {time_remaining:.1f} weeks")
        
        # Log initial samples
        if logger:
            logger.log_iteration(
                iteration=0,
                X_history=X.cpu().numpy(),
                Y_history=Y.cpu().numpy(),
                n_new_points=init_n,
                strategy=None,
                timing=0.0,
                extra_info={"agent_kernel_reasoning": kernel_reasoning}
            )
        
        # Main BO loop
        iteration = 0
        while iteration < iters:
            # Check if we have budget/time remaining (need at least 1 week)
            if budget_remaining < cost_per_point or time_remaining < time_per_iteration:
                print(f"\n[BO] Stopping: Insufficient budget (${budget_remaining:.2f}) or time ({time_remaining:.1f} weeks)")
                break
            
            iteration += 1
            iter_start = time.perf_counter()

            budget_remaining, time_remaining, cost_per_point, event_messages = \
            self.agent.process_iteration_events(
                iteration, budget_remaining, time_remaining, cost_per_point
            )
        
            # Display event notifications
            if event_messages:
                print(f"\n{'='*80}")
                for msg in event_messages:
                    print(msg)
                print('='*80)
            
            print(f"\n{'='*80}")
            print(f"[BO] Iteration {iteration}/{iters}")
            print(f"[BO] Budget: ${budget_remaining:.2f} | Time: {time_remaining:.1f} weeks")
            print('='*80)
            
            # Fit GP model
            print("[BO] Fitting GP model...")
            model = self.gp_manager.fit_model(X, Y)
            
            # Compute Pareto front
            Y_mo = torch.stack([Y[:, 0], Y[:, 1]], dim=-1)
            pareto_Y, ref_point = self.acq_manager.compute_pareto_front(Y_mo)
            
            # Prepare candidate pool
            if pool_subsample is None:
                X_pool_np = self.design_space.space
            else:
                X_pool_np = self.design_space.sample(pool_subsample, method="sobol")
            X_pool = torch.tensor(X_pool_np, dtype=DTYPE, device=DEVICE)
            
            # Compute acquisition function options
            print("[BO] Computing acquisition options...")
            allocation_options = self.acq_manager.compute_all_allocation_options(
                model=model,
                X_pool=X_pool,
                pareto_Y=pareto_Y,
                ref_point=ref_point,
                mc_samples=mc_samples
            )
            
            # Agent selects allocation
            print(f"\n[BO] Consulting agent for resource allocation...")
            selected_idx, reasoning, selected_point_arrays = self.agent.select_resource_allocation(
                iteration=iteration,
                budget_remaining=budget_remaining,
                time_remaining=time_remaining,
                cost_per_point=cost_per_point,
                time_per_point=time_per_iteration,
                X_history=X.cpu().numpy(),
                Y_history=Y.cpu().numpy(),
                allocation_options=allocation_options,
                max_batch_size=5
            )
            
            # Combine selected points
            X_next = np.vstack(selected_point_arrays) if selected_point_arrays else np.array([]).reshape(0, 5)
            
            if len(X_next) == 0:
                print("[BO] Warning: Agent selected no points, using default balanced option")
                selected_batch = allocation_options.qehvi_batches[2]  # Balanced option
                X_next_list = []
                if len(selected_batch.points) > 0:
                    X_next_list.append(selected_batch.points)
                if len(selected_batch.explore_points) > 0:
                    X_next_list.append(selected_batch.explore_points)
                X_next = np.vstack(X_next_list)
            
            # Evaluate selected points
            print(f"[BO] Evaluating {len(X_next)} selected points...")
            result_next = self.evaluator.evaluate(X_next)
            
            # Update data
            X = torch.cat([X, torch.tensor(X_next, dtype=DTYPE, device=DEVICE)], dim=0)
            Y = torch.cat([
                Y,
                torch.tensor(np.column_stack([result_next.cte, result_next.k]), dtype=DTYPE, device=DEVICE)
            ], dim=0)
            
            # Update budget and time
            points_used = len(X_next)
            budget_spent = points_used * cost_per_point
            time_spent = time_per_iteration  # Always 1 week per iteration
            budget_remaining -= budget_spent
            time_remaining -= time_spent
            
            iter_time = time.perf_counter() - iter_start
            
            # Log iteration
            if logger:
                logger.log_iteration(
                    iteration=iteration,
                    X_history=X.cpu().numpy(),
                    Y_history=Y.cpu().numpy(),
                    n_new_points=len(X_next),
                    strategy=None,
                    timing=iter_time,
                    extra_info={
                        "agent_selected_option": selected_idx,
                        "agent_reasoning": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
                        "budget_remaining": budget_remaining,
                        "time_remaining": time_remaining,
                        "budget_spent_this_iter": budget_spent,
                        "time_spent_this_iter": time_spent
                    },
                    acquisition_data=allocation_options
                )
                
                # Log detailed evaluations
                Y_next = np.column_stack([result_next.cte, result_next.k])
                logger.log_evaluations(
                    iteration=iteration,
                    X_new=X_next,
                    Y_new=Y_next,
                    selection_info={"agent_selection": selected_idx}
                )
            
            # Create visualizations
            if vis_manager:
                vis_manager.create_iteration_plot(
                    iteration=iteration,
                    X_history=X.cpu().numpy(),
                    Y_history=Y.cpu().numpy(),
                    X_new=X_next,
                    model=model,
                    bounds=self.bounds,
                    selection_info={"agent_option": selected_idx}
                )
            
            # Print iteration summary
            ratios = Y[:, 1].cpu().numpy() / (np.abs(Y[:, 0].cpu().numpy()) + 1e-12)
            best_ratio = np.max(ratios)
            print(f"\n[BO] Iteration {iteration} complete in {iter_time:.2f}s")
            print(f"[BO] Best K/CTE ratio: {best_ratio:.3f}")
            print(f"[BO] Remaining resources: ${budget_remaining:.2f} / {time_remaining:.1f} weeks")
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Finalize
        print(f"\n{'='*80}")
        print(f"[BO] Optimization Complete!")
        print(f"[BO] Total iterations: {iteration}")
        print(f"[BO] Budget used: ${total_budget - budget_remaining:.2f} / ${total_budget:.2f}")
        print(f"[BO] Time used: {total_time - time_remaining:.1f} / {total_time:.1f} weeks")
        print('='*80)
        
        # Save agent memory
        if self.agent:
            self.agent.save_global_memory()
            print(f"[BO] Agent memory saved to {agent_log_dir}")
        
        # Finalize logging
        if logger:
            logger.finalize()
        
        # Create final GIF
        if vis_manager and create_gif:
            vis_manager.create_gif(duration=2.0, gif_name=f"{experiment_name}.gif")
        
        return X.cpu().numpy(), Y.cpu().numpy()