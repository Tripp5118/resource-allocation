"""Example script to run agent-driven Bayesian Optimization with optional events."""
import os
import random as rd
from experiments.base_optimizer import AgentDrivenBayesianOptimizer
from core.agent_manager import ResourceEvent

def main():
    # Configuration
    EXPERIMENT_NAME = "agent_noevents_ideal_abs_100"
    LOG_DIR = "results"
    
    # Budget and time constraints
    TOTAL_BUDGET = 500 * 100
    TOTAL_TIME = 100
    COST_PER_POINT = 100.0
    TIME_PER_ITERATION = 1.0
    
    # BO parameters
    INIT_N = 10                  # Initial random samples
    MAX_ITERS = 100              # Maximum iterations (may stop early due to budget/time)
    
    # Agent parameters
    MAX_REASONING_STEPS = 5     # Maximum reasoning iterations per decision
    
    # ==================== OPTIONAL: CONFIGURE EVENTS ====================
    # Uncomment and modify these to add resource events during optimization
    ENABLE_EVENTS = False  # Set to True to enable events
    
    events = None
    if ENABLE_EVENTS:
        events = []
        
        # Example 1: Budget cut at iteration 5
        events.append(ResourceEvent(
            iteration=5,
            event_type='budget_change',
            description="Budget cut by 30% due to funding constraints",
            modifier=lambda x: x * 0.7  # Reduce budget by 30%
        ))
        
        # Example 2: Time extension at iteration 10
        events.append(ResourceEvent(
            iteration=10,
            event_type='time_change',
            description="Time extension of 5 weeks approved",
            modifier=lambda x: x + 5.0  # Add 5 weeks
        ))
        
        # Example 3: Cost increase at iteration 15
        events.append(ResourceEvent(
            iteration=15,
            event_type='cost_change',
            description="Material costs increased by 50%",
            modifier=lambda x: x * 1.5  # Increase cost by 50%
        ))
        
        
        print("\n" + "="*80)
        print(f"EVENTS ENABLED: {len(events)} events scheduled")
        for event in events:
            print(f"  - Iteration {event.iteration}: {event.description}")
        print("="*80 + "\n")
    # ====================================================================
    
    # Ensure OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    print("="*80)
    print("Agent-Driven Bayesian Optimization for HEA Design")
    print("="*80)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Total Budget: ${TOTAL_BUDGET:.2f}")
    print(f"Total Time: {TOTAL_TIME:.1f} weeks")
    print(f"Cost per point: ${COST_PER_POINT:.2f}")
    print(f"Time per iteration: {TIME_PER_ITERATION:.1f} weeks")
    print(f"Max reasoning steps: {MAX_REASONING_STEPS}")
    print(f"Events enabled: {ENABLE_EVENTS}")
    print("\nNote: Each iteration takes 1 week regardless of batch size")
    print("      (all points synthesized/tested in parallel)")
    print("="*80)
    
    seed = rd.randint(1, 9999)
    print(f"Seed is {seed}")

    # Create optimizer
    with AgentDrivenBayesianOptimizer(
        model_dir="models/",
        seed=seed,
        min_comp=0.10,
        max_comp=0.40,
        step=0.005,
        use_priors=False,
        agent_api_key=api_key,
        agent_model="gpt-4o",
        max_reasoning_steps=MAX_REASONING_STEPS
    ) as optimizer:
        # Run optimization (with optional events)
        X_final, Y_final = optimizer.run(
            init_n=INIT_N,
            iters=MAX_ITERS,
            total_budget=TOTAL_BUDGET,
            total_time=TOTAL_TIME,
            cost_per_point=COST_PER_POINT,
            time_per_iteration=TIME_PER_ITERATION,
            mc_samples=128,
            pool_subsample=10000,
            log_dir=LOG_DIR,
            experiment_name=EXPERIMENT_NAME,
            create_visualizations=True,
            create_gif=True,
            events=events  # ADD THIS PARAMETER
        )
        
        # Print final results
        print("\n" + "="*80)
        print("Final Results")
        print("="*80)
        print(f"Total points evaluated: {len(X_final)}")
        ratios = Y_final[:, 1] / (abs(Y_final[:, 0]) + 1e-12)
        best_idx = ratios.argmax()
        print(f"\nBest composition found:")
        print(f"  Fe: {X_final[best_idx, 0]:.3f}")
        print(f"  Co: {X_final[best_idx, 1]:.3f}")
        print(f"  Cr: {X_final[best_idx, 2]:.3f}")
        print(f"  Ni: {X_final[best_idx, 3]:.3f}")
        print(f"  V:  {X_final[best_idx, 4]:.3f}")
        print(f"\n  CTE: {Y_final[best_idx, 0]:.6f}")
        print(f"  K:   {Y_final[best_idx, 1]:.4f}")
        print(f"  K/CTE ratio: {ratios[best_idx]:.3f}")
        print("="*80)

if __name__ == "__main__":
    main()