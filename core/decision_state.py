# core/decision_state.py
"""Decision state representation for resource allocation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class StrategyOutcome:
    """Record of a decision and its outcome."""
    iteration: int
    n_exploit: int
    n_explore: int
    score_before: float
    score_after: float
    improvement: float
    pareto_points_added: int
    pareto_points_improved: int
    expected_improvement: float  # From acquisition function
    actual_improvement: float    # HV change (hv_after - hv_before)
    hv_before: float = 0.0       # Absolute HV before evaluation
    hv_after: float = 0.0        # Absolute HV after evaluation


@dataclass
class ParetoStatus:
    """Current state of the Pareto front."""
    num_points: int
    hypervolume: float
    hypervolume_change: float
    hypervolume_change_pct: float
    points_added_last_iter: int
    points_improved_last_iter: int
    status: str  # "GROWING", "REFINING", "STAGNANT"


@dataclass
class ProgressVelocity:
    """Analysis of optimization progress rate (raw numbers, no labels)."""
    recent_improvement: float   # Score improvement over last 3 iterations
    previous_improvement: float  # Score improvement over the 3 iterations before that


@dataclass
class StrategyEffectiveness:
    """Performance summary by strategy type."""
    exploit_heavy: Dict[str, float]  # {avg_improvement, success_rate, n_times_used}
    balanced: Dict[str, float]
    explore_heavy: Dict[str, float]


@dataclass
class DecisionState:
    """Complete state for making resource allocation decisions."""
    
    # Iteration context
    iteration: int
    budget_remaining: float
    time_remaining: float
    cost_per_point: float
    time_per_point: float
    
    # History & patterns
    strategy_outcomes: List[StrategyOutcome]  # Recent decision→outcome pairs
    strategy_effectiveness: Optional[StrategyEffectiveness] = None
    
    # Current optimization status
    pareto_status: Optional[ParetoStatus] = None
    progress_velocity: Optional[ProgressVelocity] = None
    
    # Current best point info
    best_score: float = 0.0
    best_obj1: float = 0.0
    best_obj2: float = 0.0
    best_point_coords: str = ""
    total_points_evaluated: int = 0
    
    # Available options (your AllocationResults object)
    allocation_options: Any = None
    
    # Optional: beliefs from previous iteration (for LLM continuity)
    previous_beliefs: Optional[Dict] = None
    
    # Problem context
    problem_description: str = ""
    obj1_name: str = "Objective 1"
    obj2_name: str = "Objective 2"
    
    # Upcoming events
    upcoming_events: List[str] = field(default_factory=list)
    
    # Optional advanced metrics (can be None)
    uncertainty_landscape: Optional[Dict] = None
    prediction_accuracy: Optional[Dict] = None

    recent_events: List[str] = field(default_factory=list)  # NEW
    initial_budget: Optional[float] = None  # For phase calculation
    initial_time: Optional[float] = None    # For phase calculation


def build_strategy_outcome(
    iteration: int,
    selected_option_idx: int,
    n_exploit: int,
    n_explore: int,
    score_before: float,
    score_after: float,
    pareto_before: Dict,
    pareto_after: Dict,
    expected_improvement: float = 0.0,
) -> StrategyOutcome:
    """Helper to construct StrategyOutcome from iteration data."""
    improvement = score_after - score_before
    
    points_added = pareto_after.get('num_points', 0) - pareto_before.get('num_points', 0)
    points_improved = pareto_after.get('points_improved', 0)
    
    hv_before = pareto_before.get('hypervolume', 0.0)
    hv_after_val = pareto_after.get('hypervolume', 0.0)
    actual_improvement = hv_after_val - hv_before

    return StrategyOutcome(
        iteration=iteration,
        n_exploit=n_exploit,
        n_explore=n_explore,
        score_before=score_before,
        score_after=score_after,
        improvement=improvement,
        pareto_points_added=points_added,
        pareto_points_improved=points_improved,
        expected_improvement=expected_improvement,
        actual_improvement=actual_improvement,
        hv_before=hv_before,
        hv_after=hv_after_val,
    )


def format_coordinates(X_point: np.ndarray, max_dims_to_show: int = 10) -> str:
    """
    Format point coordinates for display.
    
    Args:
        X_point: Point coordinates (1D array)
        max_dims_to_show: Maximum dimensions to display (default: 10)
    
    Returns:
        Formatted string of coordinates
    """
    n_dims = len(X_point)
    
    if n_dims < max_dims_to_show:
        # Show all dimensions
        coords = ", ".join([f"x_{j}={X_point[j]:.3f}" for j in range(n_dims)])
    else:
        # Show first 5 dimensions with ellipsis
        coords = ", ".join([f"x_{j}={X_point[j]:.3f}" for j in range(5)])
        coords += ", ..."
    
    return coords