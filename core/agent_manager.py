"""Agent manager with DecisionState architecture - refactored for multi-stage decisions.""" 

import os
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np 

from core.decision_state import (
    DecisionState,
    StrategyOutcome,
    ParetoStatus,
    ProgressVelocity,
    StrategyEffectiveness,
    build_strategy_outcome
)
from core.decision_maker import DecisionMaker, BalancedPolicyDecisionMaker 

@dataclass
class ResourceEvent:
    """Event that modifies budget or time constraints."""
    iteration: int  # When to trigger (before this iteration)
    event_type: str  # 'budget_change', 'time_change', 'cost_change'
    description: str  # Human-readable description
    modifier: Callable[[float], float]  # Function that modifies the value 

    def apply(self, current_value: float) -> float:
        """Apply the event modifier to current value."""
        return self.modifier(current_value)

class EventManager:
    """Manages resource events during optimization.""" 
    def __init__(self):
        self.events: List[ResourceEvent] = []
        self.triggered_events: List[Dict[str, Any]] = []

    def add_event(self, event: ResourceEvent):
        self.events.append(event)
        self.events.sort(key=lambda e: e.iteration)

    def add_budget_cut(self, iteration: int, percentage: float, description: Optional[str] = None):
        desc = description or f"Budget cut by {percentage*100:.0f}% at iteration {iteration}"
        self.add_event(ResourceEvent(iteration, "budget_change", desc, lambda x: x * (1 - percentage)))

    def add_time_extension(self, iteration: int, weeks: float, description: Optional[str] = None):
        desc = description or f"Time extended by {weeks} weeks at iteration {iteration}"
        self.add_event(ResourceEvent(iteration, "time_change", desc, lambda x: x + weeks))

    def add_cost_increase(self, iteration: int, multiplier: float, description: Optional[str] = None):
        desc = description or f"Cost per point increased by {multiplier}x at iteration {iteration}"
        self.add_event(ResourceEvent(iteration, "cost_change", desc, lambda x: x * multiplier))

    def get_events_for_iteration(self, iteration: int) -> List[ResourceEvent]:
        return [e for e in self.events if e.iteration == iteration]

    def trigger_events(
        self,
        iteration: int,
        budget: float,
        time: float,
        cost_per_point: float,
    ) -> Tuple[float, float, float, List[str]]:
        """Apply all events for this iteration and return updated values."""
        events = self.get_events_for_iteration(iteration)
        messages = []

        for event in events:
            if event.event_type == "budget_change":
                old = budget
                budget = event.apply(budget)
                messages.append(f"⚠️  {event.description}: ${old:.2f} → ${budget:.2f}")
            elif event.event_type == "time_change":
                old = time
                time = event.apply(time)
                messages.append(f"⚠️  {event.description}: {old:.1f} → {time:.1f} weeks")
            elif event.event_type == "cost_change":
                old = cost_per_point
                cost_per_point = event.apply(cost_per_point)
                messages.append(f"⚠️  {event.description}: ${old:.2f} → ${cost_per_point:.2f} per point")

            self.triggered_events.append(
                {
                    "iteration": iteration,
                    "event_type": event.event_type,
                    "description": event.description,
                    "timestamp": iteration,
                }
            )

        return budget, time, cost_per_point, messages

class BOAgent:
    def __init__(
        self,
        decision_maker: Optional[DecisionMaker] = None,
        log_dir: Optional[str] = None,
        problem_description: str = "",
        obj1_name: str = "Objective 1",
        obj2_name: str = "Objective 2",
        **kwargs  # Accept other args for backward compatibility
    ):
        """
        Args:
            decision_maker: DecisionMaker instance (if None, uses balanced policy)
            log_dir: Directory to save logs
            problem_description: Description of optimization problem
            obj1_name, obj2_name: Objective names
        """
        self.decision_maker = decision_maker or BalancedPolicyDecisionMaker()
        self.problem_description = problem_description
        self.obj1_name = obj1_name
        self.obj2_name = obj2_name
        self.log_dir = log_dir
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Memory for tracking outcomes
        self.outcome_history: List[StrategyOutcome] = []
        self.pareto_history: List[Dict] = []  # Track pareto status over time
        self.iteration_history: List[Dict] = []  # Track iteration-level best scores
        
        # Event management
        self.event_manager = EventManager()
        
        # Store previous beliefs for LLM continuity
        self.previous_beliefs: Optional[Dict] = None
        
        # Store last decision for completing outcome record
        self._last_decision: Optional[Dict] = None
        self._last_pareto_before: Optional[Dict] = None

    def _log(self, level: str, message: str):
        """Logging helper."""
        print(f"[Agent:{level}] {message}")

    # ===== EVENT MANAGEMENT (unchanged) =====

    def add_event(self, event: ResourceEvent):
        self.event_manager.add_event(event)

    def add_budget_cut(self, iteration: int, percentage: float, description: Optional[str] = None):
        self.event_manager.add_budget_cut(iteration, percentage, description)

    def add_time_extension(self, iteration: int, weeks: float, description: Optional[str] = None):
        self.event_manager.add_time_extension(iteration, weeks, description)

    def add_cost_increase(self, iteration: int, multiplier: float, description: Optional[str] = None):
        self.event_manager.add_cost_increase(iteration, multiplier, description)

    def process_iteration_events(
        self, iteration: int, budget: float, time: float, cost_per_point: float
    ) -> Tuple[float, float, float, List[str]]:
        return self.event_manager.trigger_events(iteration, budget, time, cost_per_point)

    # ===== CORE DECISION METHOD (refactored) =====

    def select_resource_allocation(
        self,
        iteration: int,
        budget_remaining: float,
        time_remaining: float,
        cost_per_point: float,
        time_per_point: float,
        X_history: np.ndarray,
        Y_history: np.ndarray,
        allocation_results: Any,
        max_batch_size: int = 5,
        score_fn: Optional[Callable] = None,
        pareto_front_info: Optional[Dict] = None,  # NEW: expect this from BO loop
        **kwargs  # Backward compatibility
    ) -> Tuple[int, str, List[np.ndarray]]:
        """
        Select resource allocation using DecisionState architecture.
        
        Args:
            iteration: Current iteration number
            budget_remaining: Remaining budget ($)
            time_remaining: Remaining time (weeks)
            cost_per_point: Cost per evaluation ($)
            time_per_point: Time per iteration (weeks)
            X_history: Input history (n, d)
            Y_history: Output history (n, 2)
            allocation_results: AllocationResults with options
            max_batch_size: Maximum batch size
            score_fn: Function mapping Y -> scores (default: maximize Y[:, 1])
            pareto_front_info: Dict with {num_points, hypervolume, points_added, points_improved}
        
        Returns:
            selected_idx: Index in allocation_results.options
            reasoning: Full reasoning text
            selected_points: List of point arrays for evaluation
        """
        self._log("INFO", f"\n{'='*80}\nIteration {iteration}: Resource Allocation\n{'='*80}")
        
        if score_fn is None:
            score_fn = lambda Y: Y[:, 1]  # Default: maximize second objective
        
        # Build DecisionState
        state = self._build_decision_state(
            iteration=iteration,
            budget_remaining=budget_remaining,
            time_remaining=time_remaining,
            cost_per_point=cost_per_point,
            time_per_point=time_per_point,
            X_history=X_history,
            Y_history=Y_history,
            allocation_results=allocation_results,
            max_batch_size=max_batch_size,
            score_fn=score_fn,
            pareto_front_info=pareto_front_info,
        )
        
        # Delegate decision to decision maker
        selected_idx, reasoning, updated_beliefs = self.decision_maker.make_decision(state)
        
        # Validate feasibility
        selected_idx = self._validate_feasibility(
            selected_idx, allocation_results, budget_remaining, time_remaining,
            cost_per_point, time_per_point, max_batch_size
        )
        
        # Extract points
        selected_batch = allocation_results.options[selected_idx]
        selected_points = self._extract_points(selected_batch, max_batch_size)
        
        # Store decision for next iteration's outcome tracking
        self._record_decision(iteration, selected_idx, selected_batch, state, updated_beliefs)
        
        # Save logs
        self._save_iteration_log(iteration, {
            "iteration": iteration,
            "selected_option": selected_idx,
            "reasoning": reasoning,
            "beliefs": updated_beliefs,
            "budget_remaining": budget_remaining,
            "time_remaining": time_remaining,
        })
        
        n_exploit = selected_batch.num_exploitation
        n_explore = max_batch_size - n_exploit
        self._log("INFO", f"✓ Selected Option {selected_idx}: {n_exploit} exploit + {n_explore} explore")
        
        return selected_idx, reasoning, selected_points

    # ===== STATE BUILDING =====

    def _build_decision_state(
        self,
        iteration: int,
        budget_remaining: float,
        time_remaining: float,
        cost_per_point: float,
        time_per_point: float,
        X_history: np.ndarray,
        Y_history: np.ndarray,
        allocation_results: Any,
        max_batch_size: int,
        score_fn: Callable,
        pareto_front_info: Optional[Dict],
    ) -> DecisionState:
        """Build complete DecisionState from BO data."""
        
        # Compute current best
        if len(Y_history) > 0:
            scores = score_fn(Y_history)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_obj1 = float(Y_history[best_idx, 0])
            best_obj2 = float(Y_history[best_idx, 1])
            
            # Display coordinates based on dimensionality
            n_dims = X_history.shape[1]
            if n_dims < 10:
                # Show all dimensions if less than 10
                best_coords = ", ".join([f"x_{j}={X_history[best_idx, j]:.3f}" 
                                        for j in range(n_dims)])
            else:
                # Show first 5 dimensions with ellipsis if 10 or more
                best_coords = ", ".join([f"x_{j}={X_history[best_idx, j]:.3f}" 
                                        for j in range(5)])
                best_coords += ", ..."
        else:
            best_score = 0.0
            best_obj1 = 0.0
            best_obj2 = 0.0
            best_coords = "N/A"
        
        # Store iteration-level info for velocity tracking
        self.iteration_history.append({
            'iteration': iteration,
            'best_score': best_score,
            'points_evaluated': len(Y_history),
        })
        
        # Build pareto status
        pareto_status = self._build_pareto_status(pareto_front_info, iteration)
        
        # Build progress velocity (iteration-level, raw numbers)
        progress_velocity = self._build_progress_velocity()
        
        # Build strategy effectiveness
        strategy_effectiveness = self._build_strategy_effectiveness()
        
        # Get recent outcomes (last 5)
        recent_outcomes = self.outcome_history[-5:] if len(self.outcome_history) > 0 else []
        
        # Check for upcoming events
        upcoming = self.event_manager.get_events_for_iteration(iteration + 1)
        upcoming_events = [e.description for e in upcoming]
        
        return DecisionState(
            iteration=iteration,
            budget_remaining=budget_remaining,
            time_remaining=time_remaining,
            cost_per_point=cost_per_point,
            time_per_point=time_per_point,
            strategy_outcomes=recent_outcomes,
            strategy_effectiveness=strategy_effectiveness,
            pareto_status=pareto_status,
            progress_velocity=progress_velocity,
            best_score=best_score,
            best_obj1=best_obj1,
            best_obj2=best_obj2,
            best_point_coords=best_coords,
            total_points_evaluated=len(X_history),
            allocation_options=allocation_results,
            previous_beliefs=self.previous_beliefs,
            problem_description=self.problem_description,
            obj1_name=self.obj1_name,
            obj2_name=self.obj2_name,
            upcoming_events=upcoming_events,
        )

    def _build_pareto_status(self, pareto_info: Optional[Dict], iteration: int) -> Optional[ParetoStatus]:
        """Build ParetoStatus from info dict."""
        if pareto_info is None:
            return None
        
        # Store in history
        self.pareto_history.append(pareto_info)
        
        # Compute change from last iteration
        if len(self.pareto_history) >= 2:
            prev = self.pareto_history[-2]
            hv_change = pareto_info['hypervolume'] - prev['hypervolume']
            hv_change_pct = (hv_change / prev['hypervolume'] * 100) if prev['hypervolume'] > 0 else 0.0
        else:
            hv_change = 0.0
            hv_change_pct = 0.0
        
        # Determine status based on changes
        points_added = pareto_info.get('points_added', 0)
        points_improved = pareto_info.get('points_improved', 0)
        
        if points_added >= 2:
            status = "GROWING"
        elif points_added == 1 or points_improved >= 1:
            status = "REFINING"
        else:
            status = "STAGNANT"
        
        return ParetoStatus(
            num_points=pareto_info['num_points'],
            hypervolume=pareto_info['hypervolume'],
            hypervolume_change=hv_change,
            hypervolume_change_pct=hv_change_pct,
            points_added_last_iter=points_added,
            points_improved_last_iter=points_improved,
            status=status,
        )

    def _build_progress_velocity(self) -> Optional[ProgressVelocity]:
        """Analyze rate of improvement using iteration-level data (raw numbers, no labels)."""
        if len(self.iteration_history) < 3:
            return None
        
        # Look at last 3 iterations vs previous 3 iterations
        recent_3 = self.iteration_history[-3:]
        previous_3 = self.iteration_history[-6:-3] if len(self.iteration_history) >= 6 else []
        
        # Compute improvements (raw numbers)
        improvement_recent = recent_3[-1]['best_score'] - recent_3[0]['best_score']
        
        if previous_3:
            improvement_previous = previous_3[-1]['best_score'] - previous_3[0]['best_score']
        else:
            improvement_previous = 0.0
        
        # Return raw numbers - let LLM interpret
        return ProgressVelocity(
            recent_improvement=float(improvement_recent),
            previous_improvement=float(improvement_previous),
        )

    def _build_strategy_effectiveness(self) -> Optional[StrategyEffectiveness]:
        """Compute average performance by strategy type.
        
        Bins by explore ratio (n_explore / batch_size) so thresholds are
        meaningful regardless of batch size:
          exploit-heavy : ratio < 0.33
          balanced       : ratio 0.33–0.67
          explore-heavy  : ratio > 0.67
        """
        if len(self.outcome_history) < 2:
            return None
        
        exploit_heavy = []
        balanced = []
        explore_heavy = []
        
        for outcome in self.outcome_history:
            batch_size = outcome.n_exploit + outcome.n_explore
            if batch_size == 0:
                continue  # Shouldn't happen, but guard against divide-by-zero
            ratio = outcome.n_explore / batch_size
            if ratio < 0.33:
                exploit_heavy.append(outcome.improvement)
            elif ratio > 0.67:
                explore_heavy.append(outcome.improvement)
            else:
                balanced.append(outcome.improvement)
        
        def summarize(improvements):
            if not improvements:
                return {"avg_improvement": 0.0, "success_rate": 0.0, "n_times_used": 0}
            return {
                "avg_improvement": float(np.mean(improvements)),
                "success_rate": float(sum(1 for x in improvements if x > 0.001) / len(improvements)),
                "n_times_used": len(improvements),
            }
        
        return StrategyEffectiveness(
            exploit_heavy=summarize(exploit_heavy),
            balanced=summarize(balanced),
            explore_heavy=summarize(explore_heavy),
        )

    def _record_decision(self, iteration: int, selected_idx: int, selected_batch: Any, 
                        state: DecisionState, beliefs: Optional[Dict]):
        """Store decision info for next iteration's outcome tracking."""
        self.previous_beliefs = beliefs
        
        # Store pareto status before evaluation
        if state.pareto_status:
            self._last_pareto_before = {
                'num_points': state.pareto_status.num_points,
                'hypervolume': state.pareto_status.hypervolume,
            }
        else:
            self._last_pareto_before = {'num_points': 0, 'hypervolume': 0.0}
        
        # Store decision details
        self._last_decision = {
            'iteration': iteration,
            'selected_idx': selected_idx,
            'n_exploit': selected_batch.num_exploitation,
            'n_explore': state.allocation_options.options[0].total_batch_size - selected_batch.num_exploitation,
            'score_before': state.best_score,
            'expected_improvement': selected_batch.hypervolume_improvement,
        }

    def record_iteration_outcome(self, score_after: float, pareto_after: Dict):
        """
        Call this after evaluating points to complete the outcome record.
        
        Args:
            score_after: Best score after evaluation
            pareto_after: Dict with {num_points, hypervolume, points_added, points_improved}
        """
        if not hasattr(self, '_last_decision') or self._last_decision is None:
            return
        
        outcome = build_strategy_outcome(
            iteration=self._last_decision['iteration'],
            selected_option_idx=self._last_decision['selected_idx'],
            n_exploit=self._last_decision['n_exploit'],
            n_explore=self._last_decision['n_explore'],
            score_before=self._last_decision['score_before'],
            score_after=score_after,
            pareto_before=self._last_pareto_before,
            pareto_after=pareto_after,
            expected_improvement=self._last_decision['expected_improvement'],
        )
        
        self.outcome_history.append(outcome)
        
        # Clear last decision
        self._last_decision = None
        self._last_pareto_before = None

    # ===== HELPER METHODS =====

    def _validate_feasibility(
        self,
        selected_idx: int,
        allocation_results: Any,
        budget: float,
        time: float,
        cost_per_point: float,
        time_per_point: float,
        max_batch: int,
    ) -> int:
        """Validate selected option is feasible, otherwise find alternative."""
        num_options = len(allocation_results.options)
        
        # Clamp to valid range first
        if selected_idx < 0 or selected_idx >= num_options:
            self._log("WARNING", f"Invalid option {selected_idx}, using option 0")
            selected_idx = 0
        
        batch = allocation_results.options[selected_idx]
        total_points = batch.total_batch_size if hasattr(batch, 'total_batch_size') else max_batch
        cost = total_points * cost_per_point
        
        if cost <= budget and time_per_point <= time and total_points <= max_batch:
            return selected_idx
        
        self._log("WARNING", f"Option {selected_idx} infeasible, searching alternatives...")
        
        # Try from smallest to largest
        for i in range(num_options):
            b = allocation_results.options[i]
            total = b.total_batch_size if hasattr(b, 'total_batch_size') else max_batch
            if total * cost_per_point <= budget and time_per_point <= time and total <= max_batch:
                self._log("INFO", f"Using feasible alternative: Option {i}")
                return i
        
        self._log("WARNING", "No feasible options, defaulting to Option 0")
        return 0

    def _extract_points(self, selected_batch: Any, max_batch_size: int) -> List[np.ndarray]:
        """Extract points from batch."""
        points = []
        if hasattr(selected_batch, 'exploitation_points') and len(selected_batch.exploitation_points) > 0:
            points.append(selected_batch.exploitation_points)
        if hasattr(selected_batch, 'exploration_points') and len(selected_batch.exploration_points) > 0:
            points.append(selected_batch.exploration_points)
        return points

    def _save_iteration_log(self, iteration: int, content: Dict):
        """Save iteration log."""
        if self.log_dir:
            filepath = os.path.join(self.log_dir, f"iteration_{iteration}.json")
            with open(filepath, "w") as f:
                json.dump(content, f, indent=2, default=str)

    def save_global_memory(self):
        """Save complete history."""
        if self.log_dir:
            # Save outcomes
            filepath = os.path.join(self.log_dir, "outcome_history.json")
            with open(filepath, "w") as f:
                json.dump([vars(o) for o in self.outcome_history], f, indent=2)
            
            # Save events
            events_filepath = os.path.join(self.log_dir, "events.json")
            events_data = {
                "scheduled_events": [
                    {
                        "iteration": e.iteration,
                        "type": e.event_type,
                        "description": e.description,
                    }
                    for e in self.event_manager.events
                ],
                "triggered_events": self.event_manager.triggered_events,
            }
            with open(events_filepath, "w") as f:
                json.dump(events_data, f, indent=2)

    def cleanup(self):
        """Cleanup at end of optimization."""
        self.save_global_memory()
        self.decision_maker.cleanup()
        self._log("INFO", "Agent cleanup completed")