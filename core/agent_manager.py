# core/agent_manager.py
"""LLM Agent Manager for generic 2-objective BO with event-driven resource management."""

import os
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_classic.memory import ConversationBufferMemory


# ============================================================================
#  EVENT SYSTEM
# ============================================================================

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


# ============================================================================
#  MAIN AGENT CLASS
# ============================================================================

class BOAgent:
    """LangChain-based Agent for generic 2-objective Bayesian Optimization.
    
    
    Example:
        >>> agent = BOAgent(
        ...     model="gpt-4o",
        ...     log_dir="./logs",
        ...     problem_description="Optimize material properties",
        ...     obj1_name="Strength",
        ...     obj2_name="Toughness"
        ... )
        >>> 
        >>> # Add resource events
        >>> agent.add_budget_cut(iteration=5, percentage=0.3)
        >>> 
        >>> # Make decision
        >>> selected_idx, reasoning, points = agent.select_resource_allocation(
        ...     iteration=1,
        ...     budget_remaining=5000,
        ...     time_remaining=10.0,
        ...     X_history=X,
        ...     Y_history=Y,
        ...     allocation_results=acq_data
        ... )"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        log_dir: Optional[str] = None,
        max_reasoning_steps: int = 10,
        temperature: float = 0.7,
        problem_description: str = "",
        obj1_name: str = "Objective 1",
        obj2_name: str = "Objective 2",
    ):
        """
        Args:
            api_key: OpenAI API key (or None to use environment variable)
            model: LLM model name
            log_dir: Directory to save agent logs
            max_reasoning_steps: Maximum reasoning iterations per decision
            temperature: LLM temperature for generation
            problem_description: Short description of the optimization problem
            obj1_name, obj2_name: Human-friendly names of the two objectives
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        self.model = model
        self.temperature = temperature
        self.max_reasoning_steps = max_reasoning_steps

        self.problem_description = problem_description.strip()
        self.obj1_name = obj1_name
        self.obj2_name = obj2_name

        # Memory
        self.global_memory: List[Dict[str, Any]] = []
        self.iteration_memory: Optional[ConversationBufferMemory] = None

        # Event management
        self.event_manager = EventManager()

        # Logging
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def _log (self, level: str, message: str):
        """Structured logging helper."""
        print(f"[Agent:{level}] {message}")

    # -------------------------- Event Management --------------------------- #

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

    # ---------------------------- Memory & Logs --------------------------- #

    def _create_iteration_memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def save_global_memory(self):
        if self.log_dir:
            filepath = os.path.join(self.log_dir, "agent_global_memory.json")
            with open(filepath, "w") as f:
                json.dump(self.global_memory, f, indent=2)

            events_filepath = os.path.join(self.log_dir, "agent_events.json")
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

    def _save_iteration_log(self, iteration: int, content: Dict[str, Any]):
        if self.log_dir:
            filepath = os.path.join(self.log_dir, f"agent_iteration_{iteration}.json")
            with open(filepath, "w") as f:
                json.dump(content, f, indent=2)

    def _build_global_context(self) -> str:
        if not self.global_memory:
            return "This is the first decision in the optimization campaign."

        context = "## Optimization History\n\n"

        # Recent iterations
        summaries = [m for m in self.global_memory if m.get("step") == "iteration_summary"]
        if summaries:
            context += "**Recent Iterations**:\n"
            for summary in summaries[-3:]:
                it = summary.get("iteration", "?")
                context += f"\nIteration {it}:\n"
                context += f"  - Best score: {summary.get('best_score', 0):.3f}\n"
                context += (
                    f"  - Decision: Option {summary.get('selected_option')} "
                    f"({summary.get('decision_summary', 'N/A')})\n"
                )

        # Event history
        if self.event_manager.triggered_events:
            context += "\n**Resource Events**:\n"
            for event in self.event_manager.triggered_events[-3:]:
                context += f"  - Iter {event['iteration']}: {event['description']}\n"

        return context

    # ---------------- Resource Allocation (Main Decision) ----------------- #
    from core.acquisition_functions import AllocationResults
    def select_resource_allocation(
        self,
        iteration: int,
        budget_remaining: float,
        time_remaining: float,
        cost_per_point: float,
        time_per_point: float,
        X_history: np.ndarray,
        Y_history: np.ndarray,
        allocation_results: AllocationResults,
        max_batch_size: int = 5,
        score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[int, str, List[np.ndarray]]:
        """
        Agent selects resource allocation strategy.

        Args:
            iteration: Current iteration
            budget_remaining, time_remaining: remaining resources
            cost_per_point: cost per point
            time_per_point: time per iteration
            X_history: (n, d)
            Y_history: (n, 2)
            allocation_results: AllocationResults with 6 allocation options
            max_batch_size: maximum allowed batch size
            score_fn: function mapping Y_history -> scores

        Returns:
            selected_idx: index in allocation_results.options
            reasoning: full reasoning text
            selected_points: [exploit_points, explore_points] list for evaluation
        """
        self._log("INFO", f"\n{'='*80}\nIteration {iteration}: Resource Allocation Decision\n{'='*80}")

        if score_fn is None:
            def score_fn_default(Y: np.ndarray) -> np.ndarray:
                return Y[:, 1]
            score_fn = score_fn_default

        self.iteration_memory = self._create_iteration_memory()
        global_context = self._build_global_context()

        analysis = self._analyze_current_state_distilled(
            X_history, Y_history, budget_remaining, time_remaining,
            cost_per_point, time_per_point, score_fn
        )
        self._log("INFO", f"\nAnalysis:\n{analysis}\n")

        options_desc = self._describe_allocation_results(
            allocation_results, cost_per_point, time_per_point,
            budget_remaining, time_remaining, max_batch_size
        )

        upcoming = self.event_manager.get_events_for_iteration(iteration + 1)
        event_warning = ""
        if upcoming:
            event_warning = "\n⚠️  **UPCOMING EVENTS NEXT ITERATION**:\n"
            for e in upcoming:
                event_warning += f"  - {e.description}\n"

        system_message = f"""You are an expert AI managing a Bayesian Optimization campaign.

Problem:
{self.problem_description or '(no additional description provided)'}

Objectives:
- {self.obj1_name}
- {self.obj2_name}

Current Status:
- Iteration: {iteration}
- Budget: ${budget_remaining:.2f}
- Time: {time_remaining:.1f} weeks

Important: Each iteration takes {time_per_point:.1f} week(s) regardless of batch size (parallel execution).

{global_context}{event_warning}

When ready to decide, respond:
SELECTED_OPTION: <0-5>
REASONING: <brief justification>"""

        initial_message = f"""Make a resource allocation decision for this iteration.

{analysis}

{options_desc}

Decision framework (guideline):
- If recent progress is improving with enough runway → prefer more exploitation
- If progress is plateauing or runway is short → prefer more exploration or balanced

You have up to {self.max_reasoning_steps} reasoning steps.

When ready:
SELECTED_OPTION: <0-5>
REASONING: <justification>

Begin analysis."""

        messages = [SystemMessage(content=system_message), HumanMessage(content=initial_message)]
        self.iteration_memory.chat_memory.add_message(SystemMessage(content=system_message))
        self.iteration_memory.chat_memory.add_message(HumanMessage(content=initial_message))

        reasoning_steps = []
        final_response = ""

        for step in range(1, self.max_reasoning_steps + 1):
            self._log("INFO", f"\nReasoning step {step}/{self.max_reasoning_steps}...")
            try:
                response = self.llm.invoke(messages)
                text = response.content
            except Exception as e:
                self._log("ERROR", f"Error in LLM call at step {step}: {e}")
                if step == 1:
                    # First step failure - use default
                    self._log("INFO", "Using default option 2 due to LLM failure")
                    return 2, "LLM error: using default balanced allocation", []
                else:
                    # Use last valid response
                    self._log("INFO", "Using last valid reasoning step")
                    break
            
            self._log("INFO", f"\n{text}\n")

            reasoning_steps.append((f"Step {step}", text))
            messages.append(AIMessage(content=text))
            self.iteration_memory.chat_memory.add_message(AIMessage(content=text))

            if "SELECTED_OPTION:" in text.upper():
                final_response = text
                self._log("INFO", f"Decision reached at step {step}")
                break

            if step < self.max_reasoning_steps:
                messages.append(HumanMessage(content="Continue reasoning or make final decision."))
                self.iteration_memory.chat_memory.add_message(
                    HumanMessage(content="Continue reasoning or make final decision.")
                )
            else:
                messages.append(
                    HumanMessage(
                        content="Make final decision now:\nSELECTED_OPTION: <0-5>\nREASONING: <justification>"
                    )
                )
                self.iteration_memory.chat_memory.add_message(
                    HumanMessage(
                        content="Make final decision now:\nSELECTED_OPTION: <0-5>\nREASONING: <justification>"
                    )
                )
                response = self.llm.invoke(messages)
                final_response = response.content
                reasoning_steps.append(("Final Decision", final_response))
                self.iteration_memory.chat_memory.add_message(AIMessage(content=final_response))

        selected_idx = self._parse_selection(final_response)
        selected_idx = self._validate_feasibility(
            selected_idx, allocation_results, budget_remaining, time_remaining,
            cost_per_point, time_per_point, max_batch_size
        )

        selected_batch = allocation_results.options[selected_idx]
        selected_points = []
        if len(selected_batch.exploitation_points) > 0:
            selected_points.append(selected_batch.exploitation_points)
        if len(selected_batch.exploration_points) > 0:
            selected_points.append(selected_batch.exploration_points)

        full_reasoning = "\n\n".join([f"**{t}**\n{c}" for t, c in reasoning_steps])

        iteration_summary = self._summarize_iteration(
            iteration, budget_remaining, time_remaining,
            cost_per_point, time_per_point,
            X_history, Y_history, selected_idx, selected_batch,
            full_reasoning, score_fn
        )
        self.global_memory.append(iteration_summary)

        iteration_log = {
            "iteration": iteration,
            "budget_remaining": budget_remaining,
            "time_remaining": time_remaining,
            "reasoning_steps": [{"step": t, "content": c} for t, c in reasoning_steps],
            "selected_option": selected_idx,
            "selected_batch_size": selected_batch.num_exploitation,
            "summary": iteration_summary,
        }
        self._save_iteration_log(iteration, iteration_log)

        self._log(
            "INFO",
            f"✓ Selected Option {selected_idx}: "
            f"{selected_batch.num_exploitation} exploit + {max_batch_size - selected_batch.num_exploitation} explore"
        )

        return selected_idx, full_reasoning, selected_points

    # -------------------------- Analysis Helpers -------------------------- #

    def _parse_selection(self, response: str) -> int:
        if "SELECTED_OPTION:" in response.upper():
            try:
                line = [l for l in response.split("\n") if "SELECTED_OPTION:" in l.upper()][0]
                option_str = line.split(":")[1].strip()
                for ch in option_str:
                    if ch.isdigit():
                        opt = int(ch)
                        if 0 <= opt <= 5:
                            return opt
            except Exception:
                pass
        self._log("WARNING", "Could not parse selection, defaulting to option 2 (balanced)")
        return 2

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
        batch = allocation_results.options[selected_idx]
        total_points = batch.num_exploitation + (max_batch - batch.num_exploitation)
        cost = total_points * cost_per_point
        if cost <= budget and time_per_point <= time and total_points <= max_batch:
            return selected_idx

        self._log("WARNING", f"Option {selected_idx} infeasible, searching alternatives...")
        for i in range(5, -1, -1):
            b = allocation_results.options[i]
            total = b.total_batch_size
            if total * cost_per_point <= budget and time_per_point <= time and total <= max_batch:
                self._log("INFO", f"Using feasible alternative: Option {i}")
                return i

        self._log("WARNING", "No feasible options, defaulting to Option 0")
        return 0

    def _analyze_current_state_distilled(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        budget: float,
        time: float,
        cost_per_point: float,
        time_per_point: float,
        score_fn: Callable[[np.ndarray], np.ndarray],
    ) -> str:
        n = len(X)
        if n == 0:
            return """## Key Decision Criteria
**Status**: No evaluation history available yet.
This is the initial exploration phase."""

        scores = score_fn(Y)
        best_idx = int(np.argmax(scores))

        max_iterations = int(time / time_per_point)
        max_affordable_points = int(budget / cost_per_point)
        limiting_factor = "time" if max_iterations * 5 < max_affordable_points else "budget"

        improvement_status = "N/A"
        if n >= 6:
            recent_best = np.max(scores[-3:])
            earlier_best = np.max(scores[-6:-3])
            if earlier_best > 1e-10:
                improvement_pct = (recent_best - earlier_best) / earlier_best * 100
                if improvement_pct > 5:
                    improvement_status = f"Accelerating (+{improvement_pct:.1f}%)"
                elif improvement_pct > 0:
                    improvement_status = f"Steady (+{improvement_pct:.1f}%)"
                else:
                    improvement_status = f"Plateaued ({improvement_pct:+.1f}%)"

        best_x = X[best_idx]
        best_y = Y[best_idx]

        coords_str = ", ".join([f"x_{j}={best_x[j]:.3f}" for j in range(best_x.shape[0])])

        analysis = f"""## Key Decision Criteria

**1. Runway Assessment**:
   - Approx. iterations remaining: ~{max_iterations}
   - Approx. affordable points: ~{max_affordable_points}
   - Limiting factor: {limiting_factor}

**2. Progress Momentum**:
   - Total points: {n}
   - Best score: {scores[best_idx]:.3f}
   - Trend (last 3 vs previous 3): {improvement_status}

**3. Best Observed Point**:
   - {self.obj1_name}: {best_y[0]:.4g}
   - {self.obj2_name}: {best_y[1]:.4g}
   - Coordinates: {coords_str}"""

        return analysis

    def _describe_allocation_results(
        self,
        allocation_results: Any,
        cost_per_point: float,
        time_per_point: float,
        budget: float,
        time: float,
        max_batch: int,
    ) -> str:
        desc = "## Allocation Options\n\n"
        for i, batch in enumerate(allocation_results.options):
            n_exploit = batch.num_exploitation
            n_explore = batch.total_batch_size - n_exploit if hasattr(batch, 'total_batch_size') else (max_batch - n_exploit)
            total_points = n_exploit + n_explore
            cost = total_points * cost_per_point
            feasible = cost <= budget and time_per_point <= time and total_points <= max_batch

            desc += (
                f"**Option {i}**: {n_exploit} exploit + {n_explore} explore | "
                f"Cost: ${cost:.0f} | EHVI: {batch.hypervolume_improvement:.4f} | "
                f"Entropy: {batch.information_gain:.2f} | "
                f"{'✓ Feasible' if feasible else '✗ Infeasible'}\n"
            )

        return desc

    def _summarize_iteration(
        self,
        iteration: int,
        budget_start: float,
        time_start: float,
        cost_per_point: float,
        time_per_point: float,
        X: np.ndarray,
        Y: np.ndarray,
        selected_idx: int,
        selected_batch: Any,
        reasoning: str,
        score_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Dict[str, Any]:
        total_points = selected_batch.num_exploitation + (5 - selected_batch.num_exploitation)
        budget_spent = total_points * cost_per_point
        time_spent = time_per_point

        scores = score_fn(Y)
        best_idx = int(np.argmax(scores))

        return {
            "step": "iteration_summary",
            "iteration": iteration,
            "budget_at_start": budget_start,
            "time_at_start": time_start,
            "budget_spent": budget_spent,
            "time_spent": time_spent,
            "budget_remaining": budget_start - budget_spent,
            "time_remaining": time_start - time_spent,
            "points_evaluated": total_points,
            "best_score": float(scores[best_idx]),
            "best_point": X[best_idx].tolist(),
            "best_obj1": float(Y[best_idx, 0]),
            "best_obj2": float(Y[best_idx, 1]),
            "selected_option": selected_idx,
            "decision_summary": f"{selected_batch.num_exploitation} exploit + {5 - selected_batch.num_exploitation} explore",
            "reasoning_summary": reasoning[:500] + "..." if len(reasoning) > 500 else reasoning,
        }
    
    def cleanup(self):
        """Cleanup resources at end of optimization."""
        self.save_global_memory()
        self._log("INFO", "Agent cleanup completed")