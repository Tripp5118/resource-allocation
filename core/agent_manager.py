"""LLM Agent Manager for Bayesian Optimization with event-driven resource management."""
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory


# ============================================================================
#  EVENT SYSTEM
# ============================================================================

@dataclass
class ResourceEvent:
    """Event that modifies budget or time constraints."""
    iteration: int  # When to trigger (before this iteration)
    event_type: str  # 'budget_change', 'time_change', 'cost_change', 'custom'
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
        """Add an event to the schedule."""
        self.events.append(event)
        # Sort by iteration
        self.events.sort(key=lambda e: e.iteration)
    
    def add_budget_cut(self, iteration: int, percentage: float, description: Optional[str] = None):
        """Convenience: Add a budget cut event."""
        desc = description or f"Budget cut by {percentage*100:.0f}% at iteration {iteration}"
        event = ResourceEvent(
            iteration=iteration,
            event_type='budget_change',
            description=desc,
            modifier=lambda x: x * (1 - percentage)
        )
        self.add_event(event)
    
    def add_time_extension(self, iteration: int, weeks: float, description: Optional[str] = None):
        """Convenience: Add extra time."""
        desc = description or f"Time extended by {weeks} weeks at iteration {iteration}"
        event = ResourceEvent(
            iteration=iteration,
            event_type='time_change',
            description=desc,
            modifier=lambda x: x + weeks
        )
        self.add_event(event)
    
    def add_cost_increase(self, iteration: int, multiplier: float, description: Optional[str] = None):
        """Convenience: Increase cost per point."""
        desc = description or f"Cost per point increased by {multiplier}x at iteration {iteration}"
        event = ResourceEvent(
            iteration=iteration,
            event_type='cost_change',
            description=desc,
            modifier=lambda x: x * multiplier
        )
        self.add_event(event)
    
    def get_events_for_iteration(self, iteration: int) -> List[ResourceEvent]:
        """Get all events scheduled for this iteration."""
        return [e for e in self.events if e.iteration == iteration]
    
    def trigger_events(self, iteration: int, budget: float, time: float, 
                      cost_per_point: float) -> Tuple[float, float, float, List[str]]:
        """Apply all events for this iteration and return updated values."""
        events = self.get_events_for_iteration(iteration)
        messages = []
        
        for event in events:
            if event.event_type == 'budget_change':
                old_budget = budget
                budget = event.apply(budget)
                messages.append(f"⚠️  {event.description}: ${old_budget:.2f} → ${budget:.2f}")
            elif event.event_type == 'time_change':
                old_time = time
                time = event.apply(time)
                messages.append(f"⚠️  {event.description}: {old_time:.1f} → {time:.1f} weeks")
            elif event.event_type == 'cost_change':
                old_cost = cost_per_point
                cost_per_point = event.apply(cost_per_point)
                messages.append(f"⚠️  {event.description}: ${old_cost:.2f} → ${cost_per_point:.2f} per point")
            
            # Log triggered event
            self.triggered_events.append({
                'iteration': iteration,
                'event_type': event.event_type,
                'description': event.description,
                'timestamp': iteration
            })
        
        return budget, time, cost_per_point, messages


# ============================================================================
#  MAIN AGENT CLASS
# ============================================================================

class BOAgent:
    """LangChain-based Agent for Bayesian Optimization with resource management."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 log_dir: Optional[str] = None,
                 max_reasoning_steps: int = 10,
                 temperature: float = 0.7):
        """Initialize the BO Agent.
        
        Args:
            api_key: OpenAI API key (or None to use environment variable)
            model: LLM model name
            log_dir: Directory to save agent logs
            max_reasoning_steps: Maximum reasoning iterations per decision
            temperature: LLM temperature for generation
        """
        # LLM setup
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.max_reasoning_steps = max_reasoning_steps
        
        # Memory
        self.global_memory: List[Dict[str, Any]] = []
        self.iteration_memory: Optional[ConversationBufferMemory] = None
        
        # Event management
        self.event_manager = EventManager()
        
        # Logging
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # ========================================================================
    #  EVENT MANAGEMENT
    # ========================================================================
    
    def add_event(self, event: ResourceEvent):
        """Add a resource event."""
        self.event_manager.add_event(event)
    
    def add_budget_cut(self, iteration: int, percentage: float, description: Optional[str] = None):
        """Add a budget cut event at specified iteration."""
        self.event_manager.add_budget_cut(iteration, percentage, description)
    
    def add_time_extension(self, iteration: int, weeks: float, description: Optional[str] = None):
        """Add time extension event."""
        self.event_manager.add_time_extension(iteration, weeks, description)
    
    def add_cost_increase(self, iteration: int, multiplier: float, description: Optional[str] = None):
        """Add cost increase event."""
        self.event_manager.add_cost_increase(iteration, multiplier, description)
    
    def process_iteration_events(self, iteration: int, budget: float, time: float,
                                 cost_per_point: float) -> Tuple[float, float, float, List[str]]:
        """Process any events scheduled for this iteration."""
        return self.event_manager.trigger_events(iteration, budget, time, cost_per_point)
    
    # ========================================================================
    #  MEMORY & LOGGING
    # ========================================================================
    
    def _create_iteration_memory(self) -> ConversationBufferMemory:
        """Create fresh conversation memory for an iteration."""
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def save_global_memory(self):
        """Save global memory and events to disk."""
        if self.log_dir:
            # Save memory
            filepath = os.path.join(self.log_dir, "agent_global_memory.json")
            with open(filepath, 'w') as f:
                json.dump(self.global_memory, indent=2, fp=f)
            
            # Save events
            events_filepath = os.path.join(self.log_dir, "agent_events.json")
            events_data = {
                'scheduled_events': [
                    {
                        'iteration': e.iteration,
                        'type': e.event_type,
                        'description': e.description
                    } for e in self.event_manager.events
                ],
                'triggered_events': self.event_manager.triggered_events
            }
            with open(events_filepath, 'w') as f:
                json.dump(events_data, indent=2, fp=f)
    
    def _save_iteration_log(self, iteration: int, content: Dict[str, Any]):
        """Save iteration-specific log."""
        if self.log_dir:
            filepath = os.path.join(self.log_dir, f"agent_iteration_{iteration}.json")
            with open(filepath, 'w') as f:
                json.dump(content, indent=2, fp=f)
    
    def _build_global_context(self) -> str:
        """Build summary of past decisions for context."""
        if not self.global_memory:
            return "This is the first decision in the optimization campaign."
        
        context = "## Optimization History\n\n"
        
        # GP configuration
        gp_config = next((m for m in self.global_memory if m.get("step") == "gp_configuration"), None)
        if gp_config:
            context += f"**GP Kernel**: {gp_config.get('kernel_decision', 'default')}\n\n"
        
        # Recent iterations (last 3)
        summaries = [m for m in self.global_memory if m.get("step") == "iteration_summary"]
        if summaries:
            context += "**Recent Iterations**:\n"
            for summary in summaries[-3:]:
                iter_num = summary.get('iteration', '?')
                context += f"\nIteration {iter_num}:\n"
                context += f"  - Best K/CTE ratio: {summary.get('best_ratio', 0):.3f}\n"
                context += f"  - Decision: Option {summary.get('selected_option')} "
                context += f"({summary.get('decision_summary', 'N/A')})\n"
                
                # Show improvement trend
                if len(summaries) >= 2:
                    prev_ratio = summaries[-2].get('best_ratio', 0)
                    curr_ratio = summary.get('best_ratio', 0)
                    improvement = ((curr_ratio - prev_ratio) / prev_ratio * 100) if prev_ratio > 0 else 0
                    context += f"  - Improvement: {improvement:+.1f}%\n"
        
        # Event history
        if self.event_manager.triggered_events:
            context += "\n**Resource Events**:\n"
            for event in self.event_manager.triggered_events[-3:]:
                context += f"  - Iter {event['iteration']}: {event['description']}\n"
        
        return context
    
    # ========================================================================
    #  GP KERNEL CONFIGURATION
    # ========================================================================
    
    def configure_gp_kernel(self, use_priors: bool) -> Tuple[Optional[str], str]:
        """Agent decides on GP kernel configuration.
        
        Returns:
            (kernel_code, reasoning): Python code for custom kernel (or None) and reasoning
        """
        print("\n" + "="*80)
        print("[Agent] Configuring GP Kernel")
        print("="*80)
        
        system_prompt = """You are an expert in Bayesian Optimization and Gaussian Processes for materials science.

You're configuring a GP model for High Entropy Alloy (HEA) optimization with two objectives:
1. Minimize CTE (Coefficient of Thermal Expansion)
2. Maximize K (Thermal Conductivity)

Input space: 5D compositions (Fe, Co, Cr, Ni, V) summing to 1.0, each in [0.1, 0.4]
Model: MultiTaskGP (models both outputs jointly with correlation)

Decide: Should we use a custom kernel or the default BoTorch kernel?"""
        
        user_prompt = f"""Starting new optimization run for HEA design.

Prior Data Available: {"Yes" if use_priors else "No"}

Consider:
1. What kernel properties matter for composition space?
2. Does default (RBF/Matern) handle simplex constraints well?
3. Would a custom kernel provide meaningful benefits?
4. Trade-offs between custom and default?

You have up to {self.max_reasoning_steps} reasoning steps.

When ready, respond:
REASONING: <your analysis>
KERNEL: <USE_DEFAULT or Python code>

Begin your analysis."""
        
        # Iterative reasoning
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        reasoning_steps = []
        final_response = ""
        
        for step in range(1, self.max_reasoning_steps + 1):
            print(f"\n[Agent] Reasoning step {step}/{self.max_reasoning_steps}...")
            response = self.llm.invoke(messages)
            response_text = response.content
            print(f"\n{response_text}\n")
            
            reasoning_steps.append((f"Step {step}", response_text))
            messages.append(AIMessage(content=response_text))
            
            if "KERNEL:" in response_text.upper():
                final_response = response_text
                print(f"[Agent] Decision reached at step {step}")
                break
            
            if step < self.max_reasoning_steps:
                messages.append(HumanMessage(
                    content="Continue reasoning or make final decision in specified format."
                ))
            else:
                # Force decision
                messages.append(HumanMessage(
                    content="Make your final decision now:\nREASONING: <analysis>\nKERNEL: <USE_DEFAULT or code>"
                ))
                response = self.llm.invoke(messages)
                final_response = response.content
                reasoning_steps.append(("Final Decision", final_response))
        
        # Parse response
        reasoning, kernel_code = self._parse_kernel_decision(final_response)
        full_reasoning = "\n\n".join([f"**{title}**\n{content}" for title, content in reasoning_steps])
        
        # Store in global memory
        config_summary = {
            "step": "gp_configuration",
            "use_priors": use_priors,
            "kernel_decision": "custom" if kernel_code else "default",
            "reasoning": reasoning,
            "reasoning_steps": len(reasoning_steps),
            "full_reasoning": full_reasoning
        }
        self.global_memory.append(config_summary)
        
        if self.log_dir:
            filepath = os.path.join(self.log_dir, "agent_gp_configuration.json")
            with open(filepath, 'w') as f:
                json.dump(config_summary, indent=2, fp=f)
        
        return kernel_code, full_reasoning
    
    def _parse_kernel_decision(self, response: str) -> Tuple[str, Optional[str]]:
        """Parse kernel decision from agent response."""
        reasoning = ""
        kernel_code = None
        
        if "REASONING:" in response:
            parts = response.split("KERNEL:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            
            if len(parts) > 1:
                kernel_part = parts[1].strip()
                if "USE_DEFAULT" not in kernel_part.upper():
                    # Extract code block
                    if "```python" in kernel_part:
                        kernel_code = kernel_part.split("```python")[1].split("```")[0].strip()
                    elif "```" in kernel_part:
                        kernel_code = kernel_part.split("```")[0].strip()
                    else:
                        kernel_code = kernel_part
        
        return reasoning, kernel_code
    
    # ========================================================================
    #  RESOURCE ALLOCATION DECISION (MAIN FUNCTION)
    # ========================================================================
    
    def select_resource_allocation(self,
                                   iteration: int,
                                   budget_remaining: float,
                                   time_remaining: float,
                                   cost_per_point: float,
                                   time_per_point: float,
                                   X_history: np.ndarray,
                                   Y_history: np.ndarray,
                                   allocation_options: Any,
                                   max_batch_size: int = 5) -> Tuple[int, str, List[np.ndarray]]:
        """Agent selects resource allocation strategy.
        
        Args:
            iteration: Current iteration number
            budget_remaining: Remaining budget ($)
            time_remaining: Remaining time (weeks)
            cost_per_point: Cost per experimental point ($)
            time_per_point: Time per iteration (weeks, typically 1)
            X_history: Historical compositions (n, 5)
            Y_history: Historical [CTE, K] values (n, 2)
            allocation_options: AcquisitionData with 6 allocation options
            max_batch_size: Maximum batch size
        
        Returns:
            (selected_idx, reasoning, selected_points):
                - Index into allocation_options.qehvi_batches (0-5)
                - Full reasoning text
                - List of point arrays to evaluate [exploit_points, explore_points]
        """
        print("\n" + "="*80)
        print(f"[Agent] Iteration {iteration}: Resource Allocation Decision")
        print("="*80)
        
        # Create fresh iteration memory
        self.iteration_memory = self._create_iteration_memory()
        
        # Build context
        global_context = self._build_global_context()
        
        # Analyze current state (distilled version)
        analysis = self._analyze_current_state_distilled(
            X_history, Y_history, budget_remaining, time_remaining,
            cost_per_point, time_per_point
        )
        print(f"\n[Agent Analysis]:\n{analysis}\n")
        
        # Describe options
        options_desc = self._describe_allocation_options(
            allocation_options, cost_per_point, time_per_point,
            budget_remaining, time_remaining, max_batch_size
        )
        
        # Check for upcoming events
        upcoming_events = self.event_manager.get_events_for_iteration(iteration + 1)
        event_warning = ""
        if upcoming_events:
            event_warning = "\n⚠️  **UPCOMING EVENTS NEXT ITERATION**:\n"
            for event in upcoming_events:
                event_warning += f"  - {event.description}\n"
        
        # System message
        system_message = f"""You are an expert AI managing a Bayesian Optimization campaign for High Entropy Alloy design.

**Current Status**:
- Iteration: {iteration}
- Budget: ${budget_remaining:.2f}
- Time: {time_remaining:.1f} weeks

**Goals**:
- Minimize CTE (Coefficient of Thermal Expansion)
- Maximize K (Thermal Conductivity)
- Find Pareto-optimal solutions

**Important**: Each iteration takes 1 week regardless of batch size (parallel synthesis).

{global_context}{event_warning}

When ready to decide, respond:
SELECTED_OPTION: <0-5>
REASONING: <brief justification>"""
        
        # Initial user message
        initial_message = f"""Make a resource allocation decision for this iteration.

{analysis}

{options_desc}

**Decision Framework**:
1. Iterations remaining: ~{int(time_remaining / time_per_point)} iterations, ~{int(budget_remaining / cost_per_point)} affordable points
2. Recent improvement: {"Improving" if self._is_improving(Y_history) else "Plateaued"}
3. EHVI comparison: Option {self._best_ehvi_option(allocation_options)} has highest EHVI

**Strategy**:
- If last 1-2 iterations + improving → Exploit (option 0-1)
- If last 1-2 iterations + plateaued → Explore (option 4-5, Hail Mary)
- If 3+ iterations + improving → Balanced (option 2-3)
- If 3+ iterations + plateaued → Explore more (option 3-4)

You have up to {self.max_reasoning_steps} reasoning steps.

When ready:
SELECTED_OPTION: <0-5>
REASONING: <justification>

Begin analysis."""
        
        # Iterative reasoning
        messages = [SystemMessage(content=system_message), HumanMessage(content=initial_message)]
        self.iteration_memory.chat_memory.add_message(SystemMessage(content=system_message))
        self.iteration_memory.chat_memory.add_message(HumanMessage(content=initial_message))
        
        reasoning_steps = []
        final_response = ""
        
        for step in range(1, self.max_reasoning_steps + 1):
            print(f"\n[Agent] Reasoning step {step}/{self.max_reasoning_steps}...")
            response = self.llm.invoke(messages)
            response_text = response.content
            print(f"\n{response_text}\n")
            
            reasoning_steps.append((f"Step {step}", response_text))
            messages.append(AIMessage(content=response_text))
            self.iteration_memory.chat_memory.add_message(AIMessage(content=response_text))
            
            if "SELECTED_OPTION:" in response_text.upper():
                final_response = response_text
                print(f"[Agent] Decision reached at step {step}")
                break
            
            if step < self.max_reasoning_steps:
                messages.append(HumanMessage(
                    content="Continue reasoning or make final decision."
                ))
                self.iteration_memory.chat_memory.add_message(HumanMessage(
                    content="Continue reasoning or make final decision."
                ))
            else:
                # Force decision
                messages.append(HumanMessage(
                    content="Make final decision now:\nSELECTED_OPTION: <0-5>\nREASONING: <justification>"
                ))
                self.iteration_memory.chat_memory.add_message(HumanMessage(
                    content="Make final decision now:\nSELECTED_OPTION: <0-5>\nREASONING: <justification>"
                ))
                response = self.llm.invoke(messages)
                final_response = response.content
                reasoning_steps.append(("Final Decision", final_response))
                self.iteration_memory.chat_memory.add_message(AIMessage(content=final_response))
        
        # Parse and validate decision
        selected_idx = self._parse_selection(final_response)
        selected_idx = self._validate_feasibility(
            selected_idx, allocation_options, budget_remaining,
            time_remaining, cost_per_point, time_per_point
        )
        
        # Get selected points
        selected_batch = allocation_options.qehvi_batches[selected_idx]
        selected_points = []
        if len(selected_batch.points) > 0:
            selected_points.append(selected_batch.points)
        if len(selected_batch.explore_points) > 0:
            selected_points.append(selected_batch.explore_points)
        
        # Compile reasoning
        full_reasoning = "\n\n".join([f"**{title}**\n{content}" for title, content in reasoning_steps])
        
        # Summarize for memory
        iteration_summary = self._summarize_iteration(
            iteration, budget_remaining, time_remaining, cost_per_point,
            time_per_point, X_history, Y_history, selected_idx,
            selected_batch, full_reasoning
        )
        self.global_memory.append(iteration_summary)
        
        # Save log
        iteration_log = {
            "iteration": iteration,
            "budget_remaining": budget_remaining,
            "time_remaining": time_remaining,
            "analysis": analysis,
            "reasoning_steps": [{"step": t, "content": c} for t, c in reasoning_steps],
            "selected_option": selected_idx,
            "selected_batch_size": selected_batch.batch_size,
            "summary": iteration_summary
        }
        self._save_iteration_log(iteration, iteration_log)
        
        print(f"\n[Agent] ✓ Selected Option {selected_idx}: "
              f"{selected_batch.batch_size} exploit + {5-selected_batch.batch_size} explore")
        
        return selected_idx, full_reasoning, selected_points
    
    # ========================================================================
    #  ANALYSIS HELPERS (DISTILLED VERSION)
    # ========================================================================
    
    def _analyze_current_state_distilled(self, X: np.ndarray, Y: np.ndarray,
                                         budget: float, time: float,
                                         cost_per_point: float, time_per_point: float) -> str:
        """Distilled analysis focusing on key decision criteria."""
        n_points = len(X)
        cte_vals = Y[:, 0]
        k_vals = Y[:, 1]
        ratios = k_vals / (np.abs(cte_vals) + 1e-12)
        best_idx = np.argmax(ratios)
        
        # Calculate runway
        max_iterations = int(time / time_per_point)
        max_affordable_points = int(budget / cost_per_point)
        limiting_factor = "time" if max_iterations * 5 < max_affordable_points else "budget"
        
        # Calculate improvement trend
        improvement_status = "N/A"
        if n_points >= 6:
            recent_best = np.max(ratios[-3:])
            earlier_best = np.max(ratios[-6:-3])
            if earlier_best > 0:
                improvement_pct = ((recent_best - earlier_best) / earlier_best * 100)
                if improvement_pct > 5:
                    improvement_status = f"Accelerating (+{improvement_pct:.1f}%)"
                elif improvement_pct > 0:
                    improvement_status = f"Steady (+{improvement_pct:.1f}%)"
                else:
                    improvement_status = f"Plateaued ({improvement_pct:+.1f}%)"
        
        analysis = f"""## Key Decision Criteria

**1. Runway Assessment**:
   - Iterations remaining: ~{max_iterations} iterations
   - Points affordable: ~{max_affordable_points} points
   - Limiting factor: {limiting_factor}

**2. Progress Momentum**:
   - Total points: {n_points}
   - Best K/CTE ratio: {ratios[best_idx]:.3f}
   - Trend (last 3 vs previous 3): {improvement_status}

**3. Best Point**:
   - CTE: {cte_vals[best_idx]:.4f}, K: {k_vals[best_idx]:.2f}
   - Composition: Fe={X[best_idx,0]:.2f}, Co={X[best_idx,1]:.2f}, Cr={X[best_idx,2]:.2f}, Ni={X[best_idx,3]:.2f}, V={X[best_idx,4]:.2f}

**Note**: Each iteration = 1 week (parallel synthesis)"""
        
        return analysis
    
    def _describe_allocation_options(self, allocation_data: Any,
                                     cost_per_point: float, time_per_point: float,
                                     budget: float, time: float, max_batch: int) -> str:
        """Describe allocation options concisely."""
        desc = "## Allocation Options\n\n"
        
        for i, batch in enumerate(allocation_data.qehvi_batches):
            n_exploit = batch.batch_size
            n_explore = 5 - n_exploit
            total_points = n_exploit + n_explore
            cost = total_points * cost_per_point
            feasible = cost <= budget and time_per_point <= time and total_points <= max_batch
            
            desc += f"**Option {i}**: {n_exploit} exploit + {n_explore} explore | "
            desc += f"Cost: ${cost:.0f} | EHVI: {batch.full_qehvi:.4f} | "
            desc += f"Entropy: {batch.full_entropy:.2f} | "
            desc += f"{'✓ Feasible' if feasible else '✗ Infeasible'}\n"
        
        return desc
    
    def _is_improving(self, Y: np.ndarray) -> bool:
        """Check if recent iterations show improvement."""
        if len(Y) < 6:
            return True  # Assume improving early on
        ratios = Y[:, 1] / (np.abs(Y[:, 0]) + 1e-12)
        recent_best = np.max(ratios[-3:])
        earlier_best = np.max(ratios[-6:-3])
        return recent_best > earlier_best * 1.01  # 1% improvement threshold
    
    def _best_ehvi_option(self, allocation_data: Any) -> int:
        """Find option with highest EHVI."""
        best_idx = 0
        best_ehvi = allocation_data.qehvi_batches[0].full_qehvi
        for i, batch in enumerate(allocation_data.qehvi_batches[1:], 1):
            if batch.full_qehvi > best_ehvi:
                best_ehvi = batch.full_qehvi
                best_idx = i
        return best_idx
    
    def _parse_selection(self, response: str) -> int:
        """Parse selected option from response."""
        if "SELECTED_OPTION:" in response.upper():
            try:
                line = [l for l in response.split('\n') if 'SELECTED_OPTION:' in l.upper()][0]
                option_str = line.split(':')[1].strip()
                for char in option_str:
                    if char.isdigit():
                        option = int(char)
                        if 0 <= option <= 5:
                            return option
            except:
                pass
        print("[Agent] Warning: Could not parse selection, defaulting to option 2 (balanced)")
        return 2
    
    def _validate_feasibility(self, selected_idx: int, allocation_data: Any,
                             budget: float, time: float,
                             cost_per_point: float, time_per_point: float) -> int:
        """Validate and potentially override selection if infeasible."""
        selected_batch = allocation_data.qehvi_batches[selected_idx]
        total_points = selected_batch.batch_size + (5 - selected_batch.batch_size)
        cost = total_points * cost_per_point
        
        if cost <= budget and time_per_point <= time:
            return selected_idx
        
        print(f"[Agent] Warning: Option {selected_idx} exceeds constraints, finding alternative...")
        
        # Find largest feasible option
        for i in range(5, -1, -1):
            batch = allocation_data.qehvi_batches[i]
            total = batch.batch_size + (5 - batch.batch_size)
            if total * cost_per_point <= budget and time_per_point <= time:
                print(f"[Agent] Using feasible alternative: Option {i}")
                return i
        
        print("[Agent] Warning: No feasible options, defaulting to Option 0")
        return 0
    
    def _compute_pareto_mask(self, Y: np.ndarray) -> np.ndarray:
        """Compute Pareto front mask (minimize CTE, maximize K)."""
        n = len(Y)
        pareto_mask = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # j dominates i if: j.CTE <= i.CTE AND j.K >= i.K AND (strictly better in at least one)
                    if (Y[j, 0] <= Y[i, 0] and Y[j, 1] >= Y[i, 1] and
                        (Y[j, 0] < Y[i, 0] or Y[j, 1] > Y[i, 1])):
                        pareto_mask[i] = False
                        break
        return pareto_mask
    
    def _summarize_iteration(self, iteration: int, budget_start: float, time_start: float,
                            cost_per_point: float, time_per_point: float,
                            X: np.ndarray, Y: np.ndarray, selected_idx: int,
                            selected_batch: Any, reasoning: str) -> Dict[str, Any]:
        """Create iteration summary for global memory."""
        total_points = selected_batch.batch_size + (5 - selected_batch.batch_size)
        budget_spent = total_points * cost_per_point
        time_spent = time_per_point
        
        ratios = Y[:, 1] / (np.abs(Y[:, 0]) + 1e-12)
        best_ratio = np.max(ratios)
        best_idx = np.argmax(ratios)
        
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
            "best_ratio": best_ratio,
            "best_composition": X[best_idx].tolist(),
            "best_cte": float(Y[best_idx, 0]),
            "best_k": float(Y[best_idx, 1]),
            "selected_option": selected_idx,
            "decision_summary": f"{selected_batch.batch_size} exploit + {5-selected_batch.batch_size} explore",
            "reasoning_summary": reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
        }