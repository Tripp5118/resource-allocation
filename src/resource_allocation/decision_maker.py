# resource_allocation/decision_maker.py
"""Decision maker interface and implementations."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import numpy as np

from resource_allocation.decision_state import DecisionState


class DecisionMaker(ABC):
    """Abstract interface for resource allocation decisions."""
    
    @abstractmethod
    def make_decision(self, state: DecisionState) -> Tuple[int, str, Optional[Dict]]:
        """
        Make resource allocation decision.

        Args:
            state: Complete decision state

        Returns:
            selected_option: int (index into allocation_options)
            reasoning: str (explanation of decision)
            updated_beliefs: Optional[Dict] (new belief state if applicable, else None)
        """
        pass

    def select_beta(self, state: DecisionState) -> float:
        """
        Select qUCB beta before computing allocation options.
        Default: return 2.0 (no-op for fixed-policy baselines).
        Override in LLM decision makers.
        """
        return 2.0

    def cleanup(self):
        """Optional cleanup (e.g., save logs for LLM agents)."""
        pass


class FixedPolicyDecisionMaker(DecisionMaker):
    """Fixed policy that always selects same strategy."""
    
    def __init__(self, policy_type: str):
        """
        Args:
            policy_type: "pure_exploit" (always option 0) or "pure_explore" (always last option)
        """
        self.policy_type = policy_type
        if policy_type == "pure_exploit":
            self.reasoning_template = "Fixed policy: Pure exploitation (qEHVI only)"
        elif policy_type == "pure_explore":
            self.reasoning_template = "Fixed policy: Pure exploration (qUCB only)"
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def make_decision(self, state: DecisionState) -> Tuple[int, str, Optional[Dict]]:
        # Handle dynamic number of options
        num_options = len(state.allocation_options.options)
        
        if self.policy_type == "pure_exploit":
            option = 0  # Always first (most exploitation)
        else:  # pure_explore
            option = num_options - 1  # Always last (most exploration)
        
        return option, self.reasoning_template, None


class BalancedPolicyDecisionMaker(DecisionMaker):
    """Always selects balanced option (middle option)."""
    
    def __init__(self, option_idx: Optional[int] = None):
        """
        Args:
            option_idx: Fixed option index, or None to use middle option
        """
        self.fixed_option = option_idx
    
    def make_decision(self, state: DecisionState) -> Tuple[int, str, Optional[Dict]]:
        num_options = len(state.allocation_options.options)
        
        if self.fixed_option is not None:
            option = min(self.fixed_option, num_options - 1)  # Clamp to valid range
        else:
            option = num_options // 2  # Middle option
        
        reasoning = f"Fixed balanced policy: Option {option}"
        return option, reasoning, None


# Placeholder for LLM decision maker (will implement in later stages)
class LLMDecisionMaker(DecisionMaker):
    """Placeholder - will be implemented in Phase 3."""
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LLMDecisionMaker will be implemented in Phase 3")
    
    def make_decision(self, state: DecisionState) -> Tuple[int, str, Optional[Dict]]:
        raise NotImplementedError("LLMDecisionMaker will be implemented in Phase 3")