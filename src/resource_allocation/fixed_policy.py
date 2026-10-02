# resource_allocation/fixed_policy.py
"""
Fixed policy implementations for BO comparison baselines.

Includes:
- FixedMixedPolicy: phase-based DecisionMaker (explore 25% → balanced 50% → exploit 25%,
  driven by resource fraction remaining). β=2.0 throughout. No LLM.
- FixedBetaExplorePolicy: pure qUCB at a configurable fixed beta. Use for qUCB_B2/5/10/100.
- FixedExploitPolicy: pure qEHVI (option 0). Use as a DecisionMaker-style qEHVI baseline.
- PureExploitation / PureExploration / BalancedPolicy: legacy helpers (old select_points
  interface — kept for backwards compat with old-style runners).
"""

import numpy as np
from typing import List, Optional, Tuple

from resource_allocation.decision_maker import DecisionMaker
from resource_allocation.decision_state import DecisionState


class FixedMixedPolicy(DecisionMaker):
    """
    Phase-based policy for comparison baselines. No LLM calls.

    Three phases driven by fraction of resources (budget/time) remaining:
      Explore   (remaining > explore_threshold=0.75) : pure qUCB (last option)
      Balanced  (exploit_threshold < remaining ≤ 0.75): middle option
      Exploit   (remaining ≤ exploit_threshold=0.25) : pure qEHVI (option 0)

    Endgame override (remaining ≤ endgame_fraction=0.20) forces option 0.
    All phases use beta = 2.0 (default exploration beta).
    """

    def __init__(
        self,
        beta: float = 2.0,
        explore_threshold: float = 0.75,
        exploit_threshold: float = 0.25,
        endgame_fraction:  float = 0.20,
    ):
        self.beta              = beta
        self.explore_threshold = explore_threshold
        self.exploit_threshold = exploit_threshold
        self.endgame_fraction  = endgame_fraction

    def _resource_fraction(self, state: DecisionState) -> Optional[float]:
        fracs = []
        ib = getattr(state, "initial_budget", None)
        it = getattr(state, "initial_time",   None)
        if ib and ib > 0:
            fracs.append(state.budget_remaining / ib)
        if it and it > 0:
            fracs.append(state.time_remaining / it)
        return min(fracs) if fracs else None

    def _is_endgame(self, state: DecisionState) -> bool:
        frac = self._resource_fraction(state)
        return frac is not None and frac <= self.endgame_fraction

    def select_beta(self, state: DecisionState) -> float:
        return self.beta

    def make_decision(
        self, state: DecisionState
    ) -> Tuple[int, str, Optional[dict]]:
        num_options = len(state.allocation_options.options)

        if self._is_endgame(state):
            option = 0
            phase  = "endgame"
        else:
            frac = self._resource_fraction(state)
            if frac is None or frac > self.explore_threshold:
                option = num_options - 1          # pure qUCB
                phase  = "explore"
            elif frac > self.exploit_threshold:
                option = (num_options - 1) // 2  # balanced (lower-middle)
                phase  = "balanced"
            else:
                option = 0                        # pure qEHVI
                phase  = "exploit"

        reasoning = f"FixedMixedPolicy ({phase}): option {option}, β={self.beta:.1f}"
        return option, reasoning, None


class FixedBetaExplorePolicy(DecisionMaker):
    """
    Pure exploration at a fixed beta. Always selects the last option (all qUCB).
    Use to instantiate qUCB_B2, qUCB_B5, qUCB_B10, qUCB_B100 baselines.
    """

    def __init__(self, beta: float):
        self._beta = beta

    def select_beta(self, state: DecisionState) -> float:
        return self._beta

    def make_decision(
        self, state: DecisionState
    ) -> Tuple[int, str, Optional[dict]]:
        num_options = len(state.allocation_options.options)
        option = num_options - 1  # pure qUCB
        reasoning = f"FixedBetaExplorePolicy: pure qUCB, β={self._beta:.1f}"
        return option, reasoning, None


class FixedExploitPolicy(DecisionMaker):
    """
    Pure exploitation. Always selects option 0 (all qEHVI, no qUCB).
    DecisionMaker-interface equivalent of the legacy PureExploitation.
    """

    def __init__(self, beta: float = 2.0):
        self._beta = beta

    def select_beta(self, state: DecisionState) -> float:
        return self._beta

    def make_decision(
        self, state: DecisionState
    ) -> Tuple[int, str, Optional[dict]]:
        reasoning = f"FixedExploitPolicy: pure qEHVI, β={self._beta:.1f}"
        return 0, reasoning, None


# ---------------------------------------------------------------------------
# Legacy helpers (old select_points interface — kept for backwards compat)
# ---------------------------------------------------------------------------

class PureExploitation:
    """Pure exploitation policy: Select all 5 points from exploitation acquisition."""
    
    def __init__(self):
        self.name = "Pure_Exploitation"
    
    def select_points(self, allocation_results) -> Tuple[np.ndarray, str]:
        """
        Select all 5 points from exploitation (option 0).
        
        Args:
            allocation_results: AllocationResults object with options
        
        Returns:
            points: (5, d) array of selected points
            reasoning: Explanation string
        """
        # Option 0 is always full exploitation (5 exploit, 0 explore)
        option = allocation_results.options[0]
        
        # Get all exploitation points
        points = option.exploitation_points
        
        # Get metadata for informative reasoning
        metadata = allocation_results.metadata or {}
        exploit_beta = metadata.get('exploitation_beta', 'N/A')
        
        reasoning = (
            f"Pure Exploitation Strategy: Selected all {option.num_exploitation} points "
            f"from exploitation acquisition (beta={exploit_beta}, full greedy optimization)."
        )
        
        return points, reasoning


class PureExploration:
    """Pure exploration policy: Select all 5 points from exploration acquisition."""
    
    def __init__(self):
        self.name = "Pure_Exploration"
    
    def select_points(self, allocation_results) -> Tuple[np.ndarray, str]:
        """
        Select all 5 points from exploration (option 5).
        
        Args:
            allocation_results: AllocationResults object with options
        
        Returns:
            points: (5, d) array of selected points
            reasoning: Explanation string
        """
        # Option 5 is always full exploration (0 exploit, 5 explore)
        option = allocation_results.options[-1]
        
        # Get all exploration points
        points = option.exploration_points
        
        # Get metadata for informative reasoning
        metadata = allocation_results.metadata or {}
        explore_beta = metadata.get('exploration_beta', 'N/A')
        
        reasoning = (
            f"Pure Exploration Strategy: Selected all {option.num_exploration} points "
            f"from exploration acquisition (beta={explore_beta}, full diverse sampling)."
        )
        
        return points, reasoning


class BalancedPolicy:
    """Balanced policy: Select mix of exploitation and exploration (option 2)."""
    
    def __init__(self):
        self.name = "Balanced_Policy"
    
    def select_points(self, allocation_results) -> Tuple[np.ndarray, str]:
        """
        Select balanced mix of exploitation and exploration (option 2).
        
        Args:
            allocation_results: AllocationResults object with options
        
        Returns:
            points: (5, d) array of selected points
            reasoning: Explanation string
        """
        # Option 2 is balanced for batch size 5 (typically 2 exploit, 3 explore)
        option = allocation_results.options[2]
        
        # Get all points (exploitation + exploration)
        points = option.all_points
        
        # Get metadata for informative reasoning
        metadata = allocation_results.metadata or {}
        exploit_beta = metadata.get('exploitation_beta', 'N/A')
        explore_beta = metadata.get('exploration_beta', 'N/A')
        
        reasoning = (
            f"Balanced Strategy: Selected {option.num_exploitation} exploitation points "
            f"(beta={exploit_beta}) and {option.num_exploration} exploration points "
            f"(beta={explore_beta}) for balanced optimization."
        )
        
        return points, reasoning