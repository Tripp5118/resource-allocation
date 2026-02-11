# core/fixed_policy.py
"""
Simple fixed policy implementations for BO comparison.

Two scenarios:
1. Pure Exploitation: All 5 points optimized for exploitation (low beta/greedy)
2. Pure Exploration: All 5 points optimized for exploration (high beta/diverse)

These policies are agnostic to the specific acquisition function used (qEHVI, qUCB, etc.)
"""

import numpy as np
from typing import Tuple


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