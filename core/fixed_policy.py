# core/fixed_policy.py
"""
Simple fixed policy implementations for BO comparison.

Two scenarios:
1. Pure QEHVI: All 5 points from qEHVI (full exploitation)
2. Pure MESMO: All 5 points from MO-MESMO (full exploration)
"""

import numpy as np
from typing import Tuple


class PureQEHVI:
    """Pure QEHVI policy: Select all 5 points from qEHVI acquisition."""
    
    def __init__(self):
        self.name = "Pure_QEHVI"
    
    def select_points(self, allocation_options) -> Tuple[np.ndarray, str]:
        """
        Select all 5 points from qEHVI (option 0).
        
        Args:
            allocation_options: AcquisitionData object with qehvi_batches
            
        Returns:
            points: (5, d) array of selected points
            reasoning: Explanation string
        """
        # Option 0 is always full qEHVI (5 points, 0 exploration)
        batch = allocation_options.qehvi_batches[0]
        
        # Get all qEHVI points
        points = batch.points
        
        reasoning = (
            f"Pure QEHVI Strategy: Select all {batch.batch_size} points "
            f"from qEHVI acquisition (full exploitation, no exploration)."
        )
        
        return points, reasoning


class PureMESMO:
    """Pure MESMO policy: Select all 5 points from MO-MESMO acquisition."""
    
    def __init__(self):
        self.name = "Pure_MESMO"
    
    def select_points(self, allocation_options) -> Tuple[np.ndarray, str]:
        """
        Select all 5 points from MO-MESMO (option 5).
        
        Args:
            allocation_options: AcquisitionData object with qehvi_batches
            
        Returns:
            points: (5, d) array of selected points
            reasoning: Explanation string
        """
        # Option 5 is always pure exploration (0 points from qEHVI, 5 from MESMO)
        batch = allocation_options.qehvi_batches[5]
        
        # Get all exploration points
        points = batch.explore_points
        
        reasoning = (
            f"Pure MESMO Strategy: Select all {batch.total_batch_size - batch.batch_size} points "
            f"from MO-MESMO acquisition (full exploration, no exploitation)."
        )
        
        return points, reasoning
