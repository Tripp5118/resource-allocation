"""Design space generation for HEA compositions."""
import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from typing import Tuple

ELEM_ORDER = ["Fe", "Co", "Cr", "Ni", "V"]

class DesignSpace:
    """Manages the discretized composition space for HEA."""
    
    def __init__(self,
                 min_comp: float = 0.1,
                 max_comp: float = 0.4,
                 step: float = 0.005,
                 seed: int = 42):
        self.min_comp = min_comp
        self.max_comp = max_comp
        self.step = step
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.space = self._create_discretized_space()
        print(f"[DesignSpace] Generated {len(self.space)} valid compositions")
    
    def _create_discretized_space(self):
        vals = np.arange(self.min_comp, self.max_comp + self.step, self.step)
        valid = []
        tol = self.step / 2
        for fe in vals:
            for co in vals:
                for cr in vals:
                    for ni in vals:
                        v = 1.0 - (fe + co + cr + ni)
                        if self.min_comp <= v <= self.max_comp and abs(v - round(v / self.step) * self.step) < tol:
                            comp = [fe, co, cr, ni, v]
                            # Validate sum exactly 1 (with tolerance), each in allowed range
                            if abs(sum(comp) - 1.0) < tol and all(self.min_comp <= x <= self.max_comp for x in comp):
                                valid.append(comp)
        return np.array(valid)
    
    def sample(self, n: int, method: str = "random") -> np.ndarray:
        """Sample n compositions from the design space.
        
        Args:
            n: Number of samples to draw
            method: Sampling method ('random' or 'sobol')
            
        Returns:
            Array of n compositions (n, 5)
        """
        n_available = len(self.space)
        if n >= n_available:
            return self.space.copy()
        
        if method == "random":
            idx = self.rng.choice(n_available, size=n, replace=False)
            return self.space[idx]
        
        elif method == "sobol":
            # Use BoTorch's Sobol sampling to generate indices
            # Draw n samples from [0, 1] in 1D
            bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
            sobol_samples = draw_sobol_samples(
                bounds=bounds,
                n=n,
                q=1,
                seed=self.seed
            ).squeeze()  # Shape: (n,)
            
            # Scale to indices in the design space
            indices = (sobol_samples * n_available).long()
            # Clip to valid range
            indices = torch.clamp(indices, 0, n_available - 1)
            
            return self.space[indices.cpu().numpy()]
        
        else:
            raise ValueError(f"Sampling method '{method}' not implemented. Use 'random' or 'sobol'.")
    
    def get_bounds(self) -> Tuple[float, float]:
        """Return composition bounds for normalization."""
        return self.min_comp, self.max_comp