# core/design_space.py

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from typing import List, Tuple, Optional

class DesignSpace:
    """
    Discretized composition simplex with per-element bounds.

    - Compositions x in R^d
    - For each component i:
        min_i <= x_i <= max_i
        x_i on a regular grid with spacing `step`
    - Sum_i x_i == 1 (within tolerance)
    """

    def __init__(
        self,
        components: List[Tuple[str, float, float]],
        step: float = 0.025,
        seed: int = 42,
    ):
        """
        Args:
            components: list of (label, min_comp, max_comp) for each dimension.
                        Example:
                          [("Ti", 0.0, 0.35),
                           ("V",  0.0, 0.50),
                           ...]
            step: grid spacing for all components.
            seed: RNG seed for sampling.
        """
        if not components:
            raise ValueError("At least one component must be provided.")

        self.labels = [c[0] for c in components]
        self.mins = np.array([c[1] for c in components], dtype=float)
        self.maxs = np.array([c[2] for c in components], dtype=float)
        self.step = float(step)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.n_components = len(self.labels)

        # Basic checks
        if np.any(self.mins < 0.0) or np.any(self.maxs > 1.0):
            raise ValueError("All min/max must be within [0, 1].")
        if np.any(self.mins > self.maxs):
            raise ValueError("For each component, min must be <= max.")

        # Enumerate grid
        self.space = self._create_discretized_simplex()
        if self.space.size == 0:
            raise RuntimeError(
                "Design space enumeration produced zero points. "
                "Check your ranges and step size."
            )

        print(
            f"[DesignSpace] Generated {len(self.space)} valid compositions "
            f"for {self.n_components} components: {', '.join(self.labels)}"
        )

    # ------------------------------------------------------------------ #
    #  Grid enumeration
    # ------------------------------------------------------------------ #

    def _create_discretized_simplex(self) -> np.ndarray:
        """
        Enumerate all compositions satisfying:
            mins[i] <= x_i <= maxs[i],
            sum(x) == 1,
            x_i on a grid defined by mins[i] + k * step (within tolerance).
        """
        d = self.n_components
        step = self.step
        tol = step / 2.0

        # Precompute discrete values per component
        value_lists = []
        for lo, hi in zip(self.mins, self.maxs):
            # For numerical robustness, extend the hi slightly before np.arange
            vals = np.arange(lo, hi + step * 0.5, step)
            # Clip to [lo, hi]
            vals = vals[(vals >= lo - tol) & (vals <= hi + tol)]
            # Round to nearest grid to avoid cumulative floating errors
            vals = np.round(vals / step) * step
            vals = np.unique(vals)
            value_lists.append(vals)

        valid = []

        def recurse(idx: int, partial: List[float], remaining: float):
            if idx == d - 1:
                # Last component is determined by remaining sum
                x_last = remaining
                lo = self.mins[idx]
                hi = self.maxs[idx]
                if lo - tol <= x_last <= hi + tol:
                    # Snap to nearest grid
                    x_last_grid = round(x_last / step) * step
                    if lo - tol <= x_last_grid <= hi + tol:
                        comp = partial + [x_last_grid]
                        if abs(sum(comp) - 1.0) < tol:
                            valid.append(comp)
                return

            lo = self.mins[idx]
            hi = self.maxs[idx]
            vals = value_lists[idx]

            # For each possible value at idx, ensure the rest can still fit in min/max
            for v in vals:
                if v < lo - tol or v > hi + tol:
                    continue
                if v > remaining + tol:
                    break  # cannot exceed remaining sum

                dims_left = (d - 1) - idx
                min_possible = np.sum(self.mins[idx + 1 :])
                max_possible = np.sum(self.maxs[idx + 1 :])

                new_remaining = remaining - v
                # We need min_possible <= new_remaining <= max_possible
                if not (min_possible - tol <= new_remaining <= max_possible + tol):
                    continue

                recurse(idx + 1, partial + [v], new_remaining)

        # We want sum(x) == 1
        recurse(idx=0, partial=[], remaining=1.0)
        return np.array(valid, dtype=float)

    # ------------------------------------------------------------------ #
    #  Sampling
    # ------------------------------------------------------------------ #

    def sample(self, n: int, method: str = "random") -> np.ndarray:
        """
        Sample n points from the enumerated simplex.

        Args:
            n: number of samples
            method: "random" or "sobol"

        Returns:
            (n, n_components) array
        """
        n_available = len(self.space)
        if n_available == 0:
            raise RuntimeError("Design space is empty.")

        if n >= n_available:
            return self.space.copy()

        if method == "random":
            idx = self.rng.choice(n_available, size=n, replace=False)
            return self.space[idx]

        elif method == "sobol":
            bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
            sobol_samples = draw_sobol_samples(
                bounds=bounds,
                n=n,
                q=1,
                seed=self.seed,
            ).squeeze()
            indices = (sobol_samples * n_available).long()
            indices = torch.clamp(indices, 0, n_available - 1)
            return self.space[indices.cpu().numpy()]

        else:
            raise ValueError("Sampling method must be 'random' or 'sobol'.")

    # ------------------------------------------------------------------ #
    #  Bounds for GP / acquisition
    # ------------------------------------------------------------------ #

    def get_bounds(self) -> np.ndarray:
        """
        Return (2, d) array of lower/upper bounds per dimension.

        (Simply min/max for each component.)
        """
        return np.vstack([self.mins, self.maxs])