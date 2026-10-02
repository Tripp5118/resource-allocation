# resource_allocation/design_space.py

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from typing import List, Tuple, Optional


class DesignSpace:
    """
    Discretized composition simplex with per-element bounds and optional
    group-level sum constraints.

    Without groups (original behaviour):
        - mins[i] <= x_i <= maxs[i]
        - sum(x) == 1

    With groups:
        - mins[i] <= x_i <= maxs[i]                                    (per-element)
        - group_min_g <= sum(x_i for i in group g) <= group_max_g      (per-group)
        - sum(x) == 1

    Components are expressed in fractions (0-1), consistent with the original
    library convention. If your design space is in at%, divide by 100 first.

    Group usage example (Fe-Co-Ni system with dopants, values in fractions):

        components = [
            ("Fe", 0.0, 1.0),
            ("Co", 0.0, 1.0),
            ("Ni", 0.0, 1.0),
            ("V",  0.0, 0.05),
            ("Mo", 0.0, 0.05),
            ("C",  0.0, 0.05),
            ("Si", 0.0, 0.05),
            ("Nb", 0.0, 0.05),
            ("Ti", 0.0, 0.05),
        ]
        groups = [
            ("majors",  ["Fe", "Co", "Ni"], 0.90, 1.00),
            ("dopants", ["V", "Mo", "C", "Si", "Nb", "Ti"], 0.00, 0.10),
        ]
        ds = DesignSpace(components, step=0.02, groups=groups)
    """

    def __init__(
        self,
        components: List[Tuple[str, float, float]],
        step: float = 0.025,
        seed: int = 42,
        groups: Optional[List[Tuple[str, List[str], float, float]]] = None,
    ):
        """
        Args:
            components: list of (label, min_frac, max_frac).
            step:       grid spacing in the same units as components.
            seed:       RNG seed.
            groups:     optional list of (group_name, [element_labels], group_min, group_max).
                        Elements not in any group are subject only to per-element bounds
                        and the global sum==1 constraint.
        """
        if not components:
            raise ValueError("At least one component must be provided.")

        self.labels       = [c[0] for c in components]
        self.mins         = np.array([c[1] for c in components], dtype=float)
        self.maxs         = np.array([c[2] for c in components], dtype=float)
        self.step         = float(step)
        self.seed         = seed
        self.rng          = np.random.default_rng(seed)
        self.n_components = len(self.labels)

        if np.any(self.mins < 0.0) or np.any(self.maxs > 1.0):
            raise ValueError("All min/max must be within [0, 1].")
        if np.any(self.mins > self.maxs):
            raise ValueError("For each component, min must be <= max.")

        self.groups = None
        if groups is not None:
            label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
            self.groups  = []
            for g_name, g_labels, g_min, g_max in groups:
                missing = [l for l in g_labels if l not in label_to_idx]
                if missing:
                    raise ValueError(f"Group '{g_name}' references unknown elements: {missing}")
                if g_min > g_max:
                    raise ValueError(f"Group '{g_name}': group_min > group_max.")
                self.groups.append({
                    "name":    g_name,
                    "labels":  g_labels,
                    "indices": [label_to_idx[l] for l in g_labels],
                    "min":     float(g_min),
                    "max":     float(g_max),
                })

        self.space = self._create_discretized_simplex()
        if self.space.size == 0:
            raise RuntimeError(
                "Design space enumeration produced zero points. "
                "Check your ranges, step size, and group constraints."
            )

        group_str = f" with {len(self.groups)} group(s)" if self.groups else ""
        print(
            f"[DesignSpace] Generated {len(self.space):,} valid compositions "
            f"for {self.n_components} components{group_str}: {', '.join(self.labels)}"
        )

    # ------------------------------------------------------------------ #
    #  Grid enumeration
    # ------------------------------------------------------------------ #

    def _create_discretized_simplex(self) -> np.ndarray:
        if self.groups is None:
            return self._enumerate_ungrouped()
        return self._enumerate_grouped()

    def _enumerate_ungrouped(self) -> np.ndarray:
        """Original recursive enumeration — unchanged from v1."""
        d    = self.n_components
        step = self.step
        tol  = step / 2.0

        value_lists = []
        for lo, hi in zip(self.mins, self.maxs):
            vals = np.arange(lo, hi + step * 0.5, step)
            vals = vals[(vals >= lo - tol) & (vals <= hi + tol)]
            vals = np.round(vals / step) * step
            vals = np.unique(vals)
            value_lists.append(vals)

        valid = []

        def recurse(idx, partial, remaining):
            if idx == d - 1:
                x_last_grid = round(remaining / step) * step
                lo, hi      = self.mins[idx], self.maxs[idx]
                if (lo - tol <= x_last_grid <= hi + tol
                        and abs(sum(partial) + x_last_grid - 1.0) < tol):
                    valid.append(partial + [x_last_grid])
                return
            for v in value_lists[idx]:
                if v > remaining + tol:
                    break
                new_rem = remaining - v
                if (self.mins[idx+1:].sum() - tol <= new_rem
                        <= self.maxs[idx+1:].sum() + tol):
                    recurse(idx + 1, partial + [v], new_rem)

        recurse(0, [], 1.0)
        return np.array(valid, dtype=float)

    def _enumerate_grouped(self) -> np.ndarray:
        """
        Group-aware enumeration.

        1. Find all valid group-sum vectors (one total per group) that satisfy
           each group's [min, max] and sum to 1 globally.
        2. For each group-sum vector, independently enumerate element
           compositions within each group (sub-simplex per group).
        3. Take the Cartesian product across groups and assemble full rows.
        """
        import itertools

        step = self.step
        tol  = step / 2.0

        # Per-element value lists (respects individual mins/maxs)
        value_lists = []
        for lo, hi in zip(self.mins, self.maxs):
            vals = np.arange(lo, hi + step * 0.5, step)
            vals = vals[(vals >= lo - tol) & (vals <= hi + tol)]
            vals = np.round(vals / step) * step
            value_lists.append(np.unique(vals))

        # Validate group bounds are on-grid
        for g in self.groups:
            for bound in (g["min"], g["max"]):
                if abs(bound / step - round(bound / step)) > 1e-6:
                    raise ValueError(
                        f"Group '{g['name']}' bound {bound} is not divisible "
                        f"by step {step}. Adjust bounds or step size."
                    )

        n_groups = len(self.groups)
        g_mins   = np.array([g["min"] for g in self.groups])
        g_maxs   = np.array([g["max"] for g in self.groups])

        # Step 1: enumerate valid group-sum vectors
        group_sum_lists = []
        for g in self.groups:
            lo, hi = g["min"], g["max"]
            gvals  = np.arange(lo, hi + step * 0.5, step)
            gvals  = np.round(gvals / step) * step
            group_sum_lists.append(gvals[(gvals >= lo - tol) & (gvals <= hi + tol)])

        valid_group_sums = []

        def recurse_groups(g_idx, partial, remaining):
            if g_idx == n_groups - 1:
                s_last = round(remaining / step) * step
                lo, hi = g_mins[g_idx], g_maxs[g_idx]
                if lo - tol <= s_last <= hi + tol and abs(s_last - remaining) < tol:
                    valid_group_sums.append(tuple(partial + [s_last]))
                return
            for s in group_sum_lists[g_idx]:
                new_rem  = remaining - s
                min_rest = g_mins[g_idx+1:].sum()
                max_rest = g_maxs[g_idx+1:].sum()
                if min_rest - tol <= new_rem <= max_rest + tol:
                    recurse_groups(g_idx + 1, partial + [s], new_rem)

        recurse_groups(0, [], 1.0)

        if not valid_group_sums:
            return np.array([], dtype=float)

        # Step 2: for each group-sum vector, enumerate sub-compositions per group
        all_rows = []

        for group_sums in valid_group_sums:
            group_sub_comps = []   # one array of sub-compositions per group

            for g_idx, g in enumerate(self.groups):
                s_g    = group_sums[g_idx]
                g_idxs = g["indices"]
                n_elem = len(g_idxs)
                g_vals = [value_lists[i] for i in g_idxs]
                g_lo   = self.mins[g_idxs]
                g_hi   = self.maxs[g_idxs]

                sub_valid = []

                def recurse_sub(e_idx, partial, rem,
                                _vals=g_vals, _lo=g_lo, _hi=g_hi, _n=n_elem):
                    if e_idx == _n - 1:
                        # Last element: determined by remaining, but must
                        # still respect its individual per-element bound.
                        x_last = round(rem / step) * step
                        if (_lo[e_idx] - 1e-9 <= x_last <= _hi[e_idx] + 1e-9
                                and abs(x_last - rem) < tol):
                            sub_valid.append(partial + [x_last])
                        return
                    for v in _vals[e_idx]:
                        if v > rem + tol:
                            break
                        new_rem  = rem - v
                        min_rest = _lo[e_idx+1:].sum()
                        max_rest = _hi[e_idx+1:].sum()
                        if min_rest - tol <= new_rem <= max_rest + tol:
                            recurse_sub(e_idx + 1, partial + [v], new_rem)

                recurse_sub(0, [], s_g)

                if not sub_valid:
                    break   # this group-sum combo is infeasible, skip
                group_sub_comps.append((g_idxs, np.array(sub_valid, dtype=float)))
            else:
                # All groups enumerated — take Cartesian product across groups
                for combo in itertools.product(*[range(len(gc[1])) for gc in group_sub_comps]):
                    row = np.zeros(self.n_components, dtype=float)
                    for g_i, row_i in enumerate(combo):
                        row[group_sub_comps[g_i][0]] = group_sub_comps[g_i][1][row_i]
                    all_rows.append(row)

        return np.array(all_rows, dtype=float) if all_rows else np.array([], dtype=float)

    # ------------------------------------------------------------------ #
    #  Sampling
    # ------------------------------------------------------------------ #

    def sample(self, n: int, method: str = "random", seed: Optional[int] = None) -> np.ndarray:
        """
        Sample n points from the enumerated space.

        Args:
            n:      number of samples.
            method: "random" or "sobol".
            seed:   override the scramble seed for Sobol draws. When None the
                    DesignSpace constructor seed is used (same points every call).
                    Pass a different integer each iteration to get a fresh
                    space-filling subsample without reconstructing the space.

        Returns:
            (n, n_components) array.
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
            sobol_seed    = seed if seed is not None else self.seed
            bounds        = torch.tensor([[0.0], [1.0]], dtype=torch.double)
            sobol_samples = draw_sobol_samples(
                bounds=bounds, n=n, q=1, seed=sobol_seed
            ).squeeze()
            indices = torch.clamp((sobol_samples * n_available).long(), 0, n_available - 1)
            return self.space[indices.cpu().numpy()]

        else:
            raise ValueError("method must be 'random' or 'sobol'.")

    # ------------------------------------------------------------------ #
    #  Bounds for GP / acquisition
    # ------------------------------------------------------------------ #

    def get_bounds(self) -> np.ndarray:
        """Return (2, d) array of lower/upper bounds per dimension."""
        return np.vstack([self.mins, self.maxs])