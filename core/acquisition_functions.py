# core/acquisition_functions.py
"""Acquisition function computation for 2-objective BO with MultiTaskGP.

- Continuous optimization with qEHVI and MO-MESMO via optimize_acqf.
- Arbitrary input dimension d (1 <= d <= 10).
- Enforces a simplex constraint in normalized space so that:
    sum_i x_i = 1, with each x_i in [min_i, max_i], min_i >= 0.
"""

import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from botorch.models import MultiTaskGP
from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


# ============================================================================
#  DATA STRUCTURES
# ============================================================================


@dataclass
class QEHVIBatch:
    """Represents one allocation option (n exploit + k explore points)."""

    batch_size: int              # Number of qEHVI (exploitation) points
    points: np.ndarray           # qEHVI points (batch_size, d)
    explore_points: np.ndarray   # MO-MESMO points (5-batch_size, d)
    full_entropy: float          # Combined entropy of all points
    full_qehvi: float            # Combined EHVI of all points
    total_batch_size: int = 5    # Total batch size (default 5)

    def __repr__(self):
        return (
            f"QEHVIBatch(q={self.batch_size}, "
            f"Entropy={self.full_entropy:.4f}, "
            f"QEHVI={self.full_qehvi:.4f})"
        )


@dataclass
class EntropyPoint:
    """Single point ranked by entropy (reference only)."""

    rank: int
    point: np.ndarray
    entropy: float
    qehvi: float

    def __repr__(self):
        return f"EntropyPoint(rank={self.rank}, entropy={self.entropy:.4f})"


@dataclass
class AcquisitionData:
    """Container for all allocation options."""

    qehvi_batches: List[QEHVIBatch]
    entropy_points: List[EntropyPoint]

    def __repr__(self):
        s = "AcquisitionData:\n"
        s += "  Allocation Options:\n"
        for batch in self.qehvi_batches:
            s += f"    {batch}\n"
        return s


# ============================================================================
#  MAIN ACQUISITION FUNCTION MANAGER
# ============================================================================


class AcquisitionFunctionManager:
    """
    Manages acquisition function computation for 2-objective BO with MultiTaskGP.

    - Assumes bounds is (2, d) with per-dimension [min_i, max_i].
    - Enforces simplex constraint sum_i x_i = 1 via equality_constraints in normalized space.
    - Total batch size per iteration is fixed at 5 (5/0, 4/1, ..., 0/5).

    Example:
        >>> bounds = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 1.0]])
        >>> acq_mgr = AcquisitionFunctionManager(bounds, num_restarts=5)
        >>> pareto_Y, ref_point = acq_mgr.compute_pareto_front(Y_observed)
        >>> options = acq_mgr.compute_all_allocation_options(model, X_pool, pareto_Y, ref_point)
    """

    def __init__(self, bounds: torch.Tensor, num_restarts: int = 3, raw_samples: int = 128):
        """
        Args:
            bounds: (2, d) tensor of lower/upper bounds per input dimension
            num_restarts: Number of restarts for acquisition optimization
            raw_samples: Number of raw samples for initialization
        """
        if bounds.shape[0] != 2:
            raise ValueError(f"bounds must have shape (2, d), got {bounds.shape}")
        if bounds.shape[1] < 1 or bounds.shape[1] > 10:
            raise ValueError(f"Input dimension must be between 1 and 10, got {bounds.shape[1]}")
        if torch.any(bounds[0] >= bounds[1]):
            raise ValueError("Lower bounds must be strictly less than upper bounds")
        
        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)
        self.d = self.bounds.shape[1]
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    # ========================================================================
    #  MAIN INTERFACE
    # ========================================================================

    def compute_pareto_front(
        self, Y_mo: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Pareto front and reference point from objectives.

        Args:
            Y_mo: Objective values (n, 2) - both objectives should be "maximized".

        Returns:
            pareto_Y: Pareto front points (m, 2)
            ref_point: Reference point for hypervolume (2,)
        """
        if Y_mo.shape[0] == 0:
            raise ValueError("Cannot compute Pareto front from empty objective values")
        mask = is_non_dominated(Y_mo)
        pareto = Y_mo[mask]
        if len(pareto) == 0:
            raise ValueError("No non-dominated points found after filtering")
        # Simple heuristic: reference point slightly below the min of Pareto set
        ref_point = pareto.min(dim=0).values - 0.1 * torch.ones(
            2, dtype=DTYPE, device=DEVICE
        )
        return pareto, ref_point

    def compute_all_allocation_options(
        self,
        model: MultiTaskGP,
        X_pool: torch.Tensor,
        pareto_Y: torch.Tensor,
        ref_point: torch.Tensor,
        mc_samples: int = 128,
    ) -> AcquisitionData:
        """
        Compute all 6 allocation options for a total batch size of 5.

        Options:
            Option 0: 5 exploit + 0 explore
            Option 1: 4 exploit + 1 explore
            Option 2: 3 exploit + 2 explore
            Option 3: 2 exploit + 3 explore
            Option 4: 1 exploit + 4 explore
            Option 5: 0 exploit + 5 explore

        Args:
            model: Fitted MultiTaskGP model.
            X_pool: Candidate pool (n_pool, d) for entropy ranking (not used in optimize_acqf).
            pareto_Y: Current Pareto front (m, 2) in "maximize" space.
            ref_point: Reference point for hypervolume (2,).
            mc_samples: Monte Carlo samples for acquisition.

        Returns:
            AcquisitionData with 6 allocation options.
        """
        print("[AcqFn] Computing acquisition functions.")

        total_batch_size = 5

        # Reduce samples for MO-MESMO (it's more expensive)
        mc_samples_entropy = min(mc_samples // 2, 64)
        print(
            f"[AcqFn] Using {mc_samples} samples for qEHVI, "
            f"{mc_samples_entropy} for MO-MESMO"
        )

        # Step 1: Compute top entropy reference points (for reporting only)
        print("[AcqFn] Computing top entropy reference points...")
        entropy_points = self._compute_top_entropy_points(
            model=model,
            X_pool=X_pool,
            pareto_Y=pareto_Y,
            ref_point=ref_point,
            n_points=5,
            mc_samples=mc_samples,
        )

        # Step 2: Shared qEHVI acquisition for evaluation
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        qehvi_acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )

        # Step 3: Build all 6 allocation options
        qehvi_batches: List[QEHVIBatch] = []

        for q_exploit in range(total_batch_size, -1, -1):
            q_explore = total_batch_size - q_exploit
            option_idx = total_batch_size - q_exploit

            print(
                f"\n[AcqFn] Option {option_idx}: "
                f"{q_exploit} exploit + {q_explore} explore"
            )

            # Exploitation points via qEHVI
            if q_exploit > 0:
                exploit_points = self._get_qehvi_points(
                    model=model,
                    pareto_Y=pareto_Y,
                    ref_point=ref_point,
                    q=q_exploit,
                    mc_samples=mc_samples,
                )
            else:
                exploit_points = np.empty((0, self.d))

            # Exploration points via MO-MESMO (with fallback)
            if q_explore > 0:
                explore_points = self._get_entropy_points(
                    model=model,
                    pareto_Y=pareto_Y,
                    ref_point=ref_point,
                    q=q_explore,
                    mc_samples=mc_samples_entropy,
                )
            else:
                explore_points = np.empty((0, self.d))

            # Combine all points
            if exploit_points.shape[0] > 0 and explore_points.shape[0] > 0:
                all_points = np.vstack([exploit_points, explore_points])
            elif exploit_points.shape[0] > 0:
                all_points = exploit_points
            elif explore_points.shape[0] > 0:
                all_points = explore_points
            else:
                all_points = np.empty((0, self.d))

            # Compute combined metrics
            if all_points.shape[0] > 0:
                all_points_torch = torch.tensor(
                    all_points, dtype=DTYPE, device=DEVICE
                )
                combined_entropy = self._compute_joint_entropy(
                    model, all_points_torch
                )
                combined_qehvi = self._evaluate_qehvi(qehvi_acq, all_points_torch)
            else:
                combined_entropy = 0.0
                combined_qehvi = 0.0

            batch = QEHVIBatch(
                batch_size=q_exploit,
                points=exploit_points,
                explore_points=explore_points,
                full_entropy=float(combined_entropy),
                full_qehvi=float(combined_qehvi),
                total_batch_size=total_batch_size,
            )
            qehvi_batches.append(batch)

            print(
                f"  → Entropy: {batch.full_entropy:.4f}, "
                f"QEHVI: {batch.full_qehvi:.4f}"
            )

        return AcquisitionData(qehvi_batches=qehvi_batches, entropy_points=entropy_points)

    # ========================================================================
    #  POINT SELECTION
    # ========================================================================

    def _get_qehvi_points(
        self,
        model: MultiTaskGP,
        pareto_Y: torch.Tensor,
        ref_point: torch.Tensor,
        q: int,
        mc_samples: int,
    ) -> np.ndarray:
        """
        Optimize q points using qEHVI (exploitation) with simplex constraint.

        Returns:
            points: (q, d) numpy array of compositions in original space.
        """
        print(f"  Optimizing qEHVI with q={q}...")

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)

        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )

        d = self.d
        eq_constraints = self._get_simplex_equality_tuple()

        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack(
                [
                    torch.zeros(d, dtype=DTYPE, device=DEVICE),
                    torch.ones(d, dtype=DTYPE, device=DEVICE),
                ]
            ),
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            equality_constraints=eq_constraints,
            options={"batch_limit": 5, "maxiter": 200},
        )

        # Denormalize to original bounds
        candidates_denorm = self._denormalize(candidates, self.bounds)
        return candidates_denorm.cpu().numpy()

    def _get_entropy_points(
        self,
        model: MultiTaskGP,
        pareto_Y: torch.Tensor,
        ref_point: torch.Tensor,
        q: int,
        mc_samples: int,
    ) -> np.ndarray:
        """
        Optimize q points using MO-MESMO (exploration) with simplex constraint and fallback.

        Returns:
            points: (q, d) numpy array of compositions in original space.
        """
        print(f"  Optimizing MO-MESMO with q={q}...")

        try:
            points = self._optimize_mesmo(
                model=model,
                pareto_Y=pareto_Y,
                ref_point=ref_point,
                q=q,
                mc_samples=mc_samples,
            )
            print("  ✓ MO-MESMO succeeded")
            return points

        except Exception as e:
            print(f"  ✗ MO-MESMO failed: {e}")
            print("  Using fallback: variance-based entropy sampling")
            return self._fallback_entropy_sampling(model, q)

    def _optimize_mesmo(
        self,
        model: MultiTaskGP,
        pareto_Y: torch.Tensor,
        ref_point: torch.Tensor,
        q: int,
        mc_samples: int,
    ) -> np.ndarray:
        """Optimize using MO-MESMO, with simplex constraint."""

        try:
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
            hypercell_bounds = partitioning.hypercell_bounds.unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Failed to create DominatedPartitioning: {e}")
        
        try:
            acq = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                model=model,
                hypercell_bounds=hypercell_bounds,
                estimation_type="LB",
                num_samples=mc_samples,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create MO-MESMO acquisition: {e}")

        # Wrap to catch NaN/Inf during optimization
        class SafeAcqWrapper:
            def __init__(self, acq_func):
                self.acq_func = acq_func

            def __call__(self, X):
                values = self.acq_func(X)
                if torch.any(torch.isnan(values)) or torch.any(torch.isinf(values)):
                    raise ValueError("NaN/Inf in acquisition values")
                return torch.clamp(values, min=-1e6, max=1e6)

        safe_acq = SafeAcqWrapper(acq)

        d = self.d
        eq_constraints = self._get_simplex_equality_tuple()

        candidates, _ = optimize_acqf(
            acq_function=safe_acq,
            bounds=torch.stack(
                [
                    torch.zeros(d, dtype=DTYPE, device=DEVICE),
                    torch.ones(d, dtype=DTYPE, device=DEVICE),
                ]
            ),
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            equality_constraints=eq_constraints,
            options={"batch_limit": 5, "maxiter": 200},
        )

        candidates_denorm = self._denormalize(candidates, self.bounds)
        return candidates_denorm.cpu().numpy()

    def _fallback_entropy_sampling(self, model: MultiTaskGP, q: int) -> np.ndarray:
        """
        Fallback: sample points with highest GP uncertainty satisfying simplex constraint.
        """
        print("  [Fallback] Entropy-based sampling satisfying simplex constraint")

        d = self.d
        n_candidates = 2000

        # Generate simplex-constrained candidates
        from botorch.utils.sampling import sample_simplex
        
        # Sample on unit simplex
        simplex_samples = sample_simplex(n=n_candidates, d=d, dtype=DTYPE, device=DEVICE)
        
        # Scale to bounds while maintaining simplex constraint
        mins = self.bounds[0, :]
        maxs = self.bounds[1, :]
        ranges = maxs - mins
        
        # Transform: x_i = min_i + alpha_i * (max_i - min_i)
        # where alpha_i are the simplex samples
        candidates = mins.unsqueeze(0) + simplex_samples * ranges.unsqueeze(0)
        
        # Renormalize to ensure sum = 1 (due to floating point errors)
        candidates = candidates / candidates.sum(dim=1, keepdim=True)
        
        # Filter to valid range
        valid_mask = torch.all((candidates >= mins - 1e-6) & (candidates <= maxs + 1e-6), dim=1)
        candidates = candidates[valid_mask]
        
        if len(candidates) < q:
            raise RuntimeError(
                f"Fallback sampling produced only {len(candidates)} valid points, need {q}"
            )

        # Compute uncertainty (sum of log variances for both tasks)
        n = candidates.shape[0]
        X_norm = normalize(candidates, self.bounds)

        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        X_task_0 = torch.cat([X_norm, task_0], dim=-1)
        X_task_1 = torch.cat([X_norm, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)

        with torch.no_grad():
            post = model.posterior(X_both)
            var_flat = post.variance.squeeze(-1)  # (2n,)
            var_task_0 = var_flat[:n]
            var_task_1 = var_flat[n:]
            uncertainty = torch.log(var_task_0.clamp_min(1e-12)) + torch.log(
                var_task_1.clamp_min(1e-12)
            )

        top_indices = torch.argsort(uncertainty, descending=True)[:q]
        return candidates[top_indices].cpu().numpy()

    # ========================================================================
    #  EVALUATION HELPERS
    # ========================================================================

    def _evaluate_qehvi(self, acq_function, X: torch.Tensor) -> float:
        """
        Evaluate qEHVI on a batch of points in original space.

        Args:
            acq_function: qLogExpectedHypervolumeImprovement instance.
            X: Points (q, d) in original space.

        Returns:
            QEHVI value (scalar).
        """
        if X.shape[0] == 0:
            return 0.0

        X_norm = normalize(X, self.bounds)

        with torch.no_grad():
            acq_val = acq_function(X_norm.unsqueeze(0))  # (1, q)
        return float(acq_val)

    def _compute_joint_entropy(self, model: MultiTaskGP, X: torch.Tensor) -> float:
        """
        Compute joint entropy for a batch of points (both tasks).

        Args:
            model: MultiTaskGP model.
            X: Points (n, d) in original space.

        Returns:
            Joint entropy value (scalar).
        """
        if X.shape[0] == 0:
            return 0.0

        n = X.shape[0]
        X_norm = normalize(X, self.bounds)

        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        X_task_0 = torch.cat([X_norm, task_0], dim=-1)
        X_task_1 = torch.cat([X_norm, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)

        with torch.no_grad():
            post = model.posterior(X_both)
            variance = post.variance.clamp_min(1e-12)  # (2n, 1)
            log_det = torch.sum(torch.log(variance))
            d_total = 2 * n  # n points × 2 tasks
            entropy = 0.5 * (d_total * np.log(2 * np.pi * np.e) + log_det)

        return float(entropy)

    def _compute_top_entropy_points(
        self,
        model: MultiTaskGP,
        X_pool: torch.Tensor,
        pareto_Y: torch.Tensor,
        ref_point: torch.Tensor,
        n_points: int,
        mc_samples: int,
    ) -> List[EntropyPoint]:
        """
        Compute top-n points from a candidate pool ranked by entropy (for reporting).

        Args:
            model: MultiTaskGP model.
            X_pool: (N_pool, d) candidate pool in original space.
        """
        n = X_pool.shape[0]
        X_pool_norm = normalize(X_pool, self.bounds)

        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        X_task_0 = torch.cat([X_pool_norm, task_0], dim=-1)
        X_task_1 = torch.cat([X_pool_norm, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)

        with torch.no_grad():
            post = model.posterior(X_both)
            var_flat = post.variance.squeeze(-1)  # (2n,)
            var_task_0 = var_flat[:n]
            var_task_1 = var_flat[n:]
            entropy = 0.5 * torch.log(
                2 * np.pi * np.e * var_task_0.clamp_min(1e-12)
            ) + 0.5 * torch.log(
                2 * np.pi * np.e * var_task_1.clamp_min(1e-12)
            )

        top_indices = torch.argsort(entropy, descending=True)[:n_points]

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )

        entropy_points: List[EntropyPoint] = []
        for rank, idx in enumerate(top_indices, start=1):
            point = X_pool[idx].cpu().numpy()
            ent = float(entropy[idx])

            X_single = X_pool_norm[idx : idx + 1]
            with torch.no_grad():
                qehvi_val = float(acq(X_single.unsqueeze(0)))

            entropy_points.append(
                EntropyPoint(rank=rank, point=point, entropy=ent, qehvi=qehvi_val)
            )

        return entropy_points

    # ========================================================================
    #  UTILITIES
    # ========================================================================

    def _denormalize(self, X_norm: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0, 1] to original bounds."""
        return bounds[0] + X_norm * (bounds[1] - bounds[0])

    def _get_simplex_equality_tuple(self):
        """
        Equality constraint for optimize_acqf in normalized space:

        Let x_norm be in [0,1]^d. Original x is:
            x_i = min_i + (max_i - min_i) * x_norm_i

        We want sum_i x_i = 1.0 ⇒ sum_i (max_i - min_i) * x_norm_i = 1 - sum_i(min_i).

        Returns:
            list with one tuple: (indices, coefficients, rhs)
            where indices: LongTensor[d], coefficients: Tensor[d], rhs: float.
        """
        d = self.d
        mins = self.bounds[0, :]  # (d,)
        maxs = self.bounds[1, :]  # (d,)
        coef = maxs - mins        # (d,)

        rhs = 1.0 - float(mins.sum().item())

        indices = torch.arange(d, device=DEVICE, dtype=torch.long)
        coefficients = coef.to(dtype=DTYPE, device=DEVICE)

        return [(indices, coefficients, rhs)]