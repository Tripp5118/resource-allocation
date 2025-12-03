"""Acquisition function computation - MultiTaskGP version (Refactored)."""
import torch
import numpy as np
from botorch.models import MultiTaskGP
from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.max_value_entropy_search import qLowerBoundMultiObjectiveMaxValueEntropySearch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from typing import List, Tuple
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


# ============================================================================
#  DATA STRUCTURES
# ============================================================================

@dataclass
class QEHVIBatch:
    """Represents one allocation option (n exploit + k explore points)."""
    batch_size: int              # Number of qEHVI (exploitation) points
    points: np.ndarray           # qEHVI points (batch_size, 5)
    explore_points: np.ndarray   # MO-MESMO points (5-batch_size, 5)
    full_entropy: float          # Combined entropy of all points
    full_qehvi: float            # Combined EHVI of all points
    
    def __repr__(self):
        return (f"QEHVIBatch(q={self.batch_size}, "
                f"Entropy={self.full_entropy:.4f}, "
                f"QEHVI={self.full_qehvi:.4f})")


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
    """Manages acquisition function computation for multi-objective BO."""
    
    def __init__(self, bounds: torch.Tensor):
        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)
    
    # ========================================================================
    #  MAIN INTERFACE
    # ========================================================================
    
    def compute_pareto_front(self, Y_mo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Pareto front and reference point from objectives.
        
        Args:
            Y_mo: Objective values (n, 2) - both objectives should be maximized
        
        Returns:
            pareto_Y: Pareto front points (m, 2)
            ref_point: Reference point for hypervolume (2,)
        """
        mask = is_non_dominated(Y_mo)
        pareto = Y_mo[mask]
        ref_point = pareto.min(dim=0).values - 0.1 * torch.ones(2, dtype=DTYPE, device=DEVICE)
        return pareto, ref_point
    
    def compute_all_allocation_options(self,
                                       model: MultiTaskGP,
                                       X_pool: torch.Tensor,
                                       pareto_Y: torch.Tensor,
                                       ref_point: torch.Tensor,
                                       mc_samples: int = 128) -> AcquisitionData:
        """Compute all 6 allocation options (0-5 exploit points).
        
        Args:
            model: Fitted MultiTaskGP model
            X_pool: Candidate pool for entropy ranking (not used for optimization)
            pareto_Y: Current Pareto front (m, 2)
            ref_point: Reference point for hypervolume (2,)
            mc_samples: MC samples for acquisition functions
        
        Returns:
            AcquisitionData with 6 allocation options
        """
        print("[AcqFn] Computing acquisition functions.")
        
        # Reduce samples for MO-MESMO (it's more expensive)
        mc_samples_entropy = min(mc_samples // 2, 64)
        print(f"[AcqFn] Using {mc_samples} samples for qEHVI, {mc_samples_entropy} for MO-MESMO")
        
        # Step 1: Compute top entropy reference points (for reporting only)
        print("[AcqFn] Computing top entropy reference points...")
        entropy_points = self._compute_top_entropy_points(
            model=model,
            X_pool=X_pool,
            pareto_Y=pareto_Y,
            ref_point=ref_point,
            n_points=5,
            mc_samples=mc_samples
        )
        
        # Step 2: Create shared qEHVI acquisition function for evaluation
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        qehvi_acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler
        )
        
        # Step 3: Build all 6 allocation options
        qehvi_batches: List[QEHVIBatch] = []
        
        for q_exploit in range(5, -1, -1):  # 5, 4, 3, 2, 1, 0
            q_explore = 5 - q_exploit
            
            print(f"\n[AcqFn] Option {5-q_exploit}: {q_exploit} exploit + {q_explore} explore")
            
            # Get exploitation points
            if q_exploit > 0:
                exploit_points = self._get_qehvi_points(
                    model=model,
                    pareto_Y=pareto_Y,
                    ref_point=ref_point,
                    q=q_exploit,
                    mc_samples=mc_samples
                )
            else:
                exploit_points = np.empty((0, 5))
            
            # Get exploration points
            if q_explore > 0:
                explore_points = self._get_entropy_points(
                    model=model,
                    pareto_Y=pareto_Y,
                    ref_point=ref_point,
                    q=q_explore,
                    mc_samples=mc_samples_entropy
                )
            else:
                explore_points = np.empty((0, 5))
            
            # Combine all points
            if exploit_points.shape[0] > 0 and explore_points.shape[0] > 0:
                all_points = np.vstack([exploit_points, explore_points])
            elif exploit_points.shape[0] > 0:
                all_points = exploit_points
            elif explore_points.shape[0] > 0:
                all_points = explore_points
            else:
                all_points = np.empty((0, 5))
            
            # Compute combined metrics
            if all_points.shape[0] > 0:
                all_points_torch = torch.tensor(all_points, dtype=DTYPE, device=DEVICE)
                combined_entropy = self._compute_joint_entropy(model, all_points_torch)
                combined_qehvi = self._evaluate_qehvi(qehvi_acq, all_points_torch)
            else:
                combined_entropy = 0.0
                combined_qehvi = 0.0
            
            # Create batch
            batch = QEHVIBatch(
                batch_size=q_exploit,
                points=exploit_points,
                explore_points=explore_points,
                full_entropy=float(combined_entropy),
                full_qehvi=float(combined_qehvi)
            )
            qehvi_batches.append(batch)
            
            print(f"  → Entropy: {batch.full_entropy:.4f}, QEHVI: {batch.full_qehvi:.4f}")
        
        return AcquisitionData(
            qehvi_batches=qehvi_batches,
            entropy_points=entropy_points
        )
    
    # ========================================================================
    #  POINT SELECTION
    # ========================================================================
    
    def _get_qehvi_points(self,
                         model: MultiTaskGP,
                         pareto_Y: torch.Tensor,
                         ref_point: torch.Tensor,
                         q: int,
                         mc_samples: int) -> np.ndarray:
        """Optimize q points using qEHVI (exploitation).
        
        Returns:
            points: (q, 5) numpy array of compositions
        """
        print(f"  Optimizing qEHVI with q={q}...")
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler
        )
        
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([
                torch.zeros(5, dtype=DTYPE, device=DEVICE),
                torch.ones(5, dtype=DTYPE, device=DEVICE),
            ]),
            q=q,
            num_restarts=3,
            raw_samples=128,
            equality_constraints=self._get_simplex_equality_tuple(),
            options={"batch_limit": 5, "maxiter": 200}
        )
        
        # Denormalize
        candidates_denorm = self._denormalize(candidates, self.bounds)
        return candidates_denorm.cpu().numpy()
    
    def _get_entropy_points(self,
                           model: MultiTaskGP,
                           pareto_Y: torch.Tensor,
                           ref_point: torch.Tensor,
                           q: int,
                           mc_samples: int) -> np.ndarray:
        """Optimize q points using MO-MESMO (exploration) with fallback.
        
        Returns:
            points: (q, 5) numpy array of compositions
        """
        print(f"  Optimizing MO-MESMO with q={q}...")
        
        try:
            # Try MO-MESMO
            points = self._optimize_mesmo(
                model=model,
                pareto_Y=pareto_Y,
                ref_point=ref_point,
                q=q,
                mc_samples=mc_samples
            )
            print(f"  ✓ MO-MESMO succeeded")
            return points
            
        except Exception as e:
            print(f"  ✗ MO-MESMO failed: {e}")
            print(f"  Using fallback: Simple entropy sampling")
            return self._fallback_entropy_sampling(model, q)
    
    def _optimize_mesmo(self,
                       model: MultiTaskGP,
                       pareto_Y: torch.Tensor,
                       ref_point: torch.Tensor,
                       q: int,
                       mc_samples: int) -> np.ndarray:
        """Optimize using MO-MESMO (can raise exceptions)."""
        partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        hypercell_bounds = partitioning.hypercell_bounds.unsqueeze(0)
        
        acq = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model=model,
            hypercell_bounds=hypercell_bounds,
            estimation_type='LB',
            num_samples=mc_samples
        )
        
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
        
        candidates, _ = optimize_acqf(
            acq_function=safe_acq,
            bounds=torch.stack([
                torch.zeros(5, dtype=DTYPE, device=DEVICE),
                torch.ones(5, dtype=DTYPE, device=DEVICE),
            ]),
            q=q,
            num_restarts=3,
            raw_samples=128,
            equality_constraints=self._get_simplex_equality_tuple(),
            options={"batch_limit": 5, "maxiter": 200}
        )
        
        candidates_denorm = self._denormalize(candidates, self.bounds)
        return candidates_denorm.cpu().numpy()
    
    def _fallback_entropy_sampling(self, model: MultiTaskGP, q: int) -> np.ndarray:
        """Fallback: Sample points with highest GP uncertainty.
        
        This is a simple entropy estimator based on posterior variance.
        """
        from core.design_space import DesignSpace
        
        # Create temporary design space
        design_space = DesignSpace(
            min_comp=float(self.bounds[0, 0].item()),
            max_comp=float(self.bounds[1, 0].item()),
            step=0.01  # Coarser for speed
        )
        
        # Sample candidates
        n_candidates = min(2000, len(design_space.space))
        candidates = design_space.sample(n_candidates, method="sobol")
        candidates_torch = torch.tensor(candidates, dtype=DTYPE, device=DEVICE)
        
        # Compute uncertainty (sum of variances for both tasks)
        n = candidates_torch.shape[0]
        X_norm = normalize(candidates_torch, self.bounds)
        
        # Query both tasks
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
            
            # Total uncertainty (sum of log variances)
            uncertainty = torch.log(var_task_0.clamp_min(1e-12)) + torch.log(var_task_1.clamp_min(1e-12))
        
        # Select top-q most uncertain points
        top_indices = torch.argsort(uncertainty, descending=True)[:q]
        return candidates[top_indices.cpu().numpy()]
    
    # ========================================================================
    #  EVALUATION HELPERS
    # ========================================================================
    
    def _evaluate_qehvi(self, acq_function, X: torch.Tensor) -> float:
        """Evaluate qEHVI on a batch of points.
        
        Args:
            acq_function: qLogExpectedHypervolumeImprovement instance
            X: Points to evaluate (q, 5) - unnormalized
        
        Returns:
            QEHVI value (scalar)
        """
        if X.shape[0] == 0:
            return 0.0
        
        X_norm = normalize(X, self.bounds)
        
        with torch.no_grad():
            # For MultiTaskGP with multi-output, pass normalized X
            # The acquisition function handles task indices internally
            acq_val = acq_function(X_norm.unsqueeze(0))  # Shape: (1, q, 5)
        return float(acq_val)
    
    def _compute_joint_entropy(self, model: MultiTaskGP, X: torch.Tensor) -> float:
        """Compute joint entropy for a batch of points (both tasks).
        
        Args:
            model: MultiTaskGP model
            X: Points (n, 5) - unnormalized
        
        Returns:
            Joint entropy value (scalar)
        """
        if X.shape[0] == 0:
            return 0.0
        
        n = X.shape[0]
        X_norm = normalize(X, self.bounds)
        
        # Query both tasks
        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        X_task_0 = torch.cat([X_norm, task_0], dim=-1)
        X_task_1 = torch.cat([X_norm, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)
        
        with torch.no_grad():
            post = model.posterior(X_both)
            variance = post.variance.clamp_min(1e-12)  # (2n, 1)
            log_det = torch.sum(torch.log(variance))
            d = 2 * n  # n points × 2 tasks
            entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + log_det)
        
        return float(entropy)
    
    def _compute_top_entropy_points(self,
                                    model: MultiTaskGP,
                                    X_pool: torch.Tensor,
                                    pareto_Y: torch.Tensor,
                                    ref_point: torch.Tensor,
                                    n_points: int,
                                    mc_samples: int) -> List[EntropyPoint]:
        """Compute top-n points from pool ranked by entropy (for reference).
        
        Returns:
            List of EntropyPoint objects
        """
        n = X_pool.shape[0]
        X_pool_norm = normalize(X_pool, self.bounds)
        
        # Compute entropy for each point
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
            
            # Entropy = sum of log variances
            entropy = 0.5 * torch.log(2 * np.pi * np.e * var_task_0.clamp_min(1e-12)) + \
                      0.5 * torch.log(2 * np.pi * np.e * var_task_1.clamp_min(1e-12))
        
        top_indices = torch.argsort(entropy, descending=True)[:n_points]
        
        # Setup qEHVI for evaluating these points
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler
        )
        
        entropy_points = []
        for rank, idx in enumerate(top_indices, start=1):
            point = X_pool[idx].cpu().numpy()
            ent = float(entropy[idx])
            
            # Evaluate qEHVI at this point
            X_single = X_pool_norm[idx:idx+1]
            with torch.no_grad():
                qehvi_val = float(acq(X_single.unsqueeze(0)))
            
            entropy_points.append(EntropyPoint(
                rank=rank,
                point=point,
                entropy=ent,
                qehvi=qehvi_val
            ))
        
        return entropy_points
    
    # ========================================================================
    #  UTILITIES
    # ========================================================================
    
    def _denormalize(self, X_norm: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0, 1] to original bounds."""
        return bounds[0] + X_norm * (bounds[1] - bounds[0])
    
    def _get_simplex_equality_tuple(self):
        """Constraint: compositions sum to 1 (for normalized space)."""
        lower = float(self.bounds[0, 0].item())
        upper = float(self.bounds[1, 0].item())
        coef = upper - lower
        rhs = 1.0 - 5.0 * lower
        
        indices = torch.arange(5, device=DEVICE, dtype=torch.long)
        coefficients = torch.full((5,), coef, device=DEVICE, dtype=DTYPE)
        
        return [(indices, coefficients, rhs)]