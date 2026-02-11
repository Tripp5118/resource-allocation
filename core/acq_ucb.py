# core/acq_ucb.py

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from botorch.models import MultiTaskGP
from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement   
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


@dataclass
class AllocationOption:
    """Represents one exploitation/exploration allocation strategy.
    
    Args:
        num_exploitation: Number of exploitation points
        num_exploration: Number of exploration points
        exploitation_points: Exploitation candidate points (num_exploitation, d)
        exploration_points: Exploration candidate points (num_exploration, d)
        hypervolume_improvement: qEHVI score for exploitation points
        information_gain: Joint entropy score for exploration points
        total_batch_size: Total points per iteration (default 5)
    """
    num_exploitation: int
    num_exploration: int
    exploitation_points: np.ndarray
    exploration_points: np.ndarray
    hypervolume_improvement: float
    information_gain: float
    total_batch_size: int = 5

    @property
    def all_points(self) -> np.ndarray:
        """Get all points (exploitation + exploration) as a single array."""
        if self.exploitation_points.shape[0] > 0 and self.exploration_points.shape[0] > 0:
            return np.vstack([self.exploitation_points, self.exploration_points])
        elif self.exploitation_points.shape[0] > 0:
            return self.exploitation_points
        elif self.exploration_points.shape[0] > 0:
            return self.exploration_points
        else:
            return np.empty((0, self.exploitation_points.shape[1]))

    def __repr__(self):
        return (
            f"AllocationOption("
            f"exploit={self.num_exploitation}, "
            f"explore={self.num_exploration}, "
            f"HVI={self.hypervolume_improvement:.4f}, "
            f"InfoGain={self.information_gain:.4f})"
        )


@dataclass
class AllocationResults:
    """Container for all allocation options computed during acquisition.
    
    Args:
        options: List of allocation options with different exploit/explore splits
        metadata: Optional dictionary for debugging information
    """
    options: List[AllocationOption]
    metadata: Optional[dict] = None

    def __repr__(self):
        result = "AllocationResults:\n"
        for i, option in enumerate(self.options):
            result += f"  Option {i}: {option}\n"
        return result


class AcquisitionFunctionManager:
    """Manages acquisition function computation for 2-objective BO with qUCB.
    
    Uses qUpperConfidenceBound with different beta values for exploitation vs exploration:
    - Low beta (exploitation): Favors high-mean regions (greedy)
    - High beta (exploration): Favors high-uncertainty regions (diverse)
    
    Points are optimized using qUCB, but evaluated using qEHVI and joint entropy
    for interpretable exploitation/exploration metrics.
    
    Enforces simplex constraint: sum(x_i) = 1 with x_i in [min_i, max_i].
    
    Beta value guidelines:
        - Exploitation: β ∈ [0.01, 0.5]
            * 0.01-0.05: Very greedy, minimal exploration
            * 0.1-0.3: Moderate exploitation
            * 0.5+: Balanced
        - Exploration: β ∈ [1.0, 10.0]
            * 1.0-2.0: Conservative exploration
            * 3.0-5.0: Moderate exploration (recommended)
            * 5.0-10.0: Aggressive exploration
    
    Example:
        >>> bounds = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 1.0]])
        >>> manager = AcquisitionFunctionManager(
        ...     bounds=bounds,
        ...     exploitation_beta=0.05,
        ...     exploration_beta=2.0
        ... )
        >>> pareto_front, ref_point = manager.compute_pareto_front(objectives)
        >>> results = manager.compute_all_allocations(
        ...     model=model,
        ...     pareto_front=pareto_front,
        ...     reference_point=ref_point
        ... )
        >>> best = results.get_best_option("balanced")
    """
    
    def __init__(
        self,
        bounds: torch.Tensor,
        num_restarts: int = 3,
        raw_samples: int = 256,
        exploitation_beta: float = 0.05,
        exploration_beta: float = 2.0,
        objective_weights: Optional[List[float]] = None
    ):
        """Initialize the acquisition function manager.
        
        Args:
            bounds: (2, d) tensor of [lower_bounds, upper_bounds]
            num_restarts: Number of restarts for acquisition optimization
            raw_samples: Number of raw samples for initialization
            exploitation_beta: Beta for exploitation - lower = more greedy
                Recommended range: [0.01, 0.5], default 0.05
            exploration_beta: Beta for exploration - higher = more diverse
                Recommended range: [1.0, 10.0], default 2.0
            objective_weights: Weights for scalarizing objectives [w1, w2]
                Default [0.5, 0.5] gives equal weight to both objectives
        """
        if bounds.shape[0] != 2:
            raise ValueError(f"bounds must have shape (2, d), got {bounds.shape}")
        if bounds.shape[1] < 1 or bounds.shape[1] > 10:
            raise ValueError(f"Input dimension must be between 1 and 10, got {bounds.shape[1]}")
        if torch.any(bounds[0] >= bounds[1]):
            raise ValueError("Lower bounds must be strictly less than upper bounds")
        if exploitation_beta < 0:
            raise ValueError(f"exploitation_beta must be non-negative, got {exploitation_beta}")
        if exploration_beta < 0:
            raise ValueError(f"exploration_beta must be non-negative, got {exploration_beta}")
        if exploitation_beta >= exploration_beta:
            raise ValueError(
                f"exploitation_beta ({exploitation_beta}) should be less than "
                f"exploration_beta ({exploration_beta}) for meaningful distinction"
            )
        
        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)
        self.input_dim = self.bounds.shape[1]
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.exploitation_beta = exploitation_beta
        self.exploration_beta = exploration_beta
        
        if objective_weights is None:
            objective_weights = [0.5, 0.5]
        if len(objective_weights) != 2:
            raise ValueError(f"objective_weights must have length 2, got {len(objective_weights)}")
        if not np.isclose(sum(objective_weights), 1.0):
            raise ValueError(f"objective_weights must sum to 1.0, got {sum(objective_weights)}")
        
        self.objective_weights = torch.tensor(
            objective_weights, dtype=DTYPE, device=DEVICE
        )

    def compute_pareto_front(
        self,
        objectives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Pareto front and reference point from objective values.
        
        Args:
            objectives: (n, 2) tensor of objective values (both maximized)
        
        Returns:
            pareto_front: (m, 2) tensor of Pareto-optimal points
            reference_point: (2,) tensor for hypervolume calculations
        """
        if objectives.shape[0] == 0:
            raise ValueError("Cannot compute Pareto front from empty objectives")
        
        pareto_mask = is_non_dominated(objectives)
        pareto_front = objectives[pareto_mask]
        
        if len(pareto_front) == 0:
            raise ValueError("No non-dominated points found")
        
        reference_point = pareto_front.min(dim=0).values - 0.1 * torch.ones(
            2, dtype=DTYPE, device=DEVICE
        )
        
        return pareto_front, reference_point

    def compute_single_allocation(
        self,
        model: MultiTaskGP,
        num_exploitation: int,
        num_exploration: int,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256
    ) -> AllocationResults:
        """Compute a single allocation option with fixed exploit/explore split.
        
        More efficient than compute_all_allocations when you only need one split.
        
        Args:
            model: Fitted MultiTaskGP model
            num_exploitation: Number of exploitation points
            num_exploration: Number of exploration points
            pareto_front: Current Pareto front (m, 2)
            reference_point: Reference point for hypervolume (2,)
            mc_samples: Monte Carlo samples for acquisition
        
        Returns:
            AllocationResults with a single option
        """
        total_batch = num_exploitation + num_exploration
        if total_batch != 5:
            raise ValueError(f"Total batch size must be 5, got {total_batch}")
        
        print(f"[AcqFn] Computing allocation: {num_exploitation} exploit + {num_exploration} explore")
        
        option = self._compute_allocation_option(
            model=model,
            num_exploitation=num_exploitation,
            num_exploration=num_exploration,
            pareto_front=pareto_front,
            reference_point=reference_point,
            mc_samples=mc_samples
        )
        
        print(f"  → HVI: {option.hypervolume_improvement:.4f}, InfoGain: {option.information_gain:.4f}")
        
        metadata = {
            "exploitation_beta": self.exploitation_beta,
            "exploration_beta": self.exploration_beta,
            "mc_samples": mc_samples
        }
        
        return AllocationResults(options=[option], metadata=metadata)

    def compute_all_allocations(
        self,
        model: MultiTaskGP,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256
    ) -> AllocationResults:
        """Compute all allocation options (6 options for batch size 5).
        
        Options:
            - Option 0: 5 exploit + 0 explore (pure exploitation)
            - Option 1: 4 exploit + 1 explore
            - Option 2: 3 exploit + 2 explore (balanced)
            - Option 3: 2 exploit + 3 explore
            - Option 4: 1 exploit + 4 explore
            - Option 5: 0 exploit + 5 explore (pure exploration)
        
        Args:
            model: Fitted MultiTaskGP model
            pareto_front: Current Pareto front (m, 2)
            reference_point: Reference point for hypervolume (2,)
            mc_samples: Monte Carlo samples for acquisition
        
        Returns:
            AllocationResults with all 6 options
        """
        print("[AcqFn] Computing all allocation options with qUCB")
        
        total_batch_size = 5
        options = []
        
        for num_exploitation in range(total_batch_size, -1, -1):
            num_exploration = total_batch_size - num_exploitation
            option_index = total_batch_size - num_exploitation
            
            print(f"\n[AcqFn] Option {option_index}: {num_exploitation} exploit + {num_exploration} explore")
            
            option = self._compute_allocation_option(
                model=model,
                num_exploitation=num_exploitation,
                num_exploration=num_exploration,
                pareto_front=pareto_front,
                reference_point=reference_point,
                mc_samples=mc_samples
            )
            
            options.append(option)
            print(f"  → HVI: {option.hypervolume_improvement:.4f}, InfoGain: {option.information_gain:.4f}")
        
        metadata = {
            "exploitation_beta": self.exploitation_beta,
            "exploration_beta": self.exploration_beta,
            "mc_samples": mc_samples,
            "total_options": len(options)
        }
        
        return AllocationResults(options=options, metadata=metadata)

    def _compute_allocation_option(
        self,
        model: MultiTaskGP,
        num_exploitation: int,
        num_exploration: int,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int
    ) -> AllocationOption:
        """Compute a single allocation option with qUCB optimization and qEHVI/entropy evaluation."""
        
        if num_exploitation > 0:
            exploitation_points = self._optimize_acquisition(
                model=model,
                batch_size=num_exploitation,
                beta=self.exploitation_beta,
                mc_samples=mc_samples,
                mode="exploitation"
            )
        else:
            exploitation_points = np.empty((0, self.input_dim))
        
        if num_exploration > 0:
            exploration_points = self._optimize_acquisition(
                model=model,
                batch_size=num_exploration,
                beta=self.exploration_beta,
                mc_samples=mc_samples,
                mode="exploration"
            )
        else:
            exploration_points = np.empty((0, self.input_dim))
        
        option = AllocationOption(
            num_exploitation=num_exploitation,
            num_exploration=num_exploration,
            exploitation_points=exploitation_points,
            exploration_points=exploration_points,
            hypervolume_improvement=0.0,
            information_gain=0.0,
            total_batch_size=5
        )
        
        hv_improvement, info_gain = self._compute_allocation_metrics(
            model=model,
            option=option,
            pareto_front=pareto_front,
            reference_point=reference_point,
            mc_samples=mc_samples
        )
        
        option.hypervolume_improvement = hv_improvement
        option.information_gain = info_gain
        
        return option

    def _optimize_acquisition(
        self,
        model: MultiTaskGP,
        batch_size: int,
        beta: float,
        mc_samples: int,
        mode: str
    ) -> np.ndarray:
        """Optimize qUCB acquisition function to find candidate points.
        
        Args:
            model: MultiTaskGP model
            batch_size: Number of points to optimize
            beta: UCB beta parameter
            mc_samples: Monte Carlo samples
            mode: "exploitation" or "exploration" (for logging)
        
        Returns:
            Optimized points in original space (batch_size, d)
        """
        print(f"  Optimizing {mode} qUCB (beta={beta:.2f}, q={batch_size})")
        
        from botorch.acquisition.objective import ScalarizedPosteriorTransform
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        
        posterior_transform = ScalarizedPosteriorTransform(
            weights=self.objective_weights
        )
        
        acquisition_function = qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            posterior_transform=posterior_transform
        )
        
        simplex_constraint = self._get_simplex_constraint()
        
        candidates, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=torch.stack([
                torch.zeros(self.input_dim, dtype=DTYPE, device=DEVICE),
                torch.ones(self.input_dim, dtype=DTYPE, device=DEVICE)
            ]),
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            equality_constraints=simplex_constraint,
            options={"batch_limit": 5, "maxiter": 200}
        )
        
        denormalized_candidates = self._denormalize(candidates, self.bounds)
        return denormalized_candidates.cpu().numpy()

    def _compute_allocation_metrics(
        self,
        model: MultiTaskGP,
        option: AllocationOption,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256
    ) -> Tuple[float, float]:
        """Compute evaluation metrics for an allocation option.
        
        Evaluates BOTH qEHVI and entropy on ALL points in the allocation,
        regardless of whether they came from exploitation or exploration optimization.
        
        Args:
            model: Fitted GP model
            option: AllocationOption to evaluate
            pareto_front: Current Pareto front
            reference_point: Reference point for hypervolume
            mc_samples: MC samples for qEHVI
        
        Returns:
            (hypervolume_improvement, information_gain) tuple
        """
        # Combine all points from this allocation
        all_points = option.all_points  # Uses the property to get exploitation + exploration
        
        # Evaluate qEHVI on ALL points
        hv_improvement = self._evaluate_hypervolume_improvement(
            model=model,
            points=all_points,
            pareto_front=pareto_front,
            reference_point=reference_point,
            mc_samples=mc_samples
        )
        
        # Evaluate entropy on ALL points
        info_gain = self._evaluate_mutual_information(
            model=model,
            points=all_points
        )
        
        return hv_improvement, info_gain


    def _evaluate_hypervolume_improvement(
        self,
        model: MultiTaskGP,
        points: np.ndarray,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256
    ) -> float:
        """Evaluate qLogEHVI on a set of points.
        
        This is NOT used for optimization, only for scoring allocation options.
        Uses log-scale for better numerical stability.
        
        Args:
            model: Fitted GP model
            points: Points to evaluate (n, d) in original space
            pareto_front: Current Pareto front
            reference_point: Reference point for hypervolume
            mc_samples: MC samples for qEHVI computation
        
        Returns:
            Expected hypervolume improvement in log scale (scalar)
        """
        if points.shape[0] == 0:
            return 0.0
        
        points_tensor = torch.tensor(points, dtype=DTYPE, device=DEVICE)
        normalized_points = normalize(points_tensor, self.bounds)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=reference_point, Y=pareto_front)
        
        qehvi = qExpectedHypervolumeImprovement   (
            model=model,
            ref_point=reference_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
        
        with torch.no_grad():
            value = qehvi(normalized_points.unsqueeze(0))
        
        return float(value.item())

    def _evaluate_mutual_information(
        self,
        model: MultiTaskGP,
        points: np.ndarray,
        num_optimal_samples: int = 10
    ) -> float:
        """
        Calculate mutual information between predictions at candidate points
        and the optimal input-output pairs for both tasks.
        
        Automatically samples optimal points by:
        1. Drawing posterior samples from the GP
        2. Finding Pareto front for each sample using compute_pareto_front
        3. Computing conditional entropy for each optimal sample
        
        Args:
            model: Fitted MultiTaskGP model
            points: Candidate points to evaluate (n x d) - already normalized in [0,1]
            num_optimal_samples: Number of optimal point samples (default 10)
        
        Returns:
            Mutual information score (scalar)
        """
        if points.shape[0] == 0:
            return 0.0
        
        points_tensor = torch.tensor(points, dtype=DTYPE, device=DEVICE)
        n = points_tensor.shape[0]
        
        # Create task indicators for both tasks
        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        
        X_task_0 = torch.cat([points_tensor, task_0], dim=-1)
        X_task_1 = torch.cat([points_tensor, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)  # Shape: (2n, d+1)
        
        with torch.no_grad():
            # ============================================
            # TERM 1: Prior entropy H[p(y|D_n)]
            # ============================================
            posterior = model.posterior(X_both)
            covariance_matrix = posterior.covariance_matrix.squeeze(0)
            covariance_matrix = covariance_matrix + 1e-6 * torch.eye(
                covariance_matrix.shape[0],
                dtype=DTYPE,
                device=DEVICE
            )
            
            try:
                log_det_prior = torch.logdet(covariance_matrix)
                if torch.isnan(log_det_prior) or torch.isinf(log_det_prior):
                    variances = posterior.variance.squeeze(-1).clamp_min(1e-12)
                    log_det_prior = torch.sum(torch.log(variances))
            except RuntimeError:
                variances = posterior.variance.squeeze(-1).clamp_min(1e-12)
                log_det_prior = torch.sum(torch.log(variances))
            
            d_total = 2 * n
            H_prior = 0.5 * (d_total * np.log(2 * np.pi * np.e) + log_det_prior)
            
            # ============================================
            # SAMPLE OPTIMAL POINTS
            # ============================================
            optimal_samples = self._sample_optimal_points(
                model=model,
                num_samples=num_optimal_samples
            )
            
            # ============================================
            # TERM 2: Expected posterior entropy E[H[p(y|x, D_n, (X*, Y*))]]
            # ============================================
            H_posterior_expected = 0.0
            
            for i in range(optimal_samples.shape[0]):
                # Get optimal point sample (already in normalized [0,1] space)
                opt_point = optimal_samples[i:i+1]  # Shape: (1, d)
                
                # Evaluate model at optimal point for both tasks
                opt_task_0 = torch.cat([opt_point, torch.zeros(1, 1, dtype=DTYPE, device=DEVICE)], dim=-1)
                opt_task_1 = torch.cat([opt_point, torch.ones(1, 1, dtype=DTYPE, device=DEVICE)], dim=-1)
                opt_both = torch.cat([opt_task_0, opt_task_1], dim=0)
                
                # Create augmented dataset including the optimal point observations
                X_augmented = torch.cat([X_both, opt_both], dim=0)
                
                # Compute joint posterior over all points
                posterior_cond = model.posterior(X_augmented)
                cov_augmented = posterior_cond.covariance_matrix.squeeze(0)
                
                # Use Schur complement to get conditional covariance
                # Cov(y_candidates | y_optimal) = Sigma_11 - Sigma_12 @ Sigma_22^{-1} @ Sigma_21
                cov_11 = cov_augmented[:2*n, :2*n]  # Candidate-candidate
                cov_12 = cov_augmented[:2*n, 2*n:]  # Candidate-optimal
                cov_22 = cov_augmented[2*n:, 2*n:]  # Optimal-optimal
                
                # Add jitter for numerical stability
                cov_22 = cov_22 + 1e-6 * torch.eye(cov_22.shape[0], dtype=DTYPE, device=DEVICE)
                
                try:
                    # Conditional covariance via Schur complement
                    cov_22_inv = torch.linalg.inv(cov_22)
                    cov_conditional = cov_11 - cov_12 @ cov_22_inv @ cov_12.T
                    cov_conditional = cov_conditional + 1e-6 * torch.eye(
                        2*n, dtype=DTYPE, device=DEVICE
                    )
                except RuntimeError:
                    # Fallback: use marginal covariance if inversion fails
                    cov_conditional = cov_11 + 1e-6 * torch.eye(2*n, dtype=DTYPE, device=DEVICE)
                
                try:
                    log_det_cond = torch.logdet(cov_conditional)
                    if torch.isnan(log_det_cond) or torch.isinf(log_det_cond):
                        variances_cond = torch.diagonal(cov_conditional).clamp_min(1e-12)
                        log_det_cond = torch.sum(torch.log(variances_cond))
                except RuntimeError:
                    variances_cond = torch.diagonal(cov_conditional).clamp_min(1e-12)
                    log_det_cond = torch.sum(torch.log(variances_cond))
                
                H_cond = 0.5 * (d_total * np.log(2 * np.pi * np.e) + log_det_cond)
                H_posterior_expected += H_cond / optimal_samples.shape[0]
            
            # ============================================
            # Mutual Information = Prior Entropy - Expected Posterior Entropy
            # ============================================
            mutual_info = H_prior - H_posterior_expected
        
        return float(mutual_info.item())


    def _sample_optimal_points(
        self,
        model: MultiTaskGP,
        num_samples: int = 10,
        num_grid_points: int = 1000
    ) -> torch.Tensor:
        """
        Sample potential optimal points by:
        1. Drawing posterior samples from the GP
        2. For each sample, finding the Pareto front
        3. Randomly selecting points from each Pareto front
        
        All inputs/outputs are already normalized in [0,1] space.
        
        Args:
            model: Fitted MultiTaskGP model
            num_samples: Number of optimal point samples to return
            num_grid_points: Number of grid points to evaluate
        
        Returns:
            Tensor of sampled optimal points (num_samples, d) in normalized [0,1] space
        """
        # Generate grid of candidate points in normalized [0, 1] space
        grid_points = torch.rand(
            num_grid_points, 
            self.input_dim, 
            dtype=DTYPE, 
            device=DEVICE
        )
        
        # Apply simplex constraint: sum(x_i) = 1 in the ORIGINAL space
        # Since bounds are [min_i, max_i] and we're in normalized [0,1]:
        # original_i = min_i + (max_i - min_i) * normalized_i
        # sum(original) = sum(min) + sum((max-min) * normalized) = 1
        # => sum((max-min) * normalized) = 1 - sum(min)
        
        lower_bounds = self.bounds[0, :]
        upper_bounds = self.bounds[1, :]
        range_bounds = upper_bounds - lower_bounds
        target_sum = 1.0 - lower_bounds.sum().item()
        
        # Adjust last dimension to satisfy constraint
        # sum_{i=0}^{d-2} range_i * x_i + range_{d-1} * x_{d-1} = target_sum
        # => x_{d-1} = (target_sum - sum_{i=0}^{d-2} range_i * x_i) / range_{d-1}
        grid_sum = (grid_points[:, :-1] * range_bounds[:-1]).sum(dim=1)
        grid_points[:, -1] = (target_sum - grid_sum) / range_bounds[-1]
        
        # Clamp to [0, 1] to ensure validity
        grid_points = torch.clamp(grid_points, 0.0, 1.0)
        
        optimal_samples = []
        
        for sample_idx in range(num_samples):
            # Create task indices for both objectives
            n_grid = grid_points.shape[0]
            task_0 = torch.zeros(n_grid, 1, dtype=DTYPE, device=DEVICE)
            task_1 = torch.ones(n_grid, 1, dtype=DTYPE, device=DEVICE)
            
            X_task_0 = torch.cat([grid_points, task_0], dim=-1)
            X_task_1 = torch.cat([grid_points, task_1], dim=-1)
            
            # Draw posterior samples for both objectives
            posterior_0 = model.posterior(X_task_0)
            posterior_1 = model.posterior(X_task_1)
            
            sample_0 = posterior_0.rsample(torch.Size([1])).squeeze(0).squeeze(-1)  # (n_grid,)
            sample_1 = posterior_1.rsample(torch.Size([1])).squeeze(0).squeeze(-1)  # (n_grid,)
            
            # Stack into objectives matrix (n_grid, 2)
            objectives = torch.stack([sample_0, sample_1], dim=-1)
            
            # Find Pareto front using your existing method
            try:
                pareto_front, _ = self.compute_pareto_front(objectives)
                
                # Find which grid points are on the Pareto front
                pareto_mask = is_non_dominated(objectives)
                pareto_inputs = grid_points[pareto_mask]
                
                # Randomly select one point from the Pareto front
                if pareto_inputs.shape[0] > 0:
                    random_idx = torch.randint(0, pareto_inputs.shape[0], (1,))
                    optimal_samples.append(pareto_inputs[random_idx])
                else:
                    # Fallback: use random point if no Pareto front found
                    random_idx = torch.randint(0, n_grid, (1,))
                    optimal_samples.append(grid_points[random_idx])
                    
            except (ValueError, RuntimeError) as e:
                # Fallback: use random point if Pareto computation fails
                print(f"[Warning] Pareto computation failed in sample {sample_idx}: {e}")
                random_idx = torch.randint(0, n_grid, (1,))
                optimal_samples.append(grid_points[random_idx])
        
        return torch.cat(optimal_samples, dim=0)  # (num_samples, d)

    def _concatenate_points(
        self,
        exploitation_points: np.ndarray,
        exploration_points: np.ndarray
    ) -> torch.Tensor:
        """Concatenate exploitation and exploration points into single tensor."""
        if exploitation_points.shape[0] > 0 and exploration_points.shape[0] > 0:
            combined = np.vstack([exploitation_points, exploration_points])
        elif exploitation_points.shape[0] > 0:
            combined = exploitation_points
        elif exploration_points.shape[0] > 0:
            combined = exploration_points
        else:
            return torch.empty((0, self.input_dim), dtype=DTYPE, device=DEVICE)
        
        return torch.tensor(combined, dtype=DTYPE, device=DEVICE)

    def _denormalize(
        self,
        normalized_points: torch.Tensor,
        bounds: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize points from [0, 1] to original bounds."""
        return bounds[0] + normalized_points * (bounds[1] - bounds[0])

    def _get_simplex_constraint(self):
        """Get equality constraint for simplex: sum(x_i) = 1.
        
        In normalized space [0,1]^d, if x = min + (max - min) * x_norm,
        then sum(x) = 1 implies sum((max - min) * x_norm) = 1 - sum(min).
        
        Returns:
            List with constraint tuple (indices, coefficients, rhs)
        """
        lower_bounds = self.bounds[0, :]
        upper_bounds = self.bounds[1, :]
        coefficients = upper_bounds - lower_bounds
        
        right_hand_side = 1.0 - float(lower_bounds.sum().item())
        
        indices = torch.arange(self.input_dim, device=DEVICE, dtype=torch.long)
        constraint_coefficients = coefficients.to(dtype=DTYPE, device=DEVICE)
        
        return [(indices, constraint_coefficients, right_hand_side)]