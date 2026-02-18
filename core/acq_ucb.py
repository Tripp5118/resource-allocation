# core/acq_ucb.py

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Sequence
from dataclasses import dataclass

from botorch.models import MultiTaskGP
from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


@dataclass
class AllocationOption:
    """Represents one exploitation/exploration allocation strategy."""
    num_exploitation: int
    num_exploration: int
    exploitation_points: np.ndarray
    exploration_points: np.ndarray
    hypervolume_improvement: float
    information_gain: float
    total_batch_size: int = 5

    @property
    def all_points(self) -> np.ndarray:
        if self.exploitation_points.shape[0] > 0 and self.exploration_points.shape[0] > 0:
            return np.vstack([self.exploitation_points, self.exploration_points])
        elif self.exploitation_points.shape[0] > 0:
            return self.exploitation_points
        elif self.exploration_points.shape[0] > 0:
            return self.exploration_points
        else:
            return np.empty((0, self.exploitation_points.shape[1]))


@dataclass
class AllocationResults:
    """Container for all allocation options computed during acquisition."""
    options: List[AllocationOption]
    metadata: Optional[dict] = None


class AcquisitionFunctionManager:
    """Manages acquisition function computation for 2-objective BO.
    
    Now supports discrete optimization via design space specification.
    """
    
    def __init__(
        self,
        bounds: torch.Tensor,
        num_restarts: int = 10,
        raw_samples: int = 512,
        exploration_beta: float = 2.0,
        objective_weights: Optional[List[float]] = None,
        # NEW: Design space parameters for discrete optimization
        design_space: Optional[object] = None,  # Your DesignSpace object
        discrete_choices: Optional[torch.Tensor] = None,  # Pre-computed discrete choices
        use_discrete: bool = False,  # Whether to use discrete optimization
    ):
        """Initialize the acquisition function manager.
        
        Args:
            bounds: (2, d) tensor of [lower_bounds, upper_bounds]
            num_restarts: Number of restarts for acquisition optimization
            raw_samples: Number of raw samples for initialization
            exploration_beta: Beta for exploration
            objective_weights: Weights for scalarizing objectives [w1, w2]
            design_space: DesignSpace object with discrete grid
            discrete_choices: Pre-computed tensor of valid discrete points (n_choices, d)
            use_discrete: Whether to use discrete optimization
        """
        if bounds.shape[0] != 2:
            raise ValueError(f"bounds must have shape (2, d), got {bounds.shape}")
            
        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)
        self.input_dim = self.bounds.shape[1]
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.exploration_beta = exploration_beta
        
        # Discrete optimization settings
        self.use_discrete = use_discrete
        self.design_space = design_space
        self.discrete_choices = None
        
        # Build discrete choices from design space if provided
        if design_space is not None and use_discrete:
            self._build_discrete_choices_from_design_space(design_space)
        elif discrete_choices is not None and use_discrete:
            self.discrete_choices = discrete_choices.to(dtype=DTYPE, device=DEVICE)
            
        # Objective weights
        if objective_weights is None:
            objective_weights = [0.5, 0.5]
        self.objective_weights = torch.tensor(
            objective_weights, dtype=DTYPE, device=DEVICE
        )
        
    def _build_discrete_choices_from_design_space(self, design_space):
        """Build discrete choices tensor from DesignSpace object.
        
        Assumes design_space has a 'space' attribute containing all valid compositions.
        """
        if hasattr(design_space, 'space') and design_space.space is not None:
            # Use the full discrete space
            self.discrete_choices = torch.tensor(
                design_space.space, dtype=DTYPE, device=DEVICE
            )
            print(f"[AcqFn] Built discrete choices: {self.discrete_choices.shape[0]} points")
        elif hasattr(design_space, 'get_all_points'):
            # Alternative method if available
            all_points = design_space.get_all_points()
            self.discrete_choices = torch.tensor(
                all_points, dtype=DTYPE, device=DEVICE
            )
            print(f"[AcqFn] Built discrete choices: {self.discrete_choices.shape[0]} points")
        else:
            raise ValueError("DesignSpace must have 'space' attribute or 'get_all_points' method")
    
    def _build_discrete_dims_from_step(self, step: float) -> Dict[int, List[float]]:
        """Build discrete_dims dictionary for optimize_acqf_mixed_alternating.
        
        Args:
            step: Step size for discretization (e.g., 0.025)
            
        Returns:
            Dictionary mapping dimension indices to allowed values
        """
        discrete_dims = {}
        for i in range(self.input_dim):
            lower = float(self.bounds[0, i])
            upper = float(self.bounds[1, i])
            # Generate allowed values with given step
            n_steps = int(round((upper - lower) / step)) + 1
            allowed_values = [lower + j * step for j in range(n_steps)]
            # Ensure upper bound is included
            if allowed_values[-1] < upper:
                allowed_values.append(upper)
            discrete_dims[i] = allowed_values
        return discrete_dims
    
    def compute_pareto_front(
        self,
        objectives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Pareto front and reference point from objective values."""
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

    def _optimize_qehvi(
        self,
        model: MultiTaskGP,
        batch_size: int,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int
    ) -> np.ndarray:
        """Optimize points using qEHVI (exploitation).
        
        Uses discrete optimization if enabled, otherwise continuous.
        """
        print(f"  Optimizing qEHVI with q={batch_size}...")
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=reference_point, Y=pareto_front)
        
        acquisition_function = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=reference_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
        
        if self.use_discrete and self.discrete_choices is not None:
            # DISCRETE OPTIMIZATION using optimize_acqf_discrete [4]
            print(f"    Using discrete optimization over {self.discrete_choices.shape[0]} choices")
            
            # Normalize discrete choices to [0, 1]
            normalized_choices = normalize(self.discrete_choices, self.bounds)
            
            candidates, _ = optimize_acqf_discrete(
                acq_function=acquisition_function,
                q=batch_size,
                choices=normalized_choices,
                max_batch_size=2048,
                unique=True,  # Ensure unique candidates
            )
            
            # Denormalize back to original space
            denormalized_candidates = self._denormalize(candidates, self.bounds)
            
        else:
            # CONTINUOUS OPTIMIZATION (original behavior)
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

    def _optimize_qucb(
        self,
        model: MultiTaskGP,
        batch_size: int,
        beta: float,
        mc_samples: int
    ) -> np.ndarray:
        """Optimize points using qUCB (exploration).
        
        Uses discrete optimization if enabled, otherwise continuous.
        """
        print(f"  Optimizing qUCB with q={batch_size}, beta={beta}...")
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        posterior_transform = ScalarizedPosteriorTransform(weights=self.objective_weights)
        
        acquisition_function = qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            posterior_transform=posterior_transform
        )
        
        if self.use_discrete and self.discrete_choices is not None:
            # DISCRETE OPTIMIZATION [4]
            print(f"    Using discrete optimization over {self.discrete_choices.shape[0]} choices")
            
            normalized_choices = normalize(self.discrete_choices, self.bounds)
            
            candidates, _ = optimize_acqf_discrete(
                acq_function=acquisition_function,
                q=batch_size,
                choices=normalized_choices,
                max_batch_size=2048,
                unique=True,
            )
            
            denormalized_candidates = self._denormalize(candidates, self.bounds)
            
        else:
            # CONTINUOUS OPTIMIZATION
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
            num_exploitation: Number of exploitation points (qEHVI)
            num_exploration: Number of exploration points (qUCB)
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
            - Option 0: 5 exploit + 0 explore (pure qEHVI)
            - Option 1: 4 exploit + 1 explore
            - Option 2: 3 exploit + 2 explore (balanced)
            - Option 3: 2 exploit + 3 explore
            - Option 4: 1 exploit + 4 explore
            - Option 5: 0 exploit + 5 explore (pure exploration)
        
        Args:
            model: Fitted GP model
            pareto_front: Current Pareto front (m, 2)
            reference_point: Reference point for hypervolume (2,)
            mc_samples: MC samples for acquisition
        
        Returns:
            AllocationResults with all 6 options
        """
        print("[AcqFn] Computing all allocation options (qEHVI + qUCB)")
        
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
        """Compute a single allocation option with qEHVI (exploit) and qUCB (explore)."""
        
        if num_exploitation > 0:
            exploitation_points = self._optimize_qehvi(
                model=model,
                batch_size=num_exploitation,
                pareto_front=pareto_front,
                reference_point=reference_point,
                mc_samples=mc_samples
            )
        else:
            exploitation_points = np.empty((0, self.input_dim))
            
        if num_exploration > 0:
            exploration_points = self._optimize_qucb(
                model=model,
                batch_size=num_exploration,
                beta=self.exploration_beta,
                mc_samples=mc_samples
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
        
        # Evaluate combined metrics
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

    def _compute_allocation_metrics(
        self,
        model: MultiTaskGP,
        option: AllocationOption,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256
    ) -> Tuple[float, float]:
        """Compute evaluation metrics for an allocation option.
        
        Evaluates BOTH qEHVI and entropy on ALL points in the allocation.
        
        Args:
            model: Fitted GP model
            option: AllocationOption to evaluate
            pareto_front: Current Pareto front
            reference_point: Reference point for hypervolume
            mc_samples: MC samples for qEHVI
        
        Returns:
            (hypervolume_improvement, information_gain) tuple
        """
        all_points = option.all_points
        
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
        """Evaluate qEHVI on a set of points."""
        if points.shape[0] == 0:
            return 0.0
        
        points_tensor = torch.tensor(points, dtype=DTYPE, device=DEVICE)
        normalized_points = normalize(points_tensor, self.bounds)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        partitioning = NondominatedPartitioning(ref_point=reference_point, Y=pareto_front)
        
        qehvi = qExpectedHypervolumeImprovement(
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
        """Calculate mutual information for exploration scoring."""
        if points.shape[0] == 0:
            return 0.0
        
        points_tensor = torch.tensor(points, dtype=DTYPE, device=DEVICE)
        n = points_tensor.shape[0]
        
        # Create task indicators for both tasks
        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        
        X_task_0 = torch.cat([points_tensor, task_0], dim=-1)
        X_task_1 = torch.cat([points_tensor, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)
        
        with torch.no_grad():
            # Prior entropy H[p(y|D_n)]
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
            
            # Sample optimal points
            optimal_samples = self._sample_optimal_points(
                model=model,
                num_samples=num_optimal_samples
            )
            
            # Expected posterior entropy
            H_posterior_expected = 0.0
            
            for i in range(optimal_samples.shape[0]):
                opt_point = optimal_samples[i:i+1]
                
                opt_task_0 = torch.cat([opt_point, torch.zeros(1, 1, dtype=DTYPE, device=DEVICE)], dim=-1)
                opt_task_1 = torch.cat([opt_point, torch.ones(1, 1, dtype=DTYPE, device=DEVICE)], dim=-1)
                opt_both = torch.cat([opt_task_0, opt_task_1], dim=0)
                
                X_augmented = torch.cat([X_both, opt_both], dim=0)
                
                posterior_cond = model.posterior(X_augmented)
                cov_augmented = posterior_cond.covariance_matrix.squeeze(0)
                
                cov_11 = cov_augmented[:2*n, :2*n]
                cov_12 = cov_augmented[:2*n, 2*n:]
                cov_22 = cov_augmented[2*n:, 2*n:]
                
                cov_22 = cov_22 + 1e-6 * torch.eye(cov_22.shape[0], dtype=DTYPE, device=DEVICE)
                
                try:
                    cov_22_inv = torch.linalg.inv(cov_22)
                    cov_conditional = cov_11 - cov_12 @ cov_22_inv @ cov_12.T
                    cov_conditional = cov_conditional + 1e-6 * torch.eye(
                        2*n, dtype=DTYPE, device=DEVICE
                    )
                except RuntimeError:
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
            
            mutual_info = H_prior - H_posterior_expected
        
        return float(mutual_info.item())

    def _sample_optimal_points(
        self,
        model: MultiTaskGP,
        num_samples: int = 10,
        num_grid_points: int = 1000
    ) -> torch.Tensor:
        """Sample potential optimal points by finding Pareto fronts in posterior samples."""
        grid_points = torch.rand(
            num_grid_points, 
            self.input_dim, 
            dtype=DTYPE, 
            device=DEVICE
        )
        
        # Apply simplex constraint
        lower_bounds = self.bounds[0, :]
        upper_bounds = self.bounds[1, :]
        range_bounds = upper_bounds - lower_bounds
        target_sum = 1.0 - lower_bounds.sum().item()
        
        grid_sum = (grid_points[:, :-1] * range_bounds[:-1]).sum(dim=1)
        grid_points[:, -1] = (target_sum - grid_sum) / range_bounds[-1]
        grid_points = torch.clamp(grid_points, 0.0, 1.0)
        
        optimal_samples = []
        
        for sample_idx in range(num_samples):
            n_grid = grid_points.shape[0]
            task_0 = torch.zeros(n_grid, 1, dtype=DTYPE, device=DEVICE)
            task_1 = torch.ones(n_grid, 1, dtype=DTYPE, device=DEVICE)
            
            X_task_0 = torch.cat([grid_points, task_0], dim=-1)
            X_task_1 = torch.cat([grid_points, task_1], dim=-1)
            
            posterior_0 = model.posterior(X_task_0)
            posterior_1 = model.posterior(X_task_1)
            
            sample_0 = posterior_0.rsample(torch.Size([1])).squeeze(0).squeeze(-1)
            sample_1 = posterior_1.rsample(torch.Size([1])).squeeze(0).squeeze(-1)
            
            objectives = torch.stack([sample_0, sample_1], dim=-1)
            
            try:
                pareto_front, _ = self.compute_pareto_front(objectives)
                pareto_mask = is_non_dominated(objectives)
                pareto_inputs = grid_points[pareto_mask]
                
                if pareto_inputs.shape[0] > 0:
                    random_idx = torch.randint(0, pareto_inputs.shape[0], (1,))
                    optimal_samples.append(pareto_inputs[random_idx])
                else:
                    random_idx = torch.randint(0, n_grid, (1,))
                    optimal_samples.append(grid_points[random_idx])
                    
            except (ValueError, RuntimeError) as e:
                print(f"[Warning] Pareto computation failed in sample {sample_idx}: {e}")
                random_idx = torch.randint(0, n_grid, (1,))
                optimal_samples.append(grid_points[random_idx])
        
        return torch.cat(optimal_samples, dim=0)

    def _denormalize(
        self,
        normalized_points: torch.Tensor,
        bounds: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize points from [0, 1] to original bounds."""
        return bounds[0] + normalized_points * (bounds[1] - bounds[0])

    def _get_simplex_constraint(self):
        """Get equality constraint for simplex: sum(x_i) = 1."""
        lower_bounds = self.bounds[0, :]
        upper_bounds = self.bounds[1, :]
        coefficients = upper_bounds - lower_bounds
        right_hand_side = 1.0 - float(lower_bounds.sum().item())
        indices = torch.arange(self.input_dim, device=DEVICE, dtype=torch.long)
        constraint_coefficients = coefficients.to(dtype=DTYPE, device=DEVICE)
        return [(indices, constraint_coefficients, right_hand_side)]