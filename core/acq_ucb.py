import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from botorch.models import MultiTaskGP
from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_discrete

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


@dataclass
class AllocationOption:
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
    options: List[AllocationOption]
    metadata: Optional[dict] = None


class AcquisitionFunctionManager:

    def __init__(
        self,
        bounds: torch.Tensor,
        num_restarts: int = 10,
        raw_samples: int = 512,
        exploration_beta: float = 2.0,
        objective_weights: Optional[List[float]] = None,
        distance_weight: float = 1.0,
        design_space: Optional[object] = None,
        discrete_choices: Optional[torch.Tensor] = None,
        use_discrete: bool = False,
    ):
        if bounds.shape[0] != 2:
            raise ValueError(f"bounds must have shape (2, d), got {bounds.shape}")

        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)
        self.input_dim = self.bounds.shape[1]
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.exploration_beta = exploration_beta
        self.distance_weight = distance_weight
        self.candidates_qEHVI: Optional[np.ndarray] = None

        self.use_discrete = use_discrete
        self.design_space = design_space
        self.discrete_choices = None

        if design_space is not None and use_discrete:
            self._build_discrete_choices_from_design_space(design_space)
        elif discrete_choices is not None and use_discrete:
            self.discrete_choices = discrete_choices.to(dtype=DTYPE, device=DEVICE)

        if objective_weights is None:
            objective_weights = [0.5, 0.5]
        self.objective_weights = torch.tensor(objective_weights, dtype=DTYPE, device=DEVICE)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_discrete_choices_from_design_space(self, design_space):
        if hasattr(design_space, 'space') and design_space.space is not None:
            self.discrete_choices = torch.tensor(design_space.space, dtype=DTYPE, device=DEVICE)
            print(f"[AcqFn] Built discrete choices: {self.discrete_choices.shape[0]} points")
        elif hasattr(design_space, 'get_all_points'):
            all_points = design_space.get_all_points()
            self.discrete_choices = torch.tensor(all_points, dtype=DTYPE, device=DEVICE)
            print(f"[AcqFn] Built discrete choices: {self.discrete_choices.shape[0]} points")
        else:
            raise ValueError("DesignSpace must have 'space' attribute or 'get_all_points' method")

    def _build_discrete_dims_from_step(self, step: float) -> Dict[int, List[float]]:
        discrete_dims = {}
        for i in range(self.input_dim):
            lower = float(self.bounds[0, i])
            upper = float(self.bounds[1, i])
            n_steps = int(round((upper - lower) / step)) + 1
            allowed_values = [lower + j * step for j in range(n_steps)]
            if allowed_values[-1] < upper:
                allowed_values.append(upper)
            discrete_dims[i] = allowed_values
        return discrete_dims

    def _denormalize(self, normalized_points: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        return bounds[0] + normalized_points * (bounds[1] - bounds[0])

    def _get_simplex_constraint(self):
        lower_bounds = self.bounds[0, :]
        upper_bounds = self.bounds[1, :]
        coefficients = upper_bounds - lower_bounds
        right_hand_side = 1.0 - float(lower_bounds.sum().item())
        indices = torch.arange(self.input_dim, device=DEVICE, dtype=torch.long)
        constraint_coefficients = coefficients.to(dtype=DTYPE, device=DEVICE)
        return [(indices, constraint_coefficients, right_hand_side)]

    # ------------------------------------------------------------------
    # Pareto utilities
    # ------------------------------------------------------------------

    def compute_pareto_front(self, objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if objectives.shape[0] == 0:
            raise ValueError("Cannot compute Pareto front from empty objectives")
        pareto_mask = is_non_dominated(objectives)
        pareto_front = objectives[pareto_mask]
        if len(pareto_front) == 0:
            raise ValueError("No non-dominated points found")
        reference_point = pareto_front.min(dim=0).values - 0.1 * torch.ones(2, dtype=DTYPE, device=DEVICE)
        return pareto_front, reference_point

    # ------------------------------------------------------------------
    # Acquisition objectives
    # ------------------------------------------------------------------

    def _make_qucb_far_objective(self, distance_weight: float) -> GenericMCObjective:
        if self.candidates_qEHVI is None or len(self.candidates_qEHVI) == 0:
            prev = None
        else:
            prev_raw = torch.as_tensor(self.candidates_qEHVI, dtype=DTYPE, device=DEVICE)
            prev = normalize(prev_raw, self.bounds).contiguous()

        w = torch.as_tensor(self.objective_weights, dtype=DTYPE, device=DEVICE).view(-1)

        def _objective(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            if Y.size(-1) == 1:
                YW = Y.squeeze(-1)
            else:
                if w.numel() != Y.size(-1):
                    raise RuntimeError(
                        f"objective_weights has {w.numel()} elems, but Y has {Y.size(-1)} outputs."
                    )
                YW = (Y * w).sum(dim=-1)

            if prev is None or prev.numel() == 0 or distance_weight == 0.0:
                bonus = torch.zeros_like(YW)
            else:
                dists = torch.cdist(X, prev, p=2)
                min_d = dists.min(dim=-1).values
                while min_d.dim() < YW.dim():
                    min_d = min_d.unsqueeze(0)
                bonus = min_d

            return YW + distance_weight * bonus

        return GenericMCObjective(_objective)

    # ------------------------------------------------------------------
    # Core optimizers
    # ------------------------------------------------------------------

    def _optimize_qehvi(
        self,
        model: MultiTaskGP,
        batch_size: int,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int,
    ) -> np.ndarray:
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
            print(f"    Using discrete optimization over {self.discrete_choices.shape[0]} choices")
            normalized_choices = normalize(self.discrete_choices, self.bounds)
            candidates, _ = optimize_acqf_discrete(
                acq_function=acquisition_function,
                q=batch_size,
                choices=normalized_choices,
                max_batch_size=2048,
                unique=True,
            )
        else:
            simplex_constraint = self._get_simplex_constraint()
            candidates, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=torch.stack([
                    torch.zeros(self.input_dim, dtype=DTYPE, device=DEVICE),
                    torch.ones(self.input_dim, dtype=DTYPE, device=DEVICE),
                ]),
                q=batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                equality_constraints=simplex_constraint,
                options={"batch_limit": 5, "maxiter": 200},
            )

        return self._denormalize(candidates, self.bounds).cpu().numpy()

    def _optimize_qucb(
        self,
        model: MultiTaskGP,
        batch_size: int,
        beta: float,
        mc_samples: int,
        distance_weight: float = 1.0,
    ) -> np.ndarray:
        print(f"  Optimizing qUCB with q={batch_size}, beta={beta}...")

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        objective = self._make_qucb_far_objective(distance_weight=distance_weight)

        acquisition_function = qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=objective,
        )

        if self.use_discrete and self.discrete_choices is not None:
            print(f"    Using discrete optimization over {self.discrete_choices.shape[0]} choices")
            normalized_choices = normalize(self.discrete_choices, self.bounds)
            candidates, _ = optimize_acqf_discrete(
                acq_function=acquisition_function,
                q=batch_size,
                choices=normalized_choices,
                max_batch_size=2048,
                unique=True,
            )
        else:
            simplex_constraint = self._get_simplex_constraint()
            candidates, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=torch.stack([
                    torch.zeros(self.input_dim, dtype=DTYPE, device=DEVICE),
                    torch.ones(self.input_dim, dtype=DTYPE, device=DEVICE),
                ]),
                q=batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                equality_constraints=simplex_constraint,
                options={"batch_limit": 5, "maxiter": 200},
            )

        return self._denormalize(candidates, self.bounds).cpu().numpy()

    # ------------------------------------------------------------------
    # Allocation computation
    # ------------------------------------------------------------------

    def _compute_allocation_option(
        self,
        model: MultiTaskGP,
        num_exploitation: int,
        num_exploration: int,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int,
    ) -> AllocationOption:
        if num_exploitation > 0:
            exploitation_points = self._optimize_qehvi(
                model=model,
                batch_size=num_exploitation,
                pareto_front=pareto_front,
                reference_point=reference_point,
                mc_samples=mc_samples,
            )
        else:
            exploitation_points = np.empty((0, self.input_dim))

        self.candidates_qEHVI = exploitation_points

        if num_exploration > 0:
            exploration_points = self._optimize_qucb(
                model=model,
                batch_size=num_exploration,
                beta=self.exploration_beta,
                mc_samples=mc_samples,
                distance_weight=self.distance_weight,
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
            total_batch_size=num_exploitation+num_exploration,
        )

        hv_improvement, info_gain = self._compute_allocation_metrics(
            model=model,
            option=option,
            pareto_front=pareto_front,
            reference_point=reference_point,
            mc_samples=mc_samples,
        )
        option.hypervolume_improvement = hv_improvement
        option.information_gain = info_gain

        return option

    def compute_single_allocation(
        self,
        model: MultiTaskGP,
        num_exploitation: int,
        num_exploration: int,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256,
    ) -> AllocationResults:

        print(f"[AcqFn] Computing allocation: {num_exploitation} exploit + {num_exploration} explore")

        option = self._compute_allocation_option(
            model=model,
            num_exploitation=num_exploitation,
            num_exploration=num_exploration,
            pareto_front=pareto_front,
            reference_point=reference_point,
            mc_samples=mc_samples,
        )

        print(f"  → HVI: {option.hypervolume_improvement:.4f}, InfoGain: {option.information_gain:.4f}")

        metadata = {"exploration_beta": self.exploration_beta, "mc_samples": mc_samples}
        return AllocationResults(options=[option], metadata=metadata)

    def compute_all_allocations(
        self,
        model: MultiTaskGP,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256,
        total_batch_size: int = 5,
    ) -> AllocationResults:
        print("[AcqFn] Computing all allocation options (qEHVI + qUCB)")

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
                mc_samples=mc_samples,
            )
            options.append(option)
            print(f"  → HVI: {option.hypervolume_improvement:.4f}, InfoGain: {option.information_gain:.4f}")

        metadata = {
            "exploration_beta": self.exploration_beta,
            "mc_samples": mc_samples,
            "total_options": len(options),
        }
        return AllocationResults(options=options, metadata=metadata)

    # ------------------------------------------------------------------
    # Metric evaluation
    # ------------------------------------------------------------------

    def _compute_allocation_metrics(
        self,
        model: MultiTaskGP,
        option: AllocationOption,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256,
    ) -> Tuple[float, float]:
        all_points = option.all_points
        hv_improvement = self._evaluate_hypervolume_improvement(
            model=model,
            points=all_points,
            pareto_front=pareto_front,
            reference_point=reference_point,
            mc_samples=mc_samples,
        )
        info_gain = self._evaluate_mutual_information(model=model, points=all_points)
        return hv_improvement, info_gain

    def _evaluate_hypervolume_improvement(
        self,
        model: MultiTaskGP,
        points: np.ndarray,
        pareto_front: torch.Tensor,
        reference_point: torch.Tensor,
        mc_samples: int = 256,
    ) -> float:
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
        num_optimal_samples: int = 10,
    ) -> float:
        if points.shape[0] == 0:
            return 0.0

        points_tensor = torch.tensor(points, dtype=DTYPE, device=DEVICE)
        n = points_tensor.shape[0]

        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        X_task_0 = torch.cat([points_tensor, task_0], dim=-1)
        X_task_1 = torch.cat([points_tensor, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)

        with torch.no_grad():
            posterior = model.posterior(X_both)
            covariance_matrix = posterior.covariance_matrix.squeeze(0)
            covariance_matrix = covariance_matrix + 1e-6 * torch.eye(
                covariance_matrix.shape[0], dtype=DTYPE, device=DEVICE
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

            optimal_samples = self._sample_optimal_points(model=model, num_samples=num_optimal_samples)

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
                    cov_conditional = cov_conditional + 1e-6 * torch.eye(2*n, dtype=DTYPE, device=DEVICE)
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
        num_grid_points: int = 1000,
    ) -> torch.Tensor:
        grid_points = torch.rand(num_grid_points, self.input_dim, dtype=DTYPE, device=DEVICE)

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