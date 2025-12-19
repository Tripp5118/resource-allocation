# core/gp_models.py
import torch
import numpy as np
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Optional
from core.priors import EnsemblePrior

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

class GPModelManager:
    def __init__(self, bounds: torch.Tensor, use_priors: bool = False):
        """
        Args:
            bounds: (2, d) tensor of [lower, upper] for each dimension.
            use_priors: Whether to use EnsemblePrior for mean.
        """
        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)
        self.use_priors = use_priors
        self.prior_model = EnsemblePrior() if use_priors else None

    def fit_model(self, X: torch.Tensor, Y: torch.Tensor) -> MultiTaskGP:
        """
        Fit MultiTaskGP model with 2 tasks.

        X: (n, d)
        Y: (n, 2) - two objectives
        """
        n, d = X.shape
        assert Y.shape[1] == 2, "GPModelManager currently supports 2 objectives."

        X_norm = normalize(X, self.bounds)

        # Task indices
        task_indices_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_indices_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)

        X_task_0 = torch.cat([X_norm, task_indices_0], dim=-1)  # (n, d+1)
        X_task_1 = torch.cat([X_norm, task_indices_1], dim=-1)  # (n, d+1)
        train_X = torch.cat([X_task_0, X_task_1], dim=0)        # (2n, d+1)

        train_Y = torch.cat([Y[:, 0:1], Y[:, 1:2]], dim=0)      # (2n, 1)

        # Optional prior mean can be added later; omitted for generality for now
        model = MultiTaskGP(
            train_X,
            train_Y,
            task_feature=-1,
            output_tasks=[0, 1],
            outcome_transform=None
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    def predict(self, model: MultiTaskGP, X: torch.Tensor):
        """
        Predict both tasks at inputs X.

        X: (n, d), unnormalized
        Returns:
            mean: (n, 2), std: (n, 2)
        """
        n, d = X.shape
        X_norm = normalize(X, self.bounds)

        task_indices_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_indices_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)

        X_task_0 = torch.cat([X_norm, task_indices_0], dim=-1)
        X_task_1 = torch.cat([X_norm, task_indices_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)

        with torch.no_grad():
            post = model.posterior(X_both)
            mean_flat = post.mean.squeeze(-1)
            var_flat = post.variance.squeeze(-1)

        mean = torch.stack([mean_flat[:n], mean_flat[n:]], dim=-1)
        std = torch.sqrt(torch.stack([var_flat[:n], var_flat[n:]], dim=-1))
        return mean.cpu().numpy(), std.cpu().numpy()