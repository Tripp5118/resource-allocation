"""Gaussian Process model fitting and utilities using MultiTaskGP."""
import torch
import numpy as np
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from core.priors import EnsemblePrior

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

class GPModelManager:
    def __init__(self, bounds: torch.Tensor, use_priors: bool = False):
        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)
        self.use_priors = use_priors
        self.prior_model = EnsemblePrior() if use_priors else None

    def fit_model(self, X: torch.Tensor, Y: torch.Tensor) -> MultiTaskGP:
        """
        Fit MultiTaskGP model.
        
        X: (n, 5), fractions, sum to 1
        Y: (n, 2), [CTE, K]
        
        Returns MultiTaskGP model that accepts X with task indices.
        """
        n = X.shape[0]
        
        # Normalize inputs
        X_norm = normalize(X, self.bounds)
        
        # Prepare data for MultiTaskGP format:
        # Each observation point is duplicated, once for each task
        # Task 0 = CTE, Task 1 = K
        
        # Create task indices: [0, 0, 0, ..., 1, 1, 1, ...]
        task_indices_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)  # Task 0 (CTE)
        task_indices_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)   # Task 1 (K)
        
        # Concatenate inputs with task indices
        # Shape: (n, 6) where last column is task index
        X_task_0 = torch.cat([X_norm, task_indices_0], dim=-1)  # (n, 6)
        X_task_1 = torch.cat([X_norm, task_indices_1], dim=-1)  # (n, 6)
        
        # Stack all training data: (2n, 6)
        train_X = torch.cat([X_task_0, X_task_1], dim=0)
        
        # Stack outputs: (2n, 1)
        train_Y = torch.cat([
            Y[:, 0:1],  # CTE values (n, 1)
            Y[:, 1:2]   # K values (n, 1)
        ], dim=0)
        
        # Optionally set prior mean
        mean_module = None
        if self.use_priors and self.prior_model is not None:
            # Get prior predictions
            prior_means = self.prior_model.predict(X.cpu().numpy())  # (n, 2)
            prior_means_torch = torch.tensor(prior_means, dtype=DTYPE, device=DEVICE)
            
            # Create mean module (one constant per task)
            from gpytorch.means import ConstantMean
            mean_module = ConstantMean(batch_shape=torch.Size([2]))  # 2 tasks
            with torch.no_grad():
                mean_module.constant[0] = prior_means_torch[:, 0].mean()  # CTE prior
                mean_module.constant[1] = prior_means_torch[:, 1].mean()  # K prior
        
        # Create MultiTaskGP model
        model = MultiTaskGP(
            train_X,
            train_Y,
            task_feature=-1,  # Task index is the last column
            output_tasks=[0, 1],  # We want predictions for both tasks
            outcome_transform=None  # No transform to avoid MO-MESMO issues
        )
        
        # Fit model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        return model

    def predict(self, model: MultiTaskGP, X: torch.Tensor):
        """
        Make predictions for both tasks at input points X.
        
        X: (n, 5) - unnormalized compositions
        Returns: mean (n, 2), std (n, 2)
        """
        n = X.shape[0]
        X_norm = normalize(X, self.bounds)
        
        # Create input with task indices for both tasks
        task_indices_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_indices_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        
        X_task_0 = torch.cat([X_norm, task_indices_0], dim=-1)
        X_task_1 = torch.cat([X_norm, task_indices_1], dim=-1)
        
        # Stack for batch prediction
        X_both = torch.cat([X_task_0, X_task_1], dim=0)  # (2n, 6)
        
        with torch.no_grad():
            post = model.posterior(X_both)
            mean_flat = post.mean.squeeze(-1)  # (2n,)
            var_flat = post.variance.squeeze(-1)  # (2n,)
        
        # Reshape to (n, 2)
        mean = torch.stack([mean_flat[:n], mean_flat[n:]], dim=-1)
        std = torch.sqrt(torch.stack([var_flat[:n], var_flat[n:]], dim=-1))
        
        return mean.cpu().numpy(), std.cpu().numpy()