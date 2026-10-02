# resource_allocation/gp_models.py
import torch
import numpy as np
from botorch.models import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


class GPModelManager:
    def __init__(self, bounds: torch.Tensor):
        """
        Args:
            bounds: (2, d) tensor of [lower, upper] for each dimension.
        """
        self.bounds = bounds.to(dtype=DTYPE, device=DEVICE)

    def fit_model(self, X: torch.Tensor, Y: torch.Tensor) -> MultiTaskGP:
        """
        Fit MultiTaskGP with n_obj tasks, inferred from Y.shape[1].

        X: (n, d)
        Y: (n, n_obj) — any n_obj >= 2
        """
        n, _ = X.shape
        n_obj = Y.shape[1]

        X = X.to(dtype=DTYPE, device=DEVICE)
        Y = Y.to(dtype=DTYPE, device=DEVICE)
        X_norm = normalize(X, self.bounds)

        task_X_list, task_Y_list = [], []
        for k in range(n_obj):
            task_idx = torch.full((n, 1), float(k), dtype=DTYPE, device=DEVICE)
            task_X_list.append(torch.cat([X_norm, task_idx], dim=-1))
            task_Y_list.append(Y[:, k : k + 1])

        train_X = torch.cat(task_X_list, dim=0)   # (n*n_obj, d+1)
        train_Y = torch.cat(task_Y_list, dim=0)   # (n*n_obj, 1)

        model = MultiTaskGP(
            train_X,
            train_Y,
            task_feature=-1,
            output_tasks=list(range(n_obj)),
            outcome_transform=None,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def predict(self, model: MultiTaskGP, X: torch.Tensor, chunk_size: int = 2000):
        """
        Predict all tasks at inputs X.

        X: (n, d), unnormalized
        Returns:
            mean: (n, n_obj) numpy array
            std:  (n, n_obj) numpy array
        """
        n, _ = X.shape
        n_obj = model.num_outputs
        X = X.to(dtype=DTYPE, device=DEVICE)
        X_norm = normalize(X, self.bounds)

        mean_chunks, var_chunks = [], []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            X_chunk = X_norm[start:end]
            c = end - start
            task_X_list = []
            for k in range(n_obj):
                task_idx = torch.full((c, 1), float(k), dtype=DTYPE, device=DEVICE)
                task_X_list.append(torch.cat([X_chunk, task_idx], dim=-1))
            X_all = torch.cat(task_X_list, dim=0)   # (c*n_obj, d+1)

            with torch.no_grad():
                post = model.posterior(X_all)
                mean_flat = post.mean.squeeze(-1)
                var_flat = post.variance.squeeze(-1)

            mean_chunks.append(torch.stack([mean_flat[k * c : (k + 1) * c] for k in range(n_obj)], dim=-1).cpu())
            var_chunks.append(torch.stack([var_flat[k * c : (k + 1) * c] for k in range(n_obj)], dim=-1).cpu())
            del X_all, post, mean_flat, var_flat

        mean = torch.cat(mean_chunks, dim=0)
        std  = torch.sqrt(torch.cat(var_chunks, dim=0))
        return mean.numpy(), std.numpy()
