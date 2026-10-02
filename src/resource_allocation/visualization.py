# resource_allocation/visualization.py

import os
from typing import Optional, List, Callable, Dict
import pandas as pd
from resource_allocation.logging_utils import LoggingManager

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import imageio

import torch
from botorch.utils.transforms import normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


def composition_to_rgb(X: np.ndarray, major_indices: Optional[List[int]] = None) -> np.ndarray:
    """
    Map composition X (n, d) to RGB colors (n, 3) in [0, 1].

    Extracts columns at major_indices (default [0, 1, 2] = Fe, Co, Ni),
    normalizes each row to sum=1 (so pure-Fe is (1,0,0), pure-Co is (0,1,0),
    pure-Ni is (0,0,1)), and returns the result clipped to [0, 1].

    Works when X is the full (n, 15) design matrix OR already (n, 3).
    """
    if major_indices is None:
        major_indices = [0, 1, 2]
    X_maj = X[:, major_indices].astype(float)
    row_sums = X_maj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (X_maj / row_sums).clip(0.0, 1.0)


class VisualizationManager:
    """
    Visualization manager for BO on a composition simplex.

    - Assumes design_space.space is an (N, d) array of compositions summing to 1.
    - Supports 2 <= d <= 10 by projecting onto a regular d-gon in 2D.
    - Can color points by:
        * a scalar score from a GP model (via score_fn(Y_pred)), or
        * arbitrary scalar values from a user-supplied evaluator f(x).

    KEY FEATURES:
    - Handles both normalized (for GP) and raw (for plotting) outputs.
    - Automatically denormalizes GP predictions for interpretable plots.
    - Creates comprehensive visualizations: 2D objective space, convergence, Pareto evolution.
    
    NORMALIZATION WORKFLOW:
    - GP operates on normalized outputs (mean=0, std=1).
    - Visualizations show original scale (interpretable to humans).
    - Pass normalization_params to enable automatic denormalization.
    """

    def __init__(
        self,
        log_dir: str = None,  # Backward compatibility
        design_space = None,
        experiment_name: str = "experiment",
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
        objective_names: Optional[List[str]] = None,
        major_indices: Optional[List[int]] = None,
    ):
        """
        Initialize visualization manager.

        Args:
            log_dir: Directory to save visualizations
            design_space: DesignSpace object (optional, for composition plots)
            experiment_name: Name of experiment
            normalization_params: Dict with 'mean' and 'std' arrays for denormalization
            objective_names: List of objective names (for labels)
            major_indices: Column indices of the "major" elements to use for a
                reduced n-gon projection when d > 10. The selected columns are
                normalized to sum=1 (barycentric coordinates) before projecting,
                so all points fall inside the polygon. For the 15-element magnet
                problem, pass [0,1,2] (Fe, Co, Ni) to get a proper ternary plot.
                When None and 2 <= d <= 10, the full d-gon is used (original behavior).
        """
        self.log_dir = log_dir
        self.design_space = design_space
        self.experiment_name = experiment_name
        self.normalization_params = normalization_params
        self.objective_names = objective_names or ["Obj1", "Obj2"]

        self.vis_dir = os.path.join(log_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

        # Infer dimension and labels
        self.dim = design_space.space.shape[1]
        labels = getattr(design_space, "labels", None)
        if labels is not None and len(labels) == self.dim:
            self.element_names = list(labels)
        else:
            self.element_names = [f"x_{i}" for i in range(self.dim)]

        if major_indices is not None:
            self.major_indices = list(major_indices)
            self.enabled = True
        elif 2 <= self.dim <= 10:
            self.major_indices = None
            self.enabled = True
        else:
            print(f"[Viz] d={self.dim} is outside [2,10] and no major_indices provided; disabled.")
            self.enabled = False
            self.image_paths: List[str] = []
            return

        self._setup_affine_projection()

        # Track image paths for GIF creation
        self.image_paths: List[str] = []

        print(f"[Viz] Initialized visualization in {self.vis_dir}")
        print(f"[Viz] Experiment name: {experiment_name}")
        print(f"[Viz] Dimension d={self.dim}, labels={self.element_names}")
        if normalization_params is not None:
            print(f"[Viz] Normalization enabled - will denormalize for plotting")
            for k, name in enumerate(self.objective_names):
                m = normalization_params['mean'][k] if k < len(normalization_params['mean']) else float('nan')
                s = normalization_params['std'][k] if k < len(normalization_params['std']) else float('nan')
                print(f"  {name}: mean={m:.4f}, std={s:.4f}")


    # Add to VisualizationManager class in visualization.py

    def create_hypervolume_plot(self, logger: LoggingManager, score_name: str = "Score"):
        """Create hypervolume progress plot."""
        plot_hypervolume_per_iteration(logger, score_name, save_dir=self.log_dir)

    def create_pareto_front_plot(
        self,
        Y_pareto: np.ndarray,
        Y_all: np.ndarray,
        objective_names: List[str],
        save_path: str,
        iteration: int,
        **kwargs,
    ):
        """Dispatch to 2D, 3D, or skip based on number of objectives."""
        n_obj = len(objective_names)
        if n_obj == 2:
            self._plot_pareto_2d(Y_pareto, Y_all, objective_names, save_path, iteration)
        elif n_obj == 3:
            self._plot_pareto_3d(Y_pareto, Y_all, objective_names, save_path, iteration)
        else:
            print(f"[Vis] Pareto plot skipped for {n_obj} objectives (N>3 not supported)")

    def _plot_pareto_2d(self, Y_pareto, Y_all, objective_names, save_path, iteration):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(Y_all[:, 0], Y_all[:, 1], c="gray", alpha=0.3, s=20, label="All evaluated")
        ax.scatter(Y_pareto[:, 0], Y_pareto[:, 1], c="red", s=50, zorder=5, label="Pareto front")
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_title(f"Pareto Front — Iteration {iteration}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _plot_pareto_3d(self, Y_pareto, Y_all, objective_names, save_path, iteration):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Y_all[:, 0], Y_all[:, 1], Y_all[:, 2],
                   c="gray", alpha=0.25, s=15, label="All evaluated")
        ax.scatter(Y_pareto[:, 0], Y_pareto[:, 1], Y_pareto[:, 2],
                   c="red", s=60, zorder=5, label="Pareto front")
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_zlabel(objective_names[2])
        ax.set_title(f"Pareto Front — Iteration {iteration}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    # ------------------------------------------------------------------ #
    #  NORMALIZATION UTILITIES
    # ------------------------------------------------------------------ #
    
    def denormalize_outputs(self, Y_normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize outputs from (mean=0, std=1) to original scale.
        
        Args:
            Y_normalized: Normalized outputs (n, 2)
            
        Returns:
            Y_raw: Original scale outputs (n, 2)
        """
        if self.normalization_params is None:
            return Y_normalized
        
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        return Y_normalized * std + mean
    
    def normalize_outputs(self, Y_raw: np.ndarray) -> np.ndarray:
        """
        Normalize outputs from original scale to (mean=0, std=1).
        
        Args:
            Y_raw: Original scale outputs (n, 2)
            
        Returns:
            Y_normalized: Normalized outputs (n, 2)
        """
        if self.normalization_params is None:
            return Y_raw
        
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        return (Y_raw - mean) / std
    
    # ------------------------------------------------------------------ #
    #  Projection setup
    # ------------------------------------------------------------------ #

    def _setup_affine_projection(self) -> None:
        """Build projection and precompute 2D projections of full design space."""
        if self.major_indices is not None:
            n = len(self.major_indices)
            angles = [np.pi / 2 + i * 2 * np.pi / n for i in range(n)]
            self.simplex_vertices = np.array([[np.cos(a), np.sin(a)] for a in angles])
            self.projection_matrix = self.simplex_vertices  # (n, 2)
            self.element_names = [self.element_names[i] for i in self.major_indices]
        else:
            num_vertices = self.dim
            vertices = []
            angle_step = 2.0 * np.pi / num_vertices
            initial_angle = np.pi / 2.0  # first vertex at top
            for i in range(num_vertices):
                angle = initial_angle + i * angle_step
                vertices.append([np.cos(angle), np.sin(angle)])
            self.simplex_vertices = np.array(vertices)
            self.projection_matrix = self.simplex_vertices

        self.vis_space = self.design_space.space
        n_available = len(self.vis_space)
        print(f"[Viz] Using full design space ({n_available} points) for visualization")
        self.design_space_2d = self._affine_transform(self.vis_space)
        self.vis_space_cpu = torch.tensor(self.vis_space, dtype=DTYPE, device="cpu")
        print(f"[Viz] Setup affine projection: projected {len(self.design_space_2d)} points")
        print(f"[Viz] Cached visualization space on CPU: {self.vis_space_cpu.shape}")

    def _affine_transform(self, X: np.ndarray) -> np.ndarray:
        """Project compositions X (n,d) to 2D using the d-gon or major-element vertices."""
        if self.major_indices is not None:
            # Barycentric projection: normalize the selected columns to sum=1 so
            # all points fall inside the polygon regardless of original scale.
            X_maj = X[:, self.major_indices]
            row_sums = X_maj.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return (X_maj / row_sums) @ self.projection_matrix
        return X @ self.projection_matrix  # (n,d) x (d,2) -> (n,2)

    # ------------------------------------------------------------------ #
    #  Generic evaluation over design space
    # ------------------------------------------------------------------ #

    def evaluate_on_design_space(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Evaluate a user-supplied function f on all points of the design space.

        Args:
            f: function mapping X (N,d) -> values:
                - either (N,) scalar values, or
                - (N,2) for two objectives.

        Returns:
            values: np.ndarray of shape (N,) or (N,2)
        """
        X = self.design_space.space  # (N, d)
        values = f(X)
        values = np.asarray(values)
        if values.shape[0] != X.shape[0]:
            raise ValueError("f(X) must return an array with the same first dimension as X.")
        return values

    # ------------------------------------------------------------------ #
    #  GP-based score prediction (for background heatmap)
    # ------------------------------------------------------------------ #

    def _predict_scores_from_gp(
        self,
        model,
        bounds: torch.Tensor,
        score_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Predict scalar scores on the design space using a MultiTaskGP model.

        This function handles normalization automatically.
        - GP predictions are in NORMALIZED space
        - Predictions are DENORMALIZED before applying score_fn
        - score_fn receives ORIGINAL SCALE outputs

        Args:
            model: fitted MultiTaskGP model
            bounds: (2,d) tensor
            score_fn: maps predicted objectives Y_pred (N,2) in ORIGINAL scale -> scores (N,)

        Returns:
            scores: (N,) scalar scores
        """
        vis_space_gpu = self.vis_space_cpu.to(device=DEVICE)
        bounds_gpu = bounds.to(device=DEVICE)
        X_norm = normalize(vis_space_gpu, bounds_gpu)
        del vis_space_gpu, bounds_gpu
        n = X_norm.shape[0]
        n_obj = model.num_outputs

        # Chunked prediction: same pattern as gp_models.predict to cap GPU memory.
        # Without chunking, X_all would be (n*n_obj, d+1) = ~994K rows for this design space,
        # which triggers a full posterior computation that can exceed GPU memory.
        CHUNK = 2000
        mean_chunks = []
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            c = end - start
            X_chunk = X_norm[start:end]
            task_X_list = []
            for k in range(n_obj):
                task_idx = torch.full((c, 1), float(k), dtype=DTYPE, device=DEVICE)
                task_X_list.append(torch.cat([X_chunk, task_idx], dim=-1))
            X_all = torch.cat(task_X_list, dim=0)
            del task_X_list

            with torch.no_grad():
                post = model.posterior(X_all)
                mean_flat = post.mean.squeeze(-1)  # (c*n_obj,)

            chunk_means = torch.stack(
                [mean_flat[k * c:(k + 1) * c] for k in range(n_obj)], dim=-1
            ).cpu()
            mean_chunks.append(chunk_means)
            del X_all, post, mean_flat, chunk_means

        del X_norm
        Y_pred_normalized = torch.cat(mean_chunks, dim=0).numpy()
        del mean_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Denormalize GP predictions to original scale
        Y_pred_raw = self.denormalize_outputs(Y_pred_normalized)

        # score_fn receives ORIGINAL SCALE outputs
        scores = score_fn(Y_pred_raw)
        return scores

    def _predict_objectives_from_gp(
        self,
        model,
        bounds: torch.Tensor,
    ) -> np.ndarray:
        """
        Predict per-objective means on the full design space using the GP posterior.

        Returns (n_ds, n_obj) array in ORIGINAL scale (denormalized).
        Same chunking pattern as _predict_scores_from_gp to keep GPU memory bounded.
        Uses the model's own device to avoid CPU/GPU mismatch.
        """
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = DEVICE

        vis_space_t = self.vis_space_cpu.to(device=target_device)
        bounds_t = bounds.to(device=target_device)
        X_norm = normalize(vis_space_t, bounds_t)
        del vis_space_t, bounds_t
        n = X_norm.shape[0]
        n_obj = model.num_outputs

        CHUNK = 2000
        mean_chunks = []
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            c = end - start
            X_chunk = X_norm[start:end]
            task_X_list = []
            for k in range(n_obj):
                task_idx = torch.full((c, 1), float(k), dtype=DTYPE, device=target_device)
                task_X_list.append(torch.cat([X_chunk, task_idx], dim=-1))
            X_all = torch.cat(task_X_list, dim=0)
            del task_X_list

            with torch.no_grad():
                post = model.posterior(X_all)
                mean_flat = post.mean.squeeze(-1)  # (c*n_obj,)

            chunk_means = torch.stack(
                [mean_flat[k * c:(k + 1) * c] for k in range(n_obj)], dim=-1
            ).cpu()
            mean_chunks.append(chunk_means)
            del X_all, post, mean_flat, chunk_means

        del X_norm
        Y_pred_normalized = torch.cat(mean_chunks, dim=0).numpy()
        del mean_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.denormalize_outputs(Y_pred_normalized)

    # ------------------------------------------------------------------ #
    #  Drawing helpers
    # ------------------------------------------------------------------ #

    def _draw_simplex_and_labels(self, ax: plt.Axes) -> None:
        """Draw the regular d-gon and label each vertex."""
        polygon = Polygon(
            self.simplex_vertices,
            closed=True,
            facecolor="#dddddd",
            edgecolor="black",
            lw=2,
            zorder=1,
        )
        ax.add_patch(polygon)

        for i, name in enumerate(self.element_names):
            x, y = self.simplex_vertices[i]
            label_x = x * 1.05
            label_y = y * 1.05
            ax.text(
                label_x,
                label_y,
                name,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
            )

    # ------------------------------------------------------------------ #
    #  Generic scalar field plotting
    # ------------------------------------------------------------------ #

    def plot_scalar_field(
        self,
        values: np.ndarray,
        field_name: str = "value",
        filename: Optional[str] = None,
        cmap: str = "plasma",
    ) -> Optional[str]:
        """
        Plot a scalar field over the simplex, given per-point values for the design space.

        Args:
            values: (N,) array, one value per design_space.point.
            field_name: name for colorbar/title.
            filename: optional filename (without directory). If None, constructed from field_name.
            cmap: matplotlib colormap.

        Returns:
            Full path to saved image, or None if visualization disabled.
        """
        if not self.enabled:
            print("[Viz] Visualization disabled (dimension outside [2,10]).")
            return None

        values = np.asarray(values).reshape(-1)
        if values.shape[0] != self.design_space_2d.shape[0]:
            raise ValueError("values must have the same length as design_space.space")

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_aspect("equal")
        ax.axis("off")

        self._draw_simplex_and_labels(ax)

        # Sort by value for better color layering
        idx = np.argsort(values)
        x_bg = self.design_space_2d[idx, 0]
        y_bg = self.design_space_2d[idx, 1]
        sorted_vals = values[idx]

        vmin = np.percentile(values, 0)
        vmax = np.percentile(values, 100)

        scatter = ax.scatter(
            x_bg,
            y_bg,
            c=sorted_vals,
            s=30,
            cmap=cmap,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
            alpha=0.85,
            zorder=2,
        )

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(field_name, fontsize=11)

        ax.set_title(f"{self.experiment_name} — {field_name}", fontsize=14, fontweight="bold", pad=15)

        if filename is None:
            safe_name = field_name.replace(" ", "_")
            filename = f"{safe_name}.png"

        plot_path = os.path.join(self.vis_dir, filename)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"[Viz] Created scalar field plot: {plot_path}")
        return plot_path

    def plot_objectives_from_evaluator(
        self,
        evaluator: Callable[[np.ndarray], np.ndarray],
        obj_names: List[str],
        filename_prefix: str = "truth",
    ) -> List[str]:
        """
        Evaluate a 2-objective evaluator on the design space and plot each objective separately.

        Args:
            evaluator: function f(X) -> Y, where X is (N,d) and Y is (N,2).
            obj_names: [name_obj1, name_obj2]
            filename_prefix: prefix for output filenames.

        Returns:
            List of file paths for the two plots.
        """
        Y = self.evaluate_on_design_space(evaluator)

        Y = np.abs(Y)
        paths = []
        for i, name in enumerate(obj_names):
            vals = Y[:, i]
            fname = f"{filename_prefix}_{i}_{name.replace(' ', '_')}.png"
            path = self.plot_scalar_field(vals, field_name=name, filename=fname)
            if path:
                paths.append(path)
        return paths

    # ------------------------------------------------------------------ #
    #  Per-iteration BO plots (GP-based)
    # ------------------------------------------------------------------ #

    def create_iteration_plot(
        self,
        iteration: int,
        X_history: np.ndarray,
        Y_history: np.ndarray,
        X_new: np.ndarray,
        model,
        bounds: torch.Tensor,
        selection_info: Optional[dict] = None,
        score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        score_name: str = "score",
        objective_names: Optional[List[str]] = None,
        current_hv: Optional[float] = None,
    ) -> Optional[str]:
        """
        Create a 2D visualization for this iteration, optionally using a GP model.

        Args:
            X_history, Y_history: all evaluated points and objectives (raw postprocessed space)
            X_new: newly evaluated points this iteration
            model: trained MultiTaskGP (if None, only evaluated points are shown)
            bounds: (2,d) tensor
            selection_info: metadata for title/legend
            score_fn: maps Y (n,n_obj) -> scalar HVI scores for coloring
            score_name: label for the colorbar
            objective_names: list of objective names
            current_hv: total hypervolume of the current Pareto front (for title display)
        """
        if not self.enabled:
            print("[Viz] Visualization disabled (dimension outside [2,10]).")
            return None

        if score_fn is None:
            def score_fn_default(Y: np.ndarray) -> np.ndarray:
                return Y[:, 1]
            score_fn = score_fn_default

        # Pareto front detection on the raw (postprocessed) objective history
        from botorch.utils.multi_objective.pareto import is_non_dominated
        _Y_t = torch.tensor(Y_history, dtype=torch.float64)
        pareto_mask = is_non_dominated(_Y_t).numpy()
        n_pareto = int(pareto_mask.sum())
        del _Y_t

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")
        ax.axis("off")

        self._draw_simplex_and_labels(ax)

        # Background predictions from GP, if available
        scatter = None
        min_s = max_s = None
        if model is not None:
            predicted_scores = self._predict_scores_from_gp(model, bounds, score_fn)
            min_s = np.percentile(predicted_scores, 0)
            max_s = np.percentile(predicted_scores, 100)
            print(
                f"[Viz] Predicted scores - Min: {predicted_scores.min():.4f}, "
                f"Max: {predicted_scores.max():.4f}, "
                f"Mean: {predicted_scores.mean():.4f}"
            )

            idx = np.argsort(predicted_scores)
            x_bg = self.design_space_2d[idx, 0]
            y_bg = self.design_space_2d[idx, 1]
            sorted_scores = predicted_scores[idx]

            scatter = ax.scatter(
                x_bg,
                y_bg,
                c=sorted_scores,
                s=30,
                cmap="plasma",
                edgecolors="none",
                vmin=min_s,
                vmax=max_s,
                alpha=0.8,
                zorder=2,
            )
        else:
            print("[Viz] No GP model provided; plotting evaluated points only.")

        # Evaluated history — colored by per-point HVI
        n_new = len(X_new)
        scores_hist = score_fn(Y_history)
        if len(X_history) > n_new:
            X_old = X_history[:-n_new]
            X_old_2d = self._affine_transform(X_old)
            scores_old = scores_hist[:-n_new]
            ax.scatter(
                X_old_2d[:, 0],
                X_old_2d[:, 1],
                c=scores_old,
                cmap="plasma",
                s=80,
                edgecolors="black",
                linewidths=1.5,
                vmin=min_s,
                vmax=max_s,
                alpha=0.9,
                zorder=3,
                label="Evaluated",
            )

        # New points this iteration
        if n_new > 0:
            X_new_2d = self._affine_transform(X_new)
            scores_new = scores_hist[-n_new:]
            ax.scatter(
                X_new_2d[:, 0],
                X_new_2d[:, 1],
                c=scores_new,
                cmap="plasma",
                s=200,
                marker="*",
                edgecolors="orange",
                linewidths=2,
                vmin=min_s,
                vmax=max_s,
                zorder=4,
                label="New",
            )

        # Pareto points — gold diamonds on top
        if n_pareto > 0:
            X_pareto_2d = self._affine_transform(X_history[pareto_mask])
            ax.scatter(
                X_pareto_2d[:, 0],
                X_pareto_2d[:, 1],
                c="gold",
                s=150,
                marker="D",
                edgecolors="darkorange",
                linewidths=1.5,
                zorder=5,
                label=f"Pareto ({n_pareto})",
            )

        # Colorbar
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(score_name, fontsize=11)

        # Title
        hv_str = f"{current_hv:.4f}" if current_hv is not None else "n/a"
        names = objective_names or self.objective_names
        best_idx = int(np.argmax(scores_hist))
        best_y = Y_history[best_idx]
        best_vals_str = ", ".join(
            f"{names[k] if k < len(names) else f'obj{k}'}={best_y[k]:.4g}"
            for k in range(len(best_y))
        )
        title_lines = [
            f"{self.experiment_name}",
            f"Iteration {iteration} | Evaluated: {len(X_history)} | "
            f"Pareto: {n_pareto} | HV: {hv_str}",
            f"Best point: {best_vals_str}",
        ]
        if selection_info:
            if "n_optimization" in selection_info and "n_exploration" in selection_info:
                title_lines.append(
                    f"Allocation: {selection_info['n_optimization']} opt + "
                    f"{selection_info['n_exploration']} explore"
                )
            elif "method" in selection_info:
                title_lines.append(f"Selection: {selection_info.get('method', 'Unknown')}")

        ax.set_title("\n".join(title_lines), fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="upper left", bbox_to_anchor=(1.15, 0.85), fontsize=10)

        plot_path = os.path.join(self.vis_dir, f"iteration_{iteration:03d}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        self.image_paths.append(plot_path)
        print(f"[Viz] Created plot for iteration {iteration}")
        return plot_path

    def create_objective_space_plot(
        self,
        iteration: int,
        X_history: np.ndarray,
        Y_history: np.ndarray,
        X_new: np.ndarray,
        model,
        bounds: torch.Tensor,
    ) -> Optional[str]:
        """
        Create a static 3D objective-space PNG for this iteration.

        Background: GP-predicted design-space points (surrogate model beliefs),
        RGB-colored by Fe/Co/Ni composition, depth-shaded for 3D clarity.
        Overlaid: actually-evaluated points with Pareto-optimal ones shown as
        star markers. New points this iteration shown with orange outline stars.

        Returns path to saved PNG, or None if visualization is disabled.
        """
        if not self.enabled:
            return None

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from botorch.utils.multi_objective.pareto import is_non_dominated as _isnd

        ELEV, AZIM = 25, 225
        n_new = len(X_new)

        # GP-predicted objectives over the full design space (background)
        Y_gp = self._predict_objectives_from_gp(model, bounds)  # (n_ds, n_obj)
        n_obj = Y_gp.shape[1]
        if n_obj < 3:
            return None
        obj_names = (self.objective_names + [f"Obj{i}" for i in range(n_obj, 3)])[:3]
        ds_rgb = composition_to_rgb(self.vis_space, self.major_indices or [0, 1, 2])

        # Pareto of actually-evaluated history (not GP predictions)
        _Y_t = torch.tensor(Y_history, dtype=torch.float64)
        hist_pareto_mask = _isnd(_Y_t).numpy()
        del _Y_t
        n_hist_pareto = int(hist_pareto_mask.sum())

        # Camera direction for depth shading (must match view_init below)
        elev_r = np.radians(ELEV)
        azim_r = np.radians(AZIM)
        cam_dir = np.array([
            np.cos(elev_r) * np.cos(azim_r),
            np.cos(elev_r) * np.sin(azim_r),
            np.sin(elev_r),
        ])

        def _depth_rgba(Y3: np.ndarray, rgb: np.ndarray,
                        alpha_near: float, alpha_far: float) -> np.ndarray:
            """RGBA where near-camera points are brighter; far points are darker."""
            depth = Y3 @ cam_dir
            drange = depth.max() - depth.min()
            t = (depth - depth.min()) / drange if drange > 1e-9 else np.full(len(depth), 0.5)
            alpha = alpha_far + t * (alpha_near - alpha_far)
            brightness = 0.25 + 0.75 * t[:, None]
            rgba = np.concatenate([np.clip(rgb * brightness, 0, 1), alpha[:, None]], axis=1)
            return rgba.astype(np.float32)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # --- Background: full GP-predicted design space (depth-shaded) ---
        gp_rgba = _depth_rgba(Y_gp, ds_rgb, alpha_near=0.18, alpha_far=0.03)
        ax.scatter(
            Y_gp[:, 0], Y_gp[:, 1], Y_gp[:, 2],
            c=gp_rgba,
            s=2,
            edgecolors="none",
            zorder=1,
        )

        # --- Evaluated history ---
        _rgb_idx = self.major_indices or [0, 1, 2]
        if len(X_history) > n_new:
            X_old = X_history[:-n_new]
            Y_old = Y_history[:-n_new]
            old_pareto = hist_pareto_mask[:-n_new]
            old_rgb = composition_to_rgb(X_old, _rgb_idx)

            # Non-Pareto evaluated points
            non_p = ~old_pareto
            if non_p.any():
                np_rgba = _depth_rgba(Y_old[non_p], old_rgb[non_p],
                                      alpha_near=0.90, alpha_far=0.35)
                ax.scatter(
                    Y_old[non_p, 0], Y_old[non_p, 1], Y_old[non_p, 2],
                    c=np_rgba, s=35, edgecolors="none",
                    zorder=4, label="Evaluated",
                )

            # Pareto-optimal evaluated points — star markers
            if old_pareto.any():
                p_rgba = _depth_rgba(Y_old[old_pareto], old_rgb[old_pareto],
                                     alpha_near=1.0, alpha_far=0.55)
                ax.scatter(
                    Y_old[old_pareto, 0], Y_old[old_pareto, 1], Y_old[old_pareto, 2],
                    c=p_rgba, s=120, marker="*",
                    edgecolors="black", linewidths=0.8,
                    zorder=5, label=f"Pareto (n={n_hist_pareto})",
                )

        # --- New points this iteration — orange-outlined stars ---
        if n_new > 0:
            Y_new = Y_history[-n_new:]
            new_rgb = composition_to_rgb(X_new, _rgb_idx)
            new_rgba = _depth_rgba(Y_new, new_rgb, alpha_near=1.0, alpha_far=0.80)
            ax.scatter(
                Y_new[:, 0], Y_new[:, 1], Y_new[:, 2],
                c=new_rgba, s=200, marker="*",
                edgecolors="orange", linewidths=2,
                zorder=6, label="New",
            )

        ax.set_xlabel(obj_names[0], fontsize=10, labelpad=8)
        ax.set_ylabel(obj_names[1], fontsize=10, labelpad=8)
        ax.set_zlabel(obj_names[2], fontsize=10, labelpad=8)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

        ax.set_title(
            f"{self.experiment_name} — Objective Space (GP Surrogate)\n"
            f"Iteration {iteration} | Evaluated: {len(X_history)} | "
            f"Pareto (evaluated): {n_hist_pareto}",
            fontsize=12,
            fontweight="bold",
        )

        plot_path = os.path.join(self.vis_dir, f"objective_space_{iteration:03d}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[Viz] Created objective space plot for iteration {iteration}")
        return plot_path

    # ------------------------------------------------------------------ #
    #  GIF creation
    # ------------------------------------------------------------------ #

    def create_gif(self, duration: float = 2.0, gif_name: str = "optimization.gif") -> None:
        if not self.image_paths:
            print("[Viz] No images to create GIF")
            return

        gif_path = os.path.join(self.log_dir, gif_name)

        print("[Viz] Determining minimum image dimensions for GIF...")
        min_width = float("inf")
        min_height = float("inf")
        for path in self.image_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    min_width = min(min_width, w)
                    min_height = min(min_height, h)
            except Exception as e:
                print(f"[Viz] Warning: failed to read image {path}: {e}")
        if min_width == float("inf") or min_height == float("inf"):
            print("[Viz] Could not determine dimensions for GIF")
            return
        target_size = (int(min_width), int(min_height))

        frames = []
        for path in self.image_paths:
            try:
                with Image.open(path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != target_size:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                    frames.append(np.array(img))
            except Exception as e:
                print(f"[Viz] Warning: failed to process image {path}: {e}")
        if len(frames) < 2:
            print("[Viz] Not enough frames to create GIF")
            return

        try:
            imageio.mimsave(gif_path, frames, fps=1.0 / duration)
            print(f"[Viz] Created GIF at {gif_path}")
        except Exception as e:
            print(f"[Viz] Failed to create GIF: {e}")

# Additional Graphing Functions for Batched Runs

def plot_strategy_decisions(
    logger: LoggingManager,
    seed: int,
    beta_explore: float,
    event_iteration: int = None,
    save_dir: str = None
):
    """
    Plot the exploitation vs exploration decisions over iterations.
    
    Args:
        logger: LoggingManager with convergence data
        seed: Random seed used
        beta_explore: Exploration beta value
        event_iteration: Iteration where event occurred (optional)
        save_dir: Directory to save plot (default: logger.log_dir)
    """
    df = pd.read_csv(logger.convergence_path)
    df = df[df['iteration'] > 0].copy()
    df = df.sort_values('iteration')
    
    opt_color = '#2ECC71'
    exp_color = '#6C5CE7'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot optimization points
    ax.plot(df['iteration'], df['selected_n_opt'],
            color=opt_color, linewidth=2, alpha=0.85)
    ax.scatter(df['iteration'], df['selected_n_opt'],
               color=opt_color, s=100, marker='o',
               label='Optimization Points',
               alpha=0.9, edgecolors='black', linewidth=1)
    
    # Plot exploration points
    ax.plot(df['iteration'], df['selected_n_exp'],
            color=exp_color, linewidth=2, alpha=0.85)
    ax.scatter(df['iteration'], df['selected_n_exp'],
               color=exp_color, s=100, marker='s',
               label='Exploration Points',
               alpha=0.9, edgecolors='black', linewidth=1)
    
    # Add vertical line at event if applicable
    if event_iteration is not None:
        ax.axvline(x=event_iteration, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Event (Iter {event_iteration})')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Points Selected', fontsize=12, fontweight='bold')
    
    title = f'Strategy Decisions: {logger.experiment_name}\n(Seed: {seed}, β_explore: {beta_explore}'
    if event_iteration is not None:
        title += f', Event: Iter {event_iteration})'
    else:
        title += ')'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_yticks(range(0, 6))
    ax.set_ylim(-0.5, 5.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    if save_dir is None:
        save_dir = logger.log_dir
    save_path = os.path.join(save_dir, f"{logger.experiment_name}_strategy_decisions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved strategy decisions plot to {save_path}")

def plot_convergence_comparison(
    loggers: List[LoggingManager],
    seed: int,
    beta_explore: float,
    score_name: str = "Score",
    event_iteration: int = None,
    save_dir: str = None
):
    """
    Plot convergence comparison across strategies.
    
    Args:
        loggers: List of LoggingManager objects (in order: exploit, explore, agent)
        seed: Random seed used
        beta_explore: Exploration beta value
        score_name: Name of the score metric
        event_iteration: Iteration where event occurred (optional)
        save_dir: Directory to save plot (default: parent of first logger)
    """
    logger_exploit = loggers[0]
    logger_explore = loggers[1]
    logger_agent = loggers[2]
    
    df_exploit = pd.read_csv(logger_exploit.convergence_path)
    df_explore = pd.read_csv(logger_explore.convergence_path)
    df_agent = pd.read_csv(logger_agent.convergence_path)
    
    exploit_max_iter = df_exploit.loc[df_exploit['best_score'].idxmax(), 'iteration']
    explore_max_iter = df_explore.loc[df_explore['best_score'].idxmax(), 'iteration']
    agent_max_iter = df_agent.loc[df_agent['best_score'].idxmax(), 'iteration']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(df_exploit['iteration'], df_exploit['best_score'],
            marker='o', linewidth=2.5, label='Pure Exploitation', markersize=6)
    ax.plot(df_explore['iteration'], df_explore['best_score'],
            marker='s', linewidth=2.5, label='Pure Exploration', markersize=6)
    ax.plot(df_agent['iteration'], df_agent['best_score'],
            marker='^', linewidth=2.5, label='LLM Agent', markersize=7)
    
    ax.axvline(x=exploit_max_iter, color='C0', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(x=explore_max_iter, color='C1', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvline(x=agent_max_iter, color='C2', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Add vertical line at event if applicable
    if event_iteration is not None:
        ax.axvline(x=event_iteration, color='red', linestyle='--', 
                   linewidth=2.5, alpha=0.7, label=f'Event (Iter {event_iteration})')
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Best {score_name}', fontsize=14, fontweight='bold')
    
    title = f'Strategy Comparison\n(Seed: {seed}'
    if event_iteration is not None:
        title += f', Event: Iter {event_iteration})'
    else:
        title += ')'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    
    if save_dir is None:
        save_dir = os.path.dirname(loggers[0].log_dir)
    save_path = os.path.join(save_dir, "convergence_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved convergence comparison to {save_path}")

def plot_hypervolume_per_iteration(
    logger: LoggingManager,
    score_name: str = "Score",
    save_dir: str = None
):
    """
    Plot hypervolume of Pareto front over iterations.
    
    Args:
        logger: LoggingManager with convergence data
        score_name: Name of score metric (for title)
        save_dir: Directory to save plot (default: logger.log_dir)
    """
    df = pd.read_csv(logger.convergence_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['iteration'], df['total_hypervolume'],
            color='#3498DB', linewidth=2.5, marker='o',
            markersize=8, alpha=0.85, label='Hypervolume')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize=12, fontweight='bold')
    ax.set_title(f'Hypervolume Progress: {logger.experiment_name}',
                 fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    if save_dir is None:
        save_dir = logger.log_dir
    save_path = os.path.join(save_dir, f"{logger.experiment_name}_hypervolume.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved hypervolume plot to {save_path}")


def plot_final_pareto_front(
    Y_history: np.ndarray,
    obj1_name: str,
    obj2_name: str,
    obj1_display: str,
    obj2_display: str,
    experiment_name: str,
    save_path: str
):
    """
    Plot final Pareto front.
    
    Args:
        Y_history: All objective values (n, 2)
        obj1_name: Name of objective 1
        obj2_name: Name of objective 2
        experiment_name: Name of experiment (for title)
        save_path: Path to save plot
    """
    from botorch.utils.multi_objective.pareto import is_non_dominated
    
    # Convert to torch for Pareto filtering
    Y_tensor = torch.tensor(Y_history, dtype=torch.double)
    pareto_mask = is_non_dominated(Y_tensor).cpu().numpy()
    
    # Split into Pareto and non-Pareto points
    pareto_points = Y_history[pareto_mask]
    non_pareto_points = Y_history[~pareto_mask]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot non-Pareto points
    if len(non_pareto_points) > 0:
        ax.scatter(non_pareto_points[:, 0], non_pareto_points[:, 1],
                   c='lightgray', s=80, alpha=0.5, 
                   label='Non-Pareto Points', zorder=1)
    
    # Plot Pareto front
    if len(pareto_points) > 0:
        # Sort Pareto points for line plotting
        sorted_idx = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_idx]
        
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                'r-', linewidth=2, alpha=0.6, zorder=2)
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1],
                   c='#E74C3C', s=150, marker='*',
                   edgecolors='darkred', linewidth=1.5,
                   label=f'Pareto Front ({len(pareto_points)} points)',
                   zorder=3)
    
    ax.set_xlabel(f'{obj1_display}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{obj2_display}', fontsize=12, fontweight='bold')
    ax.set_title(f'Final Pareto Front: {experiment_name}',
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved Pareto front plot to {save_path}")