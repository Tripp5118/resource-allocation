# core/visualization.py

import os
from typing import Optional, List, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import imageio

import torch
from botorch.utils.transforms import normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


class VisualizationManager:
    """
    Visualization manager for BO on a composition simplex.

    - Assumes design_space.space is an (N, d) array of compositions summing to 1.
    - Supports 2 <= d <= 10 by projecting onto a regular d-gon in 2D.
    - Can color points by:
        * a scalar score from a GP model (via score_fn(Y_pred)), or
        * arbitrary scalar values from a user-supplied evaluator f(x).
    """

    def __init__(self, log_dir: str, design_space, experiment_name: str = "experiment"):
        self.log_dir = log_dir
        self.design_space = design_space
        self.experiment_name = experiment_name
        self.vis_dir = os.path.join(log_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

        # Infer dimension and labels
        self.dim = design_space.space.shape[1]
        if not (2 <= self.dim <= 10):
            print(f"[Viz] Dimension d={self.dim} is outside [2, 10]; visualization disabled.")
            self.enabled = False
            self.image_paths: List[str] = []
            return
        self.enabled = True

        labels = getattr(design_space, "labels", None)
        if labels is not None and len(labels) == self.dim:
            self.element_names = list(labels)
        else:
            self.element_names = [f"x_{i}" for i in range(self.dim)]

        self._setup_affine_projection()

        # Track image paths for GIF creation
        self.image_paths: List[str] = []

        print(f"[Viz] Initialized visualization in {self.vis_dir}")
        print(f"[Viz] Experiment name: {experiment_name}")
        print(f"[Viz] Dimension d={self.dim}, labels={self.element_names}")

    # ------------------------------------------------------------------ #
    #  Projection setup
    # ------------------------------------------------------------------ #

    def _setup_affine_projection(self) -> None:
        """Build regular d-gon and precompute 2D projections of full design space."""
        num_vertices = self.dim
        vertices = []
        angle_step = 2.0 * np.pi / num_vertices
        initial_angle = np.pi / 2.0  # first vertex at top

        for i in range(num_vertices):
            angle = initial_angle + i * angle_step
            x = np.cos(angle)
            y = np.sin(angle)
            vertices.append([x, y])

        self.simplex_vertices = np.array(vertices)       # (d, 2)
        self.projection_matrix = self.simplex_vertices   # (d, 2)

        # Use full design space by default
        self.vis_space = self.design_space.space
        n_available = len(self.vis_space)
        print(f"[Viz] Using full design space ({n_available} points) for visualization")

        # Precompute 2D projections (CPU)
        self.design_space_2d = self._affine_transform(self.vis_space)
        self.vis_space_cpu = torch.tensor(self.vis_space, dtype=DTYPE, device="cpu")

        print(f"[Viz] Setup affine projection: projected {len(self.design_space_2d)} points")
        print(f"[Viz] Cached visualization space on CPU: {self.vis_space_cpu.shape}")

    def _affine_transform(self, X: np.ndarray) -> np.ndarray:
        """Project compositions X (n,d) to 2D using the d-gon vertices."""
        X_norm = X  # assume already valid compositions
        return X_norm @ self.projection_matrix  # (n,d) x (d,2) -> (n,2)

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

        Args:
            model: fitted MultiTaskGP model
            bounds: (2,d) tensor
            score_fn: maps predicted objectives Y_pred (N,2) -> scores (N,)

        Returns:
            scores: (N,) scalar scores
        """
        vis_space_gpu = self.vis_space_cpu.to(device=DEVICE)
        bounds_gpu = bounds.to(device=DEVICE)

        X_norm = normalize(vis_space_gpu, bounds_gpu)
        n = X_norm.shape[0]

        # MultiTaskGP with task index: prepare both tasks
        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        X_task_0 = torch.cat([X_norm, task_0], dim=-1)
        X_task_1 = torch.cat([X_norm, task_1], dim=-1)
        X_both = torch.cat([X_task_0, X_task_1], dim=0)

        with torch.no_grad():
            post = model.posterior(X_both)
            mean_flat = post.mean.squeeze(-1)  # (2n,)
            mean_obj1 = mean_flat[:n].cpu().numpy()
            mean_obj2 = mean_flat[n:].cpu().numpy()

        Y_pred = np.column_stack([mean_obj1, mean_obj2])
        scores = score_fn(Y_pred)

        # Cleanup
        del vis_space_gpu, X_norm, X_task_0, X_task_1, X_both
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

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
            s=10,
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
        if len(obj_names) != 2:
            raise ValueError("obj_names must be a list of 2 names.")

        Y = self.evaluate_on_design_space(evaluator)  # (N,2)
        if Y.shape[1] != 2:
            raise ValueError("evaluator must return an array of shape (N,2).")

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
        obj1_name: str = "obj1",
        obj2_name: str = "obj2",
    ) -> Optional[str]:
        """
        Create a 2D visualization for this iteration, optionally using a GP model.

        Args:
            X_history, Y_history: all evaluated points and objectives
            X_new: newly evaluated points this iteration
            model: trained MultiTaskGP (if None, only evaluated points are shown)
            bounds: (2,d) tensor
            selection_info: metadata for title/legend
            score_fn: maps Y (n,2) -> scalar scores for coloring
            score_name: label for the colorbar
            obj1_name, obj2_name: used in title for best point
        """
        if not self.enabled:
            print("[Viz] Visualization disabled (dimension outside [2,10]).")
            return None

        if score_fn is None:
            def score_fn_default(Y: np.ndarray) -> np.ndarray:
                return Y[:, 1]
            score_fn = score_fn_default

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
                s=10,
                cmap="plasma",
                edgecolors="none",
                vmin=min_s,
                vmax=max_s,
                alpha=0.8,
                zorder=2,
            )
        else:
            print("[Viz] No GP model provided; plotting evaluated points only.")

        # Plot evaluated history
        n_new = len(X_new)
        scores_hist = score_fn(Y_history)
        if len(X_history) > n_new:
            X_old = X_history[:-n_new]
            Y_old = Y_history[:-n_new]
            X_old_2d = self._affine_transform(X_old)
            scores_old = score_fn(Y_old)
            ax.scatter(
                X_old_2d[:, 0],
                X_old_2d[:, 1],
                c=scores_old,
                cmap="plasma",
                s=80,
                edgecolors="black",
                linewidths=1.5,
                vmin=min_s if min_s is not None else None,
                vmax=max_s if max_s is not None else None,
                alpha=0.9,
                zorder=3,
                label="Evaluated",
            )

        # New points
        if n_new > 0:
            X_new_2d = self._affine_transform(X_new)
            Y_new_hist = Y_history[-n_new:]
            scores_new = score_fn(Y_new_hist)
            ax.scatter(
                X_new_2d[:, 0],
                X_new_2d[:, 1],
                c=scores_new,
                cmap="plasma",
                s=200,
                marker="*",
                edgecolors="orange",
                linewidths=2,
                vmin=min_s if min_s is not None else None,
                vmax=max_s if max_s is not None else None,
                zorder=4,
                label="New",
            )

        # Best point (by score_fn)
        best_idx = int(np.argmax(scores_hist))
        best_2d = self._affine_transform(X_history[best_idx:best_idx + 1])
        best_y = Y_history[best_idx]
        ax.scatter(
            best_2d[:, 0],
            best_2d[:, 1],
            marker="*",
            s=300,
            facecolors="lime",
            edgecolors="darkgreen",
            linewidths=3,
            zorder=5,
            label=f"Best ({score_name}={scores_hist[best_idx]:.3f})",
        )

        # Colorbar
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(score_name, fontsize=11)

        # Title
        title_lines = [
            f"{self.experiment_name}",
            f"Iteration {iteration}",
            f"Evaluated: {len(X_history)}, Best {score_name}: {scores_hist[best_idx]:.3f}",
            f"Best point: {obj1_name}={best_y[0]:.4g}, {obj2_name}={best_y[1]:.4g}",
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