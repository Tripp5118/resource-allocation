import os
from pathlib import Path
from typing import Optional, List

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

    def __init__(self, log_dir: str, design_space, experiment_name: str = "experiment"):
        self.log_dir = log_dir
        self.design_space = design_space
        self.experiment_name = experiment_name
        self.vis_dir = os.path.join(log_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

        # Setup affine projection with subsampled design space
        self._setup_affine_projection()

        # Track image paths for GIF creation
        self.image_paths: List[str] = []

        print(f"[Viz] Initialized visualization in {self.vis_dir}")
        print(f"[Viz] Experiment name: {experiment_name}")

    def _setup_affine_projection(self) -> None:
        # Determine the element order from the design space if available
        try:
            from core.design_space import ELEM_ORDER
            self.element_names = list(ELEM_ORDER)
        except Exception:
            self.element_names = ['Fe', 'Co', 'Cr', 'Ni', 'V']

        num_vertices = len(self.element_names)
        # Generate regular pentagon vertices (first at pi/2)
        pentagon_vertices = []
        angle_step = 2 * np.pi / num_vertices
        initial_angle = np.pi / 2.0
        for i in range(num_vertices):
            angle = initial_angle + i * angle_step
            x = np.cos(angle)
            y = np.sin(angle)
            pentagon_vertices.append([x, y])
        self.pentagon_vertices = np.array(pentagon_vertices)

        # Create projection matrix: shape (5, 2), projecting compositions to 2‑D
        self.projection_matrix = self.pentagon_vertices

        n_vis_points = 100000
        n_available = len(self.design_space.space)
        
        if n_available > n_vis_points:
            # Subsample using the design_space's sample method (Sobol for better coverage)
            self.vis_space = self.design_space.sample(n_vis_points, method='sobol')
            print(f"[Viz] Subsampled design space from {n_available} to {n_vis_points} points for visualization")
        else:
            self.vis_space = self.design_space.space
            print(f"[Viz] Using full design space ({n_available} points) for visualization")

        # Precompute 2D projections for the subsampled space (CPU operation, done once)
        self.design_space_2d = self._affine_transform(self.vis_space)
        
        # Cache subsampled space on CPU (moved from GPU to avoid OOM)
        self.vis_space_cpu = torch.tensor(self.vis_space, dtype=DTYPE, device='cpu')
        
        print(f"[Viz] Setup affine projection: projected {len(self.design_space_2d)} points")
        print(f"[Viz] Cached visualization space on CPU: {self.vis_space_cpu.shape}")

    def _affine_transform(self, X: np.ndarray) -> np.ndarray:
        # Normalise compositions to sum to one
        X_norm = X #/ X.sum(axis=1, keepdims=True)
        # Project: (n,5) × (5,2) = (n,2)
        return X_norm @ self.projection_matrix

    def _predict_ratio(self, model, bounds: torch.Tensor) -> np.ndarray:
        """Predict K/|CTE| ratio on subsampled design space."""
        
        # Keep model on GPU, move data to GPU for prediction
        vis_space_gpu = self.vis_space_cpu.to(device=DEVICE)
        bounds_gpu = bounds.to(device=DEVICE)
        
        X_norm = normalize(vis_space_gpu, bounds_gpu)
        
        # Add task indices for both tasks
        n = X_norm.shape[0]
        task_0 = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
        task_1 = torch.ones(n, 1, dtype=DTYPE, device=DEVICE)
        
        X_task_0 = torch.cat([X_norm, task_0], dim=-1)
        X_task_1 = torch.cat([X_norm, task_1], dim=-1)
        
        with torch.no_grad():
            # Predict in batches
            batch_size = 500
            cte_preds = []
            k_preds = []
            
            for i in range(0, n, batch_size):
                end_idx = min(i + batch_size, n)
                batch_0 = X_norm[i:end_idx]
                post = model.posterior(batch_0)
                cte_preds.append(post.mean[:,0].squeeze(-1).cpu())  # Move to CPU immediately
                k_preds.append(post.mean[:,1].squeeze(-1).cpu())
            
            cte_pred = torch.cat(cte_preds, dim=0).numpy()
            k_pred = torch.cat(k_preds, dim=0).numpy()
        
        # Clean up GPU memory
        del vis_space_gpu, X_norm, X_task_0, X_task_1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # CTE is already in units of 1e-6 K⁻¹
        ratio_pred = k_pred / (np.abs(cte_pred) + 1e-12)
        return ratio_pred

    def _draw_pentagon_and_labels(self, ax: plt.Axes) -> None:
        # Grey pentagon background
        polygon = Polygon(self.pentagon_vertices, closed=True,
                          facecolor='#dddddd', edgecolor='black', lw=2, zorder=1)
        ax.add_patch(polygon)
        # Label each vertex slightly outside the polygon
        for i, name in enumerate(self.element_names):
            x, y = self.pentagon_vertices[i]
            # Offset position by 5% beyond the unit circle
            label_x = x * 1.05
            label_y = y * 1.05
            ax.text(label_x, label_y, name, fontsize=14, fontweight='bold',
                    ha='center', va='center', color='black')

    def create_iteration_plot(self,
                             iteration: int,
                             X_history: np.ndarray,
                             Y_history: np.ndarray,
                             X_new: np.ndarray,
                             model,
                             bounds: torch.Tensor,
                             selection_info: Optional[dict] = None) -> str:
        # Prepare figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw base pentagon and labels
        self._draw_pentagon_and_labels(ax)

        # Predict ratio over subsampled design space (on CPU)
        predicted_ratios = self._predict_ratio(model, bounds)
        # Use percentile clipping to handle outliers (recomputed each iteration)
        min_ratio = np.percentile(predicted_ratios, 0)
        max_ratio = np.percentile(predicted_ratios, 100)
        # Diagnostic logging
        print(f"[Viz] Predicted ratios - Min: {predicted_ratios.min():.4f}, "
              f"Max: {predicted_ratios.max():.4f}, Mean: {predicted_ratios.mean():.4f}")
        print(f"[Viz] Color scale - Min (0%): {min_ratio:.4f}, Max (100%): {max_ratio:.4f}")
        
        # Scatter plot of predicted performance across design space
        # Sort by predicted ratios
        sort_indices = np.argsort(predicted_ratios)
        x = self.design_space_2d[sort_indices, 0]
        y = self.design_space_2d[sort_indices, 1]
        sorted_ratios = predicted_ratios[sort_indices]

        scatter = ax.scatter(
            x,
            y,
            c=sorted_ratios,
            s=10,
            cmap='plasma',
            edgecolors='none',
            vmin=min_ratio,
            vmax=max_ratio,
            alpha=0.8,
            zorder=2
        )

        # Plot previously evaluated points (excluding new ones)
        n_new = len(X_new)
        if len(X_history) > n_new:
            X_old = X_history[:-n_new]
            Y_old = Y_history[:-n_new]
            X_old_2d = self._affine_transform(X_old)
            # CTE already in 1e-6 units
            ratios_old = Y_old[:, 1] / (np.abs(Y_old[:, 0]) + 1e-12)
            ax.scatter(
                X_old_2d[:, 0],
                X_old_2d[:, 1],
                c=ratios_old,
                cmap='plasma',
                s=80,
                edgecolors='black',
                linewidths=1.5,
                vmin=min_ratio,
                vmax=max_ratio,
                alpha=0.9,
                zorder=3,
                label='Evaluated'
            )

        # Plot new points
        if n_new > 0:
            X_new_2d = self._affine_transform(X_new)
            Y_new = Y_history[-n_new:]
            # CTE already in 1e-6 units
            ratios_new = Y_new[:, 1] / (np.abs(Y_new[:, 0]) + 1e-12)
            ax.scatter(
                X_new_2d[:, 0],
                X_new_2d[:, 1],
                c=ratios_new,
                cmap='plasma',
                s=200,
                marker='*',
                edgecolors='orange',
                linewidths=2,
                vmin=min_ratio,
                vmax=max_ratio,
                zorder=4,
                label='New'
            )

        # Highlight the best observed point
        # CTE already in 1e-6 units
        all_ratios = Y_history[:, 1] / (np.abs(Y_history[:, 0]) + 1e-12)
        best_idx = int(np.argmax(all_ratios))
        best_point_2d = self._affine_transform(X_history[best_idx:best_idx+1])
        best_cte = Y_history[best_idx, 0]  # Already in 1e-6 K⁻¹
        best_k = Y_history[best_idx, 1]
        ax.scatter(
            best_point_2d[:, 0],
            best_point_2d[:, 1],
            marker='*',
            s=300,
            facecolors='lime',
            edgecolors='darkgreen',
            linewidths=3,
            zorder=5,
            label=f'Best (K/|CTE|={all_ratios[best_idx]:.3f})'
        )

        # Colourbar on the right side with proper label
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('K / |CTE| Ratio (W·m⁻¹·K⁻¹ / 10⁻⁶·K⁻¹)', fontsize=11)

        # Title including experiment name and iteration
        title_lines = [f'{self.experiment_name}',
                       f'Iteration {iteration}',
                       f'Evaluated: {len(X_history)}, Best K/|CTE|: {all_ratios[best_idx]:.3f}']
        # Updated CTE units to show 1e-6 K⁻¹
        title_lines.append(f'Best point: CTE={best_cte:.2f}× μK⁻¹, K={best_k:.2f} W·m⁻¹·K⁻¹')
        if selection_info:
            if 'n_optimization' in selection_info and 'n_exploration' in selection_info:
                title_lines.append(f"Allocation: {selection_info['n_optimization']} opt + "
                                 f"{selection_info['n_exploration']} explore")
            elif 'method' in selection_info:
                title_lines.append(f"Selection: {selection_info.get('method', 'Unknown')}")
        ax.set_title('\n'.join(title_lines), fontsize=14, fontweight='bold', pad=15)

        # Legend positioned to the right, outside the plot area, below the colorbar
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 0.85), fontsize=10)

        # Save figure with tight layout
        plot_path = os.path.join(self.vis_dir, f'iteration_{iteration:03d}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        self.image_paths.append(plot_path)
        print(f"[Viz] Created plot for iteration {iteration}")
        return plot_path

    def create_gif(self, duration: float = 2.0, gif_name: str = "optimization.gif") -> None:
        if not self.image_paths:
            print("[Viz] No images to create GIF")
            return

        gif_path = os.path.join(self.log_dir, gif_name)

        # Determine smallest dimensions across all images
        print("[Viz] Determining minimum image dimensions for GIF...")
        min_width = float('inf')
        min_height = float('inf')
        for path in self.image_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    if w < min_width:
                        min_width = w
                    if h < min_height:
                        min_height = h
            except Exception as e:
                print(f"[Viz] Warning: failed to read image {path}: {e}")
        if min_width == float('inf') or min_height == float('inf'):
            print("[Viz] Could not determine dimensions for GIF")
            return
        target_size = (int(min_width), int(min_height))

        # Load and resize images
        frames = []
        for path in self.image_paths:
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
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