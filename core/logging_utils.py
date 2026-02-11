"""Logging utilities for generic 2-objective BO experiments."""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Callable


class LoggingManager:
    """Manages CSV logging and metadata for BO experiments."""

    def __init__(
        self, 
        log_dir: str, 
        experiment_name: str,
        clear_existing: bool = True,
    ):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of this experiment
            clear_existing: If True, delete existing CSV files before starting.
                           If False, append to existing files (can cause duplicates).
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # Setup paths
        self.convergence_path = os.path.join(log_dir, f"{experiment_name}_convergence.csv")
        self.options_path = os.path.join(log_dir, f"{experiment_name}_options.csv")
        self.evaluations_path = os.path.join(log_dir, f"{experiment_name}_evaluations.csv")
        self.metadata_path = os.path.join(log_dir, f"{experiment_name}_metadata.json")

        # Clear existing files if requested
        if clear_existing:
            for path in [self.convergence_path, self.options_path, self.evaluations_path, self.metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"[Logger] Cleared existing file: {path}")

        # Initialize metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "iterations_completed": 0,
        }

        print(f"[Logger] Initialized logging in {log_dir}")
        print(f"[Logger] Files:")
        print(f"  - Convergence: {self.convergence_path}")
        print(f"  - Options: {self.options_path}")
        print(f"  - Evaluations: {self.evaluations_path}")

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def log_iteration(
        self,
        iteration: int,
        X_history: np.ndarray,
        Y_history: np.ndarray,
        n_new_points: int,
        strategy,
        timing: float = 0.0,
        extra_info: Optional[Dict[str, Any]] = None,
        acquisition_data=None,
        score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        obj1_name: str = "obj1",
        obj2_name: str = "obj2",
    ):
        """
        Log convergence data and options for one iteration.

        Args:
            iteration: Current iteration number (0 = initial)
            X_history: All evaluated points so far (n, d)
            Y_history: All evaluated objective values so far (n, 2)
            n_new_points: Number of new points evaluated this iteration
            strategy: Strategy/agent instance (for metadata)
            timing: Time taken for this iteration (seconds)
            extra_info: Additional info (selection details, budget, etc.)
            acquisition_data: AllocationResults object with all options
            score_fn: Function taking Y_history (n, 2) and returning scores (n,)
            obj1_name, obj2_name: Names of the two objectives (for column labels)
        """
        if score_fn is None:
            # Default: maximize obj2, ignore obj1
            def score_fn_default(Y: np.ndarray) -> np.ndarray:
                return Y[:, 1]
            score_fn = score_fn_default

        self._log_convergence(
            iteration=iteration,
            X_history=X_history,
            Y_history=np.abs(Y_history),
            n_new_points=n_new_points,
            strategy=strategy,
            timing=timing,
            extra_info=extra_info,
            score_fn=score_fn,
            obj1_name=obj1_name,
            obj2_name=obj2_name,
        )

        # Log allocation options (if available and not iteration 0)
        if acquisition_data is not None and iteration > 0:
            self._log_options(iteration, acquisition_data, extra_info)

        # Update metadata
        self.metadata["iterations_completed"] = iteration
        self.metadata["last_update"] = datetime.now().isoformat()
        self._save_metadata()

    def log_evaluations(
        self,
        iteration: int,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        selection_info: Optional[Dict[str, Any]] = None,
        score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        obj1_name: str = "obj1",
        obj2_name: str = "obj2",
    ):
        """
        Log detailed evaluation results for newly evaluated points.

        Args:
            iteration: Current iteration
            X_new: Newly evaluated points (k, d)
            Y_new: Newly evaluated objectives (k, 2)
            selection_info: Info about how points were selected
            score_fn: Function mapping Y_new (k, 2) -> scores (k,)
            obj1_name, obj2_name: Names of the objectives
        """
        if score_fn is None:
            def score_fn_default(Y: np.ndarray) -> np.ndarray:
                return Y[:, 1]
            score_fn = score_fn_default

        scores = score_fn(Y_new)
        rows = []

        Y_new = np.abs(Y_new)
        for i, (x, y, s) in enumerate(zip(X_new, Y_new, scores)):
            row = {
                "iteration": iteration,
                "point_idx": i,
                obj1_name: y[0],
                obj2_name: y[1],
                "score": s,
            }
            # Add coordinates x_0 ... x_{d-1}
            for j, val in enumerate(x):
                row[f"x_{j}"] = val

            # Source information
            if selection_info and "sources" in selection_info:
                row["source"] = selection_info["sources"][i] if i < len(selection_info["sources"]) else "Unknown"
            else:
                row["source"] = "Unknown"

            rows.append(row)

        df = pd.DataFrame(rows)
        if not os.path.exists(self.evaluations_path):
            # First write - create file with header
            df.to_csv(self.evaluations_path, index=False, mode='w')
        else:
            # Subsequent writes - append without header
            df.to_csv(self.evaluations_path, index=False, mode='a', header=False)

        print(f"[Logger] Logged {len(rows)} evaluations for iteration {iteration}")

    def finalize(self):
        """Finalize logging at end of experiment."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["status"] = "completed"
        self._save_metadata()
        print(f"[Logger] Finalized logging for {self.experiment_name}")

    def load_convergence_data(self) -> pd.DataFrame:
        """Load convergence data from CSV."""
        if os.path.exists(self.convergence_path):
            return pd.read_csv(self.convergence_path)
        return pd.DataFrame()

    def load_options_data(self) -> pd.DataFrame:
        """Load options data from CSV."""
        if os.path.exists(self.options_path):
            return pd.read_csv(self.options_path)
        return pd.DataFrame()

    def load_evaluations_data(self) -> pd.DataFrame:
        """Load evaluations data from CSV."""
        if os.path.exists(self.evaluations_path):
            return pd.read_csv(self.evaluations_path)
        return pd.DataFrame()

    def get_summary_stats(self, obj1_name="obj1", obj2_name="obj2") -> Dict[str, Any]:
        """Get summary statistics from logged data."""
        df = self.load_convergence_data()
        if df.empty:
            return {}

        last_iter = df.iloc[-1]
        return {
            "total_iterations": int(last_iter["iteration"]),
            "total_evaluations": int(last_iter["total_evaluated"]),
            "best_score": float(last_iter["best_score"]),
            f"best_{obj1_name}": float(last_iter[f"best_{obj1_name}"]),
            f"best_{obj2_name}": float(last_iter[f"best_{obj2_name}"]),
            "final_mean_score": float(last_iter["mean_score"]),
            "total_time_sec": float(df["timing_sec"].sum()),
        }

    # --------------------------------------------------------------------- #
    # INTERNAL HELPERS
    # --------------------------------------------------------------------- #

    def _log_convergence(
        self,
        iteration: int,
        X_history: np.ndarray,
        Y_history: np.ndarray,
        n_new_points: int,
        strategy,
        timing: float,
        extra_info: Optional[Dict[str, Any]],
        score_fn: Callable[[np.ndarray], np.ndarray],
        obj1_name: str,
        obj2_name: str,
    ):
        """Log high-level convergence metrics."""
        scores = score_fn(Y_history)
        obj1_vals = Y_history[:, 0]
        obj2_vals = Y_history[:, 1]

        best_idx = int(np.nanargmax(scores))
        best_score = scores[best_idx]
        best_obj1 = obj1_vals[best_idx]
        best_obj2 = obj2_vals[best_idx]
        best_x = X_history[best_idx]

        row_data = {
            "iteration": iteration,
            "total_evaluated": len(X_history),
            "new_points": n_new_points,
            "timing_sec": timing,
            "strategy": type(strategy).__name__ if strategy is not None else "None",
            "best_score": best_score,
            f"best_{obj1_name}": best_obj1,
            f"best_{obj2_name}": best_obj2,
            "mean_score": np.nanmean(scores),
            "std_score": np.nanstd(scores),
            f"mean_{obj1_name}": np.nanmean(obj1_vals),
            f"mean_{obj2_name}": np.nanmean(obj2_vals),
        }

        # add best x_*
        for j, val in enumerate(best_x):
            row_data[f"best_x_{j}"] = val

        # selection info (optional)
        if extra_info:
            row_data["selected_n_opt"] = extra_info.get("n_optimization", None)
            row_data["selected_n_exp"] = extra_info.get("n_exploration", None)
            # ADD THESE TWO LINES:
            row_data["hypervolume_improvement"] = extra_info.get("hypervolume_improvement", None)
            row_data["information_gain"] = extra_info.get("information_gain", None)
        else:
            row_data["selected_n_opt"] = None
            row_data["selected_n_exp"] = None
            # ADD THESE TWO LINES:
            row_data["hypervolume_improvement"] = None
            row_data["information_gain"] = None

        # LLM info, if present
        if hasattr(strategy, "model"):
            row_data["llm_model"] = strategy.model
            row_data["llm_temp"] = getattr(strategy, "temperature", None)
        else:
            row_data["llm_model"] = None
            row_data["llm_temp"] = None

        df_row = pd.DataFrame([row_data])
        if not os.path.exists(self.convergence_path):
            # First write - create file with header
            df_row.to_csv(self.convergence_path, index=False, mode='w')
        else:
            # Subsequent writes - append without header
            df_row.to_csv(self.convergence_path, index=False, mode='a', header=False)

        print(f"[Logger] Logged convergence for iteration {iteration}")

    def _log_options(self, iteration: int, acquisition_data, extra_info: Optional[Dict[str, Any]]):
        """Log allocation options with their points and acquisition metrics."""
        selected_n_opt = extra_info.get("n_optimization", None) if extra_info else None
        selected_n_exp = extra_info.get("n_exploration", None) if extra_info else None

        rows = []
        for option_num, batch in enumerate(acquisition_data.options, start=1):
            n_opt = batch.num_exploitation
            n_exp = batch.total_batch_size - batch.num_exploitation if hasattr(batch, "total_batch_size") else None
            is_selected = (selected_n_opt == n_opt and selected_n_exp == n_exp)

            base_row = {
                "iteration": iteration,
                "option": option_num,
                "n_opt": n_opt,
                "n_exp": n_exp,
                "selected": is_selected,
                "full_qehvi": batch.hypervolume_improvement,
                "full_entropy": batch.information_gain,
            }

            all_points = []
            all_sources = []

            for point in batch.exploitation_points:
                all_points.append(point)
                all_sources.append("QEHVI")
            for point in batch.exploration_points:
                all_points.append(point)
                all_sources.append("MO-MESMO")

            for point_idx, (point, source) in enumerate(zip(all_points, all_sources)):
                row = base_row.copy()
                row["point_idx"] = point_idx
                row["source"] = source
                for j, val in enumerate(point):
                    row[f"x_{j}"] = val
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            if not os.path.exists(self.options_path):
                # First write - create file with header
                df.to_csv(self.options_path, index=False, mode='w')
            else:
                # Subsequent writes - append without header
                df.to_csv(self.options_path, index=False, mode='a', header=False)
            print(
                f"[Logger] Logged {len(acquisition_data.options)} options "
                f"with {len(rows)} total points for iteration {iteration}"
            )

    def _save_metadata(self):
        """Save metadata to JSON."""
        with open(self.metadata_path, "w") as f:
            import json
            json.dump(self.metadata, f, indent=2)