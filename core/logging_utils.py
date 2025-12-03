"""Logging utilities for BO experiments."""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import json


class LoggingManager:
    """Manages CSV logging and metadata for BO experiments."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of this experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup paths
        self.convergence_path = os.path.join(log_dir, f"{experiment_name}_convergence.csv")
        self.options_path = os.path.join(log_dir, f"{experiment_name}_options.csv")
        self.evaluations_path = os.path.join(log_dir, f"{experiment_name}_evaluations.csv")
        self.metadata_path = os.path.join(log_dir, f"{experiment_name}_metadata.json")
        
        # Initialize metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'iterations_completed': 0
        }
        
        print(f"[Logger] Initialized logging in {log_dir}")
        print(f"[Logger] Files:")
        print(f"  - Convergence: {self.convergence_path}")
        print(f"  - Options: {self.options_path}")
        print(f"  - Evaluations: {self.evaluations_path}")
    
    def log_iteration(self,
                      iteration: int,
                      X_history: np.ndarray,
                      Y_history: np.ndarray,
                      n_new_points: int,
                      strategy,
                      timing: float = 0.0,
                      extra_info: Optional[Dict[str, Any]] = None,
                      acquisition_data=None):
        """
        Log convergence data and options for one iteration.
        
        Args:
            iteration: Current iteration number (0 = initial)
            X_history: All evaluated compositions so far (n, 5)
            Y_history: All evaluated objectives so far (n, 2) [CTE, K]
            n_new_points: Number of new points evaluated this iteration
            strategy: Strategy instance
            timing: Time taken for this iteration (seconds)
            extra_info: Additional info (contains selection details)
            acquisition_data: AcquisitionData object with all options
        """
        # Log convergence summary
        self._log_convergence(iteration, X_history, Y_history, n_new_points, 
                             strategy, timing, extra_info)
        
        # Log all allocation options (if available and not iteration 0)
        if acquisition_data is not None and iteration > 0:
            self._log_options(iteration, acquisition_data, extra_info)
        
        # Update metadata
        self.metadata['iterations_completed'] = iteration
        self.metadata['last_update'] = datetime.now().isoformat()
        self._save_metadata()
    
    def _log_convergence(self, iteration: int, X_history: np.ndarray, 
                        Y_history: np.ndarray, n_new_points: int,
                        strategy, timing: float, extra_info: Optional[Dict[str, Any]]):
        """Log high-level convergence metrics."""
        # Compute metrics using absolute value of CTE
        cte_vals = Y_history[:, 0]
        k_vals = Y_history[:, 1]
        ratios = k_vals / (np.abs(cte_vals) + 1e-12)
        
        # Best ratio info
        best_ratio_idx = int(np.nanargmax(ratios))
        best_ratio = ratios[best_ratio_idx]
        best_cte = cte_vals[best_ratio_idx]
        best_k = k_vals[best_ratio_idx]
        best_composition = X_history[best_ratio_idx]
        
        # Build row data
        row_data = {
            # Iteration info
            'iteration': iteration,
            'total_evaluated': len(X_history),
            'new_points': n_new_points,
            'timing_sec': timing,
            
            # Strategy info
            'strategy': type(strategy).__name__,
            
            # Best K/|CTE| ratio
            'best_ratio': best_ratio,
            'best_cte': best_cte * 1e6,  # Convert to µK⁻¹
            'best_k': best_k,
            'best_fe': best_composition[0],
            'best_co': best_composition[1],
            'best_cr': best_composition[2],
            'best_ni': best_composition[3],
            'best_v': best_composition[4],
            
            # Statistics
            'mean_ratio': np.nanmean(ratios),
            'std_ratio': np.nanstd(ratios),
            'mean_cte': np.nanmean(np.abs(cte_vals)) * 1e6,
            'mean_k': np.nanmean(k_vals),
        }
        
        # Add selection info if available
        if extra_info:
            row_data['selected_n_opt'] = extra_info.get('n_optimization', None)
            row_data['selected_n_exp'] = extra_info.get('n_exploration', None)
        else:
            row_data['selected_n_opt'] = None
            row_data['selected_n_exp'] = None
        
        # Add strategy-specific parameters
        if hasattr(strategy, 'w_explore'):
            row_data['w_explore'] = strategy.w_explore
            row_data['w_exploit'] = strategy.w_exploit
        else:
            row_data['w_explore'] = None
            row_data['w_exploit'] = None
        
        if hasattr(strategy, 'model'):
            row_data['llm_model'] = strategy.model
            row_data['llm_temp'] = getattr(strategy, 'temperature', None)
        else:
            row_data['llm_model'] = None
            row_data['llm_temp'] = None
        
        # Write to CSV
        df_row = pd.DataFrame([row_data])
        
        if not os.path.exists(self.convergence_path):
            df_row.to_csv(self.convergence_path, index=False)
        else:
            df_row.to_csv(self.convergence_path, mode='a', header=False, index=False)
        
        print(f"[Logger] Logged convergence for iteration {iteration}")
    
    def _log_options(self, iteration: int, acquisition_data, 
                    extra_info: Optional[Dict[str, Any]]):
        """Log all 6 allocation options with their compositions and metrics."""
        # Determine which option was selected
        selected_n_opt = extra_info.get('n_optimization', None) if extra_info else None
        selected_n_exp = extra_info.get('n_exploration', None) if extra_info else None
        
        rows = []
        
        # Log each allocation option
        for option_num, batch in enumerate(acquisition_data.qehvi_batches, start=1):
            n_opt = batch.batch_size
            n_exp = 5 - n_opt
            
            # Determine if this option was selected
            is_selected = (selected_n_opt == n_opt and selected_n_exp == n_exp)
            
            # Base row for this option
            base_row = {
                'iteration': iteration,
                'option': option_num,
                'n_opt': n_opt,
                'n_exp': n_exp,
                'selected': is_selected,
                'full_qehvi': batch.full_qehvi,
                'full_entropy': batch.full_entropy,
            }
            
            # Log each point in this option
            all_points = []
            all_sources = []
            
            # Add exploitation points
            for point in batch.points:
                all_points.append(point)
                all_sources.append('QEHVI')
            
            # Add exploration points
            for point in batch.explore_points:
                all_points.append(point)
                all_sources.append('MO-MESMO')
            
            # Create a row for each point in the option
            for point_idx, (point, source) in enumerate(zip(all_points, all_sources)):
                row = base_row.copy()
                row.update({
                    'point_idx': point_idx,
                    'source': source,
                    'fe': point[0],
                    'co': point[1],
                    'cr': point[2],
                    'ni': point[3],
                    'v': point[4],
                })
                rows.append(row)
        
        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            
            if not os.path.exists(self.options_path):
                df.to_csv(self.options_path, index=False)
            else:
                df.to_csv(self.options_path, mode='a', header=False, index=False)
            
            print(f"[Logger] Logged {len(acquisition_data.qehvi_batches)} options "
                  f"with {len(rows)} total points for iteration {iteration}")
    
    def log_evaluations(self,
                       iteration: int,
                       X_new: np.ndarray,
                       Y_new: np.ndarray,
                       selection_info: Optional[Dict[str, Any]] = None):
        """
        Log detailed evaluation results for newly evaluated points.
        
        Args:
            iteration: Current iteration
            X_new: Newly evaluated compositions (k, 5)
            Y_new: Newly evaluated objectives (k, 2)
            selection_info: Info about how points were selected
        """
        rows = []
        ratios = Y_new[:, 1] / (np.abs(Y_new[:, 0]) + 1e-12)
        
        for i, (x, y, ratio) in enumerate(zip(X_new, Y_new, ratios)):
            row = {
                'iteration': iteration,
                'point_idx': i,
                'fe': x[0],
                'co': x[1],
                'cr': x[2],
                'ni': x[3],
                'v': x[4],
                'cte': y[0] * 1e6,  # Convert to µK⁻¹
                'k': y[1],
                'k_abs_cte_ratio': ratio,
            }
            
            # Add source information
            if selection_info and 'sources' in selection_info:
                if i < len(selection_info['sources']):
                    row['source'] = selection_info['sources'][i]
                else:
                    row['source'] = 'Unknown'
            else:
                row['source'] = 'Unknown'
            
            rows.append(row)
        
        # Write to CSV
        df = pd.DataFrame(rows)
        
        if not os.path.exists(self.evaluations_path):
            df.to_csv(self.evaluations_path, index=False)
        else:
            df.to_csv(self.evaluations_path, mode='a', header=False, index=False)
        
        print(f"[Logger] Logged {len(rows)} evaluations for iteration {iteration}")
    
    def _save_metadata(self):
        """Save metadata to JSON."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def finalize(self):
        """Finalize logging at end of experiment."""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['status'] = 'completed'
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
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from logged data."""
        df = self.load_convergence_data()
        if df.empty:
            return {}
        
        last_iter = df.iloc[-1]
        return {
            'total_iterations': int(last_iter['iteration']),
            'total_evaluations': int(last_iter['total_evaluated']),
            'best_k_abs_cte_ratio': float(last_iter['best_ratio']),
            'best_cte': float(last_iter['best_cte']),
            'best_k': float(last_iter['best_k']),
            'final_mean_ratio': float(last_iter['mean_ratio']),
            'total_time': float(df['timing_sec'].sum()),
        }