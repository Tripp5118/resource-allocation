# resource_allocation/truth_interface.py
import numpy as np
import joblib
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Dict

@dataclass
class EvaluationResult:
    """Container for objective evaluations."""
    y: np.ndarray  # (n, m) array of m objectives

class TruthModelEvaluator:
    """
    Generic RandomForest-based truth model evaluator.

    Assumes:
      - An sklearn model that outputs m targets.
      - X and y are scaled by scalers (optional).
      - A postprocess function maps raw RF outputs -> final objective array.
      - Output normalization for numerical stability in BO.
    """

    def __init__(self,
                 model_path: str,
                 x_scaler_path: Optional[str] = None,
                 y_scaler_path: Optional[str] = None,
                 postprocess_outputs: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 normalize_outputs: bool = True,
                 normalization_params: Optional[Dict[str, np.ndarray]] = None):
        """
        Args:
            model_path: Path to trained sklearn model.
            x_scaler_path: Path to X scaler (StandardScaler or similar).
            y_scaler_path: Path to y scaler.
            postprocess_outputs: Function f(pred_raw) -> y_used_in_BO.
                                 If None, identity is used.
            normalize_outputs: Whether to normalize outputs for BO stability.
            normalization_params: Dict with 'mean' and 'std' arrays. If None and
                                  normalize_outputs=True, will be computed from first batch.
        """
        print(f"[TruthEvaluator] Loading model from {model_path}")
        self.model = joblib.load(model_path)

        self.x_scaler = joblib.load(x_scaler_path) if x_scaler_path else None
        self.y_scaler = joblib.load(y_scaler_path) if y_scaler_path else None

        if postprocess_outputs is None:
            postprocess_outputs = lambda arr: arr
        self.postprocess_outputs = postprocess_outputs

        self.normalize_outputs = normalize_outputs
        
        if normalize_outputs:
            if normalization_params is not None:
                self.y_mean = normalization_params['mean']
                self.y_std = normalization_params['std']
                print(f"[TruthEvaluator] Using provided normalization:")
                print(f"  Mean: {self.y_mean}")
                print(f"  Std:  {self.y_std}")
            else:
                self.y_mean = None
                self.y_std = None
                print("[TruthEvaluator] Will compute normalization from first batch")
        else:
            self.y_mean = None
            self.y_std = None

        print("[TruthEvaluator] Model loaded successfully")
        n_outputs = getattr(self.model, "n_outputs_", None)
        if n_outputs is not None:
            print(f"[TruthEvaluator] RF outputs: {n_outputs}")
        else:
            print("[TruthEvaluator] RF outputs: unknown (sklearn multioutput?)")

    def evaluate(self, X: np.ndarray, **kwargs) -> EvaluationResult:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.x_scaler is not None:
            X_scaled = self.x_scaler.transform(X)
        else:
            X_scaled = X

        preds_scaled = self.model.predict(X_scaled)

        if self.y_scaler is not None:
            preds_raw = self.y_scaler.inverse_transform(preds_scaled)
        else:
            preds_raw = preds_scaled

        y = self.postprocess_outputs(preds_raw)
        
        # Apply output normalization
        if self.normalize_outputs:
            if self.y_mean is None or self.y_std is None:
                # First evaluation - compute normalization from this batch
                print("[TruthEvaluator] Computing normalization from first batch")
                self.y_mean = np.mean(y, axis=0)
                self.y_std = np.std(y, axis=0) + 1e-8  # Add small constant to avoid division by zero
                print(f"  Mean: {self.y_mean}")
                print(f"  Std:  {self.y_std}")
            
            y_normalized = (y - self.y_mean) / self.y_std
            return EvaluationResult(y=y_normalized)
        
        return EvaluationResult(y=y)
    
    def denormalize(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized outputs back to original scale.
        
        Args:
            y_normalized: Normalized objective values (n, m)
            
        Returns:
            y_raw: Denormalized objective values (n, m)
        """
        if not self.normalize_outputs or self.y_mean is None:
            return y_normalized
        
        return y_normalized * self.y_std + self.y_mean
    
    def cleanup(self):
        print("[TruthEvaluator] Cleanup called")