# core/truth_interface.py
import numpy as np
import joblib
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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
    """

    def __init__(self,
                 model_path: str,
                 x_scaler_path: Optional[str] = None,
                 y_scaler_path: Optional[str] = None,
                 postprocess_outputs: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Args:
            model_path: Path to trained sklearn model.
            x_scaler_path: Path to X scaler (StandardScaler or similar).
            y_scaler_path: Path to y scaler.
            postprocess_outputs: Function f(pred_raw) -> y_used_in_BO.
                                 If None, identity is used.
        """
        print(f"[TruthEvaluator] Loading model from {model_path}")
        self.model = joblib.load(model_path)

        self.x_scaler = joblib.load(x_scaler_path) if x_scaler_path else None
        self.y_scaler = joblib.load(y_scaler_path) if y_scaler_path else None

        if postprocess_outputs is None:
            postprocess_outputs = lambda arr: arr
        self.postprocess_outputs = postprocess_outputs

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
        return EvaluationResult(y=y)

    def cleanup(self):
        print("[TruthEvaluator] Cleanup called")