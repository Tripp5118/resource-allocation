# core/Truth_interface.py
import numpy as np
import joblib
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    cte: np.ndarray
    k: np.ndarray

class TruthModelEvaluator:
    def __init__(self):
        '''Load pre-trained Random Forest models and scalers.'''
        print("[TruthEvaluator] Loading models from models/...")
        
        self.model = joblib.load('models/RFR_best_model.pkl')
        self.x_scaler = joblib.load('models/x_scaler.pkl')
        self.y_scaler = joblib.load('models/y_scaler.pkl')
        
        print("[TruthEvaluator] Models loaded successfully")
        print(f"[TruthEvaluator] Model: RandomForest with {self.model.n_estimators} estimators")
        print(f"[TruthEvaluator] Outputs: {self.model.n_outputs_} (CTE and K)")
    
    def evaluate(self, X: np.ndarray, **kwargs) -> EvaluationResult:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Apply input scaling
        X_scaled = self.x_scaler.transform(X)
        
        # Predict (returns scaled predictions)
        predictions_scaled = self.model.predict(X_scaled)
        
        # Inverse transform to original scale
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        
        k = predictions[:, 0] # K in W/mK
        cte_micro = -1 * predictions[:, 1] 

        cte = np.abs(cte_micro)
        print(f"cte: {cte}, k: {k}")
        return EvaluationResult(cte=cte, k=k)
    
    def cleanup(self):
        print("[TruthEvaluator] Cleanup called")