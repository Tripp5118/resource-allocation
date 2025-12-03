# core/priors.py
import os
import joblib
import numpy as np

class EnsemblePrior:
    def __init__(self, model_path="models/ensemble.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ensemble model not found at {model_path}")
        self.model = joblib.load(model_path)
        self.x_scaler = joblib.load("models/x_scaler.pkl")
        self.y_scaler = joblib.load("models/y_scaler.pkl")

    def predict(self, X_frac):
        '''Return prior mean predictions for batch X (fractions). Shape: (n, 2)'''
        # X_frac is shape (n, 5) and rows sum to 1.0
        X_scaled = self.x_scaler.transform(X_frac)
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        cte = predictions[:, 0]
        k = predictions[:, 1]
        return np.column_stack([cte, k])