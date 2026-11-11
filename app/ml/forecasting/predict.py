"""
Turning a fitted forecaster's output into demand — shared by training and the API.

Kept separate from models.py on purpose: models.py imports LightGBM (a
training-only dependency), and the API must be able to predict without it.
"""
import numpy as np

from app.ml.features import FEATURES


def predict_demand(model, X) -> np.ndarray:
    """Predictions as demand: never negative (you cannot sell -3 units)."""
    return np.clip(model.predict(X[FEATURES]), 0, None)
