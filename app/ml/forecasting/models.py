"""
The candidate forecasting models — from "no model at all" to gradient boosting.

Every candidate has the same two methods, fit(X, y) and predict(X), so the
evaluation loop treats them identically (the Strategy idea again).

Why start with baselines?
-------------------------
A model is only worth its complexity if it beats a trivial rule. "Same as
yesterday" and "same as last week" cost nothing to run; if LightGBM cannot
beat them, LightGBM is not worth deploying.

Why a log1p target for Ridge and LightGBM?
------------------------------------------
Daily demand is non-negative and heavy-tailed (median ~30 units, single bulk
orders above 2,000). Squared-error training on raw units lets those rare
spikes dominate the fit. Training on log(1 + units) shrinks the tail; the
prediction is mapped back with exp(x) - 1. log1p (not log) because demand can
be 0, and log(0) is undefined.
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import CALENDAR_FEATURES, LAG_FEATURES, PRICE_FEATURES, ROLLING_FEATURES
from app.ml.forecasting.predict import predict_demand  # re-exported: one definition, shared with the API

RANDOM_STATE = 42


class ColumnBaseline:
    """A "model" that predicts one feature column unchanged (e.g. lag_1 = yesterday's sales)."""

    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X, y):
        return self          # nothing to learn

    def predict(self, X) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


def make_ridge() -> TransformedTargetRegressor:
    """
    Linear model. Needs numbers on a common scale, so:
      - product, weekday and month are one-hot encoded (they are categories, not quantities:
        Sunday=6 is not "six times" Monday)
      - numeric features are standardised (mean 0, std 1) so the L2 penalty treats them equally
    """
    preprocess = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), ["product_id", "day_of_week", "month"]),
        ("numbers", StandardScaler(),
         LAG_FEATURES + ROLLING_FEATURES + PRICE_FEATURES + ["is_holiday", "is_christmas_break"]),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=1.0))]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def make_lightgbm() -> TransformedTargetRegressor:
    """
    Gradient-boosted trees (LightGBM).

    Why LightGBM (over XGBoost)?
      - handles the categorical product_id natively — no one-hot encoding
      - fast on CPU for a table this size (~14k rows)
      - can return per-feature SHAP contributions itself (pred_contrib=True),
        so the API can explain a forecast without the heavy `shap` package (Phase 4)
    Trees need no feature scaling, and they capture interactions
    (e.g. "Sunday AND December") that a linear model cannot.

    The settings are common, conservative defaults — NOT tuned on the test set.
    """
    return TransformedTargetRegressor(
        regressor=LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def make_candidates() -> dict:
    """Fresh, unfitted instances of every candidate, in increasing complexity."""
    return {
        "Naive (yesterday)": ColumnBaseline("lag_1"),
        "Seasonal naive (same weekday last week)": ColumnBaseline("lag_7"),
        "Moving average (last 7 days)": ColumnBaseline("roll_mean_7"),
        "Ridge regression": make_ridge(),
        "LightGBM": make_lightgbm(),
    }




# Re-exported so callers do not need to know the feature module
__all__ = ["CALENDAR_FEATURES", "ColumnBaseline", "make_candidates", "make_lightgbm", "make_ridge", "predict_demand"]
