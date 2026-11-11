"""
Forecast error metrics — written out by hand so every line can be explained.

All three compare ``actual`` and ``predicted`` arrays of the same length.
Lower is better for all of them.

    MAE   mean absolute error          average size of a miss, in units
    RMSE  root mean squared error      like MAE, but punishes big misses more
    WAPE  weighted absolute % error    total miss as a share of total demand

Why WAPE and not MAPE?
----------------------
MAPE = mean(|actual - predicted| / actual). It divides EVERY row by its own
actual value. In this dataset 22% of product-days have zero sales, so MAPE is
undefined there (division by zero), and days with tiny demand (1 unit) blow it
up. WAPE divides ONCE, by total demand:

    WAPE = sum(|actual - predicted|) / sum(|actual|)

It is defined whenever total demand is non-zero, and it reads naturally:
"WAPE 0.45 = we were off by 45% of everything that was sold".
"""
import numpy as np


def _as_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"actual and predicted differ in shape: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        raise ValueError("cannot score an empty forecast")
    return actual, predicted


def mae(actual, predicted) -> float:
    actual, predicted = _as_arrays(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted) -> float:
    actual, predicted = _as_arrays(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual, predicted) -> float:
    actual, predicted = _as_arrays(actual, predicted)
    total_demand = np.sum(np.abs(actual))
    if total_demand == 0:
        return float("nan")      # nothing was sold at all: a percentage of zero is meaningless
    return float(np.sum(np.abs(actual - predicted)) / total_demand)


def score(actual, predicted) -> dict[str, float]:
    """All three metrics at once."""
    return {"mae": mae(actual, predicted), "rmse": rmse(actual, predicted), "wape": wape(actual, predicted)}
