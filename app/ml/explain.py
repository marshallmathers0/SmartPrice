"""
Why did the model forecast this number? Exact, per-feature contributions.

Both forecasters predict log(1 + units). On that log scale each prediction
splits EXACTLY into a base value plus one contribution per feature:

    log1p(forecast) = base + sum(contribution_f for every feature f)

Ridge (linear)
--------------
For a linear model the split is just its formula:
    contribution_f = coefficient_f x (transformed value of f)
    base           = intercept
Numeric features are standardised, so "value" means "distance from the
training average": a positive contribution = above-average and pushing up.
A category (product, weekday, month) contributes its own coefficient.
For a linear model with independent features this IS the SHAP value — no
approximation, no sampling.

LightGBM (trees)
----------------
LightGBM computes TreeSHAP itself: predict(..., pred_contrib=True) returns
one column per feature plus the base value (verified identical to the
`shap` package's TreeExplainer, max difference 0.0).

Reading a contribution
----------------------
Because the scale is log(1 + units), exp(contribution) is a MULTIPLIER on
(1 + forecast): +0.30 -> x1.35 ("raises the forecast ~35%"), -0.22 -> x0.80.

This module is used by the API, so it imports neither `shap` nor `lightgbm`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.features import FEATURES

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


class ExplanationNotSupportedError(Exception):
    """The model type has no exact explanation method here."""


@dataclass(frozen=True)
class Contributions:
    base: float                  # log1p scale: the prediction before any feature is considered
    values: pd.DataFrame         # one row per input row, one column per original feature (log1p scale)


def contributions(model, X: pd.DataFrame) -> Contributions:
    """Exact per-feature contributions for every row of X (columns = FEATURES)."""
    X = X[FEATURES]
    regressor = getattr(model, "regressor_", None)           # fitted TransformedTargetRegressor
    if regressor is None:
        raise ExplanationNotSupportedError(f"{type(model).__name__} is not a fitted log-target model")

    if isinstance(regressor, Pipeline):
        return _linear_contributions(regressor, X)
    if hasattr(regressor, "booster_"):                       # LightGBM sklearn API
        raw = regressor.predict(X, pred_contrib=True)
        return Contributions(base=float(raw[0, -1]), values=pd.DataFrame(raw[:, :-1], columns=FEATURES, index=X.index))
    raise ExplanationNotSupportedError(f"No exact explanation for {type(regressor).__name__}")


def _linear_contributions(pipeline: Pipeline, X: pd.DataFrame) -> Contributions:
    preprocess, linear = pipeline.steps[0][1], pipeline.steps[-1][1]
    transformed = preprocess.transform(X)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    per_column = transformed * linear.coef_                   # coefficient x value, for every output column

    # Map each output column back to the original feature it came from:
    # a one-hot encoder produces one column per category, a scaler one per feature.
    owners: list[str] = []
    for _, transformer, columns in preprocess.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "categories_"):
            for column, categories in zip(columns, transformer.categories_):
                owners += [column] * len(categories)
        else:
            owners += list(columns)

    grouped = pd.DataFrame(per_column, columns=owners, index=X.index).T.groupby(level=0).sum().T
    return Contributions(base=float(linear.intercept_), values=grouped.reindex(columns=FEATURES, fill_value=0.0))


# ------------------------------------------------------------------
# Plain language
# ------------------------------------------------------------------

def describe(feature: str, row: pd.Series) -> str:
    """A human sentence for one feature's value in this row."""
    value = row[feature]
    match feature:
        case "product_id":
            return f"This product's usual demand level ({value})"
        case "lag_1":
            return f"Sales on the previous day: {value:.0f} units"
        case "lag_7":
            return f"Sales on the same weekday last week: {value:.0f} units"
        case "lag_14":
            return f"Sales on the same weekday two weeks ago: {value:.0f} units"
        case "roll_mean_7":
            return f"Average daily sales over the last 7 days: {value:.1f} units"
        case "roll_std_7":
            return f"Day-to-day swings over the last 7 days: +/-{value:.1f} units"
        case "roll_mean_28":
            return f"Average daily sales over the last 28 days: {value:.1f} units"
        case "roll_std_28":
            return f"Day-to-day swings over the last 28 days: +/-{value:.1f} units"
        case "day_of_week":
            return f"Day of the week: {DAY_NAMES[int(value)]}"
        case "month":
            return f"Month: {MONTH_NAMES[int(value) - 1]}"
        case "is_holiday":
            return "It is a bank holiday" if value else "Not a bank holiday"
        case "is_christmas_break":
            return "Inside the Christmas closure" if value else "Outside the Christmas closure"
        case "last_price":
            return f"Last price paid: {value:.2f}"
        case "price_vs_28d":
            return f"Price vs its 28-day median: x{value:.2f}"
    return f"{feature} = {value}"


def effect_text(contribution: float) -> str:
    multiplier = float(np.exp(contribution))
    direction = "raises" if contribution > 0 else "lowers"
    return f"{direction} the forecast (x{multiplier:.2f})"
