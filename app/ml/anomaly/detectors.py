"""
Anomaly detectors for daily demand — two simple statistical baselines and Isolation Forest.

Every detector answers the same question for a product-day that has just
happened: "how unusual is today's demand?" It returns a SCORE (higher = more
unusual). Turning a score into a yes/no flag needs a threshold, which is
chosen separately, on validation data (see evaluate.py).

Reference = the product's own recent past
-----------------------------------------
"Unusual" is judged against the previous 28 OPEN days of the same product,
on a log1p scale:
  - previous days only (.shift(1)): an anomaly must not inflate its own baseline
  - open days only: planned zeros (Saturdays, holidays) are not "normal demand"
  - log scale: demand is heavy-tailed (Phase 1); a jump from 10 to 40 units
    should count as much as 100 to 400
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.ml.anomaly.calendar import open_day_mask

REFERENCE_DAYS = 28
MIN_REFERENCE_DAYS = 10
FLAT_WINDOW = 4              # open days (including today) checked for "the value never changes"

ISOLATION_FEATURES = [
    "log_units",             # today's demand (log scale)
    "z_28",                  # distance from the recent normal, in standard deviations
    "vs_last_week",          # change vs the same weekday last week
    "vs_mean_7",             # change vs the average of the last 7 open days
    "flat_range_4",          # max - min over the last 4 open days incl. today (exactly 0 = stuck value)
    "day_of_week",
]


def build_detection_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    One row per product per OPEN day, with every detector input.

    Uses today's value (that is what is being judged) and past values for the
    reference — never future values.
    """
    frame = daily.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    # Same weekday last week, taken on the full CALENDAR before closed days are removed
    frame["last_week_units"] = frame.groupby("product_id", sort=False)["units_sold"].shift(7)
    frame = frame[open_day_mask(frame["sale_date"]).to_numpy()].reset_index(drop=True)
    frame["log_units"] = np.log1p(frame["units_sold"].astype(float))

    by_product = frame.groupby("product_id", sort=False)["log_units"]
    past = by_product.shift(1)                                     # yesterday's open-day value and before
    grouped_past = past.groupby(frame["product_id"])

    def rolling(series_grouped, window, stat, min_periods):
        return series_grouped.rolling(window, min_periods=min_periods).agg(stat).reset_index(level=0, drop=True)

    mean_28 = rolling(grouped_past, REFERENCE_DAYS, "mean", MIN_REFERENCE_DAYS)
    std_28 = rolling(grouped_past, REFERENCE_DAYS, "std", MIN_REFERENCE_DAYS)
    frame["median_28"] = rolling(grouped_past, REFERENCE_DAYS, "median", MIN_REFERENCE_DAYS)
    frame["q1_28"] = grouped_past.rolling(REFERENCE_DAYS, min_periods=MIN_REFERENCE_DAYS).quantile(0.25).reset_index(level=0, drop=True)
    frame["q3_28"] = grouped_past.rolling(REFERENCE_DAYS, min_periods=MIN_REFERENCE_DAYS).quantile(0.75).reset_index(level=0, drop=True)

    # A tiny floor stops a product with almost no variation from producing huge z-scores
    frame["z_28"] = (frame["log_units"] - mean_28) / std_28.clip(lower=0.1)
    frame["vs_last_week"] = frame["log_units"] - np.log1p(frame["last_week_units"].astype(float))
    frame["vs_mean_7"] = frame["log_units"] - rolling(grouped_past, 7, "mean", 7)
    # Includes TODAY on purpose: a stuck feed shows up as today matching the last 3 open days exactly.
    # Range (max - min), not standard deviation: pandas computes rolling std with running sums,
    # which leaves float residue (~1e-8) for identical values. max - min is exactly 0.
    window = frame["units_sold"].groupby(frame["product_id"]).rolling(FLAT_WINDOW, min_periods=FLAT_WINDOW)
    frame["flat_range_4"] = np.log1p(
        (window.max() - window.min()).reset_index(level=0, drop=True).astype(float)
    )
    frame["day_of_week"] = frame["sale_date"].dt.dayofweek

    needed = ["z_28", "q1_28", "q3_28", "vs_last_week", "vs_mean_7", "flat_range_4"]
    return frame.dropna(subset=needed).reset_index(drop=True)


class ZScoreDetector:
    """
    Baseline 1 — rolling z-score: how many standard deviations today is from
    the recent mean. Two-sided |z|, so both spikes and drops score high.
    The classic rule of thumb is |z| > 3; here the threshold is chosen on validation.
    """
    name = "Rolling z-score"

    def fit(self, frame: pd.DataFrame) -> "ZScoreDetector":
        return self          # nothing to learn — the reference window moves with the data

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["z_28"].abs().to_numpy()


class IQRDetector:
    """
    Baseline 2 — interquartile-range fences (Tukey): how far today is outside
    [Q1, Q3] of the recent past, measured in IQRs. Median and quartiles ignore
    extreme values, so one past bulk order does not widen the "normal" range
    the way it inflates a standard deviation. The textbook flag is > 1.5 IQR.
    """
    name = "IQR fences"

    def fit(self, frame: pd.DataFrame) -> "IQRDetector":
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        iqr = (frame["q3_28"] - frame["q1_28"]).clip(lower=0.1)
        below = (frame["q1_28"] - frame["log_units"]) / iqr
        above = (frame["log_units"] - frame["q3_28"]) / iqr
        return np.maximum(below, above).clip(lower=0).to_numpy()


class IsolationForestDetector:
    """
    Isolation Forest — an unsupervised tree ensemble.

    Idea: build random trees that split on random features at random values.
    An unusual point is isolated in FEW splits (it sits alone in an empty
    region); a normal point needs many. Short average path length = anomaly.

    Why it can beat the baselines: it looks at several signals at once —
    including flat_range_4, which a single-value z-score cannot see, so it can
    catch a "stuck" feed whose values each look perfectly normal.

    Trained on past data with no labels (it never sees an injected anomaly).
    """
    name = "Isolation Forest"

    def __init__(self, random_state: int = 42) -> None:
        self.model = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)

    def fit(self, frame: pd.DataFrame) -> "IsolationForestDetector":
        self.model.fit(frame[ISOLATION_FEATURES])
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        # score_samples: higher = more NORMAL, so flip the sign
        return -self.model.score_samples(frame[ISOLATION_FEATURES])


class ZScoreWithFlatlineRule(ZScoreDetector):
    """
    Rolling z-score PLUS a plain rule: flag if the last 4 open days are identical.

    Why it exists: a stuck feed repeats a perfectly normal-looking number, so
    no "how far from normal is today?" score can see it. On the validation
    window, "4 identical open days in a row" occurred on 0 of 1,465 normal
    days but on about half of the injected flatline days — a near-perfect
    signal that the other detectors do not use. (The rule can only fire from
    the 4th identical day, so the first repeats are missed: detection delay
    is inherent to flatlines.)
    """
    name = "Rolling z-score + repeated-value rule"
    RULE_SCORE = 1e6         # larger than any |z|, so a rule hit is flagged at any threshold

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        is_flat = frame["flat_range_4"].to_numpy() == 0      # exact: integer units, max == min
        return np.where(is_flat, self.RULE_SCORE, super().score(frame))


def make_detectors() -> list:
    return [ZScoreDetector(), IQRDetector(), IsolationForestDetector(), ZScoreWithFlatlineRule()]


def detector_by_name(name: str):
    """A fresh detector instance by its report name (the deployed one is recorded in metadata.json)."""
    for detector in make_detectors():
        if detector.name == name:
            return detector
    raise ValueError(f"Unknown anomaly detector: {name!r}")
