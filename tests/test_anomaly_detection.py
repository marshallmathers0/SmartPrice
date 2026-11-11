"""Phase 2 — anomaly injection, detectors, threshold selection and the evaluation protocol."""
import numpy as np
import pandas as pd
import pytest

from app.config.settings import resolve_path, settings
from app.ml.anomaly.calendar import open_day_mask
from app.ml.anomaly.detectors import (
    IQRDetector,
    IsolationForestDetector,
    ZScoreDetector,
    ZScoreWithFlatlineRule,
    build_detection_features,
)
from app.ml.anomaly.evaluate import TEST_SEED, VALIDATION_SEED, best_f1_threshold, run_evaluation, windows_for
from app.ml.anomaly.inject import FLATLINE_LENGTH, inject_anomalies
from app.ml.features import load_daily_sales

DAILY_SALES = resolve_path(settings.processed_data_dir) / "daily_sales.csv"
needs_data = pytest.mark.skipif(not DAILY_SALES.exists(), reason="processed dataset not built")

START = pd.Timestamp("2011-01-03")


def steady_daily(days: int = 120, level: int = 50, seed: int = 0) -> pd.DataFrame:
    """Two products with noisy but steady demand; closed days set to 0 like the real shop."""
    rng = np.random.default_rng(seed)
    frames = []
    for product_id in ("A", "B"):
        dates = pd.date_range(START, periods=days, freq="D")
        units = rng.poisson(level, days)
        units = np.where(open_day_mask(pd.Series(dates)).to_numpy(), units, 0)
        frames.append(pd.DataFrame({"product_id": product_id, "sale_date": dates, "units_sold": units,
                                    "avg_price": np.where(units > 0, 2.0, np.nan)}))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# Calendar
# ------------------------------------------------------------------

def test_saturdays_and_bank_holidays_are_closed():
    dates = pd.Series(pd.to_datetime(["2011-01-08", "2011-04-22", "2010-12-30", "2011-01-10"]))
    # Saturday, Good Friday, Christmas break, ordinary Monday
    assert open_day_mask(dates).tolist() == [False, False, False, True]


# ------------------------------------------------------------------
# Injection
# ------------------------------------------------------------------

@pytest.fixture
def injected():
    daily = steady_daily()
    window = (START + pd.Timedelta(days=60), START + pd.Timedelta(days=110))
    modified, labels = inject_anomalies(daily, *window, seed=7, n_spikes=5, n_drops=5, n_flatlines=2)
    return daily, modified, labels, window


def test_injection_is_labelled_and_inside_the_window(injected):
    _, _, labels, (start, end) = injected

    assert labels["anomaly_type"].value_counts().to_dict() == {
        "spike": 5, "drop": 5, "flatline": 2 * (FLATLINE_LENGTH - 1)
    }
    assert labels["sale_date"].between(start, end).all()
    assert not labels.duplicated(["product_id", "sale_date"]).any()


def test_injected_values_are_what_the_labels_say(injected):
    original, modified, labels, _ = injected
    merged = labels.merge(original, on=["product_id", "sale_date"]).merge(
        modified, on=["product_id", "sale_date"], suffixes=("_before", "_after"))

    spikes = merged[merged["anomaly_type"] == "spike"]
    drops = merged[merged["anomaly_type"] == "drop"]
    assert (spikes["units_sold_after"] >= 3 * spikes["units_sold_before"]).all()
    assert (drops["units_sold_after"] == 0).all() and (drops["units_sold_before"] > 0).all()


def test_flatline_repeats_the_first_days_value(injected):
    _, modified, labels, _ = injected
    flat = labels[labels["anomaly_type"] == "flatline"].merge(modified, on=["product_id", "sale_date"])

    for _, run in flat.groupby(flat["sale_date"].diff().dt.days.ne(1).cumsum()):
        assert run["units_sold"].nunique() == 1


def test_injection_is_reproducible():
    daily = steady_daily()
    window = (START + pd.Timedelta(days=60), START + pd.Timedelta(days=110))

    _, first = inject_anomalies(daily, *window, seed=3)
    _, second = inject_anomalies(daily, *window, seed=3)
    _, other = inject_anomalies(daily, *window, seed=4)

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_untouched_rows_are_unchanged(injected):
    original, modified, labels, _ = injected
    keys = pd.MultiIndex.from_frame(labels[["product_id", "sale_date"]])
    untouched = ~pd.MultiIndex.from_frame(modified[["product_id", "sale_date"]]).isin(keys)

    assert (original.sort_values(["product_id", "sale_date"])["units_sold"].to_numpy()[untouched]
            == modified["units_sold"].to_numpy()[untouched]).all()


# ------------------------------------------------------------------
# Detectors
# ------------------------------------------------------------------

def test_only_open_days_are_judged():
    features = build_detection_features(steady_daily())

    assert open_day_mask(features["sale_date"]).all()


def test_reference_window_never_uses_the_future():
    daily = steady_daily()
    day = START + pd.Timedelta(days=80)
    changed = daily.copy()
    changed.loc[changed["sale_date"] > day, "units_sold"] = 50_000

    def row(frame):
        f = build_detection_features(frame)
        return f[(f["product_id"] == "A") & (f["sale_date"] == day)].iloc[0]

    assert row(daily)["z_28"] == pytest.approx(row(changed)["z_28"])


def test_zscore_and_iqr_score_a_spike_and_a_drop_above_normal_days():
    daily = steady_daily()
    spike_day, drop_day = START + pd.Timedelta(days=80), START + pd.Timedelta(days=85)
    daily.loc[(daily["product_id"] == "A") & (daily["sale_date"] == spike_day), "units_sold"] = 400
    daily.loc[(daily["product_id"] == "A") & (daily["sale_date"] == drop_day), "units_sold"] = 0
    features = build_detection_features(daily)
    is_planted = features["sale_date"].isin([spike_day, drop_day]) & (features["product_id"] == "A")

    for detector in (ZScoreDetector(), IQRDetector()):
        scores = detector.fit(features).score(features)
        assert scores[is_planted.to_numpy()].min() > np.quantile(scores[~is_planted.to_numpy()], 0.99), detector.name


def test_isolation_forest_scores_an_extreme_day_as_more_anomalous():
    daily = steady_daily()
    spike_day = START + pd.Timedelta(days=80)
    daily.loc[(daily["product_id"] == "A") & (daily["sale_date"] == spike_day), "units_sold"] = 2_000
    features = build_detection_features(daily)
    scores = IsolationForestDetector().fit(features).score(features)
    is_spike = ((features["product_id"] == "A") & (features["sale_date"] == spike_day)).to_numpy()

    assert scores[is_spike][0] > np.quantile(scores[~is_spike], 0.99)


def test_repeated_value_rule_catches_a_stuck_feed():
    daily = steady_daily()
    stuck = (daily["product_id"] == "A") & daily["sale_date"].between(START + pd.Timedelta(days=80), START + pd.Timedelta(days=88))
    daily.loc[stuck, "units_sold"] = 47
    features = build_detection_features(daily)

    plain = ZScoreDetector().score(features)
    with_rule = ZScoreWithFlatlineRule().score(features)
    flat = (features["flat_range_4"] == 0).to_numpy()

    assert flat.any()
    assert (with_rule[flat] == ZScoreWithFlatlineRule.RULE_SCORE).all()
    assert (plain[flat] < 2).all()          # each stuck value looks normal to a z-score


# ------------------------------------------------------------------
# Threshold selection
# ------------------------------------------------------------------

def test_threshold_maximises_f1():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    # threshold 0.8 flags {0.9, 0.8, 0.95}: precision 2/3, recall 1 -> F1 0.8 (the best possible here)
    assert best_f1_threshold(labels, scores) == pytest.approx(0.8)


def test_perfectly_separable_scores_give_a_separating_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.9])

    threshold = best_f1_threshold(labels, scores)
    assert 0.2 < threshold <= 0.7


# ------------------------------------------------------------------
# Protocol, on the real data
# ------------------------------------------------------------------

@needs_data
def test_validation_comes_before_test_and_uses_different_anomalies():
    validation, test = windows_for(load_daily_sales(DAILY_SALES))

    assert validation.end < test.start
    assert VALIDATION_SEED != TEST_SEED


@needs_data
def test_full_evaluation_reports_every_detector():
    evaluation = run_evaluation(load_daily_sales(DAILY_SALES))

    assert list(evaluation.results.index) == [
        "Rolling z-score", "IQR fences", "Isolation Forest", "Rolling z-score + repeated-value rule"
    ]
    assert evaluation.results[["precision", "recall", "f1"]].apply(lambda c: c.between(0, 1)).all().all()
    assert evaluation.label_counts["test"]["spike"] == 30
