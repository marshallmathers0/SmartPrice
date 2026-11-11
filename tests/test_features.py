"""
Phase 1 — feature engineering, and above all: NO LEAKAGE.

The strongest leakage test does not check formulas. It changes the FUTURE
and asserts that the features of the past do not move. If any feature
secretly looked ahead, it would change.
"""
import numpy as np
import pandas as pd
import pytest

from app.ml.features import FEATURES, MIN_HISTORY_DAYS, build_features

START = pd.Timestamp("2011-01-03")     # a Monday
DAYS = 70


def make_daily(product_ids=("A", "B"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for product_id in product_ids:
        units = rng.integers(0, 50, DAYS)
        frames.append(pd.DataFrame({
            "product_id": product_id,
            "sale_date": pd.date_range(START, periods=DAYS, freq="D"),
            "units_sold": units,
            "avg_price": np.where(units > 0, rng.uniform(1, 3, DAYS).round(2), np.nan),
        }))
    return pd.concat(frames, ignore_index=True)


def features_for(frame: pd.DataFrame, product_id: str) -> pd.DataFrame:
    rows = frame[frame["product_id"] == product_id].set_index("sale_date")
    return rows[FEATURES].astype({"product_id": str})


# ------------------------------------------------------------------
# Leakage
# ------------------------------------------------------------------

@pytest.mark.parametrize("cutoff_day", [40, 55, 65])     # each leaves days AFTER the cutoff to compare
def test_changing_the_future_never_changes_past_features(cutoff_day):
    daily = make_daily()
    cutoff = START + pd.Timedelta(days=cutoff_day)

    future_changed = daily.copy()
    is_future = future_changed["sale_date"] >= cutoff
    future_changed.loc[is_future, "units_sold"] = 99_999        # wildly different future demand
    future_changed.loc[is_future, "avg_price"] = 500.0           # and future prices

    before = features_for(build_features(daily), "A")
    after = features_for(build_features(future_changed), "A")

    # Rows up to and including the cutoff day only see data from before it
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
    # ...and the change really did reach the features after it (the test can fail)
    assert not before.loc[cutoff + pd.Timedelta(days=1):].equals(after.loc[cutoff + pd.Timedelta(days=1):])


def test_todays_own_sales_and_price_are_never_features():
    daily = make_daily()
    day = START + pd.Timedelta(days=50)

    changed = daily.copy()
    today = (changed["product_id"] == "A") & (changed["sale_date"] == day)
    changed.loc[today, ["units_sold", "avg_price"]] = [12_345, 99.0]

    pd.testing.assert_series_equal(
        features_for(build_features(daily), "A").loc[day],
        features_for(build_features(changed), "A").loc[day],
    )


def test_products_never_leak_into_each_other():
    daily = make_daily()
    changed = daily.copy()
    changed.loc[changed["product_id"] == "B", "units_sold"] = 77_777

    pd.testing.assert_frame_equal(
        features_for(build_features(daily), "A"),
        features_for(build_features(changed), "A"),
    )


# ------------------------------------------------------------------
# Formulas, checked by hand
# ------------------------------------------------------------------

@pytest.fixture
def product_a():
    daily = make_daily()
    raw = daily[daily["product_id"] == "A"].set_index("sale_date")
    feats = build_features(daily)
    return raw, feats[feats["product_id"] == "A"].set_index("sale_date")


def test_lags_are_past_values(product_a):
    raw, feats = product_a
    day = START + pd.Timedelta(days=40)

    assert feats.loc[day, "lag_1"] == raw.loc[day - pd.Timedelta(days=1), "units_sold"]
    assert feats.loc[day, "lag_7"] == raw.loc[day - pd.Timedelta(days=7), "units_sold"]
    assert feats.loc[day, "lag_14"] == raw.loc[day - pd.Timedelta(days=14), "units_sold"]


def test_rolling_window_ends_yesterday(product_a):
    raw, feats = product_a
    day = START + pd.Timedelta(days=40)
    last_7 = raw.loc[day - pd.Timedelta(days=7): day - pd.Timedelta(days=1), "units_sold"]

    assert len(last_7) == 7
    assert feats.loc[day, "roll_mean_7"] == pytest.approx(last_7.mean())
    assert feats.loc[day, "roll_std_7"] == pytest.approx(last_7.std())


def test_last_price_is_the_most_recent_past_price(product_a):
    raw, feats = product_a
    day = START + pd.Timedelta(days=40)
    past_prices = raw.loc[:day - pd.Timedelta(days=1), "avg_price"].dropna()

    assert feats.loc[day, "last_price"] == past_prices.iloc[-1]


def test_calendar_features():
    daily = make_daily(product_ids=("A",))
    daily = pd.concat([daily, daily.assign(sale_date=pd.date_range("2010-11-20", periods=DAYS))])  # spans Christmas
    feats = build_features(daily.assign(product_id="A").drop_duplicates(["sale_date"])).set_index("sale_date")

    assert feats.loc[pd.Timestamp("2010-12-25"), "is_holiday"] == 1
    assert feats.loc[pd.Timestamp("2010-12-25"), "day_of_week"] == 5          # Saturday
    assert feats.loc[pd.Timestamp("2010-12-30"), "is_christmas_break"] == 1   # not a bank holiday, shop still shut
    assert feats.loc[pd.Timestamp("2011-01-04"), "is_christmas_break"] == 0


def test_rows_without_enough_history_are_dropped():
    feats = build_features(make_daily())

    assert len(feats) == 2 * (DAYS - MIN_HISTORY_DAYS)
    assert feats[FEATURES].notna().all().all()
