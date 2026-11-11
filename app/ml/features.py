"""
Leakage-safe features for next-day demand forecasting.

The rule
--------
The row for day t predicts ``units_sold`` on day t. Every feature in that row
may only use information available at the END OF DAY t-1:

    sales history   -> computed from .shift(1)   (yesterday and before)
    calendar        -> known in advance (weekday, month, public holidays)
    price           -> the last price paid BEFORE day t

Using anything from day t itself (or later) is *leakage*: the model would look
great in evaluation and then fail in production, where day t has not happened yet.

Why .shift(1) before .rolling()?
--------------------------------
    units.rolling(7).mean()           window = days t-6 .. t   <- includes today's answer!
    units.shift(1).rolling(7).mean()  window = days t-7 .. t-1 <- past only

Everything is computed per product (groupby), so one product's history never
leaks into another product's features.
"""
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

TARGET = "units_sold"

LAGS = (1, 7, 14)              # yesterday, same weekday last week, same weekday 2 weeks ago
ROLLING_WINDOWS = (7, 28)      # one week, four weeks

LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLLING_FEATURES = [f"roll_{stat}_{w}" for w in ROLLING_WINDOWS for stat in ("mean", "std")]
CALENDAR_FEATURES = ["day_of_week", "month", "is_holiday", "is_christmas_break"]
PRICE_FEATURES = ["last_price", "price_vs_28d"]
CATEGORICAL_FEATURES = ["product_id"]

# The full, ordered feature list — logged with every training run (Phase 3)
FEATURES = CATEGORICAL_FEATURES + LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES + PRICE_FEATURES

# Rows need this many past days before every lag/rolling feature exists
MIN_HISTORY_DAYS = max(max(LAGS), max(ROLLING_WINDOWS))

# NOT used: inventory. The dataset has no historical stock levels, and a value
# filled in from today's sales would leak the answer into the features.


@lru_cache
def _uk_holidays(first_year: int, last_year: int) -> frozenset:
    # The shop is in the UK and closes on English bank holidays (verified in Phase A profiling)
    return frozenset(holidays.country_holidays("GB", subdiv="ENG", years=range(first_year, last_year + 1)))


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar facts — all known in advance, so safe to use for day t itself."""
    dates = frame["sale_date"]
    uk = _uk_holidays(int(dates.dt.year.min()), int(dates.dt.year.max()))
    return frame.assign(
        day_of_week=dates.dt.dayofweek,                          # 0 = Monday ... 5 = Saturday (shop closed)
        month=dates.dt.month,
        is_holiday=dates.dt.date.isin(uk).astype(int),
        # The shop shuts from Christmas Eve until after New Year (seen in the data both years);
        # not every one of those days is a bank holiday, so it gets its own flag
        is_christmas_break=(((dates.dt.month == 12) & (dates.dt.day >= 24))
                            | ((dates.dt.month == 1) & (dates.dt.day <= 3))).astype(int),
    )


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Turn daily sales (one row per product per calendar day) into a feature table.

    Input columns:  product_id, sale_date, units_sold, avg_price
    Output:         the input columns + every name in FEATURES.
                    Rows without enough history (the first MIN_HISTORY_DAYS of
                    each product) are dropped.
    """
    frame = daily.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    by_product = frame.groupby("product_id", sort=False)

    # ---- Sales history: yesterday and earlier only ----------------------
    for k in LAGS:
        frame[f"lag_{k}"] = by_product[TARGET].shift(k)

    past_units = by_product[TARGET].shift(1)                     # value on day t = sales on day t-1
    for w in ROLLING_WINDOWS:
        window = past_units.groupby(frame["product_id"]).rolling(w, min_periods=w)
        frame[f"roll_mean_{w}"] = window.mean().reset_index(level=0, drop=True)
        frame[f"roll_std_{w}"] = window.std().reset_index(level=0, drop=True)

    # ---- Price: the last price actually paid BEFORE day t ----------------
    # avg_price on day t is an OUTCOME of day t (who bought, how much discount) — not usable.
    # Zero-sale days have no price, so carry the last known one forward.
    last_price = by_product["avg_price"].shift(1).groupby(frame["product_id"]).ffill()
    median_28 = last_price.groupby(frame["product_id"]).rolling(28, min_periods=7).median().reset_index(level=0, drop=True)
    frame["last_price"] = last_price
    frame["price_vs_28d"] = last_price / median_28               # >1 = currently pricier than usual

    frame = add_calendar_features(frame)
    frame["product_id"] = frame["product_id"].astype("category")

    has_history = frame[LAG_FEATURES + ROLLING_FEATURES + PRICE_FEATURES].notna().all(axis=1)
    return frame[has_history].reset_index(drop=True)


def load_daily_sales(path) -> pd.DataFrame:
    """Read data/processed/daily_sales.csv with the right types."""
    return pd.read_csv(path, parse_dates=["sale_date"], dtype={"product_id": str}).assign(
        units_sold=lambda f: f["units_sold"].astype(np.int64)
    )
