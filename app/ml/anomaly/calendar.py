"""
Which days is the shop open?

Phase A profiling showed the shop never trades on Saturdays and closes on
English bank holidays and from Christmas Eve to 3 January. Zero sales on
those days are PLANNED, not anomalies — so anomaly detection only judges
open days. The calendar is known in advance; it uses no sales data.
"""
import pandas as pd

from app.ml.features import add_calendar_features

SATURDAY = 5


def open_day_mask(dates: pd.Series) -> pd.Series:
    """True for days the shop is normally open."""
    calendar = add_calendar_features(pd.DataFrame({"sale_date": pd.to_datetime(dates)}))
    is_closed = (calendar["day_of_week"] == SATURDAY) | (calendar["is_holiday"] == 1) | (calendar["is_christmas_break"] == 1)
    return pd.Series(~is_closed.to_numpy(), index=dates.index)
