"""
Create a LABELLED anomaly test set by injecting known anomalies into real data.

The dataset has no anomaly labels — nobody marked which days were "wrong".
Without labels, precision and recall cannot be measured. The standard fix:
take real held-out data, plant anomalies whose location and type we KNOW,
and check how many each detector finds.

Three kinds, each a realistic failure in a sales feed:

    spike     demand 4-8x the product's normal level   (pricing error, bot orders, duplicated import)
    drop      demand 0 on a normally busy open day       (checkout outage, missing data load)
    flatline  the same number repeated for 5 days        (a stuck feed re-sending yesterday's value)

Injection is seeded, so the labelled set is identical on every run.

Honest limitation: the real data already contains natural extreme days
(one-off bulk orders) that carry no label. If a detector flags one, it counts
as a false positive here — so reported precision is a LOWER bound.
"""
import numpy as np
import pandas as pd

from app.ml.anomaly.calendar import open_day_mask

SPIKE_FACTOR = (4.0, 8.0)        # multiplier range applied to the product's normal level
DROP_MIN_NORMAL_LEVEL = 10       # only "drop" days where ~10+ units are normal, so a 0 is truly unusual
FLATLINE_LENGTH = 5              # consecutive calendar days of the same value (4 labelled repeats)
REFERENCE_DAYS = 28              # "normal level" = median of the previous 28 open days


def _normal_level(daily: pd.DataFrame) -> pd.Series:
    """Median demand over each product's previous 28 OPEN days (closed days are zero by design)."""
    open_units = daily["units_sold"].where(open_day_mask(daily["sale_date"]))
    return (
        open_units.groupby(daily["product_id"])
        .transform(lambda s: s.shift(1).rolling(REFERENCE_DAYS, min_periods=10).median())
    )


def inject_anomalies(
    daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int,
    n_spikes: int = 30,
    n_drops: int = 30,
    n_flatlines: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Plant anomalies on open days between `start` and `end` (inclusive).

    Returns (modified daily sales, labels) — labels has one row per anomalous
    product-day: product_id, sale_date, anomaly_type.
    """
    rng = np.random.default_rng(seed)
    frame = daily.sort_values(["product_id", "sale_date"]).reset_index(drop=True).copy()
    level = _normal_level(frame)
    in_window = frame["sale_date"].between(start, end) & open_day_mask(frame["sale_date"]) & level.notna()

    used: set[int] = set()
    labels: list[dict] = []

    def take(candidates: np.ndarray, n: int) -> list[int]:
        free = [i for i in candidates if i not in used]
        chosen = rng.choice(free, size=min(n, len(free)), replace=False).tolist()
        used.update(chosen)
        return chosen

    # ---- spikes: far above the normal level -------------------------------
    for i in take(np.flatnonzero(in_window), n_spikes):
        factor = rng.uniform(*SPIKE_FACTOR)
        frame.loc[i, "units_sold"] = int(round(max(frame.loc[i, "units_sold"], level[i]) * factor))
        labels.append({"product_id": frame.loc[i, "product_id"], "sale_date": frame.loc[i, "sale_date"], "anomaly_type": "spike"})

    # ---- drops: zero on a day that is normally busy -----------------------
    busy = in_window & (level >= DROP_MIN_NORMAL_LEVEL) & (frame["units_sold"] > 0)
    for i in take(np.flatnonzero(busy), n_drops):
        frame.loc[i, "units_sold"] = 0
        frame.loc[i, "avg_price"] = np.nan        # no sale, no price — consistent with real zero days
        labels.append({"product_id": frame.loc[i, "product_id"], "sale_date": frame.loc[i, "sale_date"], "anomaly_type": "drop"})

    # ---- flatlines: the first day's value repeated for the next 4 days ----
    run_starts = np.flatnonzero(in_window & (frame["units_sold"] > 0))
    placed = 0
    for i in rng.permutation(run_starts):
        if placed == n_flatlines:
            break
        run = list(range(i, i + FLATLINE_LENGTH))
        same_product = run[-1] < len(frame) and frame.loc[run[-1], "product_id"] == frame.loc[i, "product_id"]
        if not same_product or frame.loc[run[-1], "sale_date"] > end or used.intersection(run):
            continue
        used.update(run)
        for j in run[1:]:                         # the first day is genuine; the repeats are the anomaly
            frame.loc[j, ["units_sold", "avg_price"]] = frame.loc[i, ["units_sold", "avg_price"]].to_numpy()
            labels.append({"product_id": frame.loc[j, "product_id"], "sale_date": frame.loc[j, "sale_date"], "anomaly_type": "flatline"})
        placed += 1

    return frame, pd.DataFrame(labels, columns=["product_id", "sale_date", "anomaly_type"])
