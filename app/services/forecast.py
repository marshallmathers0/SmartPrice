"""
Next-day demand forecast for one product — the deployed model, used live.

    sales history (DB) -> features (same code as training) -> model -> forecast
                                                       +-> exact contributions -> plain-language reasons
                       -> anomaly detector (deployed one, frozen threshold) -> recent flags

The same build_features() function is used in training and here, so the
model always sees features computed exactly the way it was trained on
("training/serving skew" is impossible by construction).
"""
import logging
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from app.exceptions.pricing import NoSalesHistoryError
from app.ml.anomaly.calendar import open_day_mask
from app.ml.anomaly.detectors import build_detection_features, detector_by_name
from app.ml.explain import contributions, describe
from app.ml.features import FEATURES, MIN_HISTORY_DAYS, build_features
from app.ml.forecasting.predict import predict_demand
from app.ml.model_store import ModelBundle
from app.models.forecast import AnomalyFlag, DailySales, DemandForecast, ForecastFactor
from app.repositories.base import ProductRepository, SalesHistoryRepository

logger = logging.getLogger(__name__)

HISTORY_DAYS = 90          # enough for the 28-day features AND the anomaly detector's 28-open-day reference
TYPICAL_WEEKS = 4          # "typical" = the model's median forecast for the same weekday, last 4 weeks
ANOMALY_LOOKBACK_DAYS = 14 # how far back recent anomaly flags are reported
TOP_FACTORS = 5


class DemandForecastService:
    def __init__(self, products: ProductRepository, sales: SalesHistoryRepository, bundle: ModelBundle) -> None:
        self._products = products
        self._sales = sales
        self._bundle = bundle

    async def forecast(self, product_id: str) -> DemandForecast:
        await self._products.get_by_id(product_id)                 # 404 for an unknown product, before anything else
        return await self._forecast_from_history(product_id)

    async def demand_score(self, product_id: str) -> Decimal | None:
        """
        Just the demand signal for pricing — None when it cannot be computed (no history).

        Called by DemandAwareProductRepository right AFTER it loaded the product,
        so the existence check in forecast() would be a second, redundant lookup.
        """
        try:
            return (await self._forecast_from_history(product_id)).demand_score
        except NoSalesHistoryError:
            return None

    async def _forecast_from_history(self, product_id: str) -> DemandForecast:
        history = await self._sales.recent(product_id, HISTORY_DAYS)
        if len(history) < MIN_HISTORY_DAYS + 1:
            raise NoSalesHistoryError(product_id, MIN_HISTORY_DAYS + 1)

        frame = _to_frame(history)
        forecast_date = _next_open_day(frame["sale_date"].iloc[-1])
        features = build_features(_with_target_day(frame, forecast_date))
        target_row = features[features["sale_date"] == forecast_date]

        # ---- forecast + exact reasons -------------------------------------------------
        contrib = contributions(self._bundle.model, target_row[FEATURES])
        row_contrib = contrib.values.iloc[0]
        predicted = max(float(np.expm1(contrib.base + row_contrib.sum())), 0.0)

        ranked = row_contrib.abs().sort_values(ascending=False).index[:TOP_FACTORS]
        row = target_row.iloc[0]
        factors = tuple(
            ForecastFactor(
                feature=feature,
                description=describe(feature, row),
                contribution=round(float(row_contrib[feature]), 4),
                multiplier=round(float(np.exp(row_contrib[feature])), 3),
            )
            for feature in ranked
        )

        # ---- forecast -> demand score -------------------------------------------------
        # "Is tomorrow busier than usual?" needs a fair "usual":
        #  - the SAME WEEKDAY (a Sunday forecast vs recent Sundays - weekdays differ a lot);
        #  - the MODEL'S OWN forecasts for those days, not the actual sales. The model is
        #    trained on log(1 + units), so it predicts a typical day and is systematically
        #    below actual volume (out of sample: 58% of total units). Comparing it with raw
        #    actuals would make every day look quiet; comparing it with itself cancels that out.
        same_weekday = [forecast_date - timedelta(weeks=w) for w in range(1, TYPICAL_WEEKS + 1)]
        reference = features[features["sale_date"].isin(same_weekday)]
        typical = float(np.median(predict_demand(self._bundle.model, reference))) if len(reference) else 0.0
        recent_actual = float(reference["units_sold"].median()) if len(reference) else 0.0
        demand_score = demand_score_from(predicted, typical)

        logger.info(
            "Forecast | product=%s | date=%s | predicted=%.1f | typical=%.1f | demand_score=%s | model=%s",
            product_id, forecast_date, predicted, typical, demand_score, self._bundle.version,
        )
        return DemandForecast(
            product_id=product_id,
            forecast_date=forecast_date.date(),
            history_end=frame["sale_date"].iloc[-1].date(),
            predicted_units=round(predicted, 1),
            typical_units=round(typical, 1),
            recent_actual_units=round(recent_actual, 1),
            demand_score=demand_score,
            model_version=self._bundle.version,
            base_units=round(float(np.expm1(contrib.base)), 1),
            factors=factors,
            recent_anomalies=self._recent_anomalies(frame),
        )

    def _recent_anomalies(self, frame: pd.DataFrame) -> tuple[AnomalyFlag, ...]:
        """Run the deployed detector, with its frozen threshold, over the recent open days."""
        settings = self._bundle.metadata["anomaly_detector"]
        detector = detector_by_name(settings["name"])
        scored = build_detection_features(frame)
        if scored.empty:
            return ()
        detector.fit(scored)                                     # no-op for the statistical detectors
        scored = scored.assign(score=detector.score(scored))

        cutoff = frame["sale_date"].iloc[-1] - timedelta(days=ANOMALY_LOOKBACK_DAYS)
        flagged = scored[(scored["sale_date"] > cutoff) & (scored["score"] >= settings["threshold"])]
        return tuple(
            AnomalyFlag(sale_date=r.sale_date.date(), units_sold=int(r.units_sold), reason=_anomaly_reason(r))
            for r in flagged.itertuples()
        )


def demand_score_from(predicted: float, typical: float) -> Decimal:
    """0.50 = a normal day; 1.00 = double the usual demand or more; 0.00 = none expected."""
    if typical <= 0:
        return Decimal("0.50")                                   # no reference: stay neutral
    ratio = min(max(predicted / typical, 0.0), 2.0)
    return (Decimal(str(ratio)) / 2).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_frame(history: list[DailySales]) -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [h.product_id for h in history],
        "sale_date": pd.to_datetime([h.sale_date for h in history]),
        "units_sold": [h.units_sold for h in history],
        "avg_price": [float(h.avg_price) if h.avg_price is not None else np.nan for h in history],
    })


def _next_open_day(last: pd.Timestamp) -> pd.Timestamp:
    """The next day the shop trades. Forecasting a Saturday would just say "0 - closed"."""
    day = last + timedelta(days=1)
    while not open_day_mask(pd.Series([day])).iloc[0]:
        day += timedelta(days=1)
    return day


def _with_target_day(frame: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    """
    Append the day to forecast (demand unknown) and any planned closed days before it.

    A closed day between the last data and the forecast date (e.g. the Saturday
    between Friday and Sunday) is a PLANNED closure: zero sales are certain, not
    a guess, so they are filled in as 0 — exactly what the history shows for
    every past Saturday.
    """
    product_id = frame["product_id"].iloc[0]
    extra = []
    day = frame["sale_date"].iloc[-1] + timedelta(days=1)
    while day < forecast_date:
        extra.append({"product_id": product_id, "sale_date": day, "units_sold": 0, "avg_price": np.nan})
        day += timedelta(days=1)
    extra.append({"product_id": product_id, "sale_date": forecast_date, "units_sold": np.nan, "avg_price": np.nan})
    return pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)


def _anomaly_reason(row) -> str:
    if row.score >= 1e6:
        return "Same value repeated for 4 open days in a row (possible stuck data feed)"
    return "Unusually high demand" if row.z_28 > 0 else "Unusually low demand"
