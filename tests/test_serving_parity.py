"""
Phase 5 — training/serving parity, and no future data at serving time.

Two independent ways of producing the same forecast must agree exactly:

  OFFLINE  build_features() on the FULL dataset (every day, including all the
           days AFTER the forecast date), then take the row for the forecast date.
           This is exactly how the model was trained and evaluated.

  ONLINE   DemandForecastService given history only UP TO the day before,
           the way the API serves a forecast.

If they match for many past dates and products:
  - the API builds features exactly as training did (no training/serving skew);
  - the offline row cannot have used any day after the forecast date either,
    because the online path never had those days.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from app.ml.features import build_features
from app.ml.forecasting.predict import predict_demand
from app.repositories.mock import InMemorySalesHistoryRepository, MockProductRepository
from app.services.forecast import DemandForecastService

# Forecast dates spread across the data: different months, weekdays, before/after holidays.
# Each must be an OPEN day whose previous open day is the day before or a Friday (Sat closed).
CASES = [
    ("85123A", date(2011, 3, 15)),     # Tuesday
    ("22197", date(2011, 6, 5)),       # Sunday (Saturday in between is filled as a planned closure)
    ("21754", date(2011, 9, 8)),       # Thursday, low-volume product
    ("84879", date(2011, 11, 20)),     # Sunday in the November peak
    ("20725", date(2010, 10, 4)),      # Monday, first year
]


@pytest.fixture(scope="module")
def offline_features(dataset_sales):
    frame = pd.DataFrame({
        "product_id": [s.product_id for s in dataset_sales],
        "sale_date": pd.to_datetime([s.sale_date for s in dataset_sales]),
        "units_sold": [s.units_sold for s in dataset_sales],
        "avg_price": [float(s.avg_price) if s.avg_price is not None else float("nan") for s in dataset_sales],
    })
    return build_features(frame)


@pytest.mark.asyncio
@pytest.mark.parametrize("product_id, forecast_date", CASES)
async def test_online_forecast_equals_offline_pipeline(
    product_id, forecast_date, dataset_products, dataset_sales, deployed_bundle, offline_features
):
    # ONLINE: the service only ever sees days strictly before the forecast date
    # (a Saturday right before it is dropped too: the service fills planned closures itself)
    last_day = forecast_date - timedelta(days=1)
    while last_day.weekday() == 5:
        last_day -= timedelta(days=1)
    history = [s for s in dataset_sales if s.product_id == product_id and s.sale_date <= last_day]
    service = DemandForecastService(
        MockProductRepository(dataset_products), InMemorySalesHistoryRepository(history), deployed_bundle
    )
    online = await service.forecast(product_id)

    # OFFLINE: the full-dataset feature row for that date, through the same model
    row = offline_features[
        (offline_features["product_id"] == product_id) & (offline_features["sale_date"] == pd.Timestamp(forecast_date))
    ]
    offline = float(predict_demand(deployed_bundle.model, row)[0])

    assert online.forecast_date == forecast_date
    assert online.predicted_units == pytest.approx(offline, abs=0.05)       # service rounds to 0.1 units


@pytest.mark.asyncio
async def test_serving_ignores_data_after_the_history_it_is_given(dataset_products, dataset_sales, deployed_bundle):
    """Rows dated after the forecast day are never read, even if the repository holds them."""
    product_id, cutoff = "85123A", date(2011, 3, 14)
    history = [s for s in dataset_sales if s.product_id == product_id and s.sale_date <= cutoff]

    class HistoryUpTo(InMemorySalesHistoryRepository):
        async def recent(self, pid, days):
            rows = await super().recent(pid, days)
            assert all(r.sale_date <= cutoff for r in rows)                 # what the service receives
            return rows

    forecast = await DemandForecastService(
        MockProductRepository(dataset_products), HistoryUpTo(history), deployed_bundle
    ).forecast(product_id)

    assert forecast.history_end == cutoff
    assert forecast.forecast_date > cutoff
