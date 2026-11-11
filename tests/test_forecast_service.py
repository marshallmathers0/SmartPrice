"""Phase 4 — the live forecast: next open day, demand score, reasons, anomaly flags."""
from dataclasses import replace
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from app.exceptions.pricing import NoSalesHistoryError, ProductNotFoundError
from app.ml.features import FEATURES, build_features
from app.ml.forecasting.predict import predict_demand
from app.repositories.mock import SEED_PRODUCTS, InMemorySalesHistoryRepository, MockProductRepository
from app.services.forecast import (
    TOP_FACTORS,
    DemandForecastService,
    _next_open_day,
    _to_frame,
    _with_target_day,
    demand_score_from,
)

PRODUCT = "85123A"


@pytest.fixture
def service(dataset_products, dataset_sales, deployed_bundle):
    products = MockProductRepository([*dataset_products, *SEED_PRODUCTS])
    return DemandForecastService(products, InMemorySalesHistoryRepository(dataset_sales), deployed_bundle)


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "predicted, typical, expected",
    [
        (0, 10, "0.00"),        # no demand expected
        (10, 10, "0.50"),       # a normal day
        (15, 10, "0.75"),
        (20, 10, "1.00"),       # double the usual
        (90, 10, "1.00"),       # capped
        (5, 0, "0.50"),         # no reference -> neutral
    ],
)
def test_demand_score_from(predicted, typical, expected):
    assert demand_score_from(predicted, typical) == Decimal(expected)


def test_next_open_day_skips_saturday_and_holidays():
    assert _next_open_day(pd.Timestamp("2011-12-09")) == pd.Timestamp("2011-12-11")   # Fri -> Sun
    assert _next_open_day(pd.Timestamp("2011-04-21")) == pd.Timestamp("2011-04-24")   # Good Friday + Saturday skipped
    assert _next_open_day(pd.Timestamp("2011-12-05")) == pd.Timestamp("2011-12-06")   # ordinary Tuesday


def test_planned_closed_days_are_filled_with_zero(dataset_sales):
    frame = _to_frame([s for s in dataset_sales if s.product_id == PRODUCT])
    extended = _with_target_day(frame, pd.Timestamp("2011-12-11"))

    saturday = extended[extended["sale_date"] == pd.Timestamp("2011-12-10")]
    target = extended[extended["sale_date"] == pd.Timestamp("2011-12-11")]
    assert saturday["units_sold"].iat[0] == 0
    assert np.isnan(target["units_sold"].iat[0])           # the answer is unknown — and never used as a feature


# ------------------------------------------------------------------
# The service on real history with the deployed model
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forecast_for_the_next_open_day(service):
    forecast = await service.forecast(PRODUCT)

    assert forecast.history_end == date(2011, 12, 9)
    assert forecast.forecast_date == date(2011, 12, 11)
    assert forecast.predicted_units >= 0
    assert Decimal("0") <= forecast.demand_score <= Decimal("1")
    # predicted/typical are reported rounded to 0.1 units, so recomputing may differ by one rounding step
    assert abs(forecast.demand_score - demand_score_from(forecast.predicted_units, forecast.typical_units)) <= Decimal("0.01")


@pytest.mark.asyncio
async def test_forecast_matches_the_model_called_directly(service, dataset_sales, deployed_bundle):
    forecast = await service.forecast(PRODUCT)

    frame = _with_target_day(_to_frame([s for s in dataset_sales if s.product_id == PRODUCT]), pd.Timestamp("2011-12-11"))
    features = build_features(frame)
    direct = predict_demand(deployed_bundle.model, features[features["sale_date"] == pd.Timestamp("2011-12-11")])[0]

    assert forecast.predicted_units == pytest.approx(direct, abs=0.05)


@pytest.mark.asyncio
async def test_factors_are_the_largest_and_explain_the_forecast(service):
    forecast = await service.forecast(PRODUCT)
    sizes = [abs(f.contribution) for f in forecast.factors]

    assert len(forecast.factors) == TOP_FACTORS
    assert sizes == sorted(sizes, reverse=True)
    assert all(f.feature in FEATURES and f.description for f in forecast.factors)
    assert all(f.multiplier == pytest.approx(np.exp(f.contribution), rel=1e-3) for f in forecast.factors)


@pytest.mark.asyncio
async def test_typical_is_the_models_own_forecast_for_the_same_weekday(service, dataset_sales, deployed_bundle):
    forecast = await service.forecast(PRODUCT)
    frame = _with_target_day(_to_frame([s for s in dataset_sales if s.product_id == PRODUCT]), pd.Timestamp("2011-12-11"))
    features = build_features(frame)
    last_4_sundays = features[features["sale_date"].isin(pd.date_range("2011-11-13", "2011-12-04", freq="7D"))]

    assert len(last_4_sundays) == 4
    assert forecast.typical_units == pytest.approx(np.median(predict_demand(deployed_bundle.model, last_4_sundays)), abs=0.05)


@pytest.mark.asyncio
async def test_a_recent_spike_is_flagged(dataset_products, dataset_sales, deployed_bundle):
    spiked = [replace(s, units_sold=50_000) if (s.product_id == PRODUCT and s.sale_date == date(2011, 12, 8)) else s
              for s in dataset_sales]
    service = DemandForecastService(MockProductRepository(dataset_products), InMemorySalesHistoryRepository(spiked), deployed_bundle)

    flags = (await service.forecast(PRODUCT)).recent_anomalies

    assert any(f.sale_date == date(2011, 12, 8) and f.reason == "Unusually high demand" for f in flags)


@pytest.mark.asyncio
async def test_product_without_history(service):
    with pytest.raises(NoSalesHistoryError):
        await service.forecast("SKU-12345")
    assert await service.demand_score("SKU-12345") is None


@pytest.mark.asyncio
async def test_unknown_product(service):
    with pytest.raises(ProductNotFoundError):
        await service.forecast("NOT-A-PRODUCT")
    assert await service.demand_score("NOT-A-PRODUCT") is None


@pytest.mark.asyncio
async def test_too_little_history_is_refused(dataset_products, dataset_sales, deployed_bundle):
    short = [s for s in dataset_sales if s.product_id == PRODUCT][-10:]
    service = DemandForecastService(MockProductRepository(dataset_products), InMemorySalesHistoryRepository(short), deployed_bundle)

    with pytest.raises(NoSalesHistoryError, match="at least 29 days"):
        await service.forecast(PRODUCT)
