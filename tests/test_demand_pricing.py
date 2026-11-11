"""Phase 4 — the demand-based strategy and the repository decorator that feeds it."""
from dataclasses import replace
from decimal import Decimal

import pytest

from app.factories.strategy_factory import StrategyFactory
from app.models.pricing import PricingStrategy
from app.repositories.demand_aware import DemandAwareProductRepository
from app.repositories.mock import MockProductRepository
from app.services.pricing import PricingService
from app.strategies.demand_based import DemandBasedPricingStrategy


@pytest.mark.parametrize(
    "score, factor",
    [("0.00", "0.90"), ("0.29", "0.958"), ("0.50", "1.00"), ("1.00", "1.10")],
)
def test_demand_factor(make_product, score, factor):
    product = make_product(competitor="2.95")

    assert DemandBasedPricingStrategy().factor(replace(product, demand_score=Decimal(score))) == Decimal(factor)


def test_price_example_from_the_docstring(make_product):
    product = replace(make_product(cost="1.77", competitor="2.95"), demand_score=Decimal("0.29"))

    assert DemandBasedPricingStrategy().calculate_price(product) == Decimal("2.83")


def test_strategy_is_registered():
    assert isinstance(StrategyFactory.get_strategy(PricingStrategy.DEMAND_BASED), DemandBasedPricingStrategy)


@pytest.mark.asyncio
async def test_guardrails_still_protect_demand_based_prices(make_product):
    # Low demand would price 10% below a competitor who is already below our cost floor
    async def no_demand(_):
        return Decimal("0.00")

    repository = DemandAwareProductRepository(MockProductRepository([make_product("950.00", "999.00", product_id="P")]), no_demand)
    quote = await PricingService(repository).calculate_price("P", PricingStrategy.DEMAND_BASED)

    assert quote.suggested_price == Decimal("997.50")              # 999 x 0.90 = 899.10, raised to the cost floor
    assert quote.guardrails_applied[0].startswith("PriceFloorGuard")


# ------------------------------------------------------------------
# DemandAwareProductRepository
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repository_fills_in_the_forecast_score(make_product):
    async def forecast(product_id):
        return Decimal("0.83")

    product = await DemandAwareProductRepository(MockProductRepository([make_product()]), forecast).get_by_id("PROD-001")

    assert product.demand_score == Decimal("0.83")


@pytest.mark.asyncio
async def test_repository_keeps_the_stored_score_when_no_forecast(make_product):
    async def nothing(product_id):
        return None

    inner = MockProductRepository([make_product()])
    with_none = await DemandAwareProductRepository(inner, nothing).get_by_id("PROD-001")
    without_model = await DemandAwareProductRepository(inner, None).get_by_id("PROD-001")

    assert with_none.demand_score == without_model.demand_score == Decimal("0.50")


@pytest.mark.asyncio
async def test_repository_never_mutates_the_stored_product(make_product):
    async def forecast(product_id):
        return Decimal("0.99")

    inner = MockProductRepository([make_product()])
    await DemandAwareProductRepository(inner, forecast).get_by_id("PROD-001")

    assert (await inner.get_by_id("PROD-001")).demand_score == Decimal("0.50")
