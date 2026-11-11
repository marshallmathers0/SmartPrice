"""Day 3 — Strategy Pattern and Factory Pattern."""
from decimal import Decimal

import pytest

from app.factories.strategy_factory import StrategyFactory
from app.models.pricing import PricingStrategy
from app.strategies import (
    AbstractPricingStrategy,
    CompetitivePricingStrategy,
    CostPlusPricingStrategy,
    DynamicPricingStrategy,
    PenetrationPricingStrategy,
)


# DEVLOG Day 3 results table — cost=500, competitor=999 (raw strategy output, before guards)
@pytest.mark.parametrize(
    "strategy, expected",
    [
        (PricingStrategy.COMPETITIVE, Decimal("969.03")),   # 999 x 0.97
        (PricingStrategy.COST_PLUS, Decimal("700.00")),     # 500 x 1.40
        (PricingStrategy.DYNAMIC, Decimal("834.52")),       # (969.03 + 700.00) / 2
        (PricingStrategy.PENETRATION, Decimal("550.00")),   # 500 x 1.10
    ],
)
def test_raw_strategy_prices_match_devlog_table(make_product, strategy, expected):
    assert StrategyFactory.get_strategy(strategy).calculate_price(make_product()) == expected


@pytest.mark.parametrize(
    "strategy, strategy_class",
    [
        (PricingStrategy.COMPETITIVE, CompetitivePricingStrategy),
        (PricingStrategy.COST_PLUS, CostPlusPricingStrategy),
        (PricingStrategy.DYNAMIC, DynamicPricingStrategy),
        (PricingStrategy.PENETRATION, PenetrationPricingStrategy),
    ],
)
def test_factory_returns_registered_strategy(strategy, strategy_class):
    assert isinstance(StrategyFactory.get_strategy(strategy), strategy_class)


def test_factory_registers_every_strategy_enum_value():
    assert set(StrategyFactory.available_strategies()) == {s.value for s in PricingStrategy}


def test_factory_rejects_unregistered_strategy():
    with pytest.raises(ValueError):
        StrategyFactory.get_strategy("holiday")


def test_abstract_strategy_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractPricingStrategy()
