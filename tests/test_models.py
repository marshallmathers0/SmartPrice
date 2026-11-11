"""Day 5 — Product domain model."""
import dataclasses
from decimal import Decimal

import pytest

from app.models.product import Product


def test_product_is_immutable(make_product):
    product = make_product()

    with pytest.raises(dataclasses.FrozenInstanceError):
        product.cost_price = Decimal("1.00")


@pytest.mark.parametrize(
    "cost, competitor",
    [
        ("0", "999.00"),
        ("-1.00", "999.00"),
        ("500.00", "0"),
    ],
)
def test_product_rejects_non_positive_prices(cost, competitor):
    with pytest.raises(ValueError):
        Product("PROD-001", "Wireless Headphones", Decimal(cost), Decimal(competitor))


def test_demand_signals_default_to_neutral(make_product):
    product = make_product()

    assert product.inventory == 0
    assert product.demand_score == Decimal("0.50")


@pytest.mark.parametrize(
    "inventory, demand_score",
    [
        (-1, "0.50"),       # negative stock
        (10, "-0.01"),      # below the 0..1 range
        (10, "1.01"),       # above the 0..1 range
    ],
)
def test_product_rejects_invalid_demand_signals(inventory, demand_score):
    with pytest.raises(ValueError):
        Product("PROD-001", "Wireless Headphones", Decimal("500.00"), Decimal("999.00"),
                inventory=inventory, demand_score=Decimal(demand_score))


@pytest.mark.parametrize("demand_score", ["0.00", "1.00"])
def test_product_accepts_demand_score_boundaries(demand_score):
    product = Product("PROD-001", "Wireless Headphones", Decimal("500.00"), Decimal("999.00"),
                      demand_score=Decimal(demand_score))

    assert product.demand_score == Decimal(demand_score)
