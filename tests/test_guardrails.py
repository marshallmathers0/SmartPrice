"""Day 4 — Chain of Responsibility (guardrails)."""
from decimal import Decimal

from app.guardrails import PriceFloorGuard, VolatilityGuard
from app.models.product import Product
from app.repositories.mock import MockProductRepository
from app.services.pricing import PricingService


def run_chain(price: str, product: Product) -> tuple[Decimal, list[str]]:
    """Run a raw price through the service's real guard chain."""
    applied_guards: list[str] = []
    guard_chain = PricingService(MockProductRepository())._build_guard_chain()
    final_price = guard_chain.handle(Decimal(price), product, applied_guards)
    return final_price, applied_guards


# ------------------------------------------------------------------
# PriceFloorGuard
# ------------------------------------------------------------------

def test_floor_passes_price_above_floor(make_product):
    applied_guards: list[str] = []

    price = PriceFloorGuard().apply(Decimal("969.03"), make_product(), applied_guards)

    assert price == Decimal("969.03")
    assert applied_guards == []


def test_floor_raises_price_below_floor(make_product):
    # DEVLOG Day 4: cost=950, competitor=999 -> competitive 969.03 < floor 997.50
    applied_guards: list[str] = []

    price = PriceFloorGuard().apply(Decimal("969.03"), make_product(cost="950.00"), applied_guards)

    assert str(price) == "997.50"
    assert applied_guards == [
        "PriceFloorGuard: price raised from 969.03 to 997.50 (minimum margin of 5% enforced)"
    ]


def test_floor_output_is_whole_cents(make_product):
    # 333.33 x 1.05 = 349.9965 — must come back as 350.00, not 349.9965
    price = PriceFloorGuard().apply(Decimal("291.00"), make_product(cost="333.33"), [])

    assert str(price) == "350.00"


def test_floor_rounds_up_so_price_never_dips_below_floor(make_product):
    # 100.01 x 1.05 = 105.0105 — half-up would give 105.01, a fraction below the floor
    price = PriceFloorGuard().apply(Decimal("97.00"), make_product(cost="100.01"), [])

    assert str(price) == "105.02"


# ------------------------------------------------------------------
# VolatilityGuard
# ------------------------------------------------------------------

def test_volatility_passes_price_within_ceiling(make_product):
    applied_guards: list[str] = []

    price = VolatilityGuard().apply(Decimal("700.00"), make_product(), applied_guards)

    assert price == Decimal("700.00")
    assert applied_guards == []


def test_volatility_caps_price_above_ceiling(make_product):
    # DEVLOG Day 4: competitor=400 -> ceiling 600.00
    applied_guards: list[str] = []

    price = VolatilityGuard().apply(Decimal("1260.00"), make_product(cost="900.00", competitor="400.00"), applied_guards)

    assert str(price) == "600.00"
    assert applied_guards == [
        "VolatilityGuard: price capped from 1260.00 to 600.00 (max 50% above competitor enforced)"
    ]


def test_volatility_rounds_down_so_price_never_exceeds_ceiling(make_product):
    # 333.33 x 1.50 = 499.995 — half-up would give 500.00, a fraction above the ceiling
    price = VolatilityGuard().apply(Decimal("560.00"), make_product(cost="400.00", competitor="333.33"), [])

    assert str(price) == "499.99"


# ------------------------------------------------------------------
# The chain
# ------------------------------------------------------------------

def test_set_next_returns_next_guard_for_fluent_chaining():
    volatility, floor = VolatilityGuard(), PriceFloorGuard()

    assert volatility.set_next(floor) is floor


def test_chain_leaves_acceptable_price_untouched(make_product):
    price, applied_guards = run_chain("969.03", make_product())

    assert price == Decimal("969.03")
    assert applied_guards == []


def test_chain_floor_only_has_no_override_note(make_product):
    price, applied_guards = run_chain("969.03", make_product(cost="950.00"))

    assert str(price) == "997.50"
    assert applied_guards == [
        "PriceFloorGuard: price raised from 969.03 to 997.50 (minimum margin of 5% enforced)"
    ]


def test_chain_floor_wins_when_ceiling_is_below_floor(make_product):
    # cost=900, competitor=400: ceiling 600.00 < floor 945.00 — no price satisfies both.
    # The floor must win: never sell below cost.
    price, applied_guards = run_chain("1260.00", make_product(cost="900.00", competitor="400.00"))

    assert str(price) == "945.00"
    assert applied_guards == [
        "VolatilityGuard: price capped from 1260.00 to 600.00 (max 50% above competitor enforced)",
        "PriceFloorGuard: price raised from 600.00 to 945.00 (minimum margin of 5% enforced; "
        "floor takes priority over the earlier guardrail adjustment so the product is never sold below cost)",
    ]
