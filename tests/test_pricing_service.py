"""Day 5 — PricingService with an injected repository (no HTTP involved)."""
import itertools
from decimal import Decimal

import pytest

from app.exceptions.pricing import ProductNotFoundError
from app.models.pricing import PriceQuote, PricingStrategy
from app.models.product import Product
from app.repositories.base import ProductRepository
from app.repositories.mock import MockProductRepository
from app.services.pricing import PricingService


class RecordingRepository(ProductRepository):
    """
    A hand-written fake — NOT the mock repository.

    If the service works with this, it truly depends only on the
    ProductRepository abstraction (Dependency Inversion Principle).
    """

    def __init__(self, product: Product) -> None:
        self._product = product
        self.requested_ids: list[str] = []

    async def get_by_id(self, product_id: str) -> Product:
        self.requested_ids.append(product_id)
        return self._product

    async def list_all(self) -> list[Product]:
        return [self._product]


@pytest.mark.asyncio
async def test_service_loads_product_through_injected_repository(make_product):
    repository = RecordingRepository(make_product())

    quote = await PricingService(repository).calculate_price("PROD-001", PricingStrategy.COMPETITIVE)

    assert repository.requested_ids == ["PROD-001"]
    assert isinstance(quote, PriceQuote)
    assert quote.product.product_id == "PROD-001"
    assert quote.strategy is PricingStrategy.COMPETITIVE
    assert quote.suggested_price == Decimal("969.03")
    assert quote.margin_percent == Decimal("48.40")
    assert quote.guardrails_applied == ()


@pytest.mark.asyncio
async def test_missing_product_raises_business_exception_not_http_error():
    service = PricingService(MockProductRepository([]))

    with pytest.raises(ProductNotFoundError) as exc_info:
        await service.calculate_price("SKU-99999", PricingStrategy.COMPETITIVE)

    assert exc_info.value.product_id == "SKU-99999"


COSTS = ["1.00", "99.99", "333.33", "500.00", "950.00", "1000.00"]
COMPETITORS = ["1.00", "300.00", "400.00", "999.00", "5000.00"]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", list(PricingStrategy))
async def test_never_sells_below_cost_and_always_whole_cents(make_product, strategy):
    grid = [
        make_product(cost, competitor, product_id=f"{cost}/{competitor}")
        for cost, competitor in itertools.product(COSTS, COMPETITORS)
    ]
    service = PricingService(MockProductRepository(grid))

    for item in grid:
        quote = await service.calculate_price(item.product_id, strategy)
        case = f"strategy={strategy.value} cost/competitor={item.product_id} -> {quote.suggested_price}"

        assert quote.suggested_price >= item.cost_price * Decimal("1.05"), case
        assert quote.suggested_price.as_tuple().exponent == -2, case
        assert quote.margin_percent >= 0, case
