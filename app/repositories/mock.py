import logging
from collections.abc import Iterable
from decimal import Decimal

from app.exceptions.pricing import ProductNotFoundError
from app.models.forecast import DailySales
from app.models.product import Product
from app.repositories.base import ProductRepository, SalesHistoryRepository

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Seed catalog — stands in for the products table until Day 6.
# Each product exercises a different path through the pricing engine.
# ------------------------------------------------------------------
SEED_PRODUCTS: tuple[Product, ...] = (
    # Healthy margin — no guardrail fires (DEVLOG Day 3 example). Popular, well stocked.
    Product("SKU-12345", "Wireless Headphones", Decimal("500.00"), Decimal("999.00"),
            inventory=120, demand_score=Decimal("0.80")),
    # Cost close to competitor — PriceFloorGuard raises the competitive price (DEVLOG Day 4 example)
    Product("SKU-67890", "Mechanical Keyboard", Decimal("950.00"), Decimal("999.00"),
            inventory=35, demand_score=Decimal("0.55")),
    # Cost far above competitor — with cost_plus, the ceiling and floor conflict and the floor wins.
    # Overstocked and slow-selling.
    Product("SKU-11111", "Bluetooth Speaker", Decimal("900.00"), Decimal("400.00"),
            inventory=300, demand_score=Decimal("0.20")),
)


class MockProductRepository(ProductRepository):
    """
    In-memory ProductRepository backed by a Python dict.

    Day 6: the application now uses PostgresProductRepository. This class
    remains for fast tests that need no database, and SEED_PRODUCTS above
    is also what app/database/seed.py inserts into PostgreSQL.

    Pass ``products`` to use a custom catalog (tests do this);
    otherwise the seed catalog above is used.
    """

    def __init__(self, products: Iterable[Product] = SEED_PRODUCTS) -> None:
        # dict keyed by product_id → O(1) lookups, like a primary-key index
        self._products: dict[str, Product] = {product.product_id: product for product in products}

    async def get_by_id(self, product_id: str) -> Product:
        product = self._products.get(product_id)

        if product is None:
            logger.info("Product not found | product_id=%s", product_id)
            raise ProductNotFoundError(product_id)

        logger.debug("Product loaded | product_id=%s", product_id)
        return product

    async def list_all(self) -> list[Product]:
        return [self._products[key] for key in sorted(self._products)]


class InMemorySalesHistoryRepository(SalesHistoryRepository):
    """Sales history from a plain list — for tests and offline use."""

    def __init__(self, rows: Iterable[DailySales] = ()) -> None:
        self._by_product: dict[str, list[DailySales]] = {}
        for row in sorted(rows, key=lambda r: r.sale_date):
            self._by_product.setdefault(row.product_id, []).append(row)

    async def recent(self, product_id: str, days: int) -> list[DailySales]:
        return self._by_product.get(product_id, [])[-days:]
