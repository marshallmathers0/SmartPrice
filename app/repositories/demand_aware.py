import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from decimal import Decimal

from app.models.product import Product
from app.repositories.base import ProductRepository

logger = logging.getLogger(__name__)

DemandScoreProvider = Callable[[str], Awaitable[Decimal | None]]


class DemandAwareProductRepository(ProductRepository):
    """
    Decorator (again — like CachedProductRepository on Day 7): a ProductRepository
    that fills in each product's demand_score from the live demand forecast.

        PricingService
              │
              ▼
        DemandAwareProductRepository     <- demand_score = forecast (Phase 4)
              │
              ▼
        CachedProductRepository          <- Redis (Day 7)
              │
              ▼
        PostgresProductRepository        <- PostgreSQL (Day 6)

    This is how the ML model reaches pricing WITHOUT changing PricingService,
    the strategies or the guardrails: they keep receiving a plain Product,
    whose demand_score now comes from the model instead of a stored placeholder.

    It sits OUTSIDE the cache on purpose: Redis caches the product row; the
    forecast-based score is added on every request, so it is never stale.

    `demand_score` is any async function product_id -> score | None.
    None (no history, no model) keeps the product's stored score — pricing
    never fails because the ML part cannot answer.
    """

    def __init__(self, inner: ProductRepository, demand_score: DemandScoreProvider | None) -> None:
        self._inner = inner
        self._demand_score = demand_score

    async def get_by_id(self, product_id: str) -> Product:
        product = await self._inner.get_by_id(product_id)
        if self._demand_score is None:
            return product

        score = await self._demand_score(product_id)
        if score is None:
            return product
        logger.info("Demand score from forecast | product_id=%s | score=%s", product_id, score)
        return replace(product, demand_score=score)     # frozen dataclass: a new Product, nothing mutated

    async def list_all(self) -> list[Product]:
        # A forecast for every product on every listing would be slow; the list shows stored data
        return await self._inner.list_all()
