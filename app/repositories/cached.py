import logging

from app.cache.product_cache import ProductCache
from app.models.product import Product
from app.repositories.base import ProductRepository

logger = logging.getLogger(__name__)


class CachedProductRepository(ProductRepository):
    """
    Decorator Pattern: a ProductRepository that wraps another ProductRepository
    and adds caching — without changing the one it wraps.

        PricingService
               │
               ▼
        ProductRepository           <- the service still only knows this
               │
               ▼
        CachedProductRepository     <- this class
           ┌───┴──────────┐
           ▼              ▼
     ProductCache    inner repository (PostgresProductRepository)
           │              │
           ▼              ▼
         Redis        PostgreSQL

    Cache-Aside
    -----------
    1. Look in the cache.        Hit  → return it.
    2. Miss → ask the inner repository (PostgreSQL, the source of truth).
    3. Store the result in the cache, then return it.

    The cache is filled lazily — only products that are actually requested
    are ever cached.

    What is NOT cached
    ------------------
    "Product not found". ProductNotFoundError propagates untouched, so a
    product created a second later is visible immediately, and nobody can
    fill Redis by requesting millions of made-up IDs.
    """

    def __init__(self, inner: ProductRepository, cache: ProductCache) -> None:
        self._inner = inner
        self._cache = cache

    async def get_by_id(self, product_id: str) -> Product:
        cached = await self._cache.get(product_id)
        if cached is not None:
            logger.info("Cache HIT | product_id=%s", product_id)
            return cached

        logger.info("Cache MISS | product_id=%s", product_id)
        product = await self._inner.get_by_id(product_id)   # may raise ProductNotFoundError

        if await self._cache.set(product):
            logger.info("Product cached | product_id=%s", product_id)
        return product

    async def list_all(self) -> list[Product]:
        # Not cached: one cheap query, and a cached list would go stale whenever a product is added
        return await self._inner.list_all()
