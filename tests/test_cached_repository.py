"""Day 7 — CachedProductRepository: Decorator + Cache-Aside, with no Redis or database."""
import logging

import pytest

from app.cache.product_cache import ProductCache
from app.exceptions.pricing import ProductNotFoundError
from app.models.pricing import PricingStrategy
from app.models.product import Product
from app.repositories.base import ProductRepository
from app.repositories.cached import CachedProductRepository
from app.repositories.mock import MockProductRepository
from app.services.pricing import PricingService


class CountingRepository(ProductRepository):
    """Stands in for PostgreSQL and counts how often it is asked."""

    def __init__(self, products: list[Product]) -> None:
        self._inner = MockProductRepository(products)
        self.calls = 0

    async def get_by_id(self, product_id: str) -> Product:
        self.calls += 1
        return await self._inner.get_by_id(product_id)

    async def list_all(self) -> list[Product]:
        return await self._inner.list_all()


@pytest.fixture
def database(make_product) -> CountingRepository:
    return CountingRepository([make_product(product_id="PROD-001")])


@pytest.fixture
def cache(fake_redis) -> ProductCache:
    return ProductCache(fake_redis, ttl_seconds=300)


@pytest.mark.asyncio
async def test_miss_loads_from_inner_repository_and_caches(database, cache, fake_redis, make_product):
    repository = CachedProductRepository(database, cache)

    product = await repository.get_by_id("PROD-001")

    assert product == make_product(product_id="PROD-001")
    assert database.calls == 1
    assert "product:PROD-001" in fake_redis.store


@pytest.mark.asyncio
async def test_hit_never_touches_inner_repository(database, cache):
    repository = CachedProductRepository(database, cache)

    first = await repository.get_by_id("PROD-001")
    second = await repository.get_by_id("PROD-001")
    third = await repository.get_by_id("PROD-001")

    assert first == second == third
    assert database.calls == 1          # only the first request reached the "database"


@pytest.mark.asyncio
async def test_not_found_is_raised_and_not_cached(database, cache, fake_redis):
    repository = CachedProductRepository(database, cache)

    for _ in range(2):
        with pytest.raises(ProductNotFoundError):
            await repository.get_by_id("SKU-99999")

    assert database.calls == 2          # every lookup re-checks the source of truth
    assert fake_redis.store == {}


@pytest.mark.asyncio
async def test_redis_down_falls_back_to_inner_repository(database, cache, fake_redis, make_product):
    fake_redis.fail = True
    repository = CachedProductRepository(database, cache)

    product = await repository.get_by_id("PROD-001")

    assert product == make_product(product_id="PROD-001")
    assert database.calls == 1


@pytest.mark.asyncio
async def test_cache_events_are_logged(database, cache, caplog):
    repository = CachedProductRepository(database, cache)

    with caplog.at_level(logging.INFO, logger="app.repositories.cached"):
        await repository.get_by_id("PROD-001")
        await repository.get_by_id("PROD-001")

    assert [record.getMessage() for record in caplog.records] == [
        "Cache MISS | product_id=PROD-001",
        "Product cached | product_id=PROD-001",
        "Cache HIT | product_id=PROD-001",
    ]


@pytest.mark.asyncio
async def test_pricing_is_identical_from_cache_and_from_source(cache, make_product):
    # DEVLOG Day 4 floor scenario — the cached product must price exactly the same
    database = CountingRepository([make_product("950.00", "999.00", product_id="PROD-005")])
    service = PricingService(CachedProductRepository(database, cache))

    from_database = await service.calculate_price("PROD-005", PricingStrategy.COMPETITIVE)
    from_cache = await service.calculate_price("PROD-005", PricingStrategy.COMPETITIVE)

    assert database.calls == 1
    assert from_cache == from_database
    assert from_cache.suggested_price == from_database.suggested_price


@pytest.mark.asyncio
async def test_product_cached_is_not_logged_when_redis_is_down(database, cache, fake_redis, caplog):
    fake_redis.fail = True
    repository = CachedProductRepository(database, cache)

    with caplog.at_level(logging.INFO, logger="app.repositories.cached"):
        await repository.get_by_id("PROD-001")

    decorator_events = [r.getMessage() for r in caplog.records if r.name == "app.repositories.cached"]
    assert decorator_events == ["Cache MISS | product_id=PROD-001"]     # no false "Product cached"
