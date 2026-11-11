"""Day 7 — ProductCache against the real Redis server."""
from decimal import Decimal

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from app.cache.product_cache import ProductCache
from app.config.settings import settings
from app.models.product import Product

pytestmark = pytest.mark.redis

TEST_ID = "TEST-REDIS-1"


@pytest_asyncio.fixture
async def redis_client():
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    await client.delete(ProductCache.key(TEST_ID))
    yield client
    await client.delete(ProductCache.key(TEST_ID))    # never leave test keys behind
    await client.aclose()


def demo_product() -> Product:
    return Product(TEST_ID, "Test Lamp", Decimal("12.34"), Decimal("56.78"), inventory=5, demand_score=Decimal("0.33"))


@pytest.mark.asyncio
async def test_round_trip_through_real_redis(redis_client):
    cache = ProductCache(redis_client, ttl_seconds=300)

    await cache.set(demo_product())

    assert await cache.get(TEST_ID) == demo_product()


@pytest.mark.asyncio
async def test_real_ttl_is_set(redis_client):
    await ProductCache(redis_client, ttl_seconds=300).set(demo_product())

    ttl = await redis_client.ttl(ProductCache.key(TEST_ID))

    assert 0 < ttl <= 300


@pytest.mark.asyncio
async def test_invalidate_deletes_real_key(redis_client):
    cache = ProductCache(redis_client, ttl_seconds=300)
    await cache.set(demo_product())

    await cache.invalidate(TEST_ID)

    assert await redis_client.exists(ProductCache.key(TEST_ID)) == 0


@pytest.mark.asyncio
async def test_unreachable_redis_degrades_to_a_miss():
    # Nothing listens on port 1 — this is what a Redis outage looks like
    dead = Redis.from_url("redis://127.0.0.1:1/0", decode_responses=True, socket_connect_timeout=0.5)
    cache = ProductCache(dead, ttl_seconds=300)
    try:
        assert await cache.get(TEST_ID) is None
        await cache.set(demo_product())     # must not raise
    finally:
        await dead.aclose()
