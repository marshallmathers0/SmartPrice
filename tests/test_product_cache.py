"""Day 7 — ProductCache: keys, serialization, TTL, invalidation, graceful degradation."""
import json
from decimal import Decimal

import pytest

from app.cache.product_cache import ProductCache
from app.models.product import Product

TTL = 300


@pytest.fixture
def cache(fake_redis) -> ProductCache:
    return ProductCache(fake_redis, ttl_seconds=TTL, retry_seconds=30)


def demo_product() -> Product:
    return Product("SKU-12345", "Wireless Headphones", Decimal("500.00"), Decimal("999.00"),
                   inventory=120, demand_score=Decimal("0.80"))


def test_key_format():
    assert ProductCache.key("SKU-12345") == "product:SKU-12345"


@pytest.mark.asyncio
async def test_round_trip_restores_exact_types(cache):
    product = demo_product()

    await cache.set(product)
    loaded = await cache.get("SKU-12345")

    assert loaded == product
    # Same types as a product from PostgreSQL — never str or float for money
    assert type(loaded.cost_price) is Decimal
    assert type(loaded.competitor_price) is Decimal
    assert type(loaded.demand_score) is Decimal
    assert type(loaded.inventory) is int


@pytest.mark.asyncio
async def test_money_is_stored_as_exact_strings(cache, fake_redis):
    await cache.set(demo_product())

    stored = json.loads(fake_redis.store["product:SKU-12345"])

    assert stored["cost_price"] == "500.00"     # not 500.0 — Decimal precision preserved
    assert stored["demand_score"] == "0.80"


@pytest.mark.asyncio
async def test_entries_expire_after_ttl(cache, fake_redis):
    await cache.set(demo_product())

    assert fake_redis.ttls["product:SKU-12345"] == TTL


@pytest.mark.asyncio
async def test_miss_returns_none(cache):
    assert await cache.get("SKU-NOT-CACHED") is None


@pytest.mark.asyncio
async def test_invalidate_removes_entry(cache):
    await cache.set(demo_product())

    await cache.invalidate("SKU-12345")

    assert await cache.get("SKU-12345") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        json.dumps({"product_id": "SKU-12345", "name": "Old shape"}),            # missing fields
        json.dumps({**json.loads(ProductCache._serialize(demo_product())), "cost_price": "-1"}),  # invalid value
    ],
)
async def test_unreadable_entry_is_a_miss_and_is_deleted(cache, fake_redis, raw):
    fake_redis.store["product:SKU-12345"] = raw

    assert await cache.get("SKU-12345") is None
    assert "product:SKU-12345" not in fake_redis.store


@pytest.mark.asyncio
async def test_redis_failure_behaves_like_an_empty_cache(cache, fake_redis):
    fake_redis.fail = True

    assert await cache.get("SKU-12345") is None     # no exception escapes
    await cache.set(demo_product())                  # no exception escapes
    await cache.invalidate("SKU-12345")              # no exception escapes


@pytest.mark.asyncio
async def test_after_a_failure_redis_is_skipped_for_retry_window(cache, fake_redis):
    fake_redis.fail = True
    await cache.get("SKU-12345")                     # trips the circuit
    calls_after_failure = fake_redis.calls

    await cache.get("SKU-12345")
    await cache.set(demo_product())

    assert fake_redis.calls == calls_after_failure   # Redis not even contacted


@pytest.mark.asyncio
async def test_redis_is_retried_after_the_window(fake_redis):
    cache = ProductCache(fake_redis, ttl_seconds=TTL, retry_seconds=0)
    fake_redis.fail = True
    await cache.get("SKU-12345")

    fake_redis.fail = False
    await cache.set(demo_product())

    assert await cache.get("SKU-12345") == demo_product()
