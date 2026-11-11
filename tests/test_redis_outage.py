"""
Phase 5 — Redis down: every endpoint still works, just without the cache.

Redis is pointed at a port nothing listens on — exactly what an outage looks
like to the app. The real dependency graph is used (Redis cache, demand-aware
repository, forecast service, pricing); only PostgreSQL is replaced by
in-memory data so the test needs no database.
"""
import logging
from typing import Annotated

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from app.api.dependencies import get_base_product_repository, get_product_cache, get_sales_history_repository
from app.cache.product_cache import ProductCache
from app.cache.redis_client import get_redis_client
from app.config.settings import settings
from app.main import create_app
from app.repositories.cached import CachedProductRepository
from app.repositories.mock import SEED_PRODUCTS, InMemorySalesHistoryRepository, MockProductRepository

DEAD_REDIS = "redis://127.0.0.1:1/0"
PRODUCT = "85123A"


@pytest.fixture
def outage_client(monkeypatch, dataset_products, dataset_sales, deployed_bundle):
    monkeypatch.setattr(settings, "redis_url", DEAD_REDIS)
    get_redis_client.cache_clear()
    get_product_cache.cache_clear()

    products = MockProductRepository([*dataset_products, *SEED_PRODUCTS])
    sales = InMemorySalesHistoryRepository(dataset_sales)

    def products_behind_the_real_cache(cache: Annotated[ProductCache, Depends(get_product_cache)]):
        # The real Day 7 cache decorator and the real ProductCache — only the source is in memory
        return CachedProductRepository(inner=products, cache=cache)

    app = create_app()
    app.dependency_overrides[get_base_product_repository] = products_behind_the_real_cache
    app.dependency_overrides[get_sales_history_repository] = lambda: sales
    with TestClient(app) as client:
        yield client
    get_product_cache.cache_clear()


def test_every_endpoint_works_without_redis(outage_client):
    responses = {
        "pricing": outage_client.post("/api/v1/pricing/", json={"product_id": PRODUCT, "strategy": "demand_based"}),
        "forecast": outage_client.get(f"/api/v1/forecast/{PRODUCT}"),
        "explain": outage_client.get(f"/api/v1/explain/{PRODUCT}"),
        "model": outage_client.get("/api/v1/models/current"),
    }

    assert {name: r.status_code for name, r in responses.items()} == dict.fromkeys(responses, 200)
    assert responses["pricing"].json()["demand_score"] == responses["forecast"].json()["demand_score"]


def test_answers_are_identical_with_and_without_redis(outage_client, ml_client):
    body = {"product_id": PRODUCT, "strategy": "demand_based"}

    assert outage_client.post("/api/v1/pricing/", json=body).json() == ml_client.post("/api/v1/pricing/", json=body).json()


def test_health_reports_cache_down_but_app_ok(outage_client, monkeypatch):
    from app.database import session as database

    async def database_up() -> bool:
        return True

    monkeypatch.setattr(database, "ping_database", database_up)
    health = outage_client.get("/health").json()

    assert health["status"] == "ok"
    assert health["cache"] == "unavailable"


def test_outage_is_logged_once_not_per_request(outage_client, caplog):
    with caplog.at_level(logging.WARNING, logger="app.cache.product_cache"):
        for _ in range(5):
            assert outage_client.post("/api/v1/pricing/", json={"product_id": PRODUCT}).status_code == 200

    warnings = [r for r in caplog.records if "Redis unavailable" in r.getMessage()]
    assert len(warnings) == 1          # the circuit breaker skips Redis after the first failure
