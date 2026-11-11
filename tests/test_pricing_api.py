"""End-to-end scenarios from the DEVLOG, sent through the real HTTP API."""
import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from app.cache import redis_client
from app.cache.product_cache import ProductCache
from app.config.settings import settings
from app.database import session as database
from app.database.seed import seed_products
from app.main import create_app

URL = "/api/v1/pricing/"


# ------------------------------------------------------------------
# Day 1 — skeleton endpoints
# ------------------------------------------------------------------

def stub_ping(monkeypatch, module, name: str, result: bool) -> None:
    async def ping() -> bool:
        return result
    monkeypatch.setattr(module, name, ping)


def test_health(client, monkeypatch):
    stub_ping(monkeypatch, database, "ping_database", True)
    stub_ping(monkeypatch, redis_client, "ping_redis", True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "ok"
    assert response.json()["cache"] == "ok"


def test_health_reports_degraded_when_database_is_down(client, monkeypatch):
    stub_ping(monkeypatch, database, "ping_database", False)
    stub_ping(monkeypatch, redis_client, "ping_redis", True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "unavailable"


def test_health_stays_ok_when_only_redis_is_down(client, monkeypatch):
    # Redis is a cache: losing it slows the app down but does not break it
    stub_ping(monkeypatch, database, "ping_database", True)
    stub_ping(monkeypatch, redis_client, "ping_redis", False)

    response = client.get("/health")

    assert response.json()["status"] == "ok"
    assert response.json()["cache"] == "unavailable"


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_docs_are_served(client, path):
    assert client.get(path).status_code == 200


def test_list_prices_is_empty_until_database_exists(client):
    response = client.get(URL)

    assert response.status_code == 200
    assert response.json() == []


# ------------------------------------------------------------------
# Day 2 / Day 3 — strategies through the API
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "strategy, suggested, margin",
    [
        ("competitive", "969.03", "48.40"),
        ("cost_plus", "700.00", "28.57"),
        ("dynamic", "834.52", "40.09"),
        ("penetration", "550.00", "9.09"),
    ],
)
def test_devlog_day3_table(client, strategy, suggested, margin):
    data = client.post(URL, json={"product_id": "PROD-001", "strategy": strategy}).json()

    assert data["suggested_price"] == suggested
    assert data["margin_percent"] == margin
    assert data["strategy_used"] == strategy
    assert data["guardrails_applied"] is None


def test_response_is_built_from_catalog_data(client):
    data = client.post(URL, json={"product_id": "PROD-001"}).json()

    assert data["product_id"] == "PROD-001"
    assert data["product_name"] == "Wireless Headphones"
    assert data["cost_price"] == "500.00"
    assert data["competitor_price"] == "999.00"
    assert data["inventory"] == 0               # make_product() defaults
    assert data["demand_score"] == "0.50"
    assert data["message"] == (
        f"Strategy 'competitive' applied. Suggested price is {settings.currency_symbol}29.97 below competitor price."
    )


def test_strategy_defaults_to_competitive(client):
    data = client.post(URL, json={"product_id": "PROD-001"}).json()

    assert data["strategy_used"] == "competitive"
    assert data["suggested_price"] == "969.03"


@pytest.mark.parametrize(
    "body, error",
    [
        ({}, "Field required"),
        ({"product_id": ""}, "at least 1 character"),
        ({"product_id": "PROD-001", "strategy": "holiday"}, "competitive"),
        # Day 5: prices come from the catalog — clients may no longer send them
        ({"product_id": "PROD-001", "cost_price": 500.00}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_requests_are_rejected_before_business_logic(client, body, error):
    response = client.post(URL, json=body)

    assert response.status_code == 422
    assert error in response.json()["detail"][0]["msg"]


# ------------------------------------------------------------------
# Day 4 — guardrails through the API
# ------------------------------------------------------------------

def test_devlog_day4_floor_lifecycle(client):
    data = client.post(URL, json={"product_id": "PROD-005"}).json()

    assert data["suggested_price"] == "997.50"
    assert data["margin_percent"] == "4.76"
    assert data["guardrails_applied"] == [
        "PriceFloorGuard: price raised from 969.03 to 997.50 (minimum margin of 5% enforced)"
    ]


def test_volatility_cap_without_conflict(client):
    # cost_plus 700.00 > ceiling 600.00, and 600.00 is still above the floor 525.00
    data = client.post(URL, json={"product_id": "PROD-CAP", "strategy": "cost_plus"}).json()

    assert data["suggested_price"] == "600.00"
    assert data["margin_percent"] == "16.67"
    assert data["guardrails_applied"] == [
        "VolatilityGuard: price capped from 700.00 to 600.00 (max 50% above competitor enforced)"
    ]


def test_floor_wins_when_ceiling_is_below_floor(client):
    # DEVLOG Day 4 table row: cost=900, competitor=400, cost_plus.
    # Must never sell below cost.
    data = client.post(URL, json={"product_id": "PROD-CONFLICT", "strategy": "cost_plus"}).json()

    assert data["suggested_price"] == "945.00"
    assert data["margin_percent"] == "4.76"
    assert len(data["guardrails_applied"]) == 2
    assert data["guardrails_applied"][0].startswith("VolatilityGuard: price capped from 1260.00 to 600.00")
    assert data["guardrails_applied"][1].startswith("PriceFloorGuard: price raised from 600.00 to 945.00")
    assert "never sold below cost" in data["guardrails_applied"][1]


def test_guarded_price_is_whole_cents(client):
    # 333.33 x 1.05 = 349.9965 — must be returned as a whole-cent price
    data = client.post(URL, json={"product_id": "PROD-CENTS"}).json()

    assert data["suggested_price"] == "350.00"


# ------------------------------------------------------------------
# Day 5 — repository, dependency injection, exception handling
# ------------------------------------------------------------------

def test_unknown_product_returns_404_from_global_handler(client):
    response = client.post(URL, json={"product_id": "SKU-99999"})

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Product 'SKU-99999' was not found",
        "code": "product_not_found",
    }


def test_404_is_documented_in_openapi(client):
    responses = client.get("/openapi.json").json()["paths"][URL]["post"]["responses"]

    assert "404" in responses


# ------------------------------------------------------------------
# Day 6 — the real application, end to end against PostgreSQL
# ------------------------------------------------------------------

@pytest.fixture
def real_client():
    """The production wiring — no overrides: get_db() → Postgres repository wrapped in the Redis cache."""

    async def ensure_seeded() -> None:
        # Idempotent: inserts only seed products that are missing
        async with database.SessionLocal() as session:
            await seed_products(session)
            await session.commit()
        await database.dispose_engine()

    asyncio.run(ensure_seeded())
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.mark.postgres
@pytest.mark.parametrize(
    "product_id, strategy, suggested",
    [
        ("SKU-12345", "competitive", "969.03"),   # healthy margin
        ("SKU-67890", "competitive", "997.50"),   # floor raises the price
        ("SKU-11111", "cost_plus", "945.00"),     # ceiling and floor conflict, floor wins
    ],
)
def test_seed_catalog_is_served_from_postgres(real_client, product_id, strategy, suggested):
    response = real_client.post(URL, json={"product_id": product_id, "strategy": strategy})

    assert response.status_code == 200
    assert response.json()["suggested_price"] == suggested


@pytest.mark.postgres
def test_demand_signals_are_served_from_postgres(real_client):
    data = real_client.post(URL, json={"product_id": "SKU-11111"}).json()

    assert isinstance(data["inventory"], int)
    assert 0 <= float(data["demand_score"]) <= 1


@pytest.mark.postgres
def test_unknown_product_returns_404_from_postgres(real_client):
    response = real_client.post(URL, json={"product_id": "SKU-DOES-NOT-EXIST"})

    assert response.status_code == 404
    assert response.json()["code"] == "product_not_found"


@pytest.mark.postgres
def test_health_reports_real_database(real_client):
    assert real_client.get("/health").json()["database"] == "ok"


# ------------------------------------------------------------------
# Day 7 — Redis cache, end to end
# ------------------------------------------------------------------

def redis_command(command: str, key: str):
    """Run one command against the real Redis from outside the app."""
    async def run():
        client = Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            return await getattr(client, command)(key)
        finally:
            await client.aclose()

    return asyncio.run(run())


@pytest.mark.postgres
@pytest.mark.redis
def test_second_request_is_served_from_redis(real_client, caplog):
    key = ProductCache.key("SKU-12345")
    redis_command("delete", key)                     # start cold

    with caplog.at_level(logging.INFO, logger="app.repositories.cached"):
        first = real_client.post(URL, json={"product_id": "SKU-12345"})
        cached_after_first = redis_command("exists", key)
        second = real_client.post(URL, json={"product_id": "SKU-12345"})

    events = [r.getMessage() for r in caplog.records if r.name == "app.repositories.cached"]
    assert events == [
        "Cache MISS | product_id=SKU-12345",
        "Product cached | product_id=SKU-12345",
        "Cache HIT | product_id=SKU-12345",
    ]
    assert cached_after_first == 1
    assert 0 < redis_command("ttl", key) <= settings.cache_ttl_seconds
    assert second.json() == first.json()             # identical response from cache and from PostgreSQL


@pytest.mark.postgres
@pytest.mark.redis
def test_health_reports_real_cache(real_client):
    assert real_client.get("/health").json()["cache"] == "ok"


# ------------------------------------------------------------------
# Phase 4 — forecast and explanation, end to end against PostgreSQL
# ------------------------------------------------------------------

@pytest.mark.postgres
def test_forecast_and_explain_from_postgres(real_client):
    forecast = real_client.get("/api/v1/forecast/85123A")
    explain = real_client.get("/api/v1/explain/85123A")
    priced = real_client.post(URL, json={"product_id": "85123A", "strategy": "demand_based"})

    assert forecast.status_code == explain.status_code == priced.status_code == 200
    assert forecast.json()["forecast_date"] == "2011-12-11"
    assert priced.json()["demand_score"] == forecast.json()["demand_score"]
