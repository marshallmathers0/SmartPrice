"""
Phase 5 — API regression: every endpoint still exists, answers, and keeps its contract.

Clients depend on paths and response fields. These tests fail if an endpoint
disappears or is renamed, or if a field a client reads is removed or renamed.
ADDING an endpoint or a field is fine (the contract checks "at least these");
changing the inventory below should be a conscious, reviewed decision.
"""
import pytest

from app.main import create_app

PRODUCT = "85123A"

EXPECTED_ROUTES = {
    ("GET", "/health"),
    ("GET", "/api/v1/pricing/"),
    ("POST", "/api/v1/pricing/"),
    ("GET", "/api/v1/models/current"),
    ("GET", "/api/v1/forecast/{product_id}"),
    ("GET", "/api/v1/explain/{product_id}"),
    ("GET", "/api/v1/products/"),            # Phase 6: the demo's product picker
}

PRICING_FIELDS = {
    "product_id", "product_name", "cost_price", "competitor_price", "inventory", "demand_score",
    "suggested_price", "strategy_used", "margin_percent", "guardrails_applied", "message",
}
FORECAST_FIELDS = {
    "product_id", "forecast_date", "history_end", "predicted_units", "typical_units", "recent_actual_units",
    "demand_score", "model_version", "base_units", "top_factors", "recent_anomalies",
}
FACTOR_FIELDS = {"feature", "description", "effect", "multiplier", "contribution"}
EXPLAIN_FIELDS = {"product_id", "product_name", "summary", "forecast", "price"}
PRICE_EXPLANATION_FIELDS = {
    "strategy", "strategy_description", "competitor_price", "cost_price", "demand_score",
    "demand_factor", "suggested_price", "margin_percent", "guardrails_applied",
}
MODEL_FIELDS = {
    "version", "model_name", "created_at", "target", "forecast_horizon", "features", "dataset_sha256",
    "trained_from", "trained_to", "training_rows", "validation", "test", "anomaly_detector", "mlflow_run_id",
}
PRODUCT_FIELDS = {"product_id", "name", "cost_price", "competitor_price"}
HEALTH_FIELDS = {"status", "app_name", "environment", "database", "cache", "model"}
ERROR_FIELDS = {"detail", "code"}


def test_route_inventory_is_unchanged():
    # The OpenAPI schema is the public contract clients see (and it does not depend on
    # FastAPI internals: newer versions wrap included routers in private classes)
    schema = create_app().openapi()
    routes = {(method.upper(), path) for path, operations in schema["paths"].items() for method in operations}

    assert routes == EXPECTED_ROUTES


def test_every_endpoint_is_documented_in_openapi(ml_client):
    paths = ml_client.get("/openapi.json").json()["paths"]

    for method, path in EXPECTED_ROUTES:
        assert method.lower() in paths[path], f"{method} {path} missing from OpenAPI"


@pytest.mark.parametrize("strategy", ["competitive", "cost_plus", "dynamic", "penetration", "demand_based"])
def test_pricing_contract_for_every_strategy(ml_client, strategy):
    response = ml_client.post("/api/v1/pricing/", json={"product_id": PRODUCT, "strategy": strategy})

    assert response.status_code == 200
    assert PRICING_FIELDS <= set(response.json())


def test_pricing_list_still_answers(ml_client):
    response = ml_client.get("/api/v1/pricing/")

    assert response.status_code == 200 and response.json() == []


def test_forecast_contract(ml_client):
    body = ml_client.get(f"/api/v1/forecast/{PRODUCT}").json()

    assert FORECAST_FIELDS <= set(body)
    assert all(FACTOR_FIELDS <= set(factor) for factor in body["top_factors"])


def test_explain_contract(ml_client):
    body = ml_client.get(f"/api/v1/explain/{PRODUCT}").json()

    assert EXPLAIN_FIELDS <= set(body)
    assert FORECAST_FIELDS <= set(body["forecast"])
    assert PRICE_EXPLANATION_FIELDS <= set(body["price"])
    assert all(isinstance(line, str) and line for line in body["summary"])


def test_model_contract(ml_client):
    assert MODEL_FIELDS <= set(ml_client.get("/api/v1/models/current").json())


def test_health_contract(ml_client, monkeypatch):
    from app.cache import redis_client
    from app.database import session as database

    async def up() -> bool:
        return True

    monkeypatch.setattr(database, "ping_database", up)
    monkeypatch.setattr(redis_client, "ping_redis", up)

    assert HEALTH_FIELDS <= set(ml_client.get("/health").json())


@pytest.mark.parametrize(
    "method, path, body, status",
    [
        ("post", "/api/v1/pricing/", {"product_id": "NOPE"}, 404),
        ("post", "/api/v1/pricing/", {"product_id": PRODUCT, "strategy": "holiday"}, 422),
        ("get", "/api/v1/forecast/NOPE", None, 404),
        ("get", "/api/v1/forecast/SKU-12345", None, 422),
        ("get", "/api/v1/explain/NOPE", None, 404),
    ],
)
def test_error_contract(ml_client, method, path, body, status):
    response = getattr(ml_client, method)(path, **({"json": body} if body else {}))

    assert response.status_code == status
    if status != 422 or "strategy" not in (body or {}):           # FastAPI's own validation errors keep their shape
        assert set(response.json()) == ERROR_FIELDS


def test_products_contract(ml_client):
    products = ml_client.get("/api/v1/products/").json()

    assert len(products) == 23                                    # 20 real + 3 demo
    assert all(PRODUCT_FIELDS <= set(p) for p in products)
    assert [p["product_id"] for p in products] == sorted(p["product_id"] for p in products)
