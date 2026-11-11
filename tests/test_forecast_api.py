"""Phase 4 — /forecast and /explain through the real dependency graph (in-memory data, deployed model)."""
from decimal import Decimal

from fastapi.testclient import TestClient

from app.config.settings import settings

PRODUCT = "85123A"


def test_forecast_endpoint(ml_client):
    response = ml_client.get(f"/api/v1/forecast/{PRODUCT}")
    body = response.json()

    assert response.status_code == 200
    assert body["forecast_date"] == "2011-12-11" and body["history_end"] == "2011-12-09"
    assert body["model_version"] == settings.model_version
    assert len(body["top_factors"]) == 5
    assert {f["effect"] for f in body["top_factors"]} <= {"raises", "lowers"}
    assert 0 <= float(body["demand_score"]) <= 1


def test_explain_endpoint_tells_the_whole_story(ml_client):
    body = ml_client.get(f"/api/v1/explain/{PRODUCT}").json()
    price = body["price"]

    assert body["product_name"] == "White Hanging Heart T-Light Holder"
    assert body["summary"][0].startswith("Next open day (Sun 11 Dec 2011)")
    assert any(line.startswith("Recommended price:") for line in body["summary"])
    # The price explanation is consistent with the forecast and the formula
    assert price["strategy"] == "demand_based"
    assert price["demand_score"] == body["forecast"]["demand_score"]
    expected = (Decimal(price["competitor_price"]) * Decimal(price["demand_factor"])).quantize(Decimal("0.01"))
    assert Decimal(price["suggested_price"]) == expected or price["guardrails_applied"]


def test_explain_another_strategy(ml_client):
    body = ml_client.get(f"/api/v1/explain/{PRODUCT}", params={"strategy": "cost_plus"}).json()

    assert body["price"]["strategy"] == "cost_plus"
    assert body["price"]["demand_factor"] is None


def test_pricing_uses_the_forecast_demand_score(ml_client):
    forecast = ml_client.get(f"/api/v1/forecast/{PRODUCT}").json()
    quote = ml_client.post("/api/v1/pricing/", json={"product_id": PRODUCT, "strategy": "demand_based"}).json()

    assert quote["demand_score"] == forecast["demand_score"]


def test_product_without_history(ml_client):
    forecast = ml_client.get("/api/v1/forecast/SKU-12345")
    pricing = ml_client.post("/api/v1/pricing/", json={"product_id": "SKU-12345", "strategy": "demand_based"})

    assert forecast.status_code == 422
    assert forecast.json()["code"] == "no_sales_history"
    assert pricing.status_code == 200 and pricing.json()["demand_score"] == "0.80"   # stored score kept


def test_unknown_product(ml_client):
    for path in ("/api/v1/forecast/NOPE", "/api/v1/explain/NOPE"):
        response = ml_client.get(path)
        assert response.status_code == 404 and response.json()["code"] == "product_not_found"


def test_without_a_model_ml_endpoints_503_but_pricing_works(ml_app, monkeypatch):
    monkeypatch.setattr(settings, "model_version", "v999")

    with TestClient(ml_app) as client:
        forecast = client.get(f"/api/v1/forecast/{PRODUCT}")
        explain = client.get(f"/api/v1/explain/{PRODUCT}")
        pricing = client.post("/api/v1/pricing/", json={"product_id": PRODUCT, "strategy": "demand_based"})

    assert forecast.status_code == explain.status_code == 503
    assert pricing.status_code == 200 and pricing.json()["demand_score"] == "0.50"   # stored placeholder
