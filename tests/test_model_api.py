"""Phase 3 — the API serves the model version named in config."""
import pytest
from fastapi.testclient import TestClient

from app.config.settings import resolve_path, settings
from app.main import create_app

DEPLOYED = resolve_path(settings.model_dir) / settings.model_version
needs_model = pytest.mark.skipif(not (DEPLOYED / "metadata.json").exists(), reason="no trained model")

URL = "/api/v1/models/current"


@needs_model
def test_current_model_describes_what_is_served(client):
    response = client.get(URL)
    body = response.json()

    assert response.status_code == 200
    assert body["version"] == settings.model_version
    assert body["features"][0] == "product_id"
    assert len(body["dataset_sha256"]) == 64
    assert 0 < body["test"]["wape"] < 2
    assert 0 <= body["anomaly_detector"]["f1"] <= 1


def test_missing_model_version_gives_503_but_pricing_still_works(app, monkeypatch):
    monkeypatch.setattr(settings, "model_version", "v999")

    with TestClient(app) as client:
        model = client.get(URL)
        pricing = client.post("/api/v1/pricing/", json={"product_id": "PROD-001"})

    assert model.status_code == 503
    assert model.json() == {"detail": "Forecast model 'v999' is not available", "code": "model_unavailable"}
    assert pricing.status_code == 200


@needs_model
def test_health_reports_the_loaded_model(client, monkeypatch):
    from app.cache import redis_client
    from app.database import session as database

    async def up() -> bool:
        return True

    monkeypatch.setattr(database, "ping_database", up)
    monkeypatch.setattr(redis_client, "ping_redis", up)

    assert client.get("/health").json()["model"] == settings.model_version


def test_health_reports_a_missing_model(app, monkeypatch):
    monkeypatch.setattr(settings, "model_version", "v999")

    with TestClient(app) as client:
        assert client.get("/health").json()["model"] == "unavailable"
