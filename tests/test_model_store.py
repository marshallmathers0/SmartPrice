"""Phase 3 — the versioned model store the API loads from."""
import json
import logging

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from app.config.settings import resolve_path, settings
from app.ml.features import FEATURES, build_features, load_daily_sales
from app.ml.forecasting.models import predict_demand
from app.ml.model_store import (
    METADATA_FILE,
    ModelNotAvailableError,
    library_versions,
    load_bundle,
    next_version,
    save_bundle,
)


def tiny_model() -> LinearRegression:
    return LinearRegression().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))


def metadata(**overrides) -> dict:
    return {"version": "v1", "features": FEATURES, "library_versions": library_versions(), **overrides}


def test_save_and_load_round_trip(tmp_path):
    save_bundle(tiny_model(), metadata(), "v1", root=tmp_path)

    bundle = load_bundle("v1", root=tmp_path)

    assert bundle.version == "v1"
    assert bundle.features == FEATURES
    assert bundle.model.predict(np.array([[2.0]]))[0] == pytest.approx(2.0)


def test_next_version_counts_up(tmp_path):
    assert next_version(tmp_path) == "v1"
    (tmp_path / "v1").mkdir()
    (tmp_path / "v3").mkdir()
    (tmp_path / "notes").mkdir()            # not a version folder — ignored

    assert next_version(tmp_path) == "v4"


def test_versions_are_immutable(tmp_path):
    save_bundle(tiny_model(), metadata(), "v1", root=tmp_path)

    with pytest.raises(FileExistsError):
        save_bundle(tiny_model(), metadata(), "v1", root=tmp_path)


def test_missing_version_raises(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="v9"):
        load_bundle("v9", root=tmp_path)


def test_model_trained_on_other_features_is_refused(tmp_path):
    save_bundle(tiny_model(), metadata(features=["lag_1", "something_else"]), "v1", root=tmp_path)

    with pytest.raises(ModelNotAvailableError, match="different features"):
        load_bundle("v1", root=tmp_path)


def test_library_version_mismatch_is_logged(tmp_path, caplog):
    save_bundle(tiny_model(), metadata(library_versions={"scikit-learn": "0.0.1"}), "v1", root=tmp_path)

    with caplog.at_level(logging.WARNING, logger="app.ml.model_store"):
        load_bundle("v1", root=tmp_path)

    assert "0.0.1" in caplog.text


def test_default_version_comes_from_config(tmp_path, monkeypatch):
    save_bundle(tiny_model(), metadata(version="v2"), "v2", root=tmp_path)
    monkeypatch.setattr(settings, "model_version", "v2")

    assert load_bundle(root=tmp_path).version == "v2"


# ------------------------------------------------------------------
# The committed model (models/<MODEL_VERSION>)
# ------------------------------------------------------------------

DEPLOYED = resolve_path(settings.model_dir) / settings.model_version


@pytest.mark.skipif(not (DEPLOYED / METADATA_FILE).exists(), reason="no trained model (python -m app.ml.train)")
def test_deployed_model_loads_and_predicts_valid_demand():
    bundle = load_bundle()
    daily = load_daily_sales(resolve_path(settings.processed_data_dir) / "daily_sales.csv")
    features = build_features(daily).tail(200)

    predictions = predict_demand(bundle.model, features)

    assert len(predictions) == 200
    assert np.isfinite(predictions).all() and (predictions >= 0).all()
    meta = json.loads((DEPLOYED / METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["dataset"]["sha256"] == json.loads(
        (resolve_path(settings.processed_data_dir) / "metadata.json").read_text(encoding="utf-8")
    )["daily_sales_sha256"]


# ------------------------------------------------------------------
# Phase 5 — a damaged model must never crash the API
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "broken_file, content",
    [
        ("model.joblib", b"not a pickle"),           # corrupted / truncated upload
        ("model.joblib", b""),                       # empty file
        (METADATA_FILE, b"{ this is not json"),      # damaged metadata
    ],
)
def test_damaged_model_files_raise_model_not_available(tmp_path, broken_file, content):
    folder = save_bundle(tiny_model(), metadata(), "v1", root=tmp_path)
    (folder / broken_file).write_bytes(content)

    with pytest.raises(ModelNotAvailableError, match="v1"):
        load_bundle("v1", root=tmp_path)


def test_api_starts_with_a_corrupted_model(tmp_path, monkeypatch, app):
    from fastapi.testclient import TestClient

    folder = save_bundle(tiny_model(), metadata(), "v1", root=tmp_path)
    (folder / "model.joblib").write_bytes(b"not a pickle")
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "model_version", "v1")

    with TestClient(app) as client:                                   # before the fix: KeyError at startup
        assert client.get("/api/v1/models/current").status_code == 503
        assert client.post("/api/v1/pricing/", json={"product_id": "PROD-001"}).status_code == 200
