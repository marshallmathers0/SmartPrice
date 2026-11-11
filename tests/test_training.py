"""
Phase 3 — the one-command training pipeline, end to end.

Everything is redirected into a temporary folder (models, reports, MLflow
database and artifacts), so the real project files are never touched.
"""
import pytest

from app.config.settings import resolve_path, settings
from app.ml.model_store import load_bundle

DAILY_SALES = resolve_path(settings.processed_data_dir) / "daily_sales.csv"

pytestmark = pytest.mark.skipif(not DAILY_SALES.exists(), reason="processed dataset not built")


@pytest.fixture
def isolated_training(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "reports_dir", str(tmp_path / "reports"))
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(settings, "mlflow_artifact_dir", str(tmp_path / "mlartifacts"))
    monkeypatch.setattr(settings, "mlflow_experiment", "test-experiment")
    return tmp_path


def test_training_versions_tracks_and_reports(isolated_training):
    import mlflow
    from mlflow.tracking import MlflowClient

    from app.ml.train import tracking_uri, train

    first = train()
    second = train()

    # ---- versioned, immutable artifacts --------------------------------
    assert (first["version"], second["version"]) == ("v1", "v2")
    bundle = load_bundle("v2", root=isolated_training / "models")
    assert bundle.metadata["model_name"] == first["model_name"]          # same data -> same selection
    assert bundle.metadata["dataset"]["sha256"] == first["dataset"]["sha256"]

    # ---- reports ---------------------------------------------------------
    reports = {p.name for p in (isolated_training / "reports").iterdir()}
    assert reports == {"forecast_comparison.md", "forecast_actual_vs_predicted.png",
                       "anomaly_comparison.md", "anomaly_example.png",
                       "feature_importance.png", "shap_summary_lightgbm.png", "explainability.md"}

    # ---- MLflow run: params, metrics, dataset hash, artifacts -------------
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    run = client.get_run(second["mlflow_run_id"])
    assert run.data.tags["dataset_sha256"] == second["dataset"]["sha256"]
    assert run.data.params["selected_model"] == second["model_name"]
    assert run.data.metrics["test_wape"] == pytest.approx(second["metrics"]["test"]["wape"])

    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"features.json", "reports", "model"} <= artifacts
    model_files = {a.path for a in client.list_artifacts(run.info.run_id, "model/v2")}
    assert model_files == {"model/v2/model.joblib", "model/v2/metadata.json"}
