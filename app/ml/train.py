"""
Train, evaluate, version and track the demand model — one command.

    python -m app.ml.train                 # next version (v1, v2, ...)
    python -m app.ml.train --version v7    # a specific new version

What one run does
-----------------
1. Load data/processed/daily_sales.csv and hash it (the dataset version).
2. Forecasting evaluation (Phase 1): walk-forward validation picks the model,
   the held-out test set measures it once.
3. Anomaly evaluation (Phase 2): thresholds on validation, scored on test.
   (+ Phase 4: feature-importance and SHAP reports for the final model)
4. Refit the SELECTED forecaster on ALL the data. The test score from step 2
   is the honest estimate of how it performs; using every day for the final
   fit means the deployed model also knows the most recent weeks.
5. Save it as a new immutable version: models/<version>/model.joblib + metadata.json.
6. Log everything to MLflow: parameters, metrics, feature list, dataset hash,
   reports, plots and the model files.

Browse runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
Serve it:     set MODEL_VERSION=<version> (the API loads that folder).
"""
import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from sklearn.pipeline import Pipeline

from app.config.logging import setup_logging
from app.config.settings import PROJECT_ROOT, resolve_path, settings
from app.ml.anomaly import evaluate as anomaly_evaluation
from app.ml.data.prepare import sha256_of
from app.ml.explain_report import build_reports as build_explainability_reports
from app.ml.features import FEATURES, MIN_HISTORY_DAYS, TARGET, build_features, load_daily_sales
from app.ml.forecasting import evaluate as forecast_evaluation
from app.ml.forecasting.models import RANDOM_STATE, make_candidates
from app.ml.model_store import library_versions, models_root, next_version, save_bundle

logger = logging.getLogger(__name__)


def tracking_uri() -> str:
    """sqlite:///mlflow.db is relative to the current directory — pin it to the project root."""
    uri = settings.mlflow_tracking_uri
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + (PROJECT_ROOT / uri[len(prefix):]).as_posix()
    return uri


def metric_key(*parts: str) -> str:
    """MLflow metric names allow only letters, digits and _ - . / and spaces; keep them short and stable."""
    return "_".join(re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_") for part in parts)


def selected_model_params(model) -> dict:
    """
    The selected model's own hyperparameters (e.g. Ridge alpha, LightGBM n_estimators).

    Only the final estimator: a pipeline's deep params also list every
    preprocessing internal (memory, drop, ...), which is noise in a run log.
    """
    estimator = getattr(model, "regressor", model)
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]
    if not hasattr(estimator, "get_params"):
        return {"column": model.column}            # a baseline: just the column it copies
    params = estimator.get_params(deep=False)
    return {key: value for key, value in params.items() if isinstance(value, (int, float, str, bool)) or value is None}


def train(version: str | None = None) -> dict:
    data_path = resolve_path(settings.processed_data_dir) / "daily_sales.csv"
    daily = load_daily_sales(data_path)
    dataset_sha256 = sha256_of(data_path)
    recorded = json.loads((data_path.parent / "metadata.json").read_text(encoding="utf-8"))["daily_sales_sha256"]
    if dataset_sha256 != recorded:
        logger.warning("daily_sales.csv does not match the hash in metadata.json - was it edited by hand?")

    version = version or next_version()
    reports_dir = resolve_path(settings.reports_dir)

    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(settings.mlflow_experiment) is None:
        mlflow.create_experiment(
            settings.mlflow_experiment,
            artifact_location=resolve_path(settings.mlflow_artifact_dir).as_uri(),
        )
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name=f"train-{version}") as run:
        # ---- 1-3. Evaluate ---------------------------------------------------
        logger.info("Evaluating forecasters (walk-forward validation + held-out test)")
        forecast = forecast_evaluation.run_evaluation(daily)
        logger.info("Evaluating anomaly detectors (injected, labelled anomalies)")
        anomalies = anomaly_evaluation.run_evaluation(daily)
        report_files = (forecast_evaluation.write_reports(forecast, reports_dir)
                        + anomaly_evaluation.write_reports(anomalies, reports_dir))

        best = forecast.best_model_name
        best_detector = anomalies.results["f1"].idxmax()

        # ---- 4. Final fit on all data ---------------------------------------
        features = build_features(daily)
        final_model = make_candidates()[best].fit(features[FEATURES], features[TARGET])

        # Phase 4: global explainability of the deployed model (+ LightGBM SHAP for comparison)
        report_files += build_explainability_reports(final_model, daily, reports_dir)

        # ---- 5. Version it ---------------------------------------------------
        cv = forecast.cv_summary.loc[best]
        test = forecast.test_scores.loc[best]
        detector = anomalies.results.loc[best_detector]
        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": best,
            "target": TARGET,
            "target_transform": "log1p" if hasattr(final_model, "regressor") else None,
            "forecast_horizon": "next day",
            "features": FEATURES,
            "min_history_days": MIN_HISTORY_DAYS,
            "selection": {"metric": forecast_evaluation.SELECTION_METRIC, **forecast.split_info},
            "dataset": {
                "path": data_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": dataset_sha256,
                "rows": len(daily),
                "products": int(daily["product_id"].nunique()),
                "date_from": daily["sale_date"].min().date().isoformat(),
                "date_to": daily["sale_date"].max().date().isoformat(),
            },
            "training_rows": len(features),
            "metrics": {
                "validation": {name: float(cv[(name, "mean")]) for name in ("mae", "rmse", "wape")},
                "test": {name: float(test[name]) for name in ("mae", "rmse", "wape")},
                "test_wape_all_candidates": forecast.test_scores["wape"].round(4).to_dict(),
            },
            "anomaly_detector": {
                "name": best_detector,
                "threshold": float(detector["threshold"]),
                "test": {name: float(detector[name]) for name in ("precision", "recall", "f1")},
            },
            "hyperparameters": selected_model_params(final_model),
            "library_versions": library_versions(),
            "mlflow_run_id": run.info.run_id,
        }
        model_folder = save_bundle(final_model, metadata, version)

        # ---- 6. Track it -----------------------------------------------------
        mlflow.set_tags({
            "model_version": version,
            "selected_model": best,
            "dataset_sha256": dataset_sha256,
        })
        mlflow.log_params({
            "selected_model": best,
            "selection_metric": forecast_evaluation.SELECTION_METRIC,
            "test_days": forecast_evaluation.TEST_DAYS,
            "cv_folds": forecast_evaluation.CV_FOLDS,
            "cv_validation_days": forecast_evaluation.CV_VALIDATION_DAYS,
            "target_transform": metadata["target_transform"],
            "random_state": RANDOM_STATE,
            "n_features": len(FEATURES),
            "dataset_sha256": dataset_sha256,
            "training_rows": len(features),
            "anomaly_detector": best_detector,
            "anomaly_threshold": round(float(detector["threshold"]), 4),
            **{f"model__{key}": value for key, value in metadata["hyperparameters"].items()},
        })
        mlflow.log_metrics({
            **{f"validation_{name}": value for name, value in metadata["metrics"]["validation"].items()},
            **{f"test_{name}": value for name, value in metadata["metrics"]["test"].items()},
            **{metric_key("test_wape", name): float(row["wape"]) for name, row in forecast.test_scores.iterrows()},
            **{metric_key("validation_wape", name): float(row[("wape", "mean")]) for name, row in forecast.cv_summary.iterrows()},
            **{f"anomaly_{name}": value for name, value in metadata["anomaly_detector"]["test"].items()},
        })
        mlflow.log_dict({"features": FEATURES}, "features.json")
        for report in report_files:
            mlflow.log_artifact(str(report), artifact_path="reports")
        mlflow.log_artifacts(str(model_folder), artifact_path=f"model/{version}")

    logger.info(
        "Trained %s | model=%s | test WAPE %.3f | anomaly=%s (F1 %.3f) | saved %s | MLflow run %s",
        version, best, test["wape"], best_detector, detector["f1"], model_folder, run.info.run_id,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate, version and track the SmartPrice demand model.")
    parser.add_argument("--version", help="new model version name (default: next vN)")
    args = parser.parse_args()
    setup_logging()
    metadata = train(args.version)
    print(f"\nModel {metadata['version']} saved to {models_root() / metadata['version']}")
    print(f"Serve it with MODEL_VERSION={metadata['version']}")


if __name__ == "__main__":
    main()
