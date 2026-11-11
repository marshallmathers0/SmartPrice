from pydantic import BaseModel, Field


class SplitMetrics(BaseModel):
    mae: float = Field(description="Mean absolute error, units per product-day")
    rmse: float = Field(description="Root mean squared error, units per product-day")
    wape: float = Field(description="Weighted absolute percentage error (0.70 = off by 70% of demand)")


class AnomalyDetectorInfo(BaseModel):
    name: str
    threshold: float = Field(description="Score cut-off chosen on validation (best F1)")
    precision: float
    recall: float
    f1: float


class ModelInfoResponse(BaseModel):
    """What the API is serving right now — read from models/<version>/metadata.json."""

    version: str = Field(examples=["v1"])
    model_name: str = Field(examples=["Ridge regression"])
    created_at: str
    target: str
    forecast_horizon: str
    features: list[str]
    dataset_sha256: str = Field(description="Hash of the exact data the model was trained on")
    trained_from: str
    trained_to: str
    training_rows: int
    validation: SplitMetrics = Field(description="Mean over walk-forward validation folds")
    test: SplitMetrics = Field(description="Held-out test period, scored once")
    anomaly_detector: AnomalyDetectorInfo
    mlflow_run_id: str

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ModelInfoResponse":
        detector = metadata["anomaly_detector"]
        return cls(
            version=metadata["version"],
            model_name=metadata["model_name"],
            created_at=metadata["created_at"],
            target=metadata["target"],
            forecast_horizon=metadata["forecast_horizon"],
            features=metadata["features"],
            dataset_sha256=metadata["dataset"]["sha256"],
            trained_from=metadata["dataset"]["date_from"],
            trained_to=metadata["dataset"]["date_to"],
            training_rows=metadata["training_rows"],
            validation=SplitMetrics(**metadata["metrics"]["validation"]),
            test=SplitMetrics(**metadata["metrics"]["test"]),
            anomaly_detector=AnomalyDetectorInfo(name=detector["name"], threshold=detector["threshold"], **detector["test"]),
            mlflow_run_id=metadata["mlflow_run_id"],
        )
