"""
Versioned model artifacts — how a trained model gets from training to the API.

    models/
    └── v1/
        ├── model.joblib       the fitted scikit-learn pipeline
        └── metadata.json      what it is: features, dataset hash, metrics, versions, MLflow run id

Training (python -m app.ml.train) writes a NEW version folder every run; it
never overwrites an old one. The API loads exactly the version named in
config (MODEL_VERSION), so deploying a new model, or rolling back to the
previous one, is a one-line config change — not a code change.

This module is imported by the API, so it deliberately depends only on
joblib and scikit-learn — never on MLflow, LightGBM or matplotlib (training
tools stay out of the production image).
"""
import json
import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.config.settings import resolve_path, settings
from app.ml.features import FEATURES

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
_VERSION_PATTERN = re.compile(r"^v(\d+)$")


class ModelNotAvailableError(Exception):
    """The configured model version does not exist or cannot be used."""


@dataclass(frozen=True)
class ModelBundle:
    version: str
    model: Any
    metadata: dict

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]


def models_root() -> Path:
    return resolve_path(settings.model_dir)


def next_version(root: Path | None = None) -> str:
    """v1, v2, v3 ... — one more than the highest existing version."""
    root = root or models_root()
    existing = [int(m.group(1)) for p in root.glob("v*") if p.is_dir() and (m := _VERSION_PATTERN.match(p.name))]
    return f"v{max(existing, default=0) + 1}"


def library_versions() -> dict[str, str]:
    """A pickled model is only guaranteed to load under the library versions that wrote it."""
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def save_bundle(model: Any, metadata: dict, version: str, root: Path | None = None) -> Path:
    """Write model + metadata to a NEW version folder. Refuses to overwrite an existing version."""
    folder = (root or models_root()) / version
    if folder.exists():
        raise FileExistsError(f"Model version {version} already exists at {folder} - versions are immutable")
    folder.mkdir(parents=True)
    joblib.dump(model, folder / MODEL_FILE)
    (folder / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return folder


def load_bundle(version: str | None = None, root: Path | None = None) -> ModelBundle:
    """
    Load a model version (default: settings.model_version).

    Raises ModelNotAvailableError if the folder or files are missing, or if the
    model was trained on a different feature list than this code produces —
    a mismatch would silently give wrong predictions, so it fails loudly instead.
    """
    version = version or settings.model_version
    folder = (root or models_root()) / version
    model_path, metadata_path = folder / MODEL_FILE, folder / METADATA_FILE
    if not model_path.exists() or not metadata_path.exists():
        raise ModelNotAvailableError(
            f"Model version '{version}' not found in {folder.parent} (run: python -m app.ml.train)"
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelNotAvailableError(f"Model version '{version}' has unreadable metadata: {exc}") from exc
    if metadata.get("features") != FEATURES:
        raise ModelNotAvailableError(
            f"Model version '{version}' was trained on different features than the current code produces"
        )

    trained_with = metadata.get("library_versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "Model %s was trained with scikit-learn %s but %s is installed - retrain to be safe",
            version, trained_with, sklearn.__version__,
        )

    # A truncated or corrupted file raises almost anything from inside pickle
    # (KeyError, EOFError, UnpicklingError...). Any of them means "this model
    # cannot be used" — report that clearly instead of crashing app startup.
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ModelNotAvailableError(
            f"Model version '{version}' could not be loaded ({type(exc).__name__}: {exc}) - retrain or pick another version"
        ) from exc
    return ModelBundle(version=version, model=model, metadata=metadata)
