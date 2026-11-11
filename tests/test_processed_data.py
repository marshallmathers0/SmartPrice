"""
Phase A — integrity of the committed processed dataset (data/processed/).

These files are what training and the database seed read, so they are
checked like code: if someone edits or regenerates them inconsistently,
the suite fails.
"""
import json

import pandas as pd
import pytest

from app.config.settings import resolve_path, settings
from app.ml.data.prepare import sha256_of

PROCESSED = resolve_path(settings.processed_data_dir)

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "daily_sales.csv").exists(),
    reason="processed dataset not built (python -m app.ml.data.prepare)",
)


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "daily_sales.csv", parse_dates=["sale_date"], dtype={"product_id": str})


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "products.csv", dtype={"product_id": str})


def test_file_matches_recorded_hash(metadata):
    # The hash is the dataset version every training run will log
    assert sha256_of(PROCESSED / "daily_sales.csv") == metadata["daily_sales_sha256"]


def test_every_product_has_every_calendar_day(daily, metadata):
    calendar = pd.date_range(metadata["date_from"], metadata["date_to"], freq="D")

    assert len(daily) == metadata["products"] * len(calendar) == metadata["daily_rows"]
    for _, series in daily.groupby("product_id"):
        assert list(series["sale_date"]) == list(calendar)


def test_demand_values_are_valid(daily):
    assert (daily["units_sold"] >= 0).all()
    assert (daily["orders"] >= 0).all()
    # A price exists exactly on the days something was sold
    assert (daily["avg_price"].isna() == (daily["units_sold"] == 0)).all()
    assert (daily["avg_price"].dropna() > 0).all()


def test_catalog_matches_sales(daily, catalog, metadata):
    assert len(catalog) == metadata["products"]
    assert set(catalog["product_id"]) == set(daily["product_id"])
    assert (catalog["cost_price"] < catalog["competitor_price"]).all()
