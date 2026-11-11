"""Phase 1 — the evaluation protocol: chronological splits, walk-forward folds, honest selection."""
import numpy as np
import pandas as pd
import pytest

from app.config.settings import resolve_path, settings
from app.ml.features import TARGET, build_features, load_daily_sales
from app.ml.forecasting.evaluate import (
    TEST_DAYS,
    comparison_markdown,
    run_evaluation,
    split_by_date,
    walk_forward_folds,
)
from app.ml.forecasting.models import ColumnBaseline, make_candidates, predict_demand

DAILY_SALES = resolve_path(settings.processed_data_dir) / "daily_sales.csv"
needs_data = pytest.mark.skipif(not DAILY_SALES.exists(), reason="processed dataset not built")


@pytest.fixture(scope="module")
def features():
    return build_features(load_daily_sales(DAILY_SALES))


@needs_data
def test_test_set_is_the_last_days_and_never_overlaps(features):
    development, test = split_by_date(features)

    assert test["sale_date"].nunique() == TEST_DAYS
    assert test["sale_date"].max() == features["sale_date"].max()
    assert development["sale_date"].max() < test["sale_date"].min()


@needs_data
def test_walk_forward_folds_always_train_on_the_past(features):
    development, _ = split_by_date(features)

    for train_mask, val_mask in walk_forward_folds(development):
        train, val = development[train_mask], development[val_mask]
        assert train["sale_date"].max() < val["sale_date"].min()
        assert not (train_mask & val_mask).any()
        # whole days only: every validation date has all 20 products
        assert (val.groupby("sale_date").size() == 20).all()


@needs_data
def test_folds_expand_forward(features):
    development, _ = split_by_date(features)
    starts = [development[val]["sale_date"].min() for _, val in walk_forward_folds(development)]
    train_sizes = [train.sum() for train, _ in walk_forward_folds(development)]

    assert starts == sorted(starts)
    assert train_sizes == sorted(train_sizes)


def test_column_baseline_returns_the_column():
    frame = pd.DataFrame({"lag_1": [3.0, 0.0, 7.0]})

    assert list(ColumnBaseline("lag_1").fit(frame, None).predict(frame)) == [3.0, 0.0, 7.0]


def test_candidates_cover_baselines_and_learned_models():
    assert list(make_candidates()) == [
        "Naive (yesterday)",
        "Seasonal naive (same weekday last week)",
        "Moving average (last 7 days)",
        "Ridge regression",
        "LightGBM",
    ]


@needs_data
def test_predictions_are_never_negative(features):
    development, test = split_by_date(features)
    model = make_candidates()["Ridge regression"].fit(development.drop(columns=TARGET), development[TARGET])

    assert (predict_demand(model, test) >= 0).all()


@needs_data
def test_selection_uses_validation_only_and_report_is_complete():
    result = run_evaluation(load_daily_sales(DAILY_SALES))

    # The selected model is exactly the lowest mean VALIDATION WAPE — whatever the test set says
    assert result.best_model_name == result.cv_summary[("wape", "mean")].idxmin()
    assert set(result.test_scores.index) == set(make_candidates())
    assert np.isfinite(result.test_scores.to_numpy()).all()

    report = comparison_markdown(result)
    assert "(selected)" in report and "Walk-forward validation" in report
