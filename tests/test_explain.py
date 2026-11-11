"""Phase 4 — exact per-feature contributions and their plain-language descriptions."""
import numpy as np
import pytest

from app.config.settings import resolve_path, settings
from app.ml.explain import (
    ExplanationNotSupportedError,
    contributions,
    describe,
    effect_text,
)
from app.ml.features import FEATURES, TARGET, build_features, load_daily_sales
from app.ml.forecasting.models import ColumnBaseline, make_lightgbm, make_ridge

DAILY_SALES = resolve_path(settings.processed_data_dir) / "daily_sales.csv"
pytestmark = pytest.mark.skipif(not DAILY_SALES.exists(), reason="processed dataset not built")


@pytest.fixture(scope="module")
def features():
    return build_features(load_daily_sales(DAILY_SALES))


@pytest.fixture(scope="module")
def ridge(features):
    return make_ridge().fit(features[FEATURES], features[TARGET])


@pytest.mark.parametrize("make", [make_ridge, make_lightgbm], ids=["ridge", "lightgbm"])
def test_contributions_add_up_exactly_to_the_prediction(features, make):
    model = make().fit(features[FEATURES], features[TARGET])
    X = features[FEATURES].tail(300)

    result = contributions(model, X)
    raw_log_prediction = model.regressor_.predict(X)

    assert list(result.values.columns) == FEATURES
    np.testing.assert_allclose(result.base + result.values.sum(axis=1), raw_log_prediction, atol=1e-9)


def test_ridge_category_contribution_is_its_coefficient(features, ridge):
    row = features[FEATURES].iloc[[-1]]
    pipeline = ridge.regressor_
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    coefficient = pipeline.named_steps["ridge"].coef_[names.index(f"categories__product_id_{row['product_id'].iat[0]}")]

    assert contributions(ridge, row).values["product_id"].iat[0] == pytest.approx(coefficient)


def test_ridge_numeric_contribution_is_coefficient_times_scaled_value(features, ridge):
    row = features[FEATURES].iloc[[-1]]
    pipeline = ridge.regressor_
    preprocess = pipeline.named_steps["preprocess"]
    names = list(preprocess.get_feature_names_out())
    scaled = preprocess.transform(row)
    scaled = scaled.toarray() if hasattr(scaled, "toarray") else scaled
    j = names.index("numbers__roll_mean_28")

    expected = pipeline.named_steps["ridge"].coef_[j] * scaled[0, j]
    assert contributions(ridge, row).values["roll_mean_28"].iat[0] == pytest.approx(expected)


def test_unsupported_models_are_refused(features):
    with pytest.raises(ExplanationNotSupportedError):
        contributions(ColumnBaseline("lag_1"), features[FEATURES].head(1))


def test_every_feature_has_a_plain_language_description(features):
    row = features.iloc[-1]
    for feature in FEATURES:
        text = describe(feature, row)
        assert "=" not in text, f"{feature} fell through to the generic description"


@pytest.mark.parametrize("contribution, expected", [(0.30, "raises the forecast (x1.35)"), (-0.22, "lowers the forecast (x0.80)")])
def test_effect_text(contribution, expected):
    assert effect_text(contribution) == expected
