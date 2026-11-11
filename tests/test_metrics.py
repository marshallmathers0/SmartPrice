"""Phase 1 — metric functions against small examples worked out by hand."""
import math

import pytest

from app.ml.metrics import mae, rmse, score, wape

ACTUAL = [1, 2, 3]
PREDICTED = [2, 2, 1]          # errors: 1, 0, 2


def test_mae():
    assert mae(ACTUAL, PREDICTED) == pytest.approx((1 + 0 + 2) / 3)


def test_rmse():
    assert rmse(ACTUAL, PREDICTED) == pytest.approx(math.sqrt((1 + 0 + 4) / 3))


def test_wape():
    # total absolute error 3 / total demand 6
    assert wape(ACTUAL, PREDICTED) == pytest.approx(0.5)


def test_rmse_punishes_one_big_miss_more_than_mae():
    actual = [10, 10, 10, 10]
    spread_out = [11, 11, 11, 11]     # four misses of 1
    one_big = [10, 10, 10, 14]        # one miss of 4

    assert mae(actual, spread_out) == mae(actual, one_big) == 1.0
    assert rmse(actual, one_big) > rmse(actual, spread_out)


def test_wape_works_with_zero_demand_days():
    # MAPE would divide by zero on the first two days; WAPE divides once by total demand (10)
    assert wape([0, 0, 10], [1, 0, 8]) == pytest.approx((1 + 0 + 2) / 10)


def test_wape_is_undefined_when_nothing_was_sold():
    assert math.isnan(wape([0, 0], [1, 2]))


def test_perfect_forecast_scores_zero():
    assert score(ACTUAL, ACTUAL) == {"mae": 0.0, "rmse": 0.0, "wape": 0.0}


@pytest.mark.parametrize("actual, predicted", [([1, 2], [1]), ([], [])])
def test_invalid_input_is_rejected(actual, predicted):
    with pytest.raises(ValueError):
        mae(actual, predicted)
