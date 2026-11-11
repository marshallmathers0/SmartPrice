"""Phase A — the data-preparation pipeline, on tiny hand-built frames with known answers."""
import math

import pandas as pd
import pytest

from app.ml.data.prepare import (
    COST_RATIO,
    build_catalog,
    build_daily_sales,
    clean_transactions,
    select_top_products,
)


def raw_rows(*rows) -> pd.DataFrame:
    """Rows of (invoice, code, description, quantity, timestamp, price) in the raw dataset's shape."""
    return pd.DataFrame(
        [
            {"Invoice": inv, "StockCode": code, "Description": desc, "Quantity": qty,
             "InvoiceDate": pd.Timestamp(ts), "Price": price, "Customer ID": 1.0, "Country": "United Kingdom"}
            for inv, code, desc, qty, ts, price in rows
        ]
    )


# ------------------------------------------------------------------
# clean_transactions
# ------------------------------------------------------------------

def test_clean_keeps_a_normal_sale():
    clean = clean_transactions(raw_rows(("1", "85123A", "HEART", 6, "2010-01-04 10:00", 2.95)))

    assert len(clean) == 1
    assert clean["day"].iat[0] == pd.Timestamp("2010-01-04")     # time of day removed


@pytest.mark.parametrize(
    "row, reason",
    [
        (("C2", "85123A", "HEART", -6, "2010-01-04 10:00", 2.95), "cancellation invoice"),
        (("2", "85123A", "HEART", 0, "2010-01-04 10:00", 2.95), "zero quantity"),
        (("3", "85123A", "HEART", -2, "2010-01-04 10:00", 2.95), "return"),
        (("4", "85123A", "HEART", 6, "2010-01-04 10:00", 0.0), "free item"),
        (("5", "POST", "POSTAGE", 1, "2010-01-04 10:00", 18.0), "postage line"),
        (("6", "AMAZONFEE", "AMAZON FEE", 1, "2010-01-04 10:00", 5.0), "fee line"),
        (("7", "DCGS0003", "GIFT SET", 1, "2010-01-04 10:00", 5.0), "gift-set code"),
    ],
)
def test_clean_drops_non_demand_rows(row, reason):
    assert clean_transactions(raw_rows(row)).empty, reason


def test_clean_drops_exact_duplicates_from_overlapping_sheets():
    line = ("1", "85123A", "HEART", 6, "2010-12-02 10:00", 2.95)

    assert len(clean_transactions(raw_rows(line, line))) == 1


def test_clean_merges_lowercase_codes():
    clean = clean_transactions(raw_rows(("1", "85123a", "HEART", 6, "2010-01-04 10:00", 2.95)))

    assert clean["StockCode"].iat[0] == "85123A"


# ------------------------------------------------------------------
# select_top_products
# ------------------------------------------------------------------

def test_top_products_ranks_by_days_sold_not_volume():
    clean = clean_transactions(raw_rows(
        ("1", "11111", "STEADY", 1, "2010-01-04", 1.0),
        ("2", "11111", "STEADY", 1, "2010-01-05", 1.0),
        ("3", "11111", "STEADY", 1, "2010-01-06", 1.0),
        ("4", "22222", "ONE BIG ORDER", 5000, "2010-01-04", 1.0),
    ))

    assert select_top_products(clean, n=1) == ["11111"]


def test_top_products_breaks_ties_by_code():
    clean = clean_transactions(raw_rows(
        ("1", "33333", "B", 1, "2010-01-04", 1.0),
        ("2", "11111", "A", 1, "2010-01-04", 1.0),
    ))

    assert select_top_products(clean, n=2) == ["11111", "33333"]


# ------------------------------------------------------------------
# build_daily_sales
# ------------------------------------------------------------------

@pytest.fixture
def three_day_sales() -> pd.DataFrame:
    return clean_transactions(raw_rows(
        ("1", "11111", "A", 2, "2010-01-04 09:00", 1.00),
        ("2", "11111", "A", 2, "2010-01-04 15:00", 2.00),
        # 2010-01-05: no sales of anything
        ("3", "11111", "A", 1, "2010-01-06 09:00", 1.50),
    ))


def test_daily_sales_has_one_row_per_calendar_day(three_day_sales):
    daily = build_daily_sales(three_day_sales, ["11111"])

    assert list(daily["sale_date"]) == list(pd.date_range("2010-01-04", "2010-01-06"))
    assert list(daily["units_sold"]) == [4, 0, 1]


def test_daily_price_is_quantity_weighted(three_day_sales):
    daily = build_daily_sales(three_day_sales, ["11111"])

    # (2 x 1.00 + 2 x 2.00) / 4 units = 1.50
    assert daily["avg_price"].iat[0] == pytest.approx(1.50)


def test_zero_sale_days_have_no_price(three_day_sales):
    daily = build_daily_sales(three_day_sales, ["11111"])

    assert daily["orders"].iat[1] == 0
    assert math.isnan(daily["avg_price"].iat[1])


# ------------------------------------------------------------------
# build_catalog
# ------------------------------------------------------------------

def test_catalog_derives_prices_and_cleans_names():
    clean = clean_transactions(raw_rows(
        ("1", "11111", "LUNCH BAG  BLACK SKULL.", 1, "2010-01-04", 1.00),
        ("2", "11111", "LUNCH BAG  BLACK SKULL.", 1, "2010-01-05", 2.00),
        ("3", "11111", "LUNCH BAG  BLACK SKULL.", 1, "2010-01-06", 3.00),
    ))

    row = build_catalog(clean, ["11111"]).iloc[0]

    assert row["name"] == "Lunch Bag Black Skull"            # double space and trailing dot removed
    assert row["competitor_price"] == 2.00                   # median selling price
    assert row["cost_price"] == round(2.00 * COST_RATIO, 2)  # the documented 60% assumption
