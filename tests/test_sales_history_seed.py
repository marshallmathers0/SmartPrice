"""Phase A — loading the real dataset into PostgreSQL (every test is rolled back)."""
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.config.settings import resolve_path, settings
from app.database.models.sales_history import SalesHistoryORM
from app.database.seed import load_daily_sales, load_dataset_products, seed_products, seed_sales_history

pytestmark = pytest.mark.postgres

PROCESSED = resolve_path(settings.processed_data_dir)


def sales_row(product_id: str, day: str, units: int, price: str | None) -> dict:
    return {
        "product_id": product_id,
        "sale_date": date.fromisoformat(day),
        "units_sold": units,
        "orders": 1 if units else 0,
        "avg_price": Decimal(price) if price else None,
    }


@pytest.mark.asyncio
async def test_sales_seed_is_idempotent(db_session, make_product):
    await seed_products(db_session, [make_product(product_id="TEST-SALES-1")])
    rows = [sales_row("TEST-SALES-1", "2011-01-03", 12, "1.50"), sales_row("TEST-SALES-1", "2011-01-04", 0, None)]

    assert await seed_sales_history(db_session, rows) == 2
    assert await seed_sales_history(db_session, rows) == 0      # second run inserts nothing


@pytest.mark.asyncio
async def test_sales_must_belong_to_an_existing_product(db_session):
    with pytest.raises(IntegrityError, match="fk_sales_history_product_id_products"):
        await seed_sales_history(db_session, [sales_row("TEST-NO-SUCH-PRODUCT", "2011-01-03", 1, "1.00")])


@pytest.mark.asyncio
async def test_database_rejects_negative_demand(db_session, make_product):
    await seed_products(db_session, [make_product(product_id="TEST-SALES-NEG")])

    with pytest.raises(IntegrityError, match="ck_sales_history_units_sold_non_negative"):
        await seed_sales_history(db_session, [sales_row("TEST-SALES-NEG", "2011-01-03", -1, "1.00")])


@pytest.mark.asyncio
@pytest.mark.skipif(not (PROCESSED / "daily_sales.csv").exists(), reason="processed dataset not built")
async def test_full_dataset_loads(db_session):
    products = load_dataset_products(PROCESSED)
    rows = load_daily_sales(PROCESSED)

    await seed_products(db_session, products)
    await seed_sales_history(db_session, rows)

    count = await db_session.scalar(select(func.count()).select_from(SalesHistoryORM))
    assert len(products) == 20
    assert count >= len(rows) == 14_780


# ------------------------------------------------------------------
# Phase 4 — reading history back for the forecast
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recent_history_is_the_newest_days_oldest_first(db_session, make_product):
    from app.repositories.postgres import PostgresSalesHistoryRepository

    await seed_products(db_session, [make_product(product_id="TEST-SALES-RECENT")])
    days = ["2011-01-03", "2011-01-04", "2011-01-05", "2011-01-06"]
    await seed_sales_history(db_session, [sales_row("TEST-SALES-RECENT", d, i + 1, "1.00") for i, d in enumerate(days)])

    history = await PostgresSalesHistoryRepository(db_session).recent("TEST-SALES-RECENT", days=3)

    assert [h.sale_date.isoformat() for h in history] == days[1:]        # the 3 newest, oldest first
    assert [h.units_sold for h in history] == [2, 3, 4]
    assert await PostgresSalesHistoryRepository(db_session).recent("TEST-NO-HISTORY", days=3) == []
