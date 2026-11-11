"""
Seed the database with demo data.

    python -m app.database.seed            # insert missing products and sales history
    python -m app.database.seed --update   # also reset existing products to the seed values
                                           # (and invalidate their Redis cache entries)

What gets loaded
----------------
1. The 3 hand-written demo products (SKU-12345, ...) from app/repositories/mock.py.
2. Phase A: the 20 real products from the UCI Online Retail II dataset and
   their daily sales history (data/processed/, created by
   ``python -m app.ml.data.prepare``). Skipped with a warning if those
   files are missing.

Idempotent: running it again inserts nothing new, because existing rows are
skipped (INSERT ... ON CONFLICT DO NOTHING). By default rows already in the
table are never overwritten, so manual edits survive. --update opts in to
overwriting products (ON CONFLICT ... DO UPDATE) — useful after a migration
adds new columns.
"""
import argparse
import asyncio
import csv
import logging
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.product_cache import ProductCache
from app.cache.redis_client import close_redis_client, get_redis_client
from app.config.logging import setup_logging
from app.config.settings import resolve_path, settings
from app.database.models.product import ProductORM
from app.database.models.sales_history import SalesHistoryORM
from app.database.session import SessionLocal, dispose_engine
from app.models.product import Product
from app.repositories.mock import SEED_PRODUCTS

logger = logging.getLogger(__name__)

# Columns the seed owns — everything except the surrogate id and audit timestamps
_SEEDED_COLUMNS = ("name", "cost_price", "competitor_price", "inventory", "demand_score")

# asyncpg allows at most 32,767 bind parameters per statement; 5 columns x 5,000 rows stays well under
_SALES_BATCH_ROWS = 5_000


async def seed_products(
    session: AsyncSession,
    products: Iterable[Product] = SEED_PRODUCTS,
    overwrite: bool = False,
) -> int:
    """
    Insert products. Returns how many rows were written.

    overwrite=False → only missing products are inserted (existing rows untouched).
    overwrite=True  → existing products with the same product_id are reset to these values.

    The caller owns the transaction (commit / rollback), so this function
    can be reused inside tests that roll everything back afterwards.
    """
    rows = [
        {
            "product_id": product.product_id,
            "name": product.name,
            "cost_price": product.cost_price,
            "competitor_price": product.competitor_price,
            "inventory": product.inventory,
            "demand_score": product.demand_score,
        }
        for product in products
    ]
    if not rows:
        return 0

    statement = insert(ProductORM).values(rows)
    if overwrite:
        statement = statement.on_conflict_do_update(
            index_elements=[ProductORM.product_id],
            set_={column: statement.excluded[column] for column in _SEEDED_COLUMNS},
        )
    else:
        statement = statement.on_conflict_do_nothing(index_elements=[ProductORM.product_id])

    written = (await session.execute(statement.returning(ProductORM.product_id))).scalars().all()
    return len(written)


# ------------------------------------------------------------------
# Phase A — the real dataset (files written by app.ml.data.prepare)
# ------------------------------------------------------------------

def load_dataset_products(processed_dir: Path) -> list[Product]:
    """The real products from data/processed/products.csv (empty list if the file is missing)."""
    path = processed_dir / "products.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Product(
                product_id=row["product_id"],
                name=row["name"],
                cost_price=Decimal(row["cost_price"]),
                competitor_price=Decimal(row["competitor_price"]),
            )
            for row in csv.DictReader(handle)
        ]


def load_daily_sales(processed_dir: Path) -> list[dict]:
    """Rows for sales_history from data/processed/daily_sales.csv (empty list if the file is missing)."""
    path = processed_dir / "daily_sales.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "product_id": row["product_id"],
                "sale_date": date.fromisoformat(row["sale_date"]),
                "units_sold": int(row["units_sold"]),
                "orders": int(row["orders"]),
                "avg_price": Decimal(row["avg_price"]) if row["avg_price"] else None,   # empty on zero-sale days
            }
            for row in csv.DictReader(handle)
        ]


async def seed_sales_history(session: AsyncSession, rows: list[dict]) -> int:
    """
    Insert daily sales rows; existing (product_id, sale_date) pairs are skipped.

    Idempotent for the same reason as seed_products(): ON CONFLICT DO NOTHING
    on the unique (product_id, sale_date) constraint. Returns rows inserted.
    """
    inserted = 0
    for start in range(0, len(rows), _SALES_BATCH_ROWS):
        statement = (
            insert(SalesHistoryORM)
            .values(rows[start:start + _SALES_BATCH_ROWS])
            .on_conflict_do_nothing(index_elements=[SalesHistoryORM.product_id, SalesHistoryORM.sale_date])
            .returning(SalesHistoryORM.id)
        )
        inserted += len((await session.execute(statement)).scalars().all())
    return inserted


async def main(overwrite: bool) -> None:
    setup_logging()
    processed_dir = resolve_path(settings.processed_data_dir)
    dataset_products = load_dataset_products(processed_dir)
    sales_rows = load_daily_sales(processed_dir)
    all_products = [*SEED_PRODUCTS, *dataset_products]

    if not dataset_products:
        logger.warning(
            "No processed dataset in %s - only the demo products are seeded "
            "(run: python -m app.ml.data.prepare)",
            processed_dir,
        )

    try:
        async with SessionLocal() as session:
            # Products first: sales_history rows reference them by foreign key
            written = await seed_products(session, all_products, overwrite=overwrite)
            sales_inserted = await seed_sales_history(session, sales_rows)
            await session.commit()

        if overwrite:
            # Day 7: rows changed in PostgreSQL - drop their cached copies so
            # the next request reads the new values instead of a stale cache entry
            cache = ProductCache(get_redis_client(), ttl_seconds=settings.cache_ttl_seconds)
            for product in all_products:
                await cache.invalidate(product.product_id)
            logger.info("Seed complete | products inserted or updated=%d | cache entries invalidated", written)
        else:
            logger.info(
                "Seed complete | products inserted=%d | already present=%d",
                written,
                len(all_products) - written,
            )
        logger.info(
            "Sales history | rows inserted=%d | already present=%d",
            sales_inserted,
            len(sales_rows) - sales_inserted,
        )
    finally:
        await dispose_engine()
        await close_redis_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the database with demo products and sales history.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="also reset existing products to the seed values",
    )
    asyncio.run(main(overwrite=parser.parse_args().update))
