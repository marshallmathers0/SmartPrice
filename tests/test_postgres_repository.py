"""
Day 6 — PostgresProductRepository against the real database.

Every test runs inside a transaction that is rolled back afterwards,
so nothing a test writes is ever left in the database.
"""
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.database.models.product import ProductORM
from app.database.seed import seed_products
from app.exceptions.pricing import ProductNotFoundError
from app.models.pricing import PricingStrategy
from app.models.product import Product
from app.repositories.postgres import PostgresProductRepository
from app.services.pricing import PricingService

pytestmark = pytest.mark.postgres


@pytest.mark.asyncio
async def test_get_by_id_returns_domain_product_with_exact_decimals(db_session, make_product):
    await seed_products(db_session, [make_product("12.34", "56.78", product_id="TEST-PG-1", name="Test Lamp")])

    product = await PostgresProductRepository(db_session).get_by_id("TEST-PG-1")

    assert type(product) is Product            # the domain model, never the ORM row
    assert product.name == "Test Lamp"
    assert product.cost_price == Decimal("12.34")
    assert product.competitor_price == Decimal("56.78")


@pytest.mark.asyncio
async def test_missing_product_raises_product_not_found(db_session):
    with pytest.raises(ProductNotFoundError) as exc_info:
        await PostgresProductRepository(db_session).get_by_id("TEST-PG-MISSING")

    assert exc_info.value.product_id == "TEST-PG-MISSING"


@pytest.mark.asyncio
async def test_seed_is_idempotent(db_session, make_product):
    products = [
        make_product(product_id="TEST-PG-SEED-1"),
        make_product(product_id="TEST-PG-SEED-2"),
    ]

    assert await seed_products(db_session, products) == 2
    assert await seed_products(db_session, products) == 0   # second run inserts nothing

    count = await db_session.scalar(
        select(func.count()).select_from(ProductORM).where(ProductORM.product_id.like("TEST-PG-SEED-%"))
    )
    assert count == 2


@pytest.mark.asyncio
async def test_database_rejects_non_positive_prices(db_session):
    db_session.add(ProductORM(product_id="TEST-PG-BAD", name="Broken", cost_price=Decimal("0"), competitor_price=Decimal("10")))

    with pytest.raises(IntegrityError, match="ck_products_cost_price_positive"):
        await db_session.flush()


@pytest.mark.asyncio
async def test_business_logic_unchanged_on_postgres(db_session, make_product):
    # The same DEVLOG Day 4 scenario, now loaded from PostgreSQL — same answer
    await seed_products(db_session, [make_product("950.00", "999.00", product_id="TEST-PG-FLOOR")])
    service = PricingService(PostgresProductRepository(db_session))

    quote = await service.calculate_price("TEST-PG-FLOOR", PricingStrategy.COMPETITIVE)

    assert quote.suggested_price == Decimal("997.50")
    assert quote.margin_percent == Decimal("4.76")


@pytest.mark.asyncio
async def test_demand_signals_round_trip(db_session):
    product = Product("TEST-PG-DEMAND", "Test Fan", Decimal("100.00"), Decimal("150.00"),
                      inventory=42, demand_score=Decimal("0.73"))
    await seed_products(db_session, [product])

    loaded = await PostgresProductRepository(db_session).get_by_id("TEST-PG-DEMAND")

    assert loaded == product     # every field survives the trip through PostgreSQL


@pytest.mark.asyncio
async def test_seed_overwrite_resets_existing_rows(db_session, make_product):
    await seed_products(db_session, [make_product(product_id="TEST-PG-RESET")])
    changed = Product("TEST-PG-RESET", "Renamed", Decimal("1.00"), Decimal("2.00"), inventory=7, demand_score=Decimal("0.90"))

    assert await seed_products(db_session, [changed]) == 0                    # default: never overwrite
    assert await seed_products(db_session, [changed], overwrite=True) == 1   # opt-in: reset to seed values

    assert await PostgresProductRepository(db_session).get_by_id("TEST-PG-RESET") == changed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "column, value, constraint",
    [
        ("inventory", -1, "ck_products_inventory_non_negative"),
        ("demand_score", Decimal("1.50"), "ck_products_demand_score_range"),
    ],
)
async def test_database_rejects_invalid_demand_signals(db_session, column, value, constraint):
    row = ProductORM(product_id="TEST-PG-BAD-DEMAND", name="Broken", cost_price=Decimal("10"), competitor_price=Decimal("10"))
    setattr(row, column, value)
    db_session.add(row)

    with pytest.raises(IntegrityError, match=constraint):
        await db_session.flush()


@pytest.mark.asyncio
async def test_existing_rows_get_column_defaults(db_session):
    # Inserting without the new columns — what every row had when the migration ran
    db_session.add(ProductORM(product_id="TEST-PG-DEFAULTS", name="Old Row", cost_price=Decimal("10"), competitor_price=Decimal("20")))
    await db_session.flush()

    product = await PostgresProductRepository(db_session).get_by_id("TEST-PG-DEFAULTS")

    assert product.inventory == 0
    assert product.demand_score == Decimal("0.50")


@pytest.mark.asyncio
async def test_list_all_returns_domain_products_in_order(db_session, make_product):
    await seed_products(db_session, [make_product(product_id="TEST-PG-LIST-B"), make_product(product_id="TEST-PG-LIST-A")])

    listed = [p for p in await PostgresProductRepository(db_session).list_all() if p.product_id.startswith("TEST-PG-LIST")]

    assert [p.product_id for p in listed] == ["TEST-PG-LIST-A", "TEST-PG-LIST-B"]
    assert all(type(p) is Product for p in listed)
