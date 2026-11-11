import asyncio
from decimal import Decimal

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.dependencies import get_product_repository
from app.config.settings import settings
from app.main import create_app
from app.models.product import Product
from app.repositories.mock import MockProductRepository


# ------------------------------------------------------------------
# PostgreSQL integration tests
# ------------------------------------------------------------------
# Tests marked @pytest.mark.postgres need the real database
# (docker compose up -d && alembic upgrade head). When it is not
# reachable or not migrated, they are skipped — the rest of the suite
# never touches a database.
# ------------------------------------------------------------------

def _postgres_ready() -> bool:
    async def probe() -> bool:
        engine = create_async_engine(settings.database_url, poolclass=NullPool, connect_args={"timeout": 3})
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1 FROM products LIMIT 1"))   # migrated?
            return True
        except Exception:
            return False
        finally:
            await engine.dispose()

    return asyncio.run(probe())


def _redis_ready() -> bool:
    async def probe() -> bool:
        client = Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
        try:
            return bool(await client.ping())
        except Exception:
            return False
        finally:
            await client.aclose()

    return asyncio.run(probe())


def pytest_collection_modifyitems(config, items):
    postgres_items = [item for item in items if "postgres" in item.keywords]
    if postgres_items and not _postgres_ready():
        skip = pytest.mark.skip(reason="PostgreSQL not reachable or not migrated (docker compose up -d; alembic upgrade head)")
        for item in postgres_items:
            item.add_marker(skip)

    redis_items = [item for item in items if "redis" in item.keywords]
    if redis_items and not _redis_ready():
        skip = pytest.mark.skip(reason="Redis not reachable (docker compose up -d)")
        for item in redis_items:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def devlog_thresholds(monkeypatch):
    """
    Pin guardrail thresholds to the DEVLOG defaults (1.05 floor, 1.50 ceiling)
    so a change to the local .env can never change test results.
    """
    monkeypatch.setattr(settings, "minimum_margin_multiplier", 1.05)
    monkeypatch.setattr(settings, "max_volatility_multiplier", 1.50)


@pytest.fixture(autouse=True)
def no_startup_service_checks(monkeypatch):
    """Unit tests never wait on the database or Redis during app startup."""
    monkeypatch.setattr(settings, "db_check_on_startup", False)
    monkeypatch.setattr(settings, "redis_check_on_startup", False)


class FakeRedis:
    """
    In-memory stand-in for redis.asyncio.Redis — only what ProductCache uses.

    Records TTLs and call counts, and can be switched to fail like a Redis
    server that is down.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.calls = 0
        self.fail = False

    def _call(self) -> None:
        self.calls += 1
        if self.fail:
            raise RedisConnectionError("Connection refused")

    async def get(self, key: str) -> str | None:
        self._call()
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._call()
        self.store[key] = value
        self.ttls[key] = ex

    async def delete(self, key: str) -> None:
        self._call()
        self.store.pop(key, None)


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def make_product():
    """Build a Product; defaults match the DEVLOG example (cost=500, competitor=999)."""
    def _make(
        cost: str = "500.00",
        competitor: str = "999.00",
        product_id: str = "PROD-001",
        name: str = "Wireless Headphones",
    ) -> Product:
        return Product(
            product_id=product_id,
            name=name,
            cost_price=Decimal(cost),
            competitor_price=Decimal(competitor),
        )

    return _make


@pytest.fixture
def catalog(make_product) -> list[Product]:
    """The DEVLOG scenarios as catalog products."""
    return [
        make_product("500.00", "999.00", product_id="PROD-001"),        # Day 3 results table
        make_product("950.00", "999.00", product_id="PROD-005"),        # Day 4 floor lifecycle
        make_product("500.00", "400.00", product_id="PROD-CAP"),        # ceiling caps, no conflict
        make_product("900.00", "400.00", product_id="PROD-CONFLICT"),   # ceiling below floor
        make_product("333.33", "300.00", product_id="PROD-CENTS"),      # floor not a whole cent
    ]


@pytest.fixture
def app(catalog):
    """
    App with the test catalog injected in place of the real repository.

    This is Dependency Injection paying off: tests choose the data source
    without touching the service, the router, or the seed data.
    """
    application = create_app()
    repository = MockProductRepository(catalog)
    application.dependency_overrides[get_product_repository] = lambda: repository
    return application


@pytest.fixture
def client(app):
    """HTTP client for the test app (runs the lifespan startup/shutdown)."""
    with TestClient(app) as test_client:
        yield test_client


# Shared by every @pytest.mark.postgres test that writes to the database
@pytest_asyncio.fixture
async def db_session():
    """A session whose work is always rolled back — the database is left untouched."""
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        # create_savepoint: session.commit() inside a test only releases a
        # savepoint; the outer transaction is still rolled back below.
        session = AsyncSession(bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
    await engine.dispose()


# ------------------------------------------------------------------
# Phase 4 — the real dataset in memory, and the deployed model
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def dataset_products():
    from app.config.settings import resolve_path
    from app.database.seed import load_dataset_products
    return load_dataset_products(resolve_path(settings.processed_data_dir))


@pytest.fixture(scope="session")
def dataset_sales():
    from app.config.settings import resolve_path
    from app.database.seed import load_daily_sales
    from app.models.forecast import DailySales
    rows = load_daily_sales(resolve_path(settings.processed_data_dir))
    return [DailySales(r["product_id"], r["sale_date"], r["units_sold"], r["avg_price"]) for r in rows]


@pytest.fixture(scope="session")
def deployed_bundle():
    from app.ml.model_store import ModelNotAvailableError, load_bundle
    try:
        return load_bundle()
    except ModelNotAvailableError:
        pytest.skip("no trained model (python -m app.ml.train)")


@pytest.fixture
def ml_app(dataset_products, dataset_sales, deployed_bundle):
    """
    The REAL dependency graph (forecast service, demand-aware repository,
    pricing), with only the two data sources swapped for in-memory versions.
    """
    from app.api.dependencies import get_base_product_repository, get_sales_history_repository
    from app.repositories.mock import SEED_PRODUCTS, InMemorySalesHistoryRepository

    application = create_app()
    products = MockProductRepository([*dataset_products, *SEED_PRODUCTS])     # 20 real + 3 demo (no history)
    sales = InMemorySalesHistoryRepository(dataset_sales)
    application.dependency_overrides[get_base_product_repository] = lambda: products
    application.dependency_overrides[get_sales_history_repository] = lambda: sales
    return application


@pytest.fixture
def ml_client(ml_app):
    with TestClient(ml_app) as test_client:
        yield test_client
