import logging
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Engine — owns the connection pool (one per process)
# ------------------------------------------------------------------
# Creating the engine does NOT connect. Connections are opened lazily,
# on the first query, and then reused from the pool.
# ------------------------------------------------------------------
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True,     # test a pooled connection before use; drops ones the server closed
    connect_args={"timeout": settings.db_connect_timeout},
)

# ------------------------------------------------------------------
# Session factory — one short-lived session per request
# ------------------------------------------------------------------
# expire_on_commit=False: objects stay readable after a commit, instead of
# triggering a surprise lazy reload (which async sessions cannot do).
# ------------------------------------------------------------------
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency — one database session per request.

        Engine
            │
            ▼
        SessionLocal
            │
            ▼
        get_db()   ← yields a session, closes it when the request ends

    ``async with`` guarantees the session is closed (and its connection
    returned to the pool) even if the endpoint raises. Uncommitted work
    is rolled back on close.
    """
    async with SessionLocal() as session:
        yield session


async def ping_database() -> bool:
    """Return True if the database answers ``SELECT 1``."""
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database unreachable | %s: %s", type(exc).__name__, exc)
        return False


async def dispose_engine() -> None:
    """Close every pooled connection. Called on application shutdown."""
    await engine.dispose()
