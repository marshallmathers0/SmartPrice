"""Phase 6 — configuration for free-tier cloud services (Neon / Supabase Postgres, Upstash Redis, Render)."""
import asyncio

import pytest
from redis.asyncio.connection import SSLConnection

from app.cache.redis_client import get_redis_client
from app.config.settings import PROJECT_ROOT, Settings, normalise_database_url, settings


@pytest.mark.parametrize(
    "given, expected",
    [
        # Neon's copy-paste connection string
        ("postgresql://alice:s3cret@ep-cool-name-123.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
         "postgresql+asyncpg://alice:s3cret@ep-cool-name-123.eu-central-1.aws.neon.tech/neondb?ssl=require"),
        # Heroku/Render-style short scheme
        ("postgres://bob:pw@db.example.com:5432/app", "postgresql+asyncpg://bob:pw@db.example.com:5432/app"),
        # Already in the app's format (local Docker) — unchanged
        ("postgresql+asyncpg://smartprice:pw@localhost:5432/smartprice",
         "postgresql+asyncpg://smartprice:pw@localhost:5432/smartprice"),
        # URL-encoded password characters and unrelated options survive
        ("postgresql://carol:p%40ss%23word@host/db?sslmode=verify-full&application_name=smartprice",
         "postgresql+asyncpg://carol:p%40ss%23word@host/db?ssl=verify-full&application_name=smartprice"),
    ],
)
def test_cloud_database_urls_are_normalised(given, expected):
    assert normalise_database_url(given) == expected


def test_settings_apply_the_normalisation(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db?sslmode=require")

    assert Settings().database_url == "postgresql+asyncpg://u:p@host/db?ssl=require"


@pytest.mark.postgres
def test_ssl_option_really_reaches_the_driver():
    """The local Docker PostgreSQL has no TLS, so a correctly passed ssl=require MUST be refused."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    plain = settings.database_url.split("?")[0]
    url = normalise_database_url(plain.replace("postgresql+asyncpg://", "postgresql://") + "?sslmode=require")

    async def connect():
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        finally:
            await engine.dispose()

    with pytest.raises(Exception, match="(?i)ssl"):
        asyncio.run(connect())


def test_upstash_tls_redis_url_uses_an_ssl_connection(monkeypatch):
    monkeypatch.setattr(settings, "redis_url", "rediss://default:token@eu1-cool-name-12345.upstash.io:6379")
    get_redis_client.cache_clear()
    try:
        client = get_redis_client()
        assert client.connection_pool.connection_class is SSLConnection
    finally:
        get_redis_client.cache_clear()


def test_plain_redis_url_does_not_use_ssl(monkeypatch):
    monkeypatch.setattr(settings, "redis_url", "redis://localhost:6379/0")
    get_redis_client.cache_clear()
    try:
        assert get_redis_client().connection_pool.connection_class is not SSLConnection
    finally:
        get_redis_client.cache_clear()


def test_env_example_lists_every_cloud_setting_without_real_secrets():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for key in ("DATABASE_URL=", "REDIS_URL=", "MODEL_VERSION=", "CURRENCY_SYMBOL="):
        assert key in example
    assert "change-me" in example                     # placeholders, not credentials
