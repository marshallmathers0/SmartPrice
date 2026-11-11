import logging
from functools import lru_cache

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from app.config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache
def get_redis_client() -> Redis:
    """
    The application's Redis client — created once, reused everywhere.

        get_redis_client()
                │
                ▼
          Redis client  ──▶  connection pool  ──▶  Redis server

    @lru_cache makes this a singleton, like get_settings(). The client owns a
    connection pool, so every request reuses already-open connections.
    Creating the client does NOT connect; connections open on first use.

    decode_responses=True: Redis returns str instead of bytes, so cached JSON
    can be parsed directly.

    Fail fast: short socket timeouts and no automatic retries. Redis is only a
    cache — if it is slow or down, ProductCache falls back to PostgreSQL
    immediately instead of making the client wait.
    """
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.redis_socket_timeout,
        socket_connect_timeout=settings.redis_socket_timeout,
        retry=Retry(NoBackoff(), retries=0),
    )


async def ping_redis() -> bool:
    """Return True if Redis answers PING."""
    try:
        return bool(await get_redis_client().ping())
    except Exception as exc:
        logger.warning("Redis unreachable | %s: %s", type(exc).__name__, exc)
        return False


async def close_redis_client() -> None:
    """
    Close the client and its pool. Called on application shutdown.

    The cached client is discarded too, so the next application instance
    (e.g. the next test) gets a fresh client bound to its own event loop.
    """
    if get_redis_client.cache_info().currsize:
        await get_redis_client().aclose()
    get_redis_client.cache_clear()
