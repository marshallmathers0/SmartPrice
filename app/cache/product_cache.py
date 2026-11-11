import json
import logging
import time
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from redis.exceptions import RedisError

from app.models.product import Product

logger = logging.getLogger(__name__)


class ProductCache:
    """
    Every Redis interaction for products lives here — and only here.

    Responsibilities
    ----------------
    - Key generation       product:SKU-12345
    - Serialization        Product → asdict() → JSON → Redis
    - Deserialization      Redis → JSON → Decimal conversion → Product
    - Expiration (TTL)     every entry expires after ``ttl_seconds``
    - Invalidation         invalidate(product_id) deletes an entry

    Graceful degradation
    --------------------
    Redis is a performance optimisation, not a dependency. Every method
    catches Redis failures, logs them, and behaves like an empty cache:
    get() returns None (a miss), set() and invalidate() do nothing.
    The caller never sees a Redis exception.

    After a failure, Redis is skipped entirely for ``retry_seconds``
    (a tiny circuit breaker). Without it, every request during an outage
    would wait for a connection timeout before falling back.
    """

    KEY_PREFIX = "product"

    def __init__(self, redis: Any, ttl_seconds: int, retry_seconds: int = 30) -> None:
        # ``redis`` is a redis.asyncio.Redis — or anything with async get/set/delete (tests)
        self._redis = redis
        self._ttl_seconds = ttl_seconds
        self._retry_seconds = retry_seconds
        self._skip_until = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def key(cls, product_id: str) -> str:
        return f"{cls.KEY_PREFIX}:{product_id}"

    async def get(self, product_id: str) -> Product | None:
        """Return the cached product, or None on a miss (or if Redis is unavailable)."""
        if not self._available():
            return None
        key = self.key(product_id)
        try:
            raw = await self._redis.get(key)
        except (RedisError, OSError) as exc:
            self._trip(exc)
            return None

        if raw is None:
            return None

        try:
            return self._deserialize(raw)
        except (ValueError, TypeError, KeyError) as exc:
            # Corrupt or outdated entry (e.g. written before a field was added):
            # treat it as a miss and remove it so it is rebuilt from PostgreSQL.
            logger.warning("Discarding unreadable cache entry | key=%s | %s: %s", key, type(exc).__name__, exc)
            await self.invalidate(product_id)
            return None

    async def set(self, product: Product) -> bool:
        """Store a product with the configured TTL. Returns False if Redis was unavailable."""
        if not self._available():
            return False
        try:
            await self._redis.set(self.key(product.product_id), self._serialize(product), ex=self._ttl_seconds)
            return True
        except (RedisError, OSError) as exc:
            self._trip(exc)
            return False

    async def invalidate(self, product_id: str) -> None:
        """Delete a cached product, e.g. after its row changes in PostgreSQL."""
        if not self._available():
            return
        try:
            await self._redis.delete(self.key(product_id))
        except (RedisError, OSError) as exc:
            self._trip(exc)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(product: Product) -> str:
        # Decimal is not JSON-serializable; str() keeps it exact ("500.00", not 500.0)
        return json.dumps(asdict(product), default=str)

    @staticmethod
    def _deserialize(raw: str) -> Product:
        data = json.loads(raw)
        # Restore exact types so a product from Redis is identical to one from
        # PostgreSQL — strategies must never receive a str or float for money.
        return Product(
            product_id=data["product_id"],
            name=data["name"],
            cost_price=Decimal(data["cost_price"]),
            competitor_price=Decimal(data["competitor_price"]),
            inventory=int(data["inventory"]),
            demand_score=Decimal(data["demand_score"]),
        )

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _available(self) -> bool:
        return time.monotonic() >= self._skip_until

    def _trip(self, exc: Exception) -> None:
        self._skip_until = time.monotonic() + self._retry_seconds
        logger.warning(
            "Redis unavailable - serving from the database for %ds | %s: %s",
            self._retry_seconds,
            type(exc).__name__,
            exc,
        )
