from datetime import datetime
from decimal import Decimal

from sqlalchemy import CheckConstraint, DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ProductORM(Base):
    """
    ORM mapping for the ``products`` table.

    This is the bridge between Python objects and relational rows.
    It is NOT the domain model: ``app.models.product.Product`` is.
    PostgresProductRepository converts rows into ``Product`` objects, so
    the business layer never sees SQLAlchemy (Persistence Ignorance).

    Why two classes for one product?
    --------------------------------
    The table carries database concerns the business does not care about —
    a surrogate ``id``, timestamps, constraints. The domain model carries
    business rules the database does not care about. Each can change
    without dragging the other along.
    """

    __tablename__ = "products"
    __table_args__ = (
        # Same invariant as Product.__post_init__, enforced by the database too:
        # bad rows are rejected on write, not discovered later on read.
        CheckConstraint("cost_price > 0", name="cost_price_positive"),
        CheckConstraint("competitor_price > 0", name="competitor_price_positive"),
        CheckConstraint("inventory >= 0", name="inventory_non_negative"),
        CheckConstraint("demand_score >= 0 AND demand_score <= 1", name="demand_score_range"),
    )

    # Surrogate key — internal to the database, never exposed by the API
    id: Mapped[int] = mapped_column(primary_key=True)

    # Business key — the SKU clients use; unique + indexed for O(log n) lookups
    product_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    name: Mapped[str] = mapped_column(String(200))

    # NUMERIC(12, 2): exact decimal money (never FLOAT), up to 9,999,999,999.99
    cost_price: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    competitor_price: Mapped[Decimal] = mapped_column(Numeric(12, 2))

    # Day 6 follow-up — demand signals (added by the second migration).
    # server_default fills existing rows when the column is added, and any
    # insert that does not mention the column.
    inventory: Mapped[int] = mapped_column(Integer, server_default="0")
    demand_score: Mapped[Decimal] = mapped_column(Numeric(3, 2), server_default="0.50")   # 0.00 … 1.00

    # Audit timestamps, set by PostgreSQL itself
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"ProductORM(id={self.id!r}, product_id={self.product_id!r})"
