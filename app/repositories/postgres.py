import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.product import ProductORM
from app.database.models.sales_history import SalesHistoryORM
from app.exceptions.pricing import ProductNotFoundError
from app.models.forecast import DailySales
from app.models.product import Product
from app.repositories.base import ProductRepository, SalesHistoryRepository

logger = logging.getLogger(__name__)


class PostgresProductRepository(ProductRepository):
    """
    ProductRepository backed by PostgreSQL through SQLAlchemy 2.x.

        PricingService
               │
               ▼
        ProductRepository          <- the service only knows this
               │
               ▼
        PostgresProductRepository  <- this class
               │
               ▼
        SQLAlchemy (AsyncSession)
               │
               ▼
        PostgreSQL

    The session is injected (one per request, from get_db()), so this class
    never opens or closes connections itself.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, product_id: str) -> Product:
        # SQLAlchemy 2.x style: build a select(), execute it, unwrap the result.
        # (Replaces the legacy session.query(ProductORM).filter(...).first())
        statement = select(ProductORM).where(ProductORM.product_id == product_id)
        result = await self._session.execute(statement)
        row = result.scalar_one_or_none()

        if row is None:
            logger.info("Product not found | product_id=%s", product_id)
            raise ProductNotFoundError(product_id)

        logger.debug("Product loaded from PostgreSQL | product_id=%s", product_id)
        return self._to_domain(row)

    async def list_all(self) -> list[Product]:
        rows = (await self._session.execute(select(ProductORM).order_by(ProductORM.product_id))).scalars().all()
        return [self._to_domain(row) for row in rows]

    @staticmethod
    def _to_domain(row: ProductORM) -> Product:
        """
        Map a database row to the domain model.

        The ORM object never leaves this class: everything above the
        repository works with ``Product`` and knows nothing about SQLAlchemy.
        """
        return Product(
            product_id=row.product_id,
            name=row.name,
            cost_price=row.cost_price,
            competitor_price=row.competitor_price,
            inventory=row.inventory,
            demand_score=row.demand_score,
        )


class PostgresSalesHistoryRepository(SalesHistoryRepository):
    """SalesHistoryRepository backed by the sales_history table (Phase A data)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def recent(self, product_id: str, days: int) -> list[DailySales]:
        # Newest `days` rows via the (product_id, sale_date) unique index, then oldest-first for the caller
        statement = (
            select(SalesHistoryORM)
            .where(SalesHistoryORM.product_id == product_id)
            .order_by(SalesHistoryORM.sale_date.desc())
            .limit(days)
        )
        rows = (await self._session.execute(statement)).scalars().all()
        return [
            DailySales(product_id=r.product_id, sale_date=r.sale_date, units_sold=r.units_sold, avg_price=r.avg_price)
            for r in reversed(rows)
        ]
