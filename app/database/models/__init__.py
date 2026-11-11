# Import every ORM model here so that importing this package registers all
# tables on Base.metadata — Alembic relies on this to autogenerate migrations.
from app.database.models.product import ProductORM
from app.database.models.sales_history import SalesHistoryORM

__all__ = ["ProductORM", "SalesHistoryORM"]
