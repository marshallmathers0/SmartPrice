from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# ------------------------------------------------------------------
# Constraint naming convention
# ------------------------------------------------------------------
# Without this, PostgreSQL invents names like "products_pkey" or
# "products_cost_price_check" — and Alembic cannot reliably find them
# again when a later migration needs to drop or alter a constraint.
# With it, every index and constraint gets a predictable name.
# ------------------------------------------------------------------
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """
    Declarative base for every ORM model.

    All tables register themselves on ``Base.metadata``. Alembic compares
    that metadata with the live database to generate migrations.

        ProductORM
              │
              ▼
            Base
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
