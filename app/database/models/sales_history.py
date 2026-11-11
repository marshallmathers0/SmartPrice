from datetime import date
from decimal import Decimal

from sqlalchemy import CheckConstraint, Date, ForeignKey, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class SalesHistoryORM(Base):
    """
    Daily demand per product — the data the forecasting model learns from.

    One row per product per CALENDAR day, including days with zero sales
    (see app/ml/data/prepare.py for why zero days must exist).

    Source: UCI Online Retail II, aggregated from 1M+ transaction lines.
    """

    __tablename__ = "sales_history"
    __table_args__ = (
        # One row per product per day — also the index every history query uses
        UniqueConstraint("product_id", "sale_date"),   # named by convention: uq_sales_history_product_id
        CheckConstraint("units_sold >= 0", name="units_sold_non_negative"),
        CheckConstraint("orders >= 0", name="orders_non_negative"),
        CheckConstraint("avg_price IS NULL OR avg_price > 0", name="avg_price_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # Deleting a product deletes its history; a sale can never point at a missing product
    product_id: Mapped[str] = mapped_column(String(64), ForeignKey("products.product_id", ondelete="CASCADE"))
    sale_date: Mapped[date] = mapped_column(Date)

    units_sold: Mapped[int] = mapped_column(Integer)          # demand: the forecasting target
    orders: Mapped[int] = mapped_column(Integer)              # distinct invoices that day
    # Quantity-weighted average price paid; NULL on zero-sale days (nobody paid anything)
    avg_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    def __repr__(self) -> str:
        return f"SalesHistoryORM(product_id={self.product_id!r}, sale_date={self.sale_date!r}, units_sold={self.units_sold!r})"
