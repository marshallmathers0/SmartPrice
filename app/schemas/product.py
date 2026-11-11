from decimal import Decimal

from pydantic import BaseModel

from app.models.product import Product


class ProductSummary(BaseModel):
    """One catalog entry (GET /products) — what the demo's product picker shows."""

    product_id: str
    name: str
    cost_price: Decimal
    competitor_price: Decimal

    @classmethod
    def from_domain(cls, product: Product) -> "ProductSummary":
        return cls(
            product_id=product.product_id,
            name=product.name,
            cost_price=product.cost_price,
            competitor_price=product.competitor_price,
        )
