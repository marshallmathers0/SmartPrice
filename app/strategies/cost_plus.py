from decimal import Decimal

from app.models.product import Product
from app.strategies.base import AbstractPricingStrategy


class CostPlusPricingStrategy(AbstractPricingStrategy):
    """
    Guarantees a fixed profit margin on top of the cost price.

    Formula: cost_price x 1.40  (40% margin)

    When to use:
    - You want predictable, stable profit margins
    - Market price is irrelevant (niche product, no competition)
    - B2B pricing where margins are agreed contractually

    Example:
        cost_price      = 500.00
        suggested_price = 500.00 x 1.40 = 700.00
        margin          = 40% guaranteed regardless of competitor price
    """

    MARGIN_MULTIPLIER = Decimal("1.40")   # 40% above cost

    @property
    def description(self) -> str:
        return "Cost + 40% fixed margin regardless of market conditions"

    def calculate_price(self, product: Product) -> Decimal:
        return self._round(product.cost_price * self.MARGIN_MULTIPLIER)
