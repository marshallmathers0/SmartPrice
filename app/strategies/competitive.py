from decimal import Decimal

from app.models.product import Product
from app.strategies.base import AbstractPricingStrategy


class CompetitivePricingStrategy(AbstractPricingStrategy):
    """
    Prices the product slightly below the competitor.

    Formula: competitor_price x 0.97  (3% below competitor)

    When to use:
    - Market is price-sensitive
    - You want to win customers purely on price
    - You have margins to absorb the 3% reduction

    Example:
        competitor_price = 999.00
        suggested_price  = 999.00 x 0.97 = 969.03
    """

    @property
    def description(self) -> str:
        return "Price 3% below competitor to win on price"

    def calculate_price(self, product: Product) -> Decimal:
        return self._round(product.competitor_price * Decimal("0.97"))
