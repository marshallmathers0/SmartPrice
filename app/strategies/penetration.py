from decimal import Decimal

from app.models.product import Product
from app.strategies.base import AbstractPricingStrategy


class PenetrationPricingStrategy(AbstractPricingStrategy):
    """
    Aggressive low price to rapidly capture market share.

    Formula: cost_price x 1.10  (only 10% above cost)

    When to use:
    - Launching a new product in a competitive market
    - Building a user base quickly at the cost of short-term margin
    - Seasonal aggressive sales or clearance

    Risk:
    - Low margin leaves little room for error
    - Competitors may respond with further price cuts (price war)
    - Not sustainable long-term — switch to competitive or dynamic once established

    Example:
        cost_price      = 500.00
        suggested_price = 500.00 x 1.10 = 550.00
        margin          = 9.09%  (low — intentionally aggressive)
    """

    MARGIN_MULTIPLIER = Decimal("1.10")   # only 10% above cost

    @property
    def description(self) -> str:
        return "10% above cost — aggressive entry price to capture market share"

    def calculate_price(self, product: Product) -> Decimal:
        return self._round(product.cost_price * self.MARGIN_MULTIPLIER)
