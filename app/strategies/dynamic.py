from decimal import Decimal

from app.models.product import Product
from app.strategies.base import AbstractPricingStrategy
from app.strategies.competitive import CompetitivePricingStrategy
from app.strategies.cost_plus import CostPlusPricingStrategy


class DynamicPricingStrategy(AbstractPricingStrategy):
    """
    Balances competitor positioning with cost-based margin protection.

    Formula: average of CompetitivePricingStrategy and CostPlusPricingStrategy

    Why average?
    - Pure competitive pricing ignores your costs (can lead to selling below margin)
    - Pure cost-plus ignores the market (can price yourself out)
    - Dynamic finds the middle ground

    When to use:
    - Normal market conditions
    - You want to stay competitive but never sacrifice margin below a threshold
    - Good default for most products

    Example:
        cost_price      = 500.00
        competitor_price = 999.00
        competitive     = 969.03  (999 x 0.97)
        cost_plus       = 700.00  (500 x 1.40)
        dynamic         = (969.03 + 700.00) / 2 = 834.52
    """

    @property
    def description(self) -> str:
        return "Average of competitive and cost-plus — balances market rate with margin"

    def calculate_price(self, product: Product) -> Decimal:
        competitive_price = CompetitivePricingStrategy().calculate_price(product)
        cost_plus_price   = CostPlusPricingStrategy().calculate_price(product)
        return self._round((competitive_price + cost_plus_price) / 2)
