from decimal import Decimal

from app.models.product import Product
from app.strategies.base import AbstractPricingStrategy


class DemandBasedPricingStrategy(AbstractPricingStrategy):
    """
    Prices around the competitor, nudged by forecast demand (Phase 4).

    Formula: competitor_price x (0.90 + 0.20 x demand_score)

        demand_score 0.00  (no demand expected)    -> 10% below competitor
        demand_score 0.50  (a normal day)          -> match the competitor
        demand_score 1.00  (double the usual+)     -> 10% above competitor

    demand_score comes from the demand forecast (app/services/forecast.py):
    the product repository fills it in from the deployed model before the
    strategy runs. The strategy itself stays a pure formula on a Product —
    it knows nothing about models, features or history.

    Why only +/-10%? A forecast is uncertain (test WAPE ~0.7). A bounded,
    linear nudge means a wrong forecast can only move the price a little,
    and the guardrails (cost floor, competitor ceiling) still apply after it.

    Example:
        competitor_price = 2.95, demand_score = 0.29
        -> 2.95 x (0.90 + 0.20 x 0.29) = 2.95 x 0.958 = 2.83
    """

    BASE_FACTOR = Decimal("0.90")
    DEMAND_RANGE = Decimal("0.20")

    @property
    def description(self) -> str:
        return "Competitor price adjusted by forecast demand: x0.90 (no demand) to x1.10 (double the usual)"

    def factor(self, product: Product) -> Decimal:
        return self.BASE_FACTOR + self.DEMAND_RANGE * product.demand_score

    def calculate_price(self, product: Product) -> Decimal:
        return self._round(product.competitor_price * self.factor(product))
