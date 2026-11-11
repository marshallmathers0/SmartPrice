import logging
from decimal import Decimal, ROUND_FLOOR

from app.config.settings import settings
from app.guardrails.base_guard import BaseGuard
from app.models.product import Product

logger = logging.getLogger(__name__)


class VolatilityGuard(BaseGuard):
    """
    Caps the suggested price if it deviates too far from the competitor price.

    Business rule
    -------------
    The suggested price must not exceed ``competitor_price × max_volatility_multiplier``.
    If a strategy (e.g. cost_plus with very high cost) produces a price that
    would be far above the market, this guard brings it back to a sane
    ceiling relative to the competition.

    The multiplier is configurable via ``MAX_VOLATILITY_MULTIPLIER``
    (default: 1.50 → price capped at 150% of competitor price).

    The ceiling is rounded DOWN to the cent so the capped price can never
    sit even a fraction of a cent above the limit.

    This is a soft rule: it runs before ``PriceFloorGuard``, so if the
    ceiling would push the price below the cost floor, the floor wins.

    Why this guard exists
    ---------------------
    Without a volatility cap, cost-plus strategies on high-cost items
    could generate prices that are wildly above market rate, making the
    product unsellable.  This guard answers:
    "Is the calculated price acceptable relative to the market?"

    Example
    -------
        competitor_price = 999.00
        suggested_price  = 1800.00  (80% above competitor!)
        multiplier       = 1.50

        ceiling          = 999.00 × 1.50 = 1498.50
        → adjusted_price = 1498.50
    """

    def apply(
        self,
        price: Decimal,
        product: Product,
        applied_guards: list[str],
    ) -> Decimal:
        ceiling = (
            product.competitor_price * Decimal(str(settings.max_volatility_multiplier))
        ).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)

        if price > ceiling:
            logger.warning(
                "VolatilityGuard triggered | product=%s | original=%.2f | ceiling=%.2f",
                product.product_id,
                price,
                ceiling,
            )
            applied_guards.append(
                f"VolatilityGuard: price capped from {price:.2f} to {ceiling:.2f} "
                f"(max {(settings.max_volatility_multiplier - 1) * 100:.0f}% above competitor enforced)"
            )
            return ceiling

        logger.debug(
            "VolatilityGuard passed | product=%s | price=%.2f <= ceiling=%.2f",
            product.product_id,
            price,
            ceiling,
        )
        return price
