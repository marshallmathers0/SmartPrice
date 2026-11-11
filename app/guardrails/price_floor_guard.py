import logging
from decimal import Decimal, ROUND_CEILING

from app.config.settings import settings
from app.guardrails.base_guard import BaseGuard
from app.models.product import Product

logger = logging.getLogger(__name__)


class PriceFloorGuard(BaseGuard):
    """
    Ensures the suggested price never falls below the cost price.

    Business rule
    -------------
    A product must never be sold below its cost (minimum selling price).
    If a strategy suggests a price at or below cost, this guard raises
    it to ``cost_price × minimum_margin_multiplier``.

    The multiplier is configurable via the ``MINIMUM_MARGIN_MULTIPLIER``
    environment variable (default: 1.05 → 5% minimum margin).

    The floor is rounded UP to the cent so the raised price can never
    sit even a fraction of a cent below the minimum.

    Final say
    ---------
    This guard runs LAST in the chain. "Never sell below cost" is the one
    non-negotiable rule, so if an earlier guard (e.g. VolatilityGuard) left
    the price below the floor, the floor wins and the override is reported
    in ``applied_guards``.

    Why this guard exists
    ---------------------
    Penetration strategies or aggressive competitive pricing can
    occasionally produce prices below cost.  Without this guard, the
    business would lose money on every sale — which should never happen
    unless explicitly authorised.

    Example
    -------
        cost_price       = 500.00
        suggested_price  = 480.00   (below cost!)
        multiplier       = 1.05

        → adjusted_price = 500.00 × 1.05 = 525.00
    """

    def apply(
        self,
        price: Decimal,
        product: Product,
        applied_guards: list[str],
    ) -> Decimal:
        floor_price = (
            product.cost_price * Decimal(str(settings.minimum_margin_multiplier))
        ).quantize(Decimal("0.01"), rounding=ROUND_CEILING)

        if price < floor_price:
            # An earlier guard already adjusted the price and still left it
            # below the floor — the two rules conflict, and the floor wins.
            overrides_earlier_guard = bool(applied_guards)

            logger.warning(
                "PriceFloorGuard triggered | product=%s | original=%.2f | floor=%.2f | overrides_earlier_guard=%s",
                product.product_id,
                price,
                floor_price,
                overrides_earlier_guard,
            )

            reason = f"minimum margin of {(settings.minimum_margin_multiplier - 1) * 100:.0f}% enforced"
            if overrides_earlier_guard:
                reason += (
                    "; floor takes priority over the earlier guardrail adjustment "
                    "so the product is never sold below cost"
                )
            applied_guards.append(
                f"PriceFloorGuard: price raised from {price:.2f} to {floor_price:.2f} ({reason})"
            )
            return floor_price

        logger.debug(
            "PriceFloorGuard passed | product=%s | price=%.2f >= floor=%.2f",
            product.product_id,
            price,
            floor_price,
        )
        return price
