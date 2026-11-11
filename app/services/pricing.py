import logging
from decimal import Decimal, ROUND_HALF_UP

from app.config.settings import settings
from app.factories.strategy_factory import StrategyFactory
from app.guardrails.base_guard import BaseGuard
from app.guardrails.price_floor_guard import PriceFloorGuard
from app.guardrails.volatility_guard import VolatilityGuard
from app.models.pricing import PriceQuote, PricingStrategy
from app.models.product import Product
from app.repositories.base import ProductRepository

logger = logging.getLogger(__name__)


class PricingService:
    """
    Orchestrates the pricing workflow.

    Day 3 change: All pricing algorithms moved to individual strategy
    classes under app/strategies/. The service no longer contains any
    pricing math — it only orchestrates.

    Day 4 change: After the strategy calculates a raw price, the service
    now runs it through a **guard chain** (Chain of Responsibility) that
    enforces business validation rules.

    Day 5 change: Repository Pattern + Dependency Injection.
    The service no longer reads prices from the HTTP request. It receives
    a ProductRepository through its constructor and loads the product
    itself. It depends only on the abstraction, so replacing the mock
    with PostgreSQL changes nothing here.

    It also works purely with domain objects (Product in, PriceQuote out)
    and imports nothing from FastAPI or the API schemas. A missing product
    surfaces as ProductNotFoundError from the repository; the service lets
    it propagate and the global exception handler turns it into a 404.

        1. Load the product from the injected repository
        2. Ask the factory for the correct strategy
        3. Call strategy.calculate_price()  →  raw price
        4. Run raw price through the guard chain  →  validated price
        5. Calculate margin (on the validated price)
        6. Return a PriceQuote

    Guard Chain (current):
        VolatilityGuard  →  PriceFloorGuard

    Why this order?
    ---------------
    Each guard can override the ones before it, so the LAST guard has the
    final say. PriceFloorGuard goes last because "never sell below cost"
    is non-negotiable: when the volatility ceiling is below the cost floor
    (cost far above competitor), the floor wins and says so in
    guardrails_applied.

    Why a guard chain?
    ------------------
    - Strategies answer: "How should the price be calculated?"
    - Guardrails answer: "Is the calculated price acceptable?"

    Separating these concerns means:
    - Guards can be added/removed without touching strategies.
    - Each guard is independently testable.
    - The chain order is explicitly defined in one place.

    Why _build_guard_chain() is a helper method?
    ---------------------------------------------
    Keeps the orchestration logic in calculate_price() clean.
    Future additions become simple — just add a new guard to the chain
    (anything that can lower the price goes before PriceFloorGuard):

        VolatilityGuard
              ↓
        MaximumPriceGuard
              ↓
        RoundingGuard
              ↓
        PriceFloorGuard
              ↓
        TaxGuard

    Only _build_guard_chain() changes. calculate_price() is untouched.
    """

    def __init__(self, product_repository: ProductRepository) -> None:
        # Injected from outside (see app/api/dependencies.py).
        # The service never decides WHICH repository it gets — only that it
        # can call get_by_id() on it.
        self._product_repository = product_repository

    async def calculate_price(self, product_id: str, strategy: PricingStrategy) -> PriceQuote:
        # Step 1 — Load the product (raises ProductNotFoundError if missing)
        product = await self._product_repository.get_by_id(product_id)

        logger.info(
            "Calculating price | product=%s | strategy=%s | cost=%.2f | competitor=%.2f",
            product.product_id,
            strategy.value,
            product.cost_price,
            product.competitor_price,
        )

        # Step 2 — Ask the factory for the correct strategy object
        pricing_strategy = StrategyFactory.get_strategy(strategy)

        # Step 3 — Delegate pricing to the strategy (raw price, no validation)
        raw_price = pricing_strategy.calculate_price(product)
        logger.info(
            "Raw price from strategy | product=%s | strategy=%s | raw_price=%.2f",
            product.product_id,
            strategy.value,
            raw_price,
        )

        # Step 4 — Run the raw price through the guard chain
        guard_chain = self._build_guard_chain()
        applied_guards: list[str] = []
        suggested_price = guard_chain.handle(raw_price, product, applied_guards)

        if applied_guards:
            logger.info(
                "Guards modified price | product=%s | raw=%.2f | final=%.2f | guards=%s",
                product.product_id,
                raw_price,
                suggested_price,
                applied_guards,
            )

        # Step 5 — Calculate margin on the FINAL (post-guard) price
        margin_percent = (
            (suggested_price - product.cost_price) / suggested_price * 100
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.info(
            "Price calculated | product=%s | strategy=%s | suggested=%.2f | margin=%.2f%%",
            product.product_id,
            strategy.value,
            suggested_price,
            margin_percent,
        )

        # Step 6 — Return a domain result; the API layer decides how to present it
        return PriceQuote(
            product=product,
            strategy=strategy,
            suggested_price=suggested_price,
            margin_percent=margin_percent,
            guardrails_applied=tuple(applied_guards),
            message=self._build_message(strategy, suggested_price, product),
        )

    # ------------------------------------------------------------------
    # Guard chain construction
    # ------------------------------------------------------------------

    def _build_guard_chain(self) -> BaseGuard:
        """
        Assemble the guard chain.

        Current chain:
            VolatilityGuard → PriceFloorGuard

        PriceFloorGuard stays last so it always has the final say.
        To extend, link a new guard before it:
            volatility.set_next(new_guard).set_next(floor)

        Only this method changes when adding new guards.
        calculate_price() is completely untouched.
        """
        volatility = VolatilityGuard()
        floor = PriceFloorGuard()

        # Wire the chain: volatility → floor (floor last = floor wins)
        volatility.set_next(floor)

        # Return the head of the chain
        return volatility

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_message(self, strategy: PricingStrategy, suggested: Decimal, product: Product) -> str:
        """Human-readable summary of the pricing decision."""
        diff = suggested - product.competitor_price
        direction = "below" if diff < 0 else "above"
        return (
            f"Strategy '{strategy.value}' applied. "
            f"Suggested price is {settings.currency_symbol}{abs(diff):.2f} {direction} competitor price."
        )
