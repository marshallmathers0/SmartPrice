import logging

from app.models.pricing import PricingStrategy
from app.strategies.base import AbstractPricingStrategy
from app.strategies.competitive import CompetitivePricingStrategy
from app.strategies.cost_plus import CostPlusPricingStrategy
from app.strategies.demand_based import DemandBasedPricingStrategy
from app.strategies.dynamic import DynamicPricingStrategy
from app.strategies.penetration import PenetrationPricingStrategy

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    Factory responsible for creating the correct pricing strategy object.

    Why a Factory?
    --------------
    The service should not need to know HOW to build a strategy object.
    It should only ask: "Give me the strategy for 'competitive'."
    The factory encapsulates all object-creation logic in one place.

    Benefits:
    1. Single place to change object construction (e.g., if a strategy
       needs configuration or dependencies in the future, only the factory
       changes — the service is untouched).
    2. Eliminates conditional object creation from the service layer.
    3. Enforces Open/Closed Principle: adding a new strategy = one new
       class + one new line in the registry. Nothing else changes.

    Open/Closed Principle:
    ----------------------
    "Open for extension, closed for modification."
    To add HolidayStrategy: create the class, add it to _registry.
    Every existing strategy, the service, and the router are unchanged.
    """

    # ------------------------------------------------------------------
    # Registry — maps enum values to strategy classes
    # This is the ONLY place you touch when adding a new strategy.
    # ------------------------------------------------------------------
    _registry: dict[PricingStrategy, type[AbstractPricingStrategy]] = {
        PricingStrategy.COMPETITIVE: CompetitivePricingStrategy,
        PricingStrategy.COST_PLUS:   CostPlusPricingStrategy,
        PricingStrategy.DYNAMIC:     DynamicPricingStrategy,
        PricingStrategy.PENETRATION: PenetrationPricingStrategy,
        PricingStrategy.DEMAND_BASED: DemandBasedPricingStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy: PricingStrategy) -> AbstractPricingStrategy:
        """
        Look up and instantiate the correct strategy class.

        Parameters
        ----------
        strategy : PricingStrategy
            The enum value selected by the client.

        Returns
        -------
        AbstractPricingStrategy
            A ready-to-use strategy instance.

        Raises
        ------
        ValueError
            If the strategy is not registered. This should never happen
            because the Pydantic enum validates strategy before it reaches
            the factory — but we guard defensively anyway.
        """
        strategy_class = cls._registry.get(strategy)

        if strategy_class is None:
            logger.error("Unknown strategy requested: %s", strategy)
            raise ValueError(
                f"No strategy registered for '{strategy}'. "
                f"Available: {list(cls._registry.keys())}"
            )

        logger.debug("Factory creating strategy: %s", strategy_class.__name__)
        return strategy_class()

    @classmethod
    def available_strategies(cls) -> list[str]:
        """Return a list of all registered strategy names. Useful for diagnostics."""
        return [s.value for s in cls._registry]
