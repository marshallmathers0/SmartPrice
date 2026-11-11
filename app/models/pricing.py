from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from app.models.product import Product


# ------------------------------------------------------------------
# Enum — only valid strategy values accepted
# ------------------------------------------------------------------
class PricingStrategy(str, Enum):
    """
    Valid pricing strategies the engine can use.
    Using an Enum instead of plain strings prevents typos like
    strategy="competitiv" from slipping through undetected.

    Day 5 change: moved here from app/schemas/pricing.py. The strategy is a
    business concept, so it lives in the domain layer; the API schemas
    import it from here (the API depends on the domain, never the reverse).
    """
    COMPETITIVE  = "competitive"    # price based on competitors
    COST_PLUS    = "cost_plus"      # cost + fixed margin
    DYNAMIC      = "dynamic"        # blend of competitive and cost-plus
    PENETRATION  = "penetration"    # low price to capture market share
    DEMAND_BASED = "demand_based"   # competitor price nudged by the ML demand forecast (Phase 4)


# ------------------------------------------------------------------
# Result of a pricing calculation
# ------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PriceQuote:
    """
    The outcome of pricing one product — what PricingService returns.

    Why not return PricingResponse directly?
    ----------------------------------------
    PricingResponse is the HTTP contract (API layer). If the service built
    it, the business layer would depend on the API layer, and any change
    to the JSON shape would mean editing the service. Instead the service
    returns this plain domain object and the router converts it with
    PricingResponse.from_quote().
    """
    product: Product
    strategy: PricingStrategy
    suggested_price: Decimal
    margin_percent: Decimal
    guardrails_applied: tuple[str, ...]   # empty when no guard changed the price
    message: str
