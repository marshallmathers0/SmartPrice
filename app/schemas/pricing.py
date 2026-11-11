from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Day 5: PricingStrategy now lives in the domain layer (app/models/pricing.py).
# It is imported here because it is also part of the HTTP contract.
from app.models.pricing import PriceQuote, PricingStrategy


# ------------------------------------------------------------------
# Request Schema — what the client SENDS us
# ------------------------------------------------------------------
class PricingRequest(BaseModel):
    """
    Validated input for a pricing calculation request.

    Pydantic rejects the request (HTTP 422) before it ever reaches
    business logic if any field fails its constraint.

    Day 5 change: the client now sends only WHICH product to price.
    Cost and competitor price are loaded from the ProductRepository —
    the catalog is the single source of truth, not the client.

    extra="forbid": a client still sending the old cost_price /
    competitor_price fields gets a clear 422 instead of having its
    prices silently ignored.
    """
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(
        ...,                            # required field (no default)
        min_length=1,
        max_length=64,
        description="Catalog product identifier (SKU)",
        examples=["SKU-12345"],
    )
    strategy: PricingStrategy = Field(
        default=PricingStrategy.COMPETITIVE,
        description="Pricing strategy to apply",
    )


# ------------------------------------------------------------------
# Response Schema — what we RETURN to the client
# ------------------------------------------------------------------
class PricingResponse(BaseModel):
    """
    Structured response after a pricing calculation.
    Pydantic serializes this to JSON automatically.

    Day 4 change: added ``guardrails_applied`` so the client can see
    which business validation rules fired and what adjustments were made.

    Day 5 change: built from the service's PriceQuote via from_quote(),
    so the service never needs to know this HTTP shape exists.
    """
    product_id: str
    product_name: str
    cost_price: Decimal
    competitor_price: Decimal
    inventory: int = Field(description="Units in stock")
    demand_score: Decimal = Field(description="Demand signal from 0.00 (none) to 1.00 (very high)")
    suggested_price: Decimal = Field(
        description="Suggested selling price after guardrails (INR)"
    )
    strategy_used: PricingStrategy
    margin_percent: Decimal = Field(
        description="Profit margin as a percentage"
    )
    guardrails_applied: Optional[list[str]] = Field(
        default=None,
        description="List of guardrail rules that adjusted the price (null if none fired)",
    )
    message: str = Field(
        description="Human-readable summary of the pricing decision"
    )

    @classmethod
    def from_quote(cls, quote: PriceQuote) -> "PricingResponse":
        """Map the domain result (PriceQuote) onto the HTTP contract."""
        return cls(
            product_id=quote.product.product_id,
            product_name=quote.product.name,
            cost_price=quote.product.cost_price,
            competitor_price=quote.product.competitor_price,
            inventory=quote.product.inventory,
            demand_score=quote.product.demand_score,
            suggested_price=quote.suggested_price,
            strategy_used=quote.strategy,
            margin_percent=quote.margin_percent,
            guardrails_applied=list(quote.guardrails_applied) or None,
            message=quote.message,
        )
