from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Product:
    """
    Core business entity — a product the business sells.

    Why a domain model (and not just the Pydantic schemas)?
    -------------------------------------------------------
    PricingRequest / PricingResponse describe the HTTP contract: the JSON
    a client sends and receives. Product describes the business itself.
    It knows nothing about FastAPI, HTTP, JSON or the database, so the
    same object works for the API, a CLI, a background job, or a unit test.

    Products are loaded by a ProductRepository (mock today, PostgreSQL later).

    Why frozen?
    -----------
    A product cannot be modified after it is loaded, so no strategy or
    guard can accidentally change its cost or competitor price while a
    price is being calculated.
    """

    product_id: str
    name: str
    cost_price: Decimal          # your cost to produce or procure (INR)
    competitor_price: Decimal    # current competitor price (INR)

    # Day 6 follow-up — demand signals for future ML / dynamic pricing.
    # No strategy reads them yet; defaults keep every existing caller valid.
    inventory: int = 0                       # units in stock
    demand_score: Decimal = Decimal("0.50")  # 0.00 (no demand) … 1.00 (very high); 0.50 = neutral

    def __post_init__(self) -> None:
        # Invariants — every pricing formula multiplies or divides by these,
        # so a product with a non-positive price must never exist.
        # Pydantic used to guarantee this for client input; now that products
        # come from storage, the domain model guarantees it itself.
        if self.cost_price <= 0:
            raise ValueError(f"Product '{self.product_id}': cost_price must be > 0, got {self.cost_price}")
        if self.competitor_price <= 0:
            raise ValueError(
                f"Product '{self.product_id}': competitor_price must be > 0, got {self.competitor_price}"
            )
        if self.inventory < 0:
            raise ValueError(f"Product '{self.product_id}': inventory must be >= 0, got {self.inventory}")
        if not Decimal("0") <= self.demand_score <= Decimal("1"):
            raise ValueError(
                f"Product '{self.product_id}': demand_score must be between 0 and 1, got {self.demand_score}"
            )
