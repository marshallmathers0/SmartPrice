from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_HALF_UP

from app.models.product import Product


class AbstractPricingStrategy(ABC):
    """
    Abstract base class — defines the contract every pricing strategy must follow.

    Why an abstract class?
    ----------------------
    It guarantees that every concrete strategy (Competitive, CostPlus, etc.)
    exposes exactly the same interface: calculate_price(product) -> Decimal.

    The service and factory never care WHICH strategy they have.
    They only call .calculate_price(product) and get a Decimal back.
    This is called "programming to an interface, not an implementation."
    """

    @abstractmethod
    def calculate_price(self, product: Product) -> Decimal:
        """
        Calculate and return the suggested price for a product.

        Every subclass MUST implement this method.
        If it does not, Python raises TypeError at instantiation time.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this strategy."""
        ...

    def _round(self, value: Decimal) -> Decimal:
        """
        Shared rounding helper available to all strategies.
        Rounds to 2 decimal places using ROUND_HALF_UP (financial standard).
        """
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
