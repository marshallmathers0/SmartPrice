from abc import ABC, abstractmethod

from app.models.forecast import DailySales
from app.models.product import Product


class ProductRepository(ABC):
    """
    Abstraction over WHERE product data lives (Repository Pattern).

    PricingService depends only on this interface, never on a concrete
    storage. Today the implementation is MockProductRepository (a dict);
    Day 6 adds a PostgreSQL implementation and the service does not change.

        PricingService
               │
               ▼
        ProductRepository        <- this class (the abstraction)
               │
               ▼
        Mock dict / PostgreSQL   <- implementations

    Why async?
    ----------
    Real implementations do I/O (PostgreSQL, Redis). Declaring the contract
    async now means a PostgreSQL repository can use an async driver without
    blocking the event loop — and without changing this interface or the
    service that calls it. A sync interface would have to change on Day 6,
    breaking the promise that only the repository changes.
    """

    @abstractmethod
    async def get_by_id(self, product_id: str) -> Product:
        """
        Return the product with this ID.

        Raises
        ------
        ProductNotFoundError
            If no product has this ID. Implementations must raise this
            instead of returning None or leaking a storage-specific error
            (KeyError, NoResultFound, ...).
        """
        ...

    @abstractmethod
    async def list_all(self) -> list[Product]:
        """Every product, ordered by product_id (Phase 6: the demo's product picker)."""
        ...


class SalesHistoryRepository(ABC):
    """
    Abstraction over WHERE daily sales history lives (Phase 4).

    Same pattern as ProductRepository: the forecast service depends only on
    this interface — PostgreSQL in production, an in-memory list in tests.
    """

    @abstractmethod
    async def recent(self, product_id: str, days: int) -> list[DailySales]:
        """
        The product's last `days` calendar days of sales, OLDEST FIRST.

        Returns an empty list if the product has no history (not an error:
        a product can exist before it has sold anything).
        """
        ...
