from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

from app.models.product import Product

logger = logging.getLogger(__name__)


class BaseGuard(ABC):
    """
    Base class for the Guardrail chain (Chain of Responsibility).

    Architecture
    ------------
    Each guard has two responsibilities:
        1. Apply its own business rule to the price (may adjust the price).
        2. Pass the (possibly adjusted) price to the next guard in the chain.

    Template Method (handle)
    ------------------------
    ``handle()`` is implemented once here and is **not** meant to be overridden.
    It calls ``apply()`` — which each concrete guard **must** implement — then
    forwards the result to the next guard.

    This ensures all guards behave consistently:
        apply own rule → log what happened → forward to next guard

    Chain construction
    ------------------
    Guards are linked via ``set_next()``, which returns the *next* guard so
    that chaining reads naturally:

        volatility.set_next(floor)          # volatility → floor
        volatility.set_next(rounding).set_next(floor)  # volatility → rounding → floor

    Order matters: each guard can override the ones before it, so the
    last guard in the chain has the final say.
    """

    def __init__(self) -> None:
        self._next_guard: Optional[BaseGuard] = None

    # ------------------------------------------------------------------
    # Chain wiring
    # ------------------------------------------------------------------

    def set_next(self, guard: BaseGuard) -> BaseGuard:
        """
        Link the next guard in the chain.

        Returns the *next* guard so calls can be chained:
            a.set_next(b).set_next(c)   →  a → b → c
        """
        self._next_guard = guard
        return guard

    # ------------------------------------------------------------------
    # Template Method — fixed algorithm, do NOT override
    # ------------------------------------------------------------------

    def handle(
        self,
        price: Decimal,
        product: Product,
        applied_guards: list[str],
    ) -> Decimal:
        """
        Execute this guard's rule, then forward to the next guard.

        Parameters
        ----------
        price : Decimal
            The current suggested price (may have been adjusted by earlier guards).
        product : Product
            The product being priced — guards may need cost_price,
            competitor_price, etc. to make decisions.
        applied_guards : list[str]
            A mutable list that each guard appends to when it modifies the
            price.  This list is surfaced in the API response so the client
            can see which rules fired.

        Returns
        -------
        Decimal
            The (possibly adjusted) price after this guard and all
            subsequent guards have run.
        """
        # Step 1 — apply this guard's rule
        adjusted_price = self.apply(price, product, applied_guards)

        # Step 2 — forward to the next guard (if any)
        if self._next_guard is not None:
            return self._next_guard.handle(adjusted_price, product, applied_guards)

        # End of chain — return final price
        return adjusted_price

    # ------------------------------------------------------------------
    # Abstract hook — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(
        self,
        price: Decimal,
        product: Product,
        applied_guards: list[str],
    ) -> Decimal:
        """
        Apply this guard's business rule.

        If the price is acceptable, return it unchanged.
        If the price violates the rule, adjust it and append a description
        to ``applied_guards``.
        """
        ...
