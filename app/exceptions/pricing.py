class PricingException(Exception):
    """
    Base class for every business error in SmartPrice.

    Business exceptions describe WHAT went wrong in business terms.
    They know nothing about HTTP — there is no status code here.
    Turning them into HTTP responses is the job of
    app/api/exception_handlers.py, so the service and repositories stay
    independent of FastAPI.

    ``code`` is a stable, machine-readable identifier clients can branch on.
    Messages are for humans and may be reworded; codes should never change.

    Future business errors inherit from this class:
        class OutOfStockError(PricingException): ...
    """

    code: str = "pricing_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProductNotFoundError(PricingException):
    """
    Raised by a ProductRepository when no product has the requested ID.

    Why not return None or let KeyError escape?
    -------------------------------------------
    - None forces every caller to remember an ``if product is None`` check.
    - KeyError (dict), NoResultFound (SQLAlchemy), ... leak HOW the data is
      stored. Every repository raising this one exception means callers
      never change when the storage does.
    """

    code = "product_not_found"

    def __init__(self, product_id: str) -> None:
        super().__init__(f"Product '{product_id}' was not found")
        self.product_id = product_id


class ModelUnavailableError(PricingException):
    """
    The configured forecast model version is not loaded (missing folder,
    or trained on different features than the current code produces).

    The pricing endpoints do not need the model and keep working; only
    ML endpoints answer 503 Service Unavailable until a valid version is
    configured — it is a server-side problem the client cannot fix.
    """

    code = "model_unavailable"

    def __init__(self, version: str) -> None:
        super().__init__(f"Forecast model '{version}' is not available")
        self.version = version


class NoSalesHistoryError(PricingException):
    """
    A forecast was requested for a product without enough sales history.

    The forecast needs at least 28 days of past demand (its longest rolling
    window). The demo products SKU-12345 etc. have no history at all.
    """

    code = "no_sales_history"

    def __init__(self, product_id: str, days_needed: int) -> None:
        super().__init__(f"Product '{product_id}' needs at least {days_needed} days of sales history to forecast")
        self.product_id = product_id
