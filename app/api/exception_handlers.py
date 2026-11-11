import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.exceptions.pricing import ModelUnavailableError, NoSalesHistoryError, PricingException, ProductNotFoundError
from app.schemas.errors import ErrorResponse

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Business exception → HTTP status code
# ------------------------------------------------------------------
# Business exceptions know nothing about HTTP. This table is the only
# place where they meet status codes. A new exception type = one new line.
#
# Lookup walks the exception's class hierarchy, so a subclass inherits its
# parent's status code, and any unmapped PricingException falls back to 400.
# ------------------------------------------------------------------
_STATUS_CODES: dict[type[PricingException], int] = {
    ProductNotFoundError: status.HTTP_404_NOT_FOUND,
    ModelUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    NoSalesHistoryError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    PricingException: status.HTTP_400_BAD_REQUEST,
}


def _status_code_for(exc: PricingException) -> int:
    return next(_STATUS_CODES[cls] for cls in type(exc).__mro__ if cls in _STATUS_CODES)


async def pricing_exception_handler(request: Request, exc: PricingException) -> JSONResponse:
    """Turn any business exception into a JSON error response."""
    status_code = _status_code_for(exc)
    logger.info(
        "Business error | %s %s | status=%d | code=%s | %s",
        request.method,
        request.url.path,
        status_code,
        exc.code,
        exc.message,
    )
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(detail=exc.message, code=exc.code).model_dump(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last line of defence for errors nobody anticipated.

    Returns a generic JSON 500 so internal details (stack traces, SQL,
    connection strings) never leak to the client. Starlette re-raises the
    exception after this response is sent, so the server still logs the
    full traceback.
    """
    logger.error(
        "Unhandled error | %s %s | %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail="Internal server error", code="internal_error").model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers. Called once from create_app()."""
    app.add_exception_handler(PricingException, pricing_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
