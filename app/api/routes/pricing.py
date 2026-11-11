import logging
from fastapi import APIRouter, status

from app.api.dependencies import PricingServiceDep
from app.schemas.errors import ErrorResponse
from app.schemas.pricing import PricingRequest, PricingResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/pricing",
    tags=["Pricing"],
)


@router.post(
    "/",
    response_model=PricingResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Product not found"},
    },
)
async def calculate_price(request: PricingRequest, service: PricingServiceDep) -> PricingResponse:
    """
    POST /api/v1/pricing/

    Looks up the product in the catalog and returns a suggested price
    for the selected strategy.

    Day 5 change: the router no longer creates PricingService itself.
    FastAPI builds it for each request through Depends(), with its
    ProductRepository injected (see app/api/dependencies.py).

    The router contains no pricing logic and no error handling:
    an unknown product raises ProductNotFoundError, which the global
    exception handler turns into a 404.
    """
    logger.info("POST /pricing/ | product_id=%s", request.product_id)
    quote = await service.calculate_price(request.product_id, request.strategy)
    return PricingResponse.from_quote(quote)


@router.get("/", response_model=list[PricingResponse])
async def list_prices() -> list:
    """
    GET /api/v1/pricing/

    Day 2: Returns empty list.
    Future: Will return historical pricing records from the database.
    """
    logger.info("GET /pricing/ called")
    return []
