import logging

from fastapi import APIRouter, Query, status

from app.api.dependencies import ExplanationServiceDep, ForecastServiceDep
from app.models.pricing import PricingStrategy
from app.schemas.errors import ErrorResponse
from app.schemas.forecast import ExplanationResponse, ForecastResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Forecast & Explainability"])

_ERRORS = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Product not found"},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse, "description": "Not enough sales history to forecast"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "No forecast model loaded"},
}


@router.get("/forecast/{product_id}", response_model=ForecastResponse, responses=_ERRORS)
async def forecast(product_id: str, service: ForecastServiceDep) -> ForecastResponse:
    """
    GET /api/v1/forecast/{product_id}

    Next-open-day demand forecast from the deployed model, its demand score,
    the top 5 factors behind it, and anomaly flags for the last 2 weeks.
    """
    logger.info("GET /forecast | product_id=%s", product_id)
    return ForecastResponse.from_domain(await service.forecast(product_id))


@router.get("/explain/{product_id}", response_model=ExplanationResponse, responses=_ERRORS)
async def explain(
    product_id: str,
    service: ExplanationServiceDep,
    strategy: PricingStrategy = Query(PricingStrategy.DEMAND_BASED, description="Pricing strategy to explain"),
) -> ExplanationResponse:
    """
    GET /api/v1/explain/{product_id}?strategy=demand_based

    The top factors behind the current demand forecast AND the recommended
    price — in plain language (`summary`) and in numbers.
    """
    logger.info("GET /explain | product_id=%s | strategy=%s", product_id, strategy.value)
    return ExplanationResponse.from_domain(await service.explain(product_id, strategy))
