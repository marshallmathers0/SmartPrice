import logging

from fastapi import APIRouter

from app.api.dependencies import ProductRepositoryDep
from app.schemas.product import ProductSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/products", tags=["Products"])


@router.get("/", response_model=list[ProductSummary])
async def list_products(products: ProductRepositoryDep) -> list[ProductSummary]:
    """
    GET /api/v1/products/

    Every product in the catalog, ordered by product_id (Phase 6 — feeds the demo's picker).
    Forecasts are per product: call /forecast/{product_id}.
    """
    logger.info("GET /products")
    return [ProductSummary.from_domain(product) for product in await products.list_all()]
