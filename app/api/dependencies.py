from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.product_cache import ProductCache
from app.cache.redis_client import get_redis_client
from app.config.settings import settings
from app.database.session import get_db
from app.exceptions.pricing import ModelUnavailableError
from app.ml.model_store import ModelBundle
from app.repositories.base import ProductRepository, SalesHistoryRepository
from app.repositories.cached import CachedProductRepository
from app.repositories.demand_aware import DemandAwareProductRepository
from app.repositories.postgres import PostgresProductRepository, PostgresSalesHistoryRepository
from app.services.explanation import ExplanationService
from app.services.forecast import DemandForecastService
from app.services.pricing import PricingService


# ------------------------------------------------------------------
# Dependency providers — the "composition root"
# ------------------------------------------------------------------
# This module is the ONE place that decides which concrete classes the
# application uses. Everything else asks for abstractions. Routers never
# build repositories or services themselves — they only ask Depends().
#
#   get_db() ──► PostgresProductRepository ──► CachedProductRepository (Redis) ──┐
#        │                                                                        │
#        └────► PostgresSalesHistoryRepository ──► DemandForecastService ─────────┤ (model from app.state)
#                                                                                 ▼
#                                         DemandAwareProductRepository ──► PricingService ──► endpoint
#
# FastAPI caches each dependency within ONE request, so all of these share
# the same database session.
# ------------------------------------------------------------------

DatabaseSessionDep = Annotated[AsyncSession, Depends(get_db)]


@lru_cache
def get_product_cache() -> ProductCache:
    """
    One ProductCache for the whole process (Day 7).

    A singleton because it holds the shared Redis client and the circuit-breaker
    state: when Redis goes down, every request should skip it, not each one
    rediscovering the outage. Discarded on shutdown (see main.lifespan).
    """
    return ProductCache(
        get_redis_client(),
        ttl_seconds=settings.cache_ttl_seconds,
        retry_seconds=settings.cache_retry_seconds,
    )


ProductCacheDep = Annotated[ProductCache, Depends(get_product_cache)]


# ------------------------------------------------------------------
# Model (Phase 3)
# ------------------------------------------------------------------

def get_model_bundle(request: Request) -> ModelBundle:
    """
    The forecast model loaded at startup (Phase 3).

    Loaded ONCE in main.lifespan from models/<MODEL_VERSION>/ and kept on
    app.state — loading a model from disk on every request would be slow.
    If it is missing, ML endpoints answer 503; pricing keeps working.
    """
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise ModelUnavailableError(settings.model_version)
    return bundle


ModelBundleDep = Annotated[ModelBundle, Depends(get_model_bundle)]


# ------------------------------------------------------------------
# Repositories
# ------------------------------------------------------------------

def get_base_product_repository(session: DatabaseSessionDep, cache: ProductCacheDep) -> ProductRepository:
    """PostgreSQL behind the Redis cache (Days 6-7) — products exactly as stored."""
    return CachedProductRepository(inner=PostgresProductRepository(session), cache=cache)


BaseProductRepositoryDep = Annotated[ProductRepository, Depends(get_base_product_repository)]


def get_sales_history_repository(session: DatabaseSessionDep) -> SalesHistoryRepository:
    """Daily sales history from PostgreSQL (Phase 4). Tests override this with an in-memory list."""
    return PostgresSalesHistoryRepository(session)


SalesHistoryRepositoryDep = Annotated[SalesHistoryRepository, Depends(get_sales_history_repository)]


# ------------------------------------------------------------------
# Forecasting (Phase 4)
# ------------------------------------------------------------------

def get_forecast_service(
    products: BaseProductRepositoryDep, sales: SalesHistoryRepositoryDep, bundle: ModelBundleDep
) -> DemandForecastService:
    """For the ML endpoints: a missing model is an error (503)."""
    return DemandForecastService(products, sales, bundle)


ForecastServiceDep = Annotated[DemandForecastService, Depends(get_forecast_service)]


def get_optional_forecast_service(
    request: Request, products: BaseProductRepositoryDep, sales: SalesHistoryRepositoryDep
) -> DemandForecastService | None:
    """For pricing: a missing model is NOT an error — pricing simply uses the stored demand score."""
    bundle = getattr(request.app.state, "model_bundle", None)
    return DemandForecastService(products, sales, bundle) if bundle is not None else None


OptionalForecastServiceDep = Annotated[DemandForecastService | None, Depends(get_optional_forecast_service)]


def get_product_repository(base: BaseProductRepositoryDep, forecasts: OptionalForecastServiceDep) -> ProductRepository:
    """
    Provide the ProductRepository pricing uses.

    Day 6: PostgresProductRepository.
    Day 7: wrapped in the Redis caching decorator.
    Phase 4: wrapped once more, so demand_score comes from the live forecast.
    PricingService, the strategies, the guards and the router did not change.

    Tests swap it out with
        app.dependency_overrides[get_product_repository] = lambda: fake_repo
    """
    return DemandAwareProductRepository(base, demand_score=forecasts.demand_score if forecasts else None)


ProductRepositoryDep = Annotated[ProductRepository, Depends(get_product_repository)]


def get_pricing_service(product_repository: ProductRepositoryDep) -> PricingService:
    """
    Build a PricingService with its repository injected.

    Created per request — it is cheap and holds no state of its own.
    """
    return PricingService(product_repository)


PricingServiceDep = Annotated[PricingService, Depends(get_pricing_service)]


def get_explanation_service(forecasts: ForecastServiceDep, pricing: PricingServiceDep) -> ExplanationService:
    return ExplanationService(forecasts, pricing)


ExplanationServiceDep = Annotated[ExplanationService, Depends(get_explanation_service)]
