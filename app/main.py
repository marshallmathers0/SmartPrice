from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.config.settings import settings
from app.api import dependencies
from app.api.exception_handlers import register_exception_handlers
from app.api.routes import forecast, model, pricing, products
from app.config.logging import setup_logging
from app.cache import redis_client
from app.database import session as database
from app.ml.model_store import ModelNotAvailableError, load_bundle

# Configure logging at the very start so everything else can use it
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Flow:
    1. Code before 'yield' runs when the app starts.
    2. 'yield' pauses execution and the FastAPI application runs.
    3. Code after 'yield' runs when the app shuts down.
    """
    # --- Startup phase ---
    logger.info("Starting up SmartPrice application...")

    # Day 6: verify the database is reachable. The app still starts if it is
    # not (so /health can report it), but the log says so immediately.
    if settings.db_check_on_startup:
        if await database.ping_database():
            logger.info("Database connection OK")
        else:
            logger.error("Database unavailable - pricing requests will fail until it is reachable")

    # Day 7: same for Redis — but Redis is optional (a cache), so a warning, not an error
    if settings.redis_check_on_startup:
        if await redis_client.ping_redis():
            logger.info("Redis connection OK - product cache enabled")
        else:
            logger.warning("Redis unavailable - serving every request from PostgreSQL")
    # Phase 3: load the configured model version once, keep it in memory.
    # A missing model does not stop the app - pricing still works; ML endpoints answer 503.
    try:
        app.state.model_bundle = load_bundle(settings.model_version)
        logger.info(
            "Forecast model %s loaded | %s | test WAPE %.3f",
            settings.model_version,
            app.state.model_bundle.metadata["model_name"],
            app.state.model_bundle.metadata["metrics"]["test"]["wape"],
        )
    except ModelNotAvailableError as exc:
        app.state.model_bundle = None
        logger.warning("No forecast model loaded - %s", exc)

    yield  # Application runs while suspended here

    # --- Shutdown phase ---
    logger.info("Shutting down SmartPrice application...")
    # Day 6: close every pooled database connection gracefully
    await database.dispose_engine()
    # Day 7: close the Redis connection pool and drop the cached singletons
    await redis_client.close_redis_client()
    dependencies.get_product_cache.cache_clear()

def create_app() -> FastAPI:
    """
    Application factory.
    Assembles and configures the FastAPI application.
    Contains no business logic.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",      # Swagger UI
        redoc_url="/redoc"     # ReDoc UI
    )

    # Translate business exceptions into HTTP responses in one place
    register_exception_handlers(app)

    # Register routers — each feature gets its own file
    app.include_router(pricing.router, prefix=settings.api_prefix)
    app.include_router(model.router, prefix=settings.api_prefix)
    app.include_router(forecast.router, prefix=settings.api_prefix)
    app.include_router(products.router, prefix=settings.api_prefix)

    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Health check — is the app running, and can it reach its backing services?

        Always HTTP 200 while the process is alive. ``status`` becomes
        "degraded" only when the database is unreachable: Redis is a cache,
        so the app works without it (just slower) and it is reported separately.
        """
        database_ok = await database.ping_database()
        cache_ok = await redis_client.ping_redis()
        return {
            "status": "ok" if database_ok else "degraded",
            "app_name": settings.app_name,
            "environment": settings.environment,
            "database": "ok" if database_ok else "unavailable",
            "cache": "ok" if cache_ok else "unavailable",
            "model": settings.model_version if getattr(app.state, "model_bundle", None) else "unavailable",
        }

    return app

# The ASGI application instance
app = create_app()
