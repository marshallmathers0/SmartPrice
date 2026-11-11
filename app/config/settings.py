from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalise_database_url(url: str) -> str:
    """
    Accept the PostgreSQL URL a cloud provider gives you, and turn it into the
    one this app's async driver (asyncpg) needs. (Phase 6)

        Neon / Supabase / Render give:   postgresql://user:pw@host/db?sslmode=require&channel_binding=require
        The app needs:                   postgresql+asyncpg://user:pw@host/db?ssl=require

    - scheme  postgres:// or postgresql://  ->  postgresql+asyncpg://
    - sslmode=X  ->  ssl=X           (asyncpg's name for the same setting)
    - channel_binding is dropped     (a libpq-only option asyncpg rejects)

    A URL that is already in the app's format passes through unchanged, so the
    same code works for local Docker and for the cloud.
    """
    parts = urlsplit(url)
    scheme = "postgresql+asyncpg" if parts.scheme in ("postgres", "postgresql") else parts.scheme
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":
            query.append(("ssl", value))
        elif key != "channel_binding":
            query.append((key, value))
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class Settings(BaseSettings):
    """
    Central configuration for the SmartPrice application.

    All values are read from environment variables or a .env file.
    Type hints ensure automatic validation and conversion by Pydantic.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",               # silently ignore unknown env vars
        protected_namespaces=(),      # allow fields starting with "model_"
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "SmartPrice"
    app_version: str = "0.1.0"
    # Phase 6: the real dataset is a UK wholesaler, so prices are in pounds
    currency_symbol: str = "£"
    debug: bool = False
    environment: str = "development"   # development | staging | production

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["*"]

    # ------------------------------------------------------------------
    # Database – PostgreSQL
    # ------------------------------------------------------------------
    # asyncpg: async driver — works with the event loop uvicorn uses on every OS
    database_url: str = "postgresql+asyncpg://smartprice:smartprice@localhost:5432/smartprice"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_connect_timeout: int = 5          # seconds to wait for a new connection
    db_check_on_startup: bool = True     # log whether the database is reachable at startup

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300          # default cache expiry: 5 minutes
    redis_socket_timeout: float = 0.5     # seconds — a cache that answers slowly is worse than no cache
    cache_retry_seconds: int = 30         # after a Redis failure, skip Redis this long before trying again
    redis_check_on_startup: bool = True   # log whether Redis is reachable at startup
    # rediss:// (TLS, e.g. Upstash) works as-is: redis-py switches to an SSL connection itself

    # ------------------------------------------------------------------
    # Machine Learning
    # ------------------------------------------------------------------
    # Phase 3: the API loads models/<model_version>/ - change the version to
    # deploy a newer model or roll back to an older one (no code change)
    model_dir: str = "models"
    model_version: str = "v1"

    # Phase 3: experiment tracking - a local SQLite file, no server needed
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_dir: str = "mlartifacts"
    mlflow_experiment: str = "smartprice-demand"

    # Data (Phase A) — relative paths are resolved from the project root,
    # so commands work no matter which directory they are run from
    raw_data_path: str = "data/raw/online_retail_ii.zip"
    processed_data_dir: str = "data/processed"
    reports_dir: str = "reports"

    # ------------------------------------------------------------------
    # Guardrails — business validation thresholds
    # ------------------------------------------------------------------
    minimum_margin_multiplier: float = 1.05   # floor = cost × 1.05 (5% min margin)
    max_volatility_multiplier: float = 1.50   # ceiling = competitor × 1.50 (50% max above)


    @field_validator("database_url")
    @classmethod
    def _accept_cloud_database_urls(cls, url: str) -> str:
        return normalise_database_url(url)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> Path:
    """Absolute paths are kept; relative ones are taken from the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    The @lru_cache decorator ensures the .env file is read only once
    for the lifetime of the application process, giving singleton-like
    behaviour across all modules that import this function.
    """
    return Settings()


# Convenience singleton — import this directly:
#   from app.config.settings import settings
settings: Settings = get_settings()
