# SmartPrice — Engineering Development Log

> A day-by-day record of how this backend was built, what decisions were made, and why.
> Written from the perspective of a software engineer building for production.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Day 1 — Infrastructure and Application Skeleton](#day-1--infrastructure-and-application-skeleton)
- [Day 2 — Layered Architecture](#day-2--layered-architecture)
- [Day 3 — Strategy Pattern and Factory Pattern](#day-3--strategy-pattern-and-factory-pattern)
- [Day 4 — Chain of Responsibility (Guardrails)](#day-4--chain-of-responsibility-guardrails)
- [Day 4 Follow-up — Guardrail Fixes and Tests](#day-4-follow-up--guardrail-fixes-and-tests)
- [Day 5 — Repository Pattern, Dependency Injection & Exception Handling](#day-5--repository-pattern-dependency-injection--exception-handling)
- [Day 6 — PostgreSQL Integration, SQLAlchemy ORM & Database Migrations](#day-6--postgresql-integration-sqlalchemy-orm--database-migrations)
- [Day 6 Follow-up — Inventory and Demand Score](#day-6-follow-up--inventory-and-demand-score)
- [Day 7 — Redis Integration, Cache-Aside Pattern & Decorator Pattern](#day-7--redis-integration-cache-aside-pattern--decorator-pattern)
- [Day 8 — Real Sales Data (ML Phase A)](#day-8--real-sales-data-ml-phase-a)
- [Day 9 — Demand Forecasting & Honest Evaluation (ML Phase 1)](#day-9--demand-forecasting--honest-evaluation-ml-phase-1)
- [Day 10 — Anomaly Detection & Labelled Evaluation (ML Phase 2)](#day-10--anomaly-detection--labelled-evaluation-ml-phase-2)
- [Day 11 — Experiment Tracking, Versioned Models & Model Loading (ML Phase 3)](#day-11--experiment-tracking-versioned-models--model-loading-ml-phase-3)
- [Day 12 — Explainability & ML-Driven Pricing (ML Phase 4)](#day-12--explainability--ml-driven-pricing-ml-phase-4)
- [Day 13 — Test Audit & Hardening (ML Phase 5)](#day-13--test-audit--hardening-ml-phase-5)
- [Day 14 — Deployment Readiness & Live Demo (ML Phase 6)](#day-14--deployment-readiness--live-demo-ml-phase-6)
- [Day 15 — Documentation & the Final Numbers (ML Phase 7)](#day-15--documentation--the-final-numbers-ml-phase-7)
- [Day 16 — Running It Day to Day, Understanding It, and Hosting It](#day-16--running-it-day-to-day-understanding-it-and-hosting-it)
- [Day 17 — Live in Production: How the Hosting Works, and Two Debugging Stories](#day-17--live-in-production-how-the-hosting-works-and-two-debugging-stories)
- [Running the Application](#running-the-application)
- [Key Engineering Principles Applied](#key-engineering-principles-applied)

---

## Project Overview

**SmartPrice** is a FastAPI backend that forecasts next-day product demand from real retail data, explains every forecast feature by feature, and turns it into a guardrailed, explainable price.

It works from:
- the product's cost and competitor price,
- its real daily sales history (UCI Online Retail II, 1,067,371 transactions, Dec 2009 - Dec 2011),
- a chosen pricing strategy — including `demand_based`, which uses the ML forecast.

**Business Goal:** Replace manual guesswork in pricing decisions with data-driven recommendations that can be explained and audited.

> **How this log reads:** Days 1-7 build the backend (layers, patterns, PostgreSQL, Redis). Until Day 8 the project had **no machine learning** — the "ML" wording in early entries described the goal, not the code. Days 8-15 add it: real data, forecasting, anomaly detection, MLflow, explainability, tests, deployment and documentation. The [README](README.md) summarises the final system and its measured results.

---

## Tech Stack

| Category | Technology | Why |
|---|---|---|
| Web Framework | FastAPI | Async, fast, auto-generates OpenAPI docs, native Pydantic support |
| Validation | Pydantic v2 | Type-safe request/response models, auto validation |
| Configuration | Pydantic Settings | Reads `.env`, validates types, supports defaults |
| ASGI Server | Uvicorn | Production-grade async server for FastAPI |
| Database | PostgreSQL + SQLAlchemy | Reliable relational DB, excellent ORM |
| Migrations | Alembic | Version-controlled database schema changes |
| Cache | Redis | Sub-millisecond in-memory cache, pub/sub |
| ML | scikit-learn, numpy, pandas | Industry-standard Python ML stack (Ridge regression is the deployed model) |
| ML (training only) | LightGBM, SHAP, matplotlib | Tree-model candidate, SHAP summary plots, report charts (Days 9-12) |
| Experiment tracking | MLflow (local SQLite store) | Parameters, metrics, dataset hash and artifacts for every training run (Day 11) |
| Demo | Streamlit | A clickable page that calls the API (Day 14) |
| Deployment | Docker, Render, Neon / Supabase, Upstash | Slim image + free-tier hosting (Day 14) |
| Testing | pytest, pytest-asyncio, httpx, pytest-cov | Async-capable test suite with coverage |

---

# Day 1 — Infrastructure and Application Skeleton

**Goal:** The application should start correctly but contain almost no business logic.

**Philosophy:** Build the foundation right. Every future feature sits on top of this. A shaky foundation breaks everything above it.

---

## Step 1 — Folder Structure

**What was created:**
```
SmartPrice(BackendProject)/
│
├── .env
├── .gitignore
├── requirements.txt
├── main.py                    <- entry point at project root
│
└── app/
    ├── api/                   <- HTTP layer (routers, endpoints)
    ├── cache/                 <- Redis caching logic
    ├── config/                <- settings, logging
    ├── factories/             <- object creation patterns
    ├── guardrails/            <- ML output safety checks
    ├── ml/                    <- machine learning models
    ├── schemas/               <- Pydantic request/response models
    ├── services/              <- business logic
    ├── strategies/            <- pricing algorithm implementations
    └── utils/                 <- shared helpers
```

**Why this structure?**

This follows **Separation of Concerns** — one of the most important principles in software engineering. Each folder has one job and one job only:

- `api/` answers: "Who called us and what did they send?"
- `services/` answers: "What should we do with this request?"
- `schemas/` answers: "Is this data valid?"
- `config/` answers: "What are our settings?"

If pricing logic changes tomorrow, you only touch `services/`. If the API format changes, you only touch `schemas/`. Nothing else breaks.

**Why is `main.py` inside `app/`?**

`main.py` contains the actual FastAPI application.
 A root-level main.py makes sense for simple programs, but it is not recommended for large or complex applications.
---

## Step 2 — `requirements.txt`

**What it is:** A manifest of all external libraries the project depends on.

**Why it matters in production:**

When you deploy to a server (AWS, GCP, Docker), that server starts with zero libraries. `pip install -r requirements.txt` reproducibly installs the exact same environment everywhere — your laptop, your teammate's laptop, the production server — all identical.

**Key libraries and why each was chosen:**

```
fastapi              <- web framework
uvicorn[standard]    <- ASGI server (runs FastAPI)
pydantic             <- data validation
pydantic-settings    <- reads .env files into typed Python objects

sqlalchemy           <- database ORM (Object Relational Mapper)
psycopg[binary]      <- PostgreSQL driver (psycopg v3)
alembic              <- database migrations

redis[hiredis]       <- Redis client + C-extension for speed

numpy                <- numerical computation
pandas               <- data manipulation
scikit-learn         <- ML algorithms
scipy                <- scientific computing
joblib               <- saving/loading ML models

python-dotenv        <- loads .env into environment
httpx                <- async HTTP client (for testing and external API calls)
pytest               <- test runner
pytest-asyncio       <- async test support
```

**Install command:**
```bash
pip install -r requirements.txt
```

---

## Step 3 — Centralized Configuration

**File:** `app/config/settings.py`

**Problem being solved:**

Without centralized config, you scatter this pattern everywhere:
```python
# in database.py
os.getenv("DATABASE_URL")

# in cache.py
os.getenv("REDIS_URL")

# in ml/model.py
os.getenv("MODEL_PATH")
```

Problems:
- If a variable name changes, you update it in 20 files
- No validation — if `DATABASE_URL` is missing, you get a cryptic error deep inside the code
- No type conversion — `os.getenv()` always returns a string, even if you need an integer or boolean

**Solution — Pydantic Settings:**
```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "SmartPrice"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    database_url: str = "postgresql+psycopg://user:password@localhost:5432/smartprice"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300

    model_path: str = "app/ml/models"
    prediction_threshold: float = 0.75

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings: Settings = get_settings()
```

**Key Concepts:**

### `class Settings(BaseSettings)`
`Settings` inherits everything from `BaseSettings`. This gives it the superpower of automatically reading values from environment variables and `.env` files.

### Type hints provide automatic conversion
```python
debug: bool = False
cache_ttl_seconds: int = 300
```
Pydantic reads `DEBUG=true` from `.env` (a string) and automatically converts it to Python `True` (a boolean). With `os.getenv()`, you would have to convert manually everywhere.

### `@lru_cache` — The Singleton Pattern

Without cache:
```
Every import of settings.py
    -> reads .env file from disk
    -> creates new Settings object
    -> wastes I/O on every request
```

With `@lru_cache`:
```
First call -> reads .env -> creates Settings object -> stores in memory
All future calls -> returns the same cached object (zero disk I/O)
```

This is called the **Singleton Pattern** — one instance, shared everywhere.

### Usage across the entire project
```python
from app.config.settings import settings

settings.database_url      # "postgresql+psycopg://..."
settings.redis_url         # "redis://localhost:6379/0"
settings.debug             # False
```

**Why not use `os.getenv()`?**

| `os.getenv()` | Pydantic Settings |
|---|---|
| Returns raw string always | Converts to correct Python type |
| No validation | Validates on startup, fails clearly |
| No defaults | Supports defaults |
| No IDE autocomplete | Full IDE autocomplete |
| Silent if missing | Crashes at startup with a clear error |

---

## Step 4 — Centralized Logging

**File:** `app/config/logging.py`

**Problem being solved:**

`print()` statements are fine for a quick script. For a production backend handling thousands of requests, they are useless because:
- No timestamps (when exactly did this happen?)
- No severity levels (is this a crash or just informational?)
- No source identification (which of 50 files printed this?)
- Cannot be filtered (you see everything or nothing)

**Solution — Python's `logging` module, configured once:**

```python
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def setup_logging() -> None:
    log_level = logging.DEBUG if settings.debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if not root_logger.handlers:
        root_logger.addHandler(handler)
```

**Log format output:**
```
2026-09-06 16:09:35 | INFO     | app.services.pricing | Price calculated: 969.03
       ^                 ^               ^                       ^
  timestamp            level          WHERE                  message
```

**Logging Levels — from least to most severe:**

| Level | When to use |
|---|---|
| `DEBUG` | Granular step-by-step detail. Development only. |
| `INFO` | Normal operations: "app started", "user logged in", "order processed" |
| `WARNING` | Unexpected but app continues: "using fallback cached price" |
| `ERROR` | A specific operation failed: "DB query failed", "payment declined" |
| `CRITICAL` | App cannot continue: "cannot connect to DB", "disk full" |

In a real production system, a `CRITICAL` log often triggers an automated alert (PagerDuty, Slack ping) to wake an engineer at 3 AM.

**Environment-aware log level:**
```python
log_level = logging.DEBUG if settings.debug else logging.INFO
```
- `DEBUG=True` in `.env` -> you see every internal detail (development)
- `DEBUG=False` in `.env` -> only INFO and above (production, clean and readable)

**How every module uses it — always 2 lines:**
```python
import logging
logger = logging.getLogger(__name__)
# __name__ is a Python special variable that automatically becomes
# the module's dotted path, e.g. "app.services.pricing"

logger.info("Price calculated: %.2f", price)
logger.error("DB connection failed: %s", error)
logger.exception("Unexpected crash")  # logs message + full traceback automatically
```

**Why centralize?** Change the log format in one file. All 50 modules across the project automatically use the new format. No changes needed anywhere else.

---

## Step 5 — FastAPI Application Assembly

**File:** `app/main.py`

**The golden rule:** `main.py` must never contain business logic. Its only job is to assemble, configure, and start the application.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.config.settings import settings
from app.api.routes import pricing
from app.config.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up SmartPrice application...")
    # TODO: connect DB, Redis, load ML models here

    yield  # <- APP IS LIVE HERE, handling requests

    logger.info("Shutting down SmartPrice application...")
    # TODO: close all connections here

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(pricing.router, prefix=settings.api_prefix)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "app_name": settings.app_name, "environment": settings.environment}

    return app

app = create_app()
```

**Key Concepts:**

### Lifespan — Startup and Shutdown Management

The `@asynccontextmanager` decorator + `yield` splits one function into two phases:

```
Server starts
      |
Code before yield runs    <- connect to DB, Redis, load ML model into memory
      |
yield -> APP IS LIVE -> handles all incoming requests
      |
Ctrl+C pressed (or container stops)
      |
Code after yield runs     <- close connections gracefully
      |
Server exits cleanly
```

**Why not connect to the database globally at module level?**
```python
# WRONG
db = connect_to_database()  # runs the instant Python imports this file
app = FastAPI()
```
Problems:
- If DB is down, the app crashes before FastAPI even starts
- The connection is never closed gracefully — it just dies when the process is killed
- You have no control over the order of initialization

```python
# CORRECT — inside lifespan
async def lifespan(app):
    db = await connect_to_database()   # connects at the right moment, in the right order
    yield
    await db.close()                   # closes cleanly every single time
```

### Application Factory Pattern — `create_app()`

Wrapping the app in a factory function (instead of creating it at module level) gives you:
- **Testability**: In tests, call `create_app()` to get a fresh, isolated app instance
- **Explicit ordering**: You control exactly what happens and when during initialization
- This is a standard pattern in Flask and FastAPI production codebases

### `app.include_router()`
Registers an entire feature's endpoints with one line:
```python
app.include_router(pricing.router, prefix=settings.api_prefix)
# Every endpoint in pricing.py gets /api/v1 prepended automatically
```

---

## Step 6 — First API Router

**File:** `app/api/routes/pricing.py`

**What is an APIRouter?**

Instead of cramming all endpoints into `main.py`, FastAPI lets you group related endpoints in their own file. `APIRouter` is a mini-FastAPI focused on one feature.

```python
router = APIRouter(
    prefix="/pricing",
    tags=["Pricing"],   # groups endpoints in Swagger UI
)

@router.post("/")
async def calculate_price():
    return {"status": "ok", "suggested_price": None}
```

**How the final URL is constructed:**
```
api_prefix (/api/v1)  +  router prefix (/pricing)  +  decorator (/)
                    =  GET or POST /api/v1/pricing/
```

**Why APIRouter over putting routes in main.py?**

If you have 100 endpoints in `main.py`, it becomes unreadable and unmaintainable. With routers:
- Adding a new feature = create one new file + one line in `main.py`
- Finding an endpoint = go to the relevant feature file immediately
- Testing a feature = test just that router in isolation

**Day 1 endpoints:**

| Method | URL | Purpose |
|---|---|---|
| GET | `/health` | Confirms the app is running |
| POST | `/api/v1/pricing/` | Pricing (dummy response) |
| GET | `/api/v1/pricing/` | List prices (empty list) |
| GET | `/docs` | Swagger UI — auto-generated interactive docs |
| GET | `/redoc` | ReDoc — alternative docs UI |

---

## Day 1 — Final Architecture

```
main.py
|
+-- setup_logging()               <- first thing that runs
+-- lifespan()                    <- manages startup and shutdown
+-- create_app()                  <- assembles the FastAPI app
    |
    +-- app/config/settings.py    <- reads .env once, shared everywhere
    +-- app/config/logging.py     <- one log format, inherited everywhere
    +-- app/api/routes/
        +-- pricing.py            <- first router, dummy responses
```

---

---

# Day 2 — Layered Architecture

**Goal:** Separate validation, API routing, and business logic into distinct, independent layers.

**Problem with Day 1:**
```
Router -> contains business logic -> becomes huge, untestable, impossible to reuse
```

**Solution — 3-Layer Architecture:**
```
Client
   |
Router (HTTP layer)        <- receives request, delegates, returns response
   |
Service (business logic)   <- calculates, decides, processes
   |
Response
```

---

## Step 1 — Schemas (Validation Layer)

**File:** `app/schemas/pricing.py`

**What is a Schema?**

A schema is a Pydantic model that defines the exact shape and validation rules of data going in or coming out of your API. It is a formal contract between the client and the server.

```python
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field

class PricingStrategy(str, Enum):
    COMPETITIVE = "competitive"
    COST_PLUS   = "cost_plus"
    DYNAMIC     = "dynamic"
    PENETRATION = "penetration"

class PricingRequest(BaseModel):
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., min_length=1, max_length=200)
    cost_price: Decimal = Field(..., gt=0, decimal_places=2)
    competitor_price: Decimal = Field(..., gt=0, decimal_places=2)
    strategy: PricingStrategy = Field(default=PricingStrategy.COMPETITIVE)

class PricingResponse(BaseModel):
    product_id: str
    product_name: str
    cost_price: Decimal
    competitor_price: Decimal
    suggested_price: Decimal
    strategy_used: PricingStrategy
    margin_percent: Decimal
    message: str
```

**Key Concepts:**

### `Field(...)` — Required Fields and Constraints

```python
cost_price: Decimal = Field(..., gt=0)
```
- `...` (Ellipsis) means the field is required — no default value
- `gt=0` means "greater than 0" — Pydantic rejects zero or negative prices before your code runs
- Other useful constraints: `lt` (less than), `ge` (>=), `le` (<=), `min_length`, `max_length`

If the client sends `cost_price: -100`, FastAPI returns HTTP 422 automatically:
```json
{
  "detail": [{
    "loc": ["body", "cost_price"],
    "msg": "Input should be greater than 0",
    "type": "greater_than"
  }]
}
```
This rejection happens **before your business logic is even called**. Services never see invalid data.

### `Enum` — Controlled Vocabulary

Without Enum:
```python
strategy: str  # client can send "COMPETITVE", "xyz", "" — all silently accepted
```

With Enum:
```python
strategy: PricingStrategy  # only "competitive", "cost_plus", "dynamic", "penetration" accepted
```
Any other value -> automatic HTTP 422. No accidental typos or invalid strategies reach your service.

### `Decimal` vs `float` — Why Money Must Never Use Float

```python
# float (binary floating point) — imprecise:
0.1 + 0.2 == 0.30000000000000004   # WRONG

# Decimal (base-10 arithmetic) — exact:
Decimal("0.1") + Decimal("0.2") == Decimal("0.3")   # CORRECT
```

For money, even a fraction of a cent error compounds at scale into real accounting discrepancies. Always use `Decimal` for any financial value.

**Why do Decimal values appear as strings in the JSON response?**
```json
"cost_price": "500.0"   <- string, not a number
```
JSON's native number type uses floating-point internally. Serializing `Decimal` as a JSON number would silently reintroduce float precision errors. Pydantic serializes `Decimal` as a string to preserve exact precision. Any consuming client reads the string and converts it to their own `Decimal` type safely.

**Why validate in schemas rather than in the service?**

Reject bad requests as early as possible. If the service does validation:
- Invalid data travels further through the system
- You get cryptic errors from inside business logic
- You have to test validation AND business logic together

If schemas do validation:
- Invalid data is stopped at the door (HTTP 422, automatic)
- Service receives only clean, guaranteed-valid data
- Validation and business logic are independently testable

---

## Step 2 — Service Layer (Business Logic)

**File:** `app/services/pricing.py`

**Why a separate service layer?**

Analogy: Think of a restaurant.
- The **waiter** (router) takes your order and delivers the food. They do not cook.
- The **kitchen** (service) cooks the food. They do not interact with customers.

If you mix cooking into the waiter's job, the restaurant collapses at scale.

**Key benefits:**

1. **Reusability**: `PricingService.calculate_price()` can be called from a REST endpoint, a CLI command, a scheduled background job, or a unit test — without any changes to the service itself.
2. **Testability**: You can test pricing logic directly with Python, no HTTP server needed.
3. **Single Responsibility Principle**: The router handles HTTP. The service handles pricing. Neither does the other's job.

```python
from decimal import Decimal, ROUND_HALF_UP

class PricingService:

    def calculate_price(self, request: PricingRequest) -> PricingResponse:
        suggested_price = self._apply_strategy(request)
        margin_percent = (
            (suggested_price - request.cost_price) / suggested_price * 100
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return PricingResponse(
            product_id=request.product_id,
            suggested_price=suggested_price,
            margin_percent=margin_percent,
            ...
        )

    def _apply_strategy(self, request: PricingRequest) -> Decimal:
        strategies = {
            PricingStrategy.COMPETITIVE: self._competitive,
            PricingStrategy.COST_PLUS:   self._cost_plus,
            PricingStrategy.DYNAMIC:     self._dynamic,
            PricingStrategy.PENETRATION: self._penetration,
        }
        price = strategies[request.strategy](request)
        return price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _competitive(self, r):  return r.competitor_price * Decimal("0.97")
    def _cost_plus(self, r):    return r.cost_price * Decimal("1.40")
    def _dynamic(self, r):      return (self._cost_plus(r) + self._competitive(r)) / 2
    def _penetration(self, r):  return r.cost_price * Decimal("1.10")
```

**Pricing Strategies — the business logic:**

| Strategy | Formula | Real-world use case |
|---|---|---|
| `competitive` | `competitor_price x 0.97` | Undercut competitor by 3% to win on price |
| `cost_plus` | `cost_price x 1.40` | Guarantee 40% margin regardless of market conditions |
| `dynamic` | Average of competitive + cost_plus | Balance market rate with margin targets |
| `penetration` | `cost_price x 1.10` | Enter new market aggressively with a low 10% margin |

**Strategy Design Pattern:**

Notice the dictionary approach:
```python
strategies = {
    PricingStrategy.COMPETITIVE: self._competitive,
    PricingStrategy.COST_PLUS:   self._cost_plus,
}
return strategies[request.strategy](request)  # call the right method
```

This is the **Strategy Design Pattern** — instead of a long `if/elif/elif/elif` chain, each algorithm is a separate method. Adding a new pricing strategy in the future means:
1. Add one method to the class
2. Add one entry to the dictionary

Nothing else changes. No risk of breaking existing strategies.

**`ROUND_HALF_UP` — Correct Financial Rounding:**

Python's default rounding is "Banker's Rounding" (round half to even), which is designed to reduce statistical bias in large datasets but surprises business users.

```python
# Banker's rounding (Python default):
round(2.5) = 2   # rounds to even (unexpected!)
round(3.5) = 4   # rounds to even

# ROUND_HALF_UP (what humans expect):
Decimal("2.5").quantize(Decimal("1"), rounding=ROUND_HALF_UP) = 3
Decimal("3.5").quantize(Decimal("1"), rounding=ROUND_HALF_UP) = 4
```

For prices shown to customers, always use `ROUND_HALF_UP`.

---

## Step 3 — Thin Router

**File:** `app/api/routes/pricing.py`

After the service layer exists, the router's job shrinks dramatically:

**Before (Day 1) — fat router:**
```python
@router.post("/")
async def calculate_price():
    return {"status": "ok", "suggested_price": None}  # dummy, does nothing
```

**After (Day 2) — thin router:**
```python
_service = PricingService()

@router.post("/", response_model=PricingResponse)
async def calculate_price(request: PricingRequest) -> PricingResponse:
    return _service.calculate_price(request)
```

That is the entire function. Two lines. The router now does exactly what a router should do: receive, delegate, return.

**`response_model=PricingResponse` — what it does:**

1. Tells FastAPI how to serialize the return value to JSON
2. Strips out any internal fields not declared in `PricingResponse` (security — internal data never accidentally leaks to clients)
3. Generates the correct JSON schema in Swagger UI at `/docs`

---

## Day 2 — Full Request Lifecycle

```
Client sends:
POST /api/v1/pricing/
{
  "product_id": "PROD-001",
  "cost_price": 500.00,
  "competitor_price": 999.00,
  "strategy": "competitive"
}

          |
FastAPI receives the request
          |
PricingRequest runs validation
  +-- cost_price > 0?          PASS
  +-- competitor_price > 0?    PASS
  +-- strategy is valid enum?  PASS
          |
Router calls _service.calculate_price(request)
          |
PricingService._apply_strategy()
  +-- strategy = "competitive"
  +-- calls _competitive()
  +-- 999.00 x 0.97 = 969.03
          |
margin_percent = (969.03 - 500) / 969.03 x 100 = 48.40%
          |
PricingResponse object created
          |
FastAPI serializes to JSON (Decimal -> string for precision)
          |
HTTP 200 returned to client:
{
  "product_id": "PROD-001",
  "product_name": "Wireless Headphones",
  "cost_price": "500.00",
  "competitor_price": "999.00",
  "suggested_price": "969.03",
  "strategy_used": "competitive",
  "margin_percent": "48.40",
  "message": "Strategy 'competitive' applied. Suggested price is Rs.29.97 below competitor price."
}
```

---

## Day 2 — Final Architecture

```
Client
   |
app/api/routes/pricing.py     <- HTTP only: receives request, calls service, returns response
   |
app/services/pricing.py       <- Business logic: strategy selection, calculation, margin
   ^
app/schemas/pricing.py        <- Validation contract: PricingRequest and PricingResponse
```

---

---

## Running the Application

**Quickest way (Day 16):** double-click `start-demo.bat` in File Explorer, or run `.\start-demo.bat` — it does all the steps below. Stop with `.\stop-demo.bat`.

```bash
# Step 1: Install all dependencies
pip install -r requirements-dev.txt       # API + training + tests (requirements.txt = API only)

# Step 2 (Day 6/7): Start PostgreSQL + Redis, create the schema, load sample products
#   First time only: copy .env.example to .env and set a real POSTGRES_PASSWORD
#   .env = LOCAL databases only (one DATABASE_URL line). Neon/Upstash URLs go in Render's settings (see Day 14).
docker compose up -d                        # PostgreSQL + Redis
docker exec smartprice-redis redis-cli PING  # -> PONG
alembic upgrade head
python -m app.database.seed                 # demo + 20 real products and 14,780 days of sales
#   python -m app.database.seed --update   resets the sample products to the seed values

# Step 2b (ML): train, evaluate, version and track the model (writes models/vN + MLflow run)
python -m app.ml.train

# Step 2c (optional): run the API in Docker exactly as it is deployed (port 8080)
#   docker compose --profile app up -d --build

# Step 3: Start the development server
python -m uvicorn app.main:app --reload

# What --reload does: watches for file changes and auto-restarts the server.
# NEVER use --reload in production. It adds overhead.

# Step 3b: the interactive demo (needs the API running) -> http://localhost:8501
streamlit run demo/app.py

# Step 4: Run the test suite (added in the Day 4 follow-up)
#   Database tests are skipped automatically when PostgreSQL is not running
python -m pytest
```

**Visit these URLs:**

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000/health | Health check — app alive, database (Day 6) and cache (Day 7) reachable |
| http://127.0.0.1:8000/docs | Swagger UI — interactive API documentation |
| http://127.0.0.1:8000/redoc | ReDoc — cleaner, readable API docs |
| http://127.0.0.1:8000/api/v1/pricing/ | Pricing endpoints (GET and POST) |

**To stop the server:** Press `Ctrl + C` in the terminal.

**Why `127.0.0.1` and not `localhost`?**

They are the same thing. `127.0.0.1` is the loopback IP address built into every computer. `localhost` is a human-readable hostname that your operating system resolves to `127.0.0.1`. Traffic to this address never leaves your machine — no one on the internet, or even on your local Wi-Fi network, can reach it.

---

## Key Engineering Principles Applied

| Principle | Where it is applied |
|---|---|
| **Separation of Concerns** | Config, logging, routing, and business logic all live in separate files |
| **Single Responsibility Principle** | Each class and module does exactly one thing |
| **DRY (Don't Repeat Yourself)** | Settings read once and cached; logging configured once and inherited |
| **Singleton Pattern** | `@lru_cache` on `get_settings()` — one Settings object for the whole app |
| **Factory Pattern** | `create_app()` in `main.py` — explicit, testable app construction |
| **Strategy Pattern** | Pricing algorithms stored in a dict, selected at runtime |
| **Fail Fast** | Pydantic rejects invalid input before business logic is ever called |
| **Defensive Defaults** | All settings have safe, sensible defaults in the `Settings` class |
| **Graceful Shutdown** | Lifespan ensures connections close cleanly, no data loss on server stop |

---

---

# Day 3 — Strategy Pattern and Factory Pattern

**Goal:** Remove all conditional pricing logic from the service and make the system open for new pricing algorithms without modifying existing code.

**Problem with Day 2:**

The `PricingService` contained every pricing algorithm as private methods:
```python
# services/pricing.py — Day 2 (the problem)
class PricingService:
    def _competitive(self, r): return r.competitor_price * Decimal("0.97")
    def _cost_plus(self, r):   return r.cost_price * Decimal("1.40")
    def _dynamic(self, r):     ...
    def _penetration(self, r): ...
```

What happens when we need 10 more strategies (Holiday, Weekend, Surge, Clearance, B2B, ...)?
- The service file grows to 500+ lines
- Every new strategy requires modifying an existing file (risk of breaking things)
- You cannot test one strategy in isolation — they are all tangled inside one class
- Violates the **Open/Closed Principle**

**Solution — Strategy Pattern + Factory Pattern.**

---

## New File Structure

```
app/
+-- strategies/
|   +-- __init__.py
|   +-- base.py               <- Abstract contract
|   +-- competitive.py        <- Competitive algorithm
|   +-- cost_plus.py          <- Cost-plus algorithm
|   +-- dynamic.py            <- Dynamic (blended) algorithm
|   +-- penetration.py        <- Penetration algorithm
|
+-- factories/
|   +-- __init__.py
|   +-- strategy_factory.py   <- Creates strategy objects
|
+-- services/
    +-- pricing.py             <- Refactored: pure orchestrator, no algorithms
```

---

## Step 1 — Abstract Base Class

**File:** `app/strategies/base.py`

```python
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_HALF_UP
from app.schemas.pricing import PricingRequest

class AbstractPricingStrategy(ABC):

    @abstractmethod
    def calculate_price(self, request: PricingRequest) -> Decimal:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    def _round(self, value: Decimal) -> Decimal:
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

**What is an Abstract Base Class (ABC)?**

An ABC is a class that cannot be instantiated directly. It exists solely to define a **contract** that all subclasses must fulfil.

```python
AbstractPricingStrategy()   # TypeError: cannot instantiate abstract class
CompetitivePricingStrategy()  # works — implements all abstract methods
```

**Why use it here?**

The factory and service only ever call `strategy.calculate_price(request)`. They do not care whether the strategy is Competitive, Penetration, or Holiday. The ABC guarantees that every strategy exposes that exact method with that exact signature. If a developer creates a new strategy and forgets to implement `calculate_price`, Python raises `TypeError` immediately at instantiation — the bug is caught before it ever runs.

**`@abstractmethod`:**
Marks a method as required. Any subclass that does not implement it cannot be instantiated.

**`_round()` shared helper:**
Every strategy needs to round the final price to 2 decimal places using `ROUND_HALF_UP`. Putting this in the base class means it is written once and inherited by all strategies — DRY principle.

---

## Step 2 — Concrete Strategy Classes

**Files:** `app/strategies/competitive.py`, `cost_plus.py`, `dynamic.py`, `penetration.py`

Each strategy is now its own isolated class:

```python
# competitive.py
class CompetitivePricingStrategy(AbstractPricingStrategy):
    @property
    def description(self) -> str:
        return "Price 3% below competitor to win on price"

    def calculate_price(self, request: PricingRequest) -> Decimal:
        return self._round(request.competitor_price * Decimal("0.97"))
```

```python
# cost_plus.py
class CostPlusPricingStrategy(AbstractPricingStrategy):
    MARGIN_MULTIPLIER = Decimal("1.40")

    def calculate_price(self, request: PricingRequest) -> Decimal:
        return self._round(request.cost_price * self.MARGIN_MULTIPLIER)
```

```python
# dynamic.py
class DynamicPricingStrategy(AbstractPricingStrategy):
    def calculate_price(self, request: PricingRequest) -> Decimal:
        competitive = CompetitivePricingStrategy().calculate_price(request)
        cost_plus   = CostPlusPricingStrategy().calculate_price(request)
        return self._round((competitive + cost_plus) / 2)
```

```python
# penetration.py
class PenetrationPricingStrategy(AbstractPricingStrategy):
    MARGIN_MULTIPLIER = Decimal("1.10")

    def calculate_price(self, request: PricingRequest) -> Decimal:
        return self._round(request.cost_price * self.MARGIN_MULTIPLIER)
```

**Why separate files for each strategy?**

- Each file is small, focused, and independently readable
- You can run a unit test for just `CompetitivePricingStrategy` with zero setup
- A developer can read and understand one strategy in 30 seconds
- If Competitive is broken, you fix only `competitive.py` — nothing else is at risk

**Results with cost=500, competitor=999:**

| Strategy | Formula | Suggested Price | Margin |
|---|---|---|---|
| `competitive` | 999 x 0.97 | Rs. 969.03 | 48.40% |
| `cost_plus` | 500 x 1.40 | Rs. 700.00 | 28.57% |
| `dynamic` | avg of above | Rs. 834.52 | 40.09% |
| `penetration` | 500 x 1.10 | Rs. 550.00 | 9.09% |

---

## Step 3 — Strategy Factory

**File:** `app/factories/strategy_factory.py`

```python
class StrategyFactory:
    _registry: dict[PricingStrategy, type[AbstractPricingStrategy]] = {
        PricingStrategy.COMPETITIVE: CompetitivePricingStrategy,
        PricingStrategy.COST_PLUS:   CostPlusPricingStrategy,
        PricingStrategy.DYNAMIC:     DynamicPricingStrategy,
        PricingStrategy.PENETRATION: PenetrationPricingStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy: PricingStrategy) -> AbstractPricingStrategy:
        strategy_class = cls._registry.get(strategy)
        if strategy_class is None:
            raise ValueError(f"No strategy registered for '{strategy}'")
        return strategy_class()

    @classmethod
    def available_strategies(cls) -> list[str]:
        return [s.value for s in cls._registry]
```

**What is the Factory Pattern?**

Instead of the service creating strategy objects directly:
```python
# Without factory — service must know about every strategy class
if request.strategy == PricingStrategy.COMPETITIVE:
    strategy = CompetitivePricingStrategy()
elif request.strategy == PricingStrategy.COST_PLUS:
    strategy = CostPlusPricingStrategy()
```

With factory:
```python
# Service just asks the factory
strategy = StrategyFactory.get_strategy(request.strategy)
```

**Why centralize object creation in a factory?**

1. **Single responsibility**: The factory's only job is to create strategy objects. The service's only job is to orchestrate the pricing workflow. Neither does the other's job.

2. **Future-proofing**: Today, creating a strategy is just `CompetitivePricingStrategy()`. Tomorrow, it might need a database connection, a configuration file, or an ML model loaded from disk. If that happens, only the factory changes — the service is completely untouched.

3. **Open/Closed Principle**: Adding a new `HolidayPricingStrategy` requires:
   - Create `app/strategies/holiday.py` (new file, nothing modified)
   - Add one line to `_registry` in the factory
   - Done. Zero risk of breaking existing strategies, service, or router.

**The `_registry` dictionary:**
Maps `PricingStrategy` enum values (what the client sends) to strategy classes (what the factory instantiates). This is the **only place in the entire codebase that changes when you add a new strategy**.

---

## Step 4 — Refactored Service

**File:** `app/services/pricing.py`

**Before (Day 2) — fat service, 110 lines:**
```python
class PricingService:
    def _apply_strategy(self, request):
        strategies = {
            PricingStrategy.COMPETITIVE: self._competitive,
            ...
        }
        return strategies[request.strategy](request)

    def _competitive(self, r): ...   # algorithm inside service
    def _cost_plus(self, r): ...     # algorithm inside service
    def _dynamic(self, r): ...       # algorithm inside service
    def _penetration(self, r): ...   # algorithm inside service
```

**After (Day 3) — thin service, 70 lines:**
```python
class PricingService:
    def calculate_price(self, request: PricingRequest) -> PricingResponse:
        # Ask factory for the right strategy
        strategy = StrategyFactory.get_strategy(request.strategy)

        # Delegate all math to the strategy
        suggested_price = strategy.calculate_price(request)

        # Service handles cross-cutting concerns (margin calculation, logging)
        margin_percent = (
            (suggested_price - request.cost_price) / suggested_price * 100
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return PricingResponse(...)
```

The service went from knowing about 4 algorithms to knowing about **zero** algorithms. It is now a pure **orchestrator** — it coordinates the flow but delegates all decisions.

**Why does margin calculation stay in the service and not the strategy?**

Margin calculation is not a pricing algorithm — it is a business reporting metric that applies to every strategy equally. It does not belong to any single strategy. The service is the right place for this kind of cross-cutting concern.

---

## Day 3 — Design Principles Applied

### Open/Closed Principle (OCP)
"Software entities should be open for extension, but closed for modification."

```
Adding HolidayPricingStrategy:

  CREATE app/strategies/holiday.py       <- new file (open for extension)
  ADD one line to StrategyFactory        <- minimal change, low risk

  NOTHING ELSE CHANGES                   <- closed for modification
```

### Single Responsibility Principle (SRP)

| Component | Its Only Responsibility |
|---|---|
| `AbstractPricingStrategy` | Define the contract all strategies must follow |
| `CompetitivePricingStrategy` | Implement the competitive pricing algorithm |
| `StrategyFactory` | Create the correct strategy object on demand |
| `PricingService` | Orchestrate the pricing workflow |

### Strategy Design Pattern

When multiple algorithms solve the same problem and you need to select one at runtime, the Strategy Pattern makes each algorithm a separate class sharing a common interface.

```
Client selects strategy: "competitive"
          |
   PricingRequest validated
          |
   PricingService asks StrategyFactory
          |
   Factory returns CompetitivePricingStrategy()
          |
   strategy.calculate_price(request)
          |
   Decimal("969.03") returned
          |
   PricingResponse built and returned
```

### Factory Design Pattern

When object creation logic is complex or must be centralized, a Factory handles all instantiation so that callers do not need to know the implementation details.

---

## Interview Questions — Day 3

**Q: Why use the Strategy Pattern?**
When multiple algorithms solve the same problem and one needs to be selected at runtime. It eliminates `if/elif` chains, isolates each algorithm, and makes adding new ones risk-free.

**Q: Why combine Strategy and Factory?**
Strategy decides which behaviour to use. Factory decides how to create the object for that behaviour. Together they eliminate both conditional logic AND conditional object creation from the service.

**Q: Why is the Abstract Base Class important?**
It guarantees interface consistency. No matter which strategy the factory returns, the service can always call `.calculate_price(request)` safely. Python enforces this contract at class instantiation time.

**Q: Could dependency injection replace the factory?**
Yes, in larger applications using DI frameworks (like Python's `dependency-injector`). The factory approach is simpler and explicit — ideal for a project at this scale. The factory can be replaced by a DI container later with zero changes to the service or strategies.

**Q: What is the Open/Closed Principle?**
A class should be open for extension (you can add new behaviour) but closed for modification (you do not change existing, working code to do so). The Strategy + Factory combination perfectly implements this.

---

---

# Day 4 — Chain of Responsibility (Guardrails)

> **Update:** two bugs in this day's guardrails were found and fixed afterwards — unrounded guard prices, and a chain order that could sell below cost. See [Day 4 Follow-up](#day-4-follow-up--guardrail-fixes-and-tests).

**Goal:** Separate pricing algorithms from business validation rules using the Chain of Responsibility pattern.

**Problem with Day 3:**

After the strategy calculates a price, the business may require additional validation:
- Minimum selling price (never sell below cost)
- Maximum markup (do not price absurdly above market)
- Maximum discount limits
- Rounding rules
- Vendor-specific constraints

If we place these checks inside every strategy, we get:
```python
# BAD — validation logic duplicated in every strategy
class CompetitivePricingStrategy:
    def calculate_price(self, request):
        price = request.competitor_price * Decimal("0.97")
        if price < request.cost_price * Decimal("1.05"):  # floor check
            price = request.cost_price * Decimal("1.05")
        if price > request.competitor_price * Decimal("1.50"):  # volatility check
            price = request.competitor_price * Decimal("1.50")
        return price

class CostPlusPricingStrategy:
    def calculate_price(self, request):
        price = request.cost_price * Decimal("1.40")
        if price < request.cost_price * Decimal("1.05"):  # same floor check — duplicated!
            price = request.cost_price * Decimal("1.05")
        if price > request.competitor_price * Decimal("1.50"):  # same volatility check — duplicated!
            price = request.competitor_price * Decimal("1.50")
        return price
```

Problems:
- Every strategy duplicates the same validation rules
- Adding a new validation rule means modifying every single strategy
- Strategies should answer "How to calculate the price?" — not "Is this price acceptable?"
- Violates Single Responsibility, DRY, and Open/Closed Principles

**Solution — Chain of Responsibility Pattern.**

---

## New File Structure

```
app/
+-- guardrails/
|   +-- __init__.py              <- package exports
|   +-- base_guard.py            <- Abstract base with Template Method
|   +-- price_floor_guard.py     <- Ensures price ≥ cost × minimum margin
|   +-- volatility_guard.py      <- Caps price ≤ competitor × max volatility
|
+-- services/
|   +-- pricing.py               <- Refactored: builds and runs guard chain
|
+-- schemas/
|   +-- pricing.py               <- Updated: guardrails_applied field in response
|
+-- config/
    +-- settings.py              <- Updated: guardrail threshold settings
```

---

## Step 1 — BaseGuard (Chain of Responsibility + Template Method)

**File:** `app/guardrails/base_guard.py`

```python
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

from app.schemas.pricing import PricingRequest


class BaseGuard(ABC):
    def __init__(self) -> None:
        self._next_guard: Optional[BaseGuard] = None

    def set_next(self, guard: BaseGuard) -> BaseGuard:
        self._next_guard = guard
        return guard

    def handle(self, price: Decimal, request: PricingRequest, applied_guards: list[str]) -> Decimal:
        adjusted_price = self.apply(price, request, applied_guards)
        if self._next_guard is not None:
            return self._next_guard.handle(adjusted_price, request, applied_guards)
        return adjusted_price

    @abstractmethod
    def apply(self, price: Decimal, request: PricingRequest, applied_guards: list[str]) -> Decimal:
        ...
```

**What is the Chain of Responsibility Pattern?**

Instead of one object handling all validation checks, the pattern creates a chain of handlers. Each handler:
1. Inspects the request
2. Either processes it (modifies the price) or passes it through unchanged
3. Forwards the result to the next handler in the chain

```
Price from Strategy
      │
      ▼
PriceFloorGuard ──── "Is the price above the floor?"
      │                   YES → pass through unchanged
      │                   NO  → raise to floor
      ▼
VolatilityGuard ──── "Is the price below the ceiling?"
      │                   YES → pass through unchanged
      │                   NO  → cap to ceiling
      ▼
Final Validated Price
```

**Template Method within BaseGuard:**

Although the primary pattern is Chain of Responsibility, `handle()` also demonstrates the **Template Method** idea:
- The overall algorithm (`apply → forward to next guard`) is **fixed** in the base class
- Subclasses customise only the `apply()` step

```python
def handle(self, price, request, applied_guards):
    adjusted_price = self.apply(price, request, applied_guards)  # VARIABLE part — subclass
    if self._next_guard is not None:
        return self._next_guard.handle(adjusted_price, ...)      # FIXED part — always forwards
    return adjusted_price
```

This ensures every guard in the chain behaves consistently. A developer cannot accidentally forget to forward to the next guard — `handle()` does that automatically.

**`set_next()` returns the next guard — why?**

This enables fluent chaining:
```python
floor.set_next(volatility).set_next(rounding).set_next(tax)
```
Instead of:
```python
floor.set_next(volatility)
volatility.set_next(rounding)
rounding.set_next(tax)
```

**`applied_guards` — a mutable list:**

Each guard appends a description when it modifies the price. This list is surfaced in the API response so the client can see exactly which rules fired:
```json
{
  "guardrails_applied": [
    "PriceFloorGuard: price raised from 480.00 to 525.00 (minimum margin of 5% enforced)"
  ]
}
```

---

## Step 2 — PriceFloorGuard

**File:** `app/guardrails/price_floor_guard.py`

```python
class PriceFloorGuard(BaseGuard):
    def apply(self, price, request, applied_guards):
        floor_price = request.cost_price * Decimal(str(settings.minimum_margin_multiplier))
        if price < floor_price:
            applied_guards.append(f"PriceFloorGuard: price raised from {price:.2f} to {floor_price:.2f}")
            return floor_price
        return price
```

**Business Rule:**
A product must never be sold below its cost. The floor is `cost_price × minimum_margin_multiplier` (default: 1.05 → 5% minimum margin).

**When does this guard trigger?**

| Strategy | cost | competitor | Raw Price | Floor (cost×1.05) | Triggered? |
|---|---|---|---|---|---|
| `competitive` | 500 | 999 | 969.03 | 525.00 | No |
| `penetration` | 500 | 520 | 550.00 | 525.00 | No |
| `competitive` | 950 | 999 | 969.03 | 997.50 | **Yes** → 997.50 |

**Why is the multiplier configurable via `.env`?**
```
MINIMUM_MARGIN_MULTIPLIER=1.05   # default: 5% minimum margin
```
Different businesses have different minimum margin requirements. A luxury brand may set 1.30 (30% margin). A volume retailer might use 1.02 (2% margin). Changing this requires zero code changes — just update the environment variable.

---

## Step 3 — VolatilityGuard

**File:** `app/guardrails/volatility_guard.py`

```python
class VolatilityGuard(BaseGuard):
    def apply(self, price, request, applied_guards):
        ceiling = request.competitor_price * Decimal(str(settings.max_volatility_multiplier))
        if price > ceiling:
            applied_guards.append(f"VolatilityGuard: price capped from {price:.2f} to {ceiling:.2f}")
            return ceiling
        return price
```

**Business Rule:**
The suggested price must not exceed `competitor_price × max_volatility_multiplier` (default: 1.50 → max 50% above competitor).

**When does this guard trigger?**

| Strategy | cost | competitor | Raw Price | Ceiling (comp×1.50) | Triggered? |
|---|---|---|---|---|---|
| `competitive` | 500 | 999 | 969.03 | 1498.50 | No |
| `cost_plus` | 500 | 999 | 700.00 | 1498.50 | No |
| `cost_plus` | 900 | 400 | 1260.00 | 600.00 | **Yes** → 600.00 |

**Why cap relative to competitor and not as an absolute value?**
Prices vary wildly across products (Rs.50 vs Rs.50,000). A fixed ceiling would either be too restrictive for expensive products or meaningless for cheap ones. Relative capping (% above competitor) scales naturally.

---

## Step 4 — Configuration Updates

**File:** `app/config/settings.py`

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Guardrails — business validation thresholds
    minimum_margin_multiplier: float = 1.05   # floor = cost × 1.05
    max_volatility_multiplier: float = 1.50   # ceiling = competitor × 1.50
```

**File:** `.env`

```
MINIMUM_MARGIN_MULTIPLIER=1.05
MAX_VOLATILITY_MULTIPLIER=1.50
```

Both thresholds are:
- Centralised in the Settings class (read once, available everywhere)
- Configurable via environment variables (no code change needed per environment)
- Type-safe (Pydantic validates they are floats at startup)

---

## Step 5 — Schema Update

**File:** `app/schemas/pricing.py`

```python
class PricingResponse(BaseModel):
    # ... existing fields ...
    guardrails_applied: Optional[list[str]] = Field(
        default=None,
        description="List of guardrail rules that adjusted the price (empty if none fired)",
    )
```

When no guards fire, the field is `null` in the JSON response (minimal noise). When guards do fire, the client sees exactly what happened:

```json
{
  "suggested_price": "997.50",
  "guardrails_applied": [
    "PriceFloorGuard: price raised from 969.03 to 997.50 (minimum margin of 5% enforced)"
  ]
}
```

---

## Step 6 — Service Integration

**File:** `app/services/pricing.py`

**Before (Day 3) — no validation:**
```python
class PricingService:
    def calculate_price(self, request):
        strategy = StrategyFactory.get_strategy(request.strategy)
        suggested_price = strategy.calculate_price(request)   # raw, unvalidated price
        margin = ...
        return PricingResponse(suggested_price=suggested_price, ...)
```

**After (Day 4) — guard chain integrated:**
```python
class PricingService:
    def calculate_price(self, request):
        strategy = StrategyFactory.get_strategy(request.strategy)
        raw_price = strategy.calculate_price(request)

        # Run through guard chain
        guard_chain = self._build_guard_chain()
        applied_guards: list[str] = []
        suggested_price = guard_chain.handle(raw_price, request, applied_guards)

        # Margin calculated on FINAL (post-guard) price
        margin = ...
        return PricingResponse(
            suggested_price=suggested_price,
            guardrails_applied=applied_guards if applied_guards else None,
            ...
        )

    def _build_guard_chain(self) -> BaseGuard:
        floor = PriceFloorGuard()
        volatility = VolatilityGuard()
        floor.set_next(volatility)
        return floor   # head of the chain
```

**Key design decisions:**

### Why `_build_guard_chain()` is a separate helper method

If the chain is built inside `calculate_price()`:
```python
# BAD — chain construction mixed with orchestration
def calculate_price(self, request):
    ...
    floor = PriceFloorGuard()
    volatility = VolatilityGuard()
    floor.set_next(volatility)
    suggested_price = floor.handle(raw_price, ...)
    ...
```

With a helper:
```python
# GOOD — clean separation
def calculate_price(self, request):
    ...
    guard_chain = self._build_guard_chain()
    suggested_price = guard_chain.handle(raw_price, ...)
    ...

def _build_guard_chain(self):
    floor = PriceFloorGuard()
    volatility = VolatilityGuard()
    floor.set_next(volatility)
    return floor
```

Benefits:
- `calculate_price()` stays clean — it is a pure orchestrator
- Adding new guards only touches `_build_guard_chain()`
- The chain assembly is isolated and independently readable

### Why margin is calculated AFTER the guard chain

Margin must reflect the **actual selling price**, not the raw strategy output. If the floor guard raises the price from 480 to 525, the margin should be based on 525 — the price the customer actually pays.

---

## Day 4 — Full Request Lifecycle

```
Client sends:
POST /api/v1/pricing/
{
  "product_id": "PROD-005",
  "cost_price": 950.00,
  "competitor_price": 999.00,
  "strategy": "competitive"
}

          |
FastAPI receives the request
          |
PricingRequest validation (Pydantic)
          |
Router → PricingService.calculate_price()
          |
Step 1: StrategyFactory → CompetitivePricingStrategy
          |
Step 2: strategy.calculate_price()
          raw_price = 999.00 × 0.97 = 969.03
          |
Step 3: Guard Chain
          |
  PriceFloorGuard:
    floor = 950.00 × 1.05 = 997.50
    969.03 < 997.50 → TRIGGERED → price = 997.50
          |
  VolatilityGuard:
    ceiling = 999.00 × 1.50 = 1498.50
    997.50 ≤ 1498.50 → passed
          |
Step 4: margin = (997.50 - 950.00) / 997.50 × 100 = 4.76%
          |
HTTP 200:
{
  "product_id": "PROD-005",
  "suggested_price": "997.50",
  "margin_percent": "4.76",
  "guardrails_applied": [
    "PriceFloorGuard: price raised from 969.03 to 997.50 (minimum margin of 5% enforced)"
  ],
  "message": "Strategy 'competitive' applied. ..."
}
```

---

## Day 4 — Final Architecture

```
Client
    │
    ▼
Router (app/api/routes/pricing.py)
    │
    ▼
PricingService (app/services/pricing.py)
    │
    ├──────────────────┐
    ▼                  ▼
StrategyFactory       Guard Chain (_build_guard_chain)
    │                  │
    ▼                  ▼
PricingStrategy      PriceFloorGuard → VolatilityGuard
    │                  │
    └────────┬─────────┘
             ▼
       Final Response
```

---

## Day 4 — Design Principles Applied

### Separation of Concerns

| Component | Question it answers |
|---|---|
| `PricingStrategy` | How should the price be calculated? |
| `BaseGuard` chain | Is the calculated price acceptable? |

Strategies do zero validation. Guards do zero calculation. Neither knows about the other.

### Open/Closed Principle (for guardrails)

```
Adding RoundingGuard:

  CREATE app/guardrails/rounding_guard.py    <- new file
  ADD one line to _build_guard_chain()       <- minimal change

  NOTHING ELSE CHANGES
```

The chain becomes:
```python
def _build_guard_chain(self):
    floor = PriceFloorGuard()
    volatility = VolatilityGuard()
    rounding = RoundingGuard()           # new guard
    floor.set_next(volatility).set_next(rounding)
    return floor
```

### Template Method Pattern (within BaseGuard)

`handle()` defines a fixed algorithm with a variable step:
```
FIXED: handle() calls apply() then forwards to next guard
VARIABLE: each guard's apply() implements its own business rule
```

A developer cannot break the chain — `handle()` always forwards. They only need to implement `apply()`.

### Chain of Responsibility Pattern

When a request needs to pass through multiple independent processing stages, Chain of Responsibility lets each stage:
- Decide independently whether to modify the price
- Forward the result to the next stage
- Remain completely unaware of other stages

---

## Interview Questions — Day 4

**Q: Why use Chain of Responsibility instead of a simple list of if-statements?**
A simple list tightly couples all validation rules in one function. Chain of Responsibility isolates each rule in its own class, makes them independently testable, and allows reordering or adding rules without modifying existing ones.

**Q: Why does BaseGuard use the Template Method pattern?**
To guarantee consistency. `handle()` ensures every guard applies its rule AND forwards to the next guard. Without it, a developer could forget the forwarding step, silently breaking the chain.

**Q: Does the order of guards in the chain matter?**
Yes. PriceFloorGuard runs first (raises the price to a minimum) before VolatilityGuard (caps the price at a maximum). If reversed, the floor guard might raise a price that the volatility guard already capped, potentially pushing the final price above the volatility ceiling.

**Q: Why are guardrail thresholds in Settings and not hardcoded?**
Different environments (development, staging, production) and different business contexts (luxury vs budget retailer) need different thresholds. Centralising them in Settings with `.env` support allows tuning without code changes or redeployment.

**Q: How would you add a new guard, say TaxGuard?**
1. Create `app/guardrails/tax_guard.py` extending `BaseGuard`, implement `apply()`
2. Add one line in `_build_guard_chain()`: `volatility.set_next(tax)`
3. Done — zero changes to existing guards, service orchestration, or strategies.

---

---

# Day 4 Follow-up — Guardrail Fixes and Tests

**Goal:** Fix two bugs found by running the Day 4 examples against the real API, and add an automated test suite so they can never come back.

---

## Bug 1 — Guardrail prices were not whole cents

Strategies round their output with `_round()`, but the guards returned the raw result of `price × multiplier`:

| Request | Returned | Expected |
|---|---|---|
| cost 950, competitor 999, `competitive` | `"997.500"` | `"997.50"` |
| cost 333.33, competitor 300, `competitive` | `"349.9965"` | `"350.00"` |

The second case also contradicted itself: the `guardrails_applied` message said "raised … to 350.00" while `suggested_price` said 349.9965.

**Fix:** each guard rounds its own limit — in the *safe direction*:

```python
# price_floor_guard.py — a minimum rounds UP
floor_price = (
    product.cost_price * Decimal(str(settings.minimum_margin_multiplier))
).quantize(Decimal("0.01"), rounding=ROUND_CEILING)

# volatility_guard.py — a maximum rounds DOWN
ceiling = (
    product.competitor_price * Decimal(str(settings.max_volatility_multiplier))
).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)
```

**Why not `ROUND_HALF_UP` like the strategies?**

A strategy *produces* a price, so "nearest cent" is right. A guard *enforces a limit*, and rounding to the nearest cent can land on the wrong side of it:

```
cost 100.01 × 1.05 = 105.0105
ROUND_HALF_UP  → 105.01   ← a fraction BELOW the floor
ROUND_CEILING  → 105.02   ← guaranteed at or above the floor
```

Round a minimum up and a maximum down, and the limit can never be crossed.

---

## Bug 2 — The system could recommend selling below cost

The chain ran `PriceFloorGuard → VolatilityGuard`. Each guard can overwrite the price set by the guard before it, so **the last guard has the final say**. When the ceiling is below the floor — cost more than about 1.43 × competitor — no price satisfies both rules, and the ceiling won:

```
cost=900, competitor=400, strategy=cost_plus

raw price          1260.00
PriceFloorGuard    floor 945.00   → 1260.00 passes
VolatilityGuard    ceiling 600.00 → capped to 600.00

final 600.00, margin -50.00%      ← selling at a loss
```

This was the Day 4 VolatilityGuard table's own example, and the Day 4 interview answer on guard order missed it.

**Fix:** reverse the chain so the rule that can never be broken runs last:

```python
def _build_guard_chain(self) -> BaseGuard:
    volatility = VolatilityGuard()
    floor = PriceFloorGuard()
    volatility.set_next(floor)      # floor last = floor wins
    return volatility
```

When the floor overrides an earlier guard, it says so in the response:

```json
{
  "suggested_price": "945.00",
  "margin_percent": "4.76",
  "guardrails_applied": [
    "VolatilityGuard: price capped from 1260.00 to 600.00 (max 50% above competitor enforced)",
    "PriceFloorGuard: price raised from 600.00 to 945.00 (minimum margin of 5% enforced; floor takes priority over the earlier guardrail adjustment so the product is never sold below cost)"
  ]
}
```

The floor guard knows it is overriding when `applied_guards` is already non-empty: an earlier guard changed the price and still left it below the floor.

**Corrected answer — "Does the order of guards in the chain matter?"**
Yes. When each handler may overwrite the value, the last handler has the final say. Put the rule you can never break (the cost floor) last; anything that can *lower* a price goes before it.

---

## Test Suite

```
pytest.ini               <- testpaths + pythonpath, so plain `pytest` finds app/
tests/
├── conftest.py          <- shared fixtures
├── test_strategies.py   <- Day 3 results table, factory
├── test_guardrails.py   <- each guard, rounding direction, chain order
└── test_pricing_api.py  <- every DEVLOG example through the real HTTP API
```

Key decisions:

- **Pinned thresholds** — `conftest.py` sets the guardrail multipliers to 1.05 / 1.50 with `monkeypatch`, so editing `.env` can never change test results.
- **Invariant test** — every strategy prices a grid of 30 cost/competitor pairs, asserting the price is never below `cost × 1.05`, always whole cents, and never a negative margin.
- **Testing the tests** — the suite was also run against a copy with the old code restored: 14 tests failed. A test that cannot fail proves nothing.

```bash
python -m pytest
```

---

---

# Day 5 — Repository Pattern, Dependency Injection & Exception Handling

**Goal:** Decouple the business logic from data access and HTTP-specific concerns, making the application easier to maintain, test, and extend.

**Problem with Day 4:**

- `PricingService` relied directly on request data. The client sent `cost_price` and `competitor_price`, and the service, every strategy, and every guard took the HTTP schema `PricingRequest` as input.
- There was no abstraction for retrieving product information.
- There was no business-level way to report an error. The only way for the service to say "product not found" would have been raising FastAPI's `HTTPException` — coupling business logic to the web framework.
- The router created the service itself (`_service = PricingService()`), so nothing could be swapped — not for a database, not for a test.

This violates the **Dependency Inversion Principle** and makes the business layer dependent on infrastructure.

**Solution:**

- Repository Pattern
- Dependency Injection
- Custom Business Exceptions
- Global Exception Handlers

This cleanly separates **business logic**, **data access**, and the **HTTP layer**.

---

## New File Structure

```
app/
├── models/                     <- NEW: domain layer
│   ├── product.py              <- Product (core business entity)
│   └── pricing.py              <- PricingStrategy enum (moved here), PriceQuote
├── repositories/               <- NEW: data layer
│   ├── base.py                 <- ProductRepository (abstraction)
│   └── mock.py                 <- MockProductRepository (in-memory dict)
├── exceptions/                 <- NEW: business errors
│   └── pricing.py              <- PricingException, ProductNotFoundError
├── api/
│   ├── dependencies.py         <- NEW: dependency providers (composition root)
│   ├── exception_handlers.py   <- NEW: business exception → HTTP response
│   └── routes/pricing.py       <- uses Depends(); no longer builds the service
├── schemas/
│   ├── pricing.py              <- request is now {product_id, strategy}
│   └── errors.py               <- NEW: ErrorResponse
├── services/pricing.py         <- repository injected; Product in, PriceQuote out
├── strategies/, guardrails/    <- take Product instead of PricingRequest
└── main.py                     <- registers the exception handlers
```

---

## Step 1 — Domain Model

**File:** `app/models/product.py`

```python
@dataclass(frozen=True, slots=True)
class Product:
    product_id: str
    name: str
    cost_price: Decimal
    competitor_price: Decimal

    def __post_init__(self) -> None:
        if self.cost_price <= 0:
            raise ValueError(...)
        if self.competitor_price <= 0:
            raise ValueError(...)
```

**Schema vs domain model:**

| | `PricingRequest` / `PricingResponse` | `Product` |
|---|---|---|
| Represents | The HTTP contract (JSON in / out) | The business entity itself |
| Built with | Pydantic | A plain Python dataclass |
| Knows about | JSON, validation errors, OpenAPI | Nothing outside the business |
| Changes when | The API shape changes | The business changes |

`Product` is independent of FastAPI, HTTP, the database, and JSON. The same object works for the API, a CLI command, a background job, or a unit test.

**Why `frozen=True`?** A product cannot change after it is loaded, so no strategy or guard can accidentally modify its cost while a price is being calculated.

**Why the `__post_init__` checks?** Pydantic used to guarantee positive prices because the client sent them. Now they come from storage, so the domain model guards its own invariants. A product with cost 0 would break the margin calculation (division by zero) — so it must not be able to exist.

### The business layer now speaks `Product`, not `PricingRequest`

Every strategy and guard changed signature:

```python
# Before (Day 4) — strategies depended on the HTTP schema
def calculate_price(self, request: PricingRequest) -> Decimal:
    return self._round(request.competitor_price * Decimal("0.97"))

# After (Day 5) — strategies depend on the domain model
def calculate_price(self, product: Product) -> Decimal:
    return self._round(product.competitor_price * Decimal("0.97"))
```

This is the change that actually makes the business layer independent of HTTP. Adding `Product` without it would just mean converting a `Product` back into a `PricingRequest` before pricing it.

**File:** `app/models/pricing.py`

- **`PricingStrategy`** moved here from `schemas/`. A strategy is a business concept. The API schemas now import it from the domain — the API depends on the domain, never the reverse.
- **`PriceQuote`** — what the service returns:

```python
@dataclass(frozen=True, slots=True)
class PriceQuote:
    product: Product
    strategy: PricingStrategy
    suggested_price: Decimal
    margin_percent: Decimal
    guardrails_applied: tuple[str, ...]
    message: str
```

**Why not return `PricingResponse` from the service?** `PricingResponse` is the HTTP contract. If the service built it, the business layer would depend on the API layer, and every change to the JSON shape would mean editing the service. Instead the router converts:

```python
return PricingResponse.from_quote(quote)
```

---

## Step 2 — Custom Exceptions

**File:** `app/exceptions/pricing.py`

```python
class PricingException(Exception):
    code: str = "pricing_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProductNotFoundError(PricingException):
    code = "product_not_found"

    def __init__(self, product_id: str) -> None:
        super().__init__(f"Product '{product_id}' was not found")
        self.product_id = product_id
```

**`PricingException`** is the base class for all business errors. Future exceptions (`OutOfStockError`, `PriceLockedError`, ...) inherit from it.

**`ProductNotFoundError`** is raised whenever the requested product cannot be found in the repository.

**Notice what is missing — a status code.** Business exceptions describe *what* went wrong in business terms. Whether that becomes a 404, a 400, or a line of CLI output is decided elsewhere.

**`code`** is a stable, machine-readable identifier. Messages are for humans and may be reworded; clients branch on `code`, which never changes.

**Why raise instead of returning `None`?**

- `None` forces every caller to remember `if product is None:`. Forget once and you get `AttributeError: 'NoneType' object has no attribute 'cost_price'` far away from the real cause.
- A generic `KeyError` (dict) or `NoResultFound` (SQLAlchemy) leaks *how* the data is stored. If every repository raises `ProductNotFoundError`, callers never change when storage does.

---

## Step 3 — Repository Pattern

**File:** `app/repositories/base.py`

```python
class ProductRepository(ABC):
    @abstractmethod
    async def get_by_id(self, product_id: str) -> Product:
        """Return the product, or raise ProductNotFoundError."""
```

`ProductRepository` defines *how* product data is accessed, not *where* it lives. The business layer depends only on this abstraction.

```
PricingService
       │
       ▼
ProductRepository        <- abstraction
       │
       ▼
Mock dict / PostgreSQL   <- implementations
```

**Why is `get_by_id` async when the mock is just a dict?**

Real repositories do I/O — PostgreSQL, Redis. With a synchronous interface, a PostgreSQL implementation would either block the event loop (freezing every other request while one query runs) or force the interface — and the service calling it — to change on Day 6. Declaring it `async` now keeps the promise that *when PostgreSQL arrives, only the repository changes.*

**File:** `app/repositories/mock.py`

```python
SEED_PRODUCTS = (
    Product("SKU-12345", "Wireless Headphones", Decimal("500.00"), Decimal("999.00")),
    Product("SKU-67890", "Mechanical Keyboard", Decimal("950.00"), Decimal("999.00")),
    Product("SKU-11111", "Bluetooth Speaker",   Decimal("900.00"), Decimal("400.00")),
)

class MockProductRepository(ProductRepository):
    def __init__(self, products: Iterable[Product] = SEED_PRODUCTS) -> None:
        self._products = {product.product_id: product for product in products}

    async def get_by_id(self, product_id: str) -> Product:
        product = self._products.get(product_id)
        if product is None:
            raise ProductNotFoundError(product_id)
        return product
```

An in-memory repository backed by a Python dictionary. It simulates the database until PostgreSQL is introduced on Day 6. The dict is keyed by `product_id`, so lookups are O(1) — like a primary-key index.

**Each seed product exercises a different path through the engine:**

| SKU | Product | Cost | Competitor | What it demonstrates |
|---|---|---|---|---|
| `SKU-12345` | Wireless Headphones | 500.00 | 999.00 | Healthy margin — no guard fires |
| `SKU-67890` | Mechanical Keyboard | 950.00 | 999.00 | `competitive` → floor raises 969.03 to 997.50 |
| `SKU-11111` | Bluetooth Speaker | 900.00 | 400.00 | `cost_plus` → ceiling and floor conflict, floor wins at 945.00 |

**The `products` argument** lets tests build a repository with any catalog they need — no special test-only class required.

---

## Step 4 — Dependency Injection in the Service

**File:** `app/services/pricing.py`

**Before (Day 4):**
```python
class PricingService:
    def calculate_price(self, request: PricingRequest) -> PricingResponse:
        strategy = StrategyFactory.get_strategy(request.strategy)
        raw_price = strategy.calculate_price(request)      # prices came from the client
        ...
        return PricingResponse(...)                        # built the HTTP response itself
```

**After (Day 5):**
```python
class PricingService:
    def __init__(self, product_repository: ProductRepository) -> None:
        self._product_repository = product_repository      # supplied from outside

    async def calculate_price(self, product_id: str, strategy: PricingStrategy) -> PriceQuote:
        product = await self._product_repository.get_by_id(product_id)   # may raise ProductNotFoundError
        raw_price = StrategyFactory.get_strategy(strategy).calculate_price(product)
        ...
        return PriceQuote(...)
```

The service never creates its repository. It does not know whether it holds a dict, PostgreSQL, or a test fake — only that it can call `get_by_id()`.

**Without DI:**
```
PricingService ──creates──▶ MockProductRepository
```
Switching to PostgreSQL means editing the service.

**With DI:**
```
PricingService ──uses──▶ ProductRepository ◀──implements── MockProductRepository
                                           ◀──implements── PostgresProductRepository (Day 6)
```
The service depends only on the abstraction. When PostgreSQL arrives, only the injected repository changes.

**The service does not catch `ProductNotFoundError`.** It has nothing useful to do about a missing product, so it lets the exception propagate. Catching an exception you cannot handle only hides it.

---

## Step 5 — Dependency Providers (the Composition Root)

**File:** `app/api/dependencies.py`

```python
@lru_cache
def get_product_repository() -> ProductRepository:
    return MockProductRepository()

ProductRepositoryDep = Annotated[ProductRepository, Depends(get_product_repository)]


def get_pricing_service(product_repository: ProductRepositoryDep) -> PricingService:
    return PricingService(product_repository)

PricingServiceDep = Annotated[PricingService, Depends(get_pricing_service)]
```

This module is the **composition root** — the *one* place in the application that names concrete classes. Everything else asks for abstractions. On Day 6, `get_product_repository()` returns a PostgreSQL repository: a one-line change.

**`@lru_cache`** is the same Singleton trick as `get_settings()` — one in-memory catalog shared by every request. The service is cheap and stateless, so it is built per request.

**`Annotated[..., Depends(...)]` aliases** let an endpoint declare a dependency with one short type annotation instead of repeating `Depends(...)` everywhere.

**Dependency graph, resolved by FastAPI on every request:**
```
Request
    │
    ▼
Depends(get_pricing_service)
    │
    ├── Depends(get_product_repository) → MockProductRepository (cached singleton)
    │
    ▼
PricingService(repository)
    │
    ▼
Endpoint
```

To call the endpoint FastAPI needs a `PricingService`; to build one it needs a `ProductRepository`; it calls the providers in the right order automatically.

---

## Step 6 — Thin Router

**File:** `app/api/routes/pricing.py`

**Before (Day 4):**
```python
_service = PricingService()     # router created the service at import time

@router.post("/", response_model=PricingResponse)
async def calculate_price(request: PricingRequest) -> PricingResponse:
    return _service.calculate_price(request)
```

**After (Day 5):**
```python
@router.post(
    "/",
    response_model=PricingResponse,
    responses={404: {"model": ErrorResponse, "description": "Product not found"}},
)
async def calculate_price(request: PricingRequest, service: PricingServiceDep) -> PricingResponse:
    quote = await service.calculate_price(request.product_id, request.strategy)
    return PricingResponse.from_quote(quote)
```

- No `PricingService()` anywhere in the router — FastAPI injects it.
- No `try/except` — an unknown product becomes a 404 elsewhere.
- `responses={404: ...}` documents the error in Swagger UI, so API consumers see it without reading the code.

---

## Step 7 — Request Schema Change

**File:** `app/schemas/pricing.py`

```python
class PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(..., min_length=1, max_length=64, examples=["SKU-12345"])
    strategy: PricingStrategy = Field(default=PricingStrategy.COMPETITIVE)
```

The client now sends only *which* product to price. Cost and competitor price come from the catalog — the single source of truth.

**This is a breaking API change:**

```
Day 4 request                              Day 5 request
{                                          {
  "product_id": "PROD-001",                  "product_id": "SKU-12345",
  "product_name": "Wireless Headphones",     "strategy": "competitive"
  "cost_price": 500.00,                    }
  "competitor_price": 999.00,
  "strategy": "competitive"
}
```

**Why `extra="forbid"`?** By default Pydantic silently ignores unknown fields. A client still sending `"cost_price": 950` would get a price calculated from the catalog's cost instead — with no error at all. `forbid` turns that into a clear `422 "Extra inputs are not permitted"`, so outdated clients fail loudly instead of quietly getting wrong answers.

**`min_length=1`** — an empty `product_id` used to be accepted.

**Side effect:** prices in responses are now consistently formatted (`"500.00"`), because they come from catalog `Decimal`s rather than however the client typed the number (`500`, `500.0`, `"500.00"`).

---

## Step 8 — Global Exception Handlers

**File:** `app/api/exception_handlers.py`

```python
_STATUS_CODES: dict[type[PricingException], int] = {
    ProductNotFoundError: status.HTTP_404_NOT_FOUND,
    PricingException:     status.HTTP_400_BAD_REQUEST,   # fallback
}

def _status_code_for(exc: PricingException) -> int:
    return next(_STATUS_CODES[cls] for cls in type(exc).__mro__ if cls in _STATUS_CODES)


async def pricing_exception_handler(request: Request, exc: PricingException) -> JSONResponse:
    return JSONResponse(
        status_code=_status_code_for(exc),
        content=ErrorResponse(detail=exc.message, code=exc.code).model_dump(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", code="internal_error").model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(PricingException, pricing_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
```

Registered once, in `create_app()`.

**One handler plus a table of status codes.** `_STATUS_CODES` is the only place business errors meet HTTP. A new exception type is one new line in the table — not a new handler.

**`__mro__` (Method Resolution Order)** is a class's ancestry:
```
ProductNotFoundError.__mro__ == (ProductNotFoundError, PricingException, Exception, BaseException, object)
```
Walking it means a subclass inherits its parent's status code, and any business exception missing from the table falls back to 400.

**One error shape everywhere** (`app/schemas/errors.py`):
```json
{ "detail": "Product 'SKU-99999' was not found", "code": "product_not_found" }
```
Show `detail` to humans; branch on `code` in code. Using `detail` as the key matches FastAPI's own error responses.

**The catch-all 500 handler** is the last line of defence. Without it, an unexpected error could put a stack trace, an SQL query, or a connection string in front of the client. With it, the client sees only `"Internal server error"`, and Starlette re-raises the exception after responding, so the server log still has the full traceback.

**Error flow:**
```
Client
   │  POST {"product_id": "SKU-99999"}
   ▼
Router ─────────────────────── no try/except
   │
   ▼
PricingService ─────────────── does not catch it
   │
   ▼
MockProductRepository.get_by_id()
   │  raise ProductNotFoundError("SKU-99999")
   ▼
pricing_exception_handler ──── looks up 404 in _STATUS_CODES
   │
   ▼
404 {"detail": "Product 'SKU-99999' was not found", "code": "product_not_found"}
```

The server log for that request — each layer reports once:
```
INFO | app.api.routes.pricing     | POST /pricing/ | product_id=SKU-99999
INFO | app.repositories.mock      | Product not found | product_id=SKU-99999
INFO | app.api.exception_handlers | Business error | POST /api/v1/pricing/ | status=404 | code=product_not_found | Product 'SKU-99999' was not found
```

---

## Step 9 — Tests

**DI makes testing trivial.** Tests replace the repository without touching any application code:

```python
@pytest.fixture
def app(catalog):
    application = create_app()
    repository = MockProductRepository(catalog)
    application.dependency_overrides[get_product_repository] = lambda: repository
    return application
```

`dependency_overrides` tells FastAPI: "whenever something asks for `get_product_repository`, call this instead." The DEVLOG scenario products (`PROD-001`, `PROD-005`, ...) become the catalog; the production seed data is untouched.

**The service is tested with no HTTP at all** — and with a hand-written fake rather than the mock, proving it depends only on the abstraction:

```python
class RecordingRepository(ProductRepository):
    async def get_by_id(self, product_id: str) -> Product:
        self.requested_ids.append(product_id)
        return self._product

quote = await PricingService(RecordingRepository(product)).calculate_price("PROD-001", PricingStrategy.COMPETITIVE)
```

**Architecture test** (`tests/test_architecture.py`) reads the imports of every module in `models/`, `exceptions/`, `repositories/`, `services/`, `strategies/`, `guardrails/` and `factories/`, and fails if any of them imports `fastapi`, `starlette`, `app.api` or `app.schemas`. "The business layer is independent of the web framework" is no longer just a sentence in this log — the test suite enforces it. Verified by adding `from fastapi import HTTPException` to the service in a scratch copy: the test failed and named the file.

| Test file | Covers |
|---|---|
| `test_models.py` | `Product` is immutable and rejects non-positive prices |
| `test_repositories.py` | Seed catalog, custom catalog, `ProductNotFoundError` |
| `test_pricing_service.py` | Service with injected fakes; never-below-cost invariant grid |
| `test_exception_handlers.py` | 400 fallback; the 500 response never leaks internal details |
| `test_architecture.py` | Business layer never imports FastAPI or the API layer |
| `test_pricing_api.py` | Every DEVLOG scenario over HTTP; 404; seed catalog end to end |

**Result: 86 tests, all passing, in under half a second.**

---

## Day 5 — Full Request Lifecycle

```
Client sends:
POST /api/v1/pricing/
{ "product_id": "SKU-67890", "strategy": "competitive" }
          |
PricingRequest validation (Pydantic)
  product_id present? strategy valid? no extra fields?   PASS
          |
FastAPI resolves dependencies
  get_product_repository()   → MockProductRepository (cached)
  get_pricing_service(repo)  → PricingService(repo)
          |
Router → await service.calculate_price("SKU-67890", COMPETITIVE)
          |
Step 1: repository.get_by_id("SKU-67890")
          → Product("Mechanical Keyboard", cost=950.00, competitor=999.00)
          |
Step 2: StrategyFactory → CompetitivePricingStrategy
Step 3: raw_price = 999.00 × 0.97 = 969.03
          |
Step 4: Guard chain
  VolatilityGuard: ceiling 1498.50 → passed
  PriceFloorGuard: floor 997.50    → TRIGGERED → 997.50
          |
Step 5: margin = (997.50 - 950.00) / 997.50 × 100 = 4.76%
Step 6: PriceQuote returned to the router
          |
Router: PricingResponse.from_quote(quote)
          |
HTTP 200:
{
  "product_id": "SKU-67890",
  "product_name": "Mechanical Keyboard",
  "cost_price": "950.00",
  "competitor_price": "999.00",
  "suggested_price": "997.50",
  "strategy_used": "competitive",
  "margin_percent": "4.76",
  "guardrails_applied": [
    "PriceFloorGuard: price raised from 969.03 to 997.50 (minimum margin of 5% enforced)"
  ],
  "message": "Strategy 'competitive' applied. Suggested price is Rs.1.50 below competitor price."
}
```

---

## Day 5 — Final Architecture

```
                    Client
                       │
                       ▼
                 FastAPI Router ◀──────── Global Exception Handlers
                       │                   PricingException → 4xx JSON
            (Dependency Injection)         Exception        → 500 JSON
                       │
                       ▼
               PricingService
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
ProductRepository               StrategyFactory
       │                               │
       ▼                               ▼
MockProductRepository          PricingStrategy
       │                               │
    Product ───────────────────────────┘
                       │
                       ▼
                 Guard Chain
          VolatilityGuard → PriceFloorGuard
                       │
                       ▼
                  PriceQuote
                       │
                       ▼
     PricingResponse.from_quote()  (router)
```

**Which way do the dependencies point?**

| Layer | Packages | May depend on |
|---|---|---|
| API | `api/`, `schemas/` | Business, Domain |
| Business | `services/`, `strategies/`, `guardrails/`, `factories/`, `exceptions/` | Domain, repository abstraction |
| Data | `repositories/` | Domain |
| Domain | `models/` | Nothing |

Arrows only point inward, toward the domain. Nothing in the business layer knows the API exists — and `test_architecture.py` keeps it that way.

---

## Day 5 — Design Principles Applied

### Dependency Inversion Principle (DIP)

High-level modules (`PricingService`) no longer depend on low-level modules (`MockProductRepository`). Both depend on the `ProductRepository` abstraction.

### Single Responsibility Principle (SRP)

| Component | Its only responsibility |
|---|---|
| Repository | Product retrieval |
| Service | Pricing orchestration |
| Strategy | Price calculation |
| Guard chain | Business validation |
| Exception handler | Turning business errors into HTTP responses |
| Dependency providers | Choosing which concrete classes the app uses |

Each component has a single reason to change.

### Open/Closed Principle (OCP)

Adding a new storage implementation means creating another repository:

```
          ProductRepository
                  ▲
                  │
       ┌──────────┴───────────┐
       │                      │
MockProductRepository   PostgresProductRepository
```

The service remains unchanged. Likewise, a new business exception is a new class plus one line in `_STATUS_CODES` — no existing handler changes.

### Separation of Concerns

- **API layer** → routers, schemas, dependency providers, exception handlers
- **Business layer** → `PricingService`, strategies, guardrails
- **Domain layer** → `Product`, `PriceQuote`, `PricingStrategy`
- **Data layer** → repositories

### Inversion of Control (IoC)

The service no longer decides how its dependencies are created. The application supplies them through FastAPI's dependency injection system. This makes the application easier to test, mock, and extend.

---

## Interview Questions — Day 5

**Q: What is the Repository Pattern?**
An abstraction that makes a data store look like an in-memory collection of domain objects. Callers ask for "the product with ID X" without knowing whether it comes from a dict, PostgreSQL, Redis, or another HTTP API. Storage can change without touching business logic.

**Q: What is the difference between Dependency Injection and Inversion of Control?**
IoC is the principle: a component does not control the creation of its collaborators — something outside does. DI is one technique for achieving it: pass the collaborators in (here, through the constructor), and let a container — FastAPI's `Depends()` — assemble the graph.

**Q: Why not raise `HTTPException` inside the service?**
It ties business logic to one web framework. The same service may run from a CLI, a scheduled job, or a message consumer where "404" means nothing. Business code raises business exceptions; the HTTP layer translates them.

**Q: Isn't a domain model plus Pydantic schemas just duplication?**
They change for different reasons. The API shape changes when clients need something new; the domain changes when the business does. Keeping them separate means a JSON tweak cannot break pricing logic, and the business layer never imports the web layer.

**Q: How do you test a service that needs a database?**
Inject a fake. Because `PricingService` depends only on `ProductRepository`, a test can pass an in-memory implementation; over HTTP, `app.dependency_overrides` swaps it. No database, no network, and the tests run in milliseconds.

**Q: Why is the repository interface `async` when the mock does no I/O?**
The interface is designed for the real implementations, which do. Making it async now means a PostgreSQL repository can use an async driver without blocking the event loop — and without changing the interface or the service.

**Q: What happens if someone adds a new business exception but forgets to map it to a status code?**
The handler walks the exception's `__mro__`, reaches `PricingException`, and returns 400 with the standard JSON error body. Clients get a sensible response instead of a 500.

**Q: Why should a 500 response never include the exception message?**
Unexpected errors can contain internal details — SQL, file paths, credentials in connection strings. The client gets a generic message; the full details go to the server log, where only engineers can see them.

---

## Looking Ahead — Day 6

- Add `PostgresProductRepository(ProductRepository)` using async SQLAlchemy.
- Keep the SQLAlchemy table classes in the data layer, separate from the `Product` domain model in `app/models/`; the repository maps database rows to `Product`.
- Create the database engine in `lifespan()` (the Day 1 TODO) and close it on shutdown.
- Change one line: `get_product_repository()` returns the PostgreSQL repository.
- `PricingService`, the strategies, the guards, and the router should not change — the architecture test and the service tests will prove it.

---

---

# Day 6 — PostgreSQL Integration, SQLAlchemy ORM & Database Migrations

**Goal:** Replace the in-memory mock repository with a real PostgreSQL-backed persistence layer while keeping the business logic completely unchanged.

**Problem with Day 5:**

- Product data was stored in an in-memory Python dictionary.
- Data disappeared whenever the application restarted.
- The database schema was not version controlled.
- The business logic was ready for a database, but no persistence layer existed.

The Repository Pattern had already decoupled the service from data access, but there was still no real storage implementation.

**Solution:**

- PostgreSQL (in Docker)
- SQLAlchemy 2.x ORM (async)
- Alembic database migrations
- `PostgresProductRepository`
- Idempotent database seeding
- Repository integration through Dependency Injection

This replaces the mock data source **without modifying the business layer**. `PricingService`, every strategy, every guard, and the router were not edited on Day 6.

---

## New File Structure

```
SmartPrice(BackendProject)/
├── docker-compose.yml              <- NEW: PostgreSQL 17 + persistent volume
├── .env.example                    <- NEW: documented settings, no real secrets
├── .gitignore                      <- now ignores .env, .venv, __pycache__
├── alembic.ini                     <- NEW: Alembic configuration
├── alembic/                        <- NEW: migrations
│   ├── env.py                      <- uses app settings + ORM metadata
│   ├── script.py.mako
│   └── versions/
│       └── 2026_10_28_2334-b2cda35e279c_create_products_table.py
└── app/
    ├── database/                   <- NEW: persistence infrastructure
    │   ├── base.py                 <- declarative Base + constraint naming
    │   ├── session.py              <- engine, SessionLocal, get_db()
    │   ├── seed.py                 <- idempotent sample data
    │   └── models/
    │       └── product.py          <- ProductORM (the products table)
    ├── repositories/
    │   └── postgres.py             <- NEW: PostgresProductRepository
    ├── api/dependencies.py         <- injects the PostgreSQL repository
    └── main.py                     <- lifespan checks + closes the database; /health reports it
```

---

## Step 1 — PostgreSQL in Docker

**File:** `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:17
    container_name: smartprice-postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER:?set POSTGRES_USER in .env}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in .env}
      POSTGRES_DB: ${POSTGRES_DB:?set POSTGRES_DB in .env}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - smartprice_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
```

- **PostgreSQL 17**, identical on every developer's machine — no local install.
- **Named volume** `smartprice_pgdata`: data survives container restarts. Verified: restarted the container, all 3 products were still there.
- **Environment-based credentials**: Docker Compose reads `.env` automatically. `${VAR:?message}` makes Compose refuse to start with a clear error if a credential is missing, instead of silently creating a database with an empty password.
- **Healthcheck**: `pg_isready` reports when PostgreSQL actually accepts connections, not just when the container process has started.

> **Decision — Docker for local, Neon for production.** The container runs the official PostgreSQL engine — the same database a cloud provider runs, not a substitute. Docker is the *local and test* environment: free, offline, one command to create and `docker compose down -v` to erase, the exact same version for everyone. The *deployed* environment uses **Neon**, a managed (serverless) PostgreSQL, because production databases are normally managed services (backups, TLS, scaling), not containers on an app server. Same engine, same migrations; only `DATABASE_URL` differs between environments. (Recorded on Day 14, when deployment made the split concrete.)

**Secrets hygiene:**
- `.env` holds the real password and is now listed in `.gitignore`.
- `.env.example` documents every setting with placeholder values — it is the file that gets committed.
- `alembic.ini` deliberately contains no database URL.

```bash
docker compose up -d       # start
docker compose down        # stop, keep data
docker compose down -v     # stop and DELETE all data
```

---

## Step 2 — SQLAlchemy Infrastructure

### Base

**File:** `app/database/base.py`

```python
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
```

All ORM models inherit from `Base`, which registers their tables on `Base.metadata`. Alembic compares that metadata with the live database to generate migrations.

```
ProductORM
      │
      ▼
    Base
```

**Why a naming convention?** Without it, PostgreSQL invents constraint names (`products_pkey`, `products_cost_price_check`). A later migration that needs to drop or alter a constraint must know its exact name — and guessed names differ between databases. With the convention every constraint is predictable: `pk_products`, `ck_products_cost_price_positive`.

### Database Session

**File:** `app/database/session.py`

```python
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True,
    connect_args={"timeout": settings.db_connect_timeout},
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
```

```
Engine              <- one per process; owns the connection pool
    │
    ▼
SessionLocal        <- factory that creates sessions
    │
    ▼
get_db()            <- FastAPI dependency: one session per request
```

| Piece | Lifetime | Job |
|---|---|---|
| `engine` | Whole process | Holds the connection pool. Creating it does **not** connect — connections open lazily on the first query and are then reused. |
| `SessionLocal` | Whole process | Factory: every call returns a new session. |
| Session | One request | A unit of work. Borrows a pooled connection, gives it back when closed. |

**`get_db()`** is exposed as a FastAPI dependency. `async with` guarantees the session is closed — and its connection returned to the pool — even if the endpoint raises. Uncommitted work is rolled back on close.

**Pool settings** (`pool_size`, `max_overflow`, `pool_timeout`) existed in `Settings` since Day 1 and are finally used. `pool_pre_ping=True` tests a pooled connection before handing it out, so a connection the server closed overnight is silently replaced instead of failing a request.

**`expire_on_commit=False`**: by default SQLAlchemy expires objects after a commit and reloads them on next access. Async sessions cannot do that implicit reload, so objects stay readable instead.

### Lifespan — the Day 1 TODO, done

**File:** `app/main.py`

```python
# --- Startup phase ---
if settings.db_check_on_startup:
    if await database.ping_database():
        logger.info("Database connection OK")
    else:
        logger.error("Database unavailable — pricing requests will fail until it is reachable")

yield

# --- Shutdown phase ---
await database.dispose_engine()     # close every pooled connection gracefully
```

The app still starts when the database is down — so it can report the problem — but the very first log lines say so. On shutdown, every pooled connection is closed cleanly.

**`/health` now verifies database connectivity:**
```json
{ "status": "ok", "app_name": "SmartPrice", "environment": "development", "database": "ok" }
```
When PostgreSQL is unreachable: `"status": "degraded", "database": "unavailable"`.

---

## Step 3 — ORM Model

**File:** `app/database/models/product.py`

```python
class ProductORM(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("cost_price > 0", name="cost_price_positive"),
        CheckConstraint("competitor_price > 0", name="competitor_price_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    cost_price: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    competitor_price: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

The ORM model is the bridge between Python objects and relational rows.

| Column | Type | Why |
|---|---|---|
| `id` | `INTEGER` primary key | Surrogate key — internal to the database, never exposed by the API |
| `product_id` | `VARCHAR(64)` unique + indexed | Business key (the SKU clients send); the index makes lookups O(log n) |
| `name` | `VARCHAR(200)` | Matches the 200-character limit the API always had |
| `cost_price`, `competitor_price` | `NUMERIC(12, 2)` | Exact decimal money — never `FLOAT`, the same rule as `Decimal` on Day 2 |
| `created_at`, `updated_at` | `TIMESTAMPTZ` | Audit trail, filled in by PostgreSQL itself |

**`Mapped[...]` type hints** are SQLAlchemy 2.x style: the column type and nullability come from the Python annotation (`Mapped[str]` → `NOT NULL`), and editors understand the attribute types.

**Check constraints** repeat `Product.__post_init__`'s rule at the database level. Defence in depth: a bad row is rejected when it is *written* — by a script, a migration, or someone in `psql` — rather than discovered later when a request fails to read it.

### Two classes for one product?

| | `ProductORM` | `Product` |
|---|---|---|
| Lives in | `app/database/models/` | `app/models/` |
| Describes | A row in the `products` table | The business entity |
| Knows about | SQLAlchemy, columns, constraints | Nothing outside the business |
| Has | Surrogate `id`, timestamps | Business invariants |

The table carries database concerns the business does not care about; the domain model carries business rules the database does not care about. Each can change without dragging the other along.

---

## Step 4 — Alembic Migrations

**Initialized with the async template:**
```bash
alembic init -t async alembic
```

```
alembic/
├── env.py              <- how to connect + which metadata to compare
├── script.py.mako      <- template for new migration files
└── versions/           <- one file per schema change
```

**`alembic/env.py` configured to use the project's own settings and metadata:**
```python
from app.config.settings import settings
from app.database.base import Base
import app.database.models          # registers every table on Base.metadata

config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
target_metadata = Base.metadata
```

- The database URL comes from `.env` through `Settings` — one source of truth, nothing secret in `alembic.ini`.
- `import app.database.models` matters: a model that is never imported is invisible to `Base.metadata`, and autogenerate would think its table should be dropped.
- `alembic.ini` sets a date-prefixed `file_template`, so migrations sort chronologically in the file explorer.

**Generated the first migration from the ORM model:**
```bash
alembic revision --autogenerate -m "create products table"
```

```
Detected added table 'products'
Detected added index 'ix_products_product_id' on '('product_id',)'
Generating alembic/versions/2026_10_28_2334-b2cda35e279c_create_products_table.py
```

**Autogenerate is a draft, not a final answer.** The generated file was reviewed before applying: named check constraints, `NUMERIC(12, 2)` money columns, a unique index on `product_id`, and a `downgrade()` that reverses everything.

**Applied — and tested in both directions:**
```bash
alembic upgrade head      # tables: alembic_version, products
alembic downgrade base    # products table removed
alembic upgrade head      # back again
alembic current           # b2cda35e279c (head)
```

A migration whose `downgrade()` has never been run is a migration that will fail on the day you need it.

**Result:**
```
products
alembic_version     <- Alembic's bookkeeping: which revision this database is at
```

The entire database schema is now version controlled. Every future change (new column, new table) is a new migration file, reviewed in code and applied the same way on every machine.

---

## Step 5 — Database Seeding

**File:** `app/database/seed.py`

```python
async def seed_products(session: AsyncSession, products: Iterable[Product] = SEED_PRODUCTS) -> int:
    rows = [{"product_id": p.product_id, "name": p.name,
             "cost_price": p.cost_price, "competitor_price": p.competitor_price} for p in products]
    statement = (
        insert(ProductORM)
        .values(rows)
        .on_conflict_do_nothing(index_elements=[ProductORM.product_id])
        .returning(ProductORM.product_id)
    )
    inserted = (await session.execute(statement)).scalars().all()
    return len(inserted)
```

```bash
python -m app.database.seed
```

**Idempotent — verified by running it twice:**
```
Seed complete | inserted=3 | already present=0
Seed complete | inserted=0 | already present=3
```

**How:** PostgreSQL's `INSERT ... ON CONFLICT (product_id) DO NOTHING` lets the *database* decide atomically whether each row exists. The obvious alternative — "SELECT first, INSERT if missing" — has a race: two processes can both see "missing" and both insert. `RETURNING product_id` reports exactly which rows were new.

**`DO NOTHING`, not `DO UPDATE`:** re-running the seed never overwrites a product someone edited.

**One catalog, one definition:** the seed inserts the same `SEED_PRODUCTS` that `MockProductRepository` serves — `SKU-12345`, `SKU-67890`, `SKU-11111` — so the mock and the database can never disagree.

**The caller owns the transaction:** `seed_products()` never commits. The CLI commits; tests run it inside a transaction that is rolled back. Same function, both uses.

---

## Step 6 — PostgreSQL Repository

**File:** `app/repositories/postgres.py`

```python
class PostgresProductRepository(ProductRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, product_id: str) -> Product:
        statement = select(ProductORM).where(ProductORM.product_id == product_id)
        result = await self._session.execute(statement)
        row = result.scalar_one_or_none()

        if row is None:
            raise ProductNotFoundError(product_id)

        return self._to_domain(row)

    @staticmethod
    def _to_domain(row: ProductORM) -> Product:
        return Product(
            product_id=row.product_id,
            name=row.name,
            cost_price=row.cost_price,
            competitor_price=row.competitor_price,
        )
```

```
PricingService
        │
        ▼
ProductRepository              <- the service only knows this
        │
        ▼
PostgresProductRepository
        │
        ▼
SQLAlchemy (AsyncSession)
        │
        ▼
PostgreSQL
```

**The repository honours the Day 5 contract exactly:**
- It returns a domain `Product` — the ORM row never leaves this class (`_to_domain`).
- A missing row raises `ProductNotFoundError`, not `None` and not a SQLAlchemy error. The global handler turns it into the same 404 as before.

**The session is injected**, one per request from `get_db()`. The repository never opens or closes connections itself.

### SQLAlchemy 2.x Query Style

Instead of the legacy query interface:
```python
session.query(ProductORM).filter(ProductORM.product_id == product_id).first()
```

the repository uses the 2.x API:
```
select(ProductORM).where(...)     <- build the statement (a plain object, no I/O)
        │
        ▼
await session.execute(statement)  <- run it
        │
        ▼
scalar_one_or_none()              <- unwrap: the ORM object, or None
```

- `session.query()` is legacy in SQLAlchemy 2.0 and is not available on `AsyncSession` at all.
- `scalar_one_or_none()` also states an expectation: **at most one** row. If the unique constraint were ever missing and two rows matched, it raises instead of silently returning an arbitrary one, as `.first()` would.

---

## Step 7 — Dependency Injection

**File:** `app/api/dependencies.py`

**Day 5:**
```python
@lru_cache
def get_product_repository() -> ProductRepository:
    return MockProductRepository()
```

**Day 6:**
```python
DatabaseSessionDep = Annotated[AsyncSession, Depends(get_db)]

def get_product_repository(session: DatabaseSessionDep) -> ProductRepository:
    return PostgresProductRepository(session)
```

Previous flow:
```
Router
      │
      ▼
MockProductRepository
      │
      ▼
PricingService
```

Current flow:
```
Router
      │
      ▼
get_db()                       <- one AsyncSession per request
      │
      ▼
PostgresProductRepository      <- wraps that session
      │
      ▼
PricingService                 <- unchanged
```

**This is the payoff of Day 5.** Swapping the mock for PostgreSQL changed one provider function. `get_pricing_service()`, `PricingService`, the strategies, the guards, the router and the exception handlers are untouched.

**Why did `@lru_cache` go away?** The mock was a stateless singleton, safe to share. A PostgreSQL repository wraps a *session*, and a session must never be shared between requests — so the repository is now built per request around that request's session. FastAPI caches a dependency *within* one request, so `get_db()` still runs exactly once per request.

---

## Step 8 — Tests

### Unit tests stay database-free

Every Day 5 test overrides `get_product_repository` with an in-memory `MockProductRepository`. FastAPI then never calls `get_db()`, so **107 tests run with no database**, in about 7 seconds.

### Integration tests against the real PostgreSQL

**File:** `tests/test_postgres_repository.py` — every test runs inside a transaction that is rolled back:

```python
@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(bind=connection, expire_on_commit=False,
                               join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()      # nothing the test wrote survives
    await engine.dispose()
```

| Test | Proves |
|---|---|
| `test_get_by_id_returns_domain_product_with_exact_decimals` | Returns `Product` (not `ProductORM`); `12.34` stays exactly `Decimal("12.34")` |
| `test_missing_product_raises_product_not_found` | Same business exception as the mock |
| `test_seed_is_idempotent` | Second seed run inserts 0 rows |
| `test_database_rejects_non_positive_prices` | The check constraint fires, by name |
| `test_business_logic_unchanged_on_postgres` | DEVLOG Day 4 scenario: same 997.50 / 4.76% from PostgreSQL |

Plus end-to-end tests in `test_pricing_api.py` that use the real wiring with **no overrides** — `get_db()` → `PostgresProductRepository` → PostgreSQL — for the three seed SKUs, a 404, and `/health`.

**Marker + automatic skip:** these tests carry `@pytest.mark.postgres`. `conftest.py` probes the database once (reachable *and* migrated); if not, they are skipped with a message saying how to start it. The suite never fails just because Docker is not running.

```
Postgres running:   117 passed
Postgres stopped:   107 passed, 10 skipped
```

Verified after a run: zero `TEST-PG-*` rows left in the database.

### Architecture test — Persistence Ignorance, enforced

`tests/test_architecture.py` gained a second rule: no module in `models/`, `services/`, `strategies/`, `guardrails/`, `factories/`, `exceptions/` — nor the `ProductRepository` abstraction — may import `sqlalchemy`, `asyncpg`, or `app.database`. The only allowed exception is `repositories/postgres.py`, the adapter whose whole job is to talk to the database.

---

## Day 6 — Full Request Lifecycle

```
Client sends:
POST /api/v1/pricing/
{ "product_id": "SKU-11111", "strategy": "cost_plus" }
          |
PricingRequest validation (Pydantic)
          |
FastAPI resolves dependencies
  get_db()                    → AsyncSession (connection from the pool)
  get_product_repository()    → PostgresProductRepository(session)
  get_pricing_service()       → PricingService(repository)
          |
Router → await service.calculate_price("SKU-11111", COST_PLUS)
          |
Step 1: SELECT products.* FROM products WHERE products.product_id = 'SKU-11111'
          → ProductORM row → _to_domain() → Product("Bluetooth Speaker", 900.00, 400.00)
          |
Step 2–3: CostPlusPricingStrategy → 900.00 × 1.40 = 1260.00
          |
Step 4: Guard chain
  VolatilityGuard: ceiling 600.00 → capped to 600.00
  PriceFloorGuard: floor 945.00   → raised to 945.00 (overrides the cap)
          |
Step 5: margin = (945.00 - 900.00) / 945.00 × 100 = 4.76%
          |
Response sent → get_db() closes the session → connection returns to the pool
          |
HTTP 200:
{
  "product_id": "SKU-11111",
  "product_name": "Bluetooth Speaker",
  "suggested_price": "945.00",
  "margin_percent": "4.76",
  ...
}
```

---

## Day 6 — Final Architecture

```
                    Client
                       │
                       ▼
                 FastAPI Router ◀──────── Global Exception Handlers
                       │
            (Dependency Injection: get_db → repository → service)
                       │
                       ▼
               PricingService
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
ProductRepository              StrategyFactory
        │                             │
        ▼                             ▼
PostgresProductRepository      PricingStrategy
        │                             ▲
        ▼                             │
  SQLAlchemy ORM (ProductORM)         │
        │                             │
        ▼                             │
    PostgreSQL ──▶ Product (domain) ──┘
                       │
                       ▼
                 Guard Chain
          VolatilityGuard → PriceFloorGuard
                       │
                       ▼
                  PriceQuote
                       │
                       ▼
     PricingResponse.from_quote()  (router)
```

**Layers:**

| Layer | Packages | Knows about |
|---|---|---|
| API | `api/`, `schemas/` | Business, domain |
| Business | `services/`, `strategies/`, `guardrails/`, `factories/`, `exceptions/` | Domain, repository abstraction |
| Domain | `models/` | Nothing |
| Persistence | `repositories/postgres.py` | Domain, ORM, SQLAlchemy |
| ORM | `database/models/` | SQLAlchemy |
| Database | PostgreSQL (Docker) | — |
| Migrations | `alembic/` | ORM metadata |

---

## Day 6 — Design Principles Applied

### Repository Pattern

Data access is completely abstracted from the business layer.

```
          ProductRepository
                  ▲
                  │
       ┌──────────┴───────────┐
       │                      │
MockProductRepository   PostgresProductRepository
   (tests)                 (application)
```

Either implementation can be injected without modifying the service — and both are, every test run.

### Dependency Inversion Principle (DIP)

`PricingService` continues to depend only on the `ProductRepository` abstraction. The switch from mock data to PostgreSQL required no changes to the business logic.

### Single Responsibility Principle (SRP)

| Component | Its only responsibility |
|---|---|
| Router | HTTP |
| Dependency providers | Wiring: session → repository → service |
| `PricingService` | Pricing orchestration |
| Repository | Database access + mapping rows to domain objects |
| ORM model | Table mapping |
| Strategy | Price calculation |
| Guard chain | Business validation |
| Alembic | Schema versioning |
| Seed script | Sample data |

### Open/Closed Principle (OCP)

Adding another storage implementation (Redis cache, MongoDB, a REST API) requires only another `ProductRepository` subclass and one changed provider. The service remains unchanged.

### Separation of Concerns

- **API layer** → routers, schemas, dependency providers
- **Business layer** → `PricingService`, strategies, guards
- **Domain layer** → `Product`
- **Persistence layer** → repositories
- **ORM layer** → SQLAlchemy models
- **Database layer** → PostgreSQL
- **Migration layer** → Alembic

### Persistence Ignorance

The domain model (`Product`) has no knowledge of SQLAlchemy, PostgreSQL, FastAPI or Pydantic. Business logic works entirely with domain objects, independent of persistence technology — and `test_architecture.py` fails the build if that ever stops being true.

---

## Notes — Where Day 6 Differs From the Original Plan

**1. The `products` columns follow the domain that exists.**
The plan listed `base_price`, `inventory` and `demand_score`. SmartPrice's domain, strategies and guards are built on `cost_price` and `competitor_price` — nothing in the code reads a base price, inventory level or demand score. The table therefore stores what the business uses today. When an ML feature needs inventory or demand data, adding it is one new migration.

> **Update:** `inventory` and `demand_score` were added the same day by a second migration — see [Day 6 Follow-up](#day-6-follow-up--inventory-and-demand-score). `base_price` was not added: `cost_price` already plays that role.

**2. Async driver: `asyncpg`, not `psycopg`.**
The Day 5 repository interface is `async`, so the database driver must be too. Psycopg 3's async mode refuses to run on Windows' default `ProactorEventLoop` — which is exactly what uvicorn uses on Windows without `--reload`. `asyncpg` works on every event loop. `requirements.txt` now lists `sqlalchemy[asyncio]` and `asyncpg` instead of `psycopg[binary]`, and `DATABASE_URL` uses the `postgresql+asyncpg://` scheme.

**3. "Domain refactoring" and "request simplification" were already done on Day 5.**
Strategies were moved from `PricingRequest` to the `Product` domain model, and the request was reduced to `{product_id, strategy}`, as part of Day 5. Day 6 is where that preparation paid off: the database arrived and nothing above the repository changed.

---

## Interview Questions — Day 6

**Q: Why use migrations instead of `Base.metadata.create_all()`?**
`create_all()` only creates tables that do not exist — it can never add a column, change a type, or rename anything in a database that already has data. Migrations are ordered, reviewable, reversible scripts that move every environment (laptop, CI, production) through exactly the same schema changes.

**Q: Why keep the ORM model separate from the domain model?**
They change for different reasons. The table gains indexes, surrogate keys and audit columns for the database's sake; the domain gains rules for the business's sake. Keeping them apart means the business layer never imports SQLAlchemy and can be tested without a database.

**Q: What does the connection pool do, and what is `pool_pre_ping`?**
Opening a PostgreSQL connection is expensive (TCP + authentication). The pool keeps connections open and lends them to requests. `pool_pre_ping` checks a pooled connection before lending it, transparently replacing connections the server or a firewall closed while idle.

**Q: Why one session per request?**
A session is a unit of work with its own transaction and identity map. Sharing it between concurrent requests mixes their transactions and data. Per-request sessions isolate requests, and `get_db()` guarantees each one is closed.

**Q: How is the seed idempotent, and why not "check then insert"?**
`INSERT ... ON CONFLICT DO NOTHING` lets the database decide atomically, row by row, under the unique index. "SELECT, then INSERT if missing" has a race window in which two processes both see "missing" and both insert.

**Q: How do you test database code without polluting the database?**
Run each test inside a transaction and roll it back at the end. `join_transaction_mode="create_savepoint"` makes even a `commit()` inside the code under test only release a savepoint, so the outer rollback still undoes everything.

**Q: Why is money stored as `NUMERIC(12, 2)` and not `FLOAT`?**
Same reason as `Decimal` on Day 2: binary floating point cannot represent most decimal fractions exactly. `NUMERIC` stores exact base-10 values, and asyncpg returns them as Python `Decimal` — exact from the database to the JSON response.

**Q: What did switching to PostgreSQL change in `PricingService`?**
Nothing. That is the point of the Repository Pattern and Dependency Inversion from Day 5 — and the architecture tests guarantee the business layer still does not import any persistence code.

---

## Looking Ahead — Day 7

- Redis caching: a caching `ProductRepository` that wraps the PostgreSQL one — the same plug-in point, another implementation.
- `GET /api/v1/pricing/` still returns `[]`: persisting price quotes (a `price_quotes` table, a second migration) would give it real history.

---

---

# Day 6 Follow-up — Inventory and Demand Score

**Goal:** Add the two demand signals from the original Day 6 plan — `inventory` and `demand_score` — to the `products` table and the domain model, through a proper second migration on a database that already holds data.

**Why now, if no strategy uses them yet?** They are the inputs a demand-aware strategy or an ML model will need. Adding them now lets real data accumulate, and it exercises the migration workflow that every future schema change will follow: *change the model → generate → review → apply → test both directions.*

`base_price` from the original plan was not added: `cost_price` already plays that role.

---

## Step 1 — Domain Model

**File:** `app/models/product.py`

```python
@dataclass(frozen=True, slots=True)
class Product:
    product_id: str
    name: str
    cost_price: Decimal
    competitor_price: Decimal
    inventory: int = 0                       # units in stock
    demand_score: Decimal = Decimal("0.50")  # 0.00 (no demand) … 1.00 (very high); 0.50 = neutral

    def __post_init__(self) -> None:
        ...
        if self.inventory < 0:
            raise ValueError(...)
        if not Decimal("0") <= self.demand_score <= Decimal("1"):
            raise ValueError(...)
```

**Why defaults?** Every existing `Product(...)` call — the mock, the tests, the guard tests — keeps working unchanged. A new field with a sensible default is a backwards-compatible change.

**Why `0.50`?** It is the *neutral* point of the scale: "we have no reason to think demand is high or low". A default of `0.00` would claim "nobody wants this product".

**Why `Decimal` for a score?** It is stored as exact `NUMERIC` in PostgreSQL, and a future pricing formula will multiply money by it — the Day 2 rule applies: no binary floats near money.

---

## Step 2 — ORM Model

**File:** `app/database/models/product.py`

```python
CheckConstraint("inventory >= 0", name="inventory_non_negative"),
CheckConstraint("demand_score >= 0 AND demand_score <= 1", name="demand_score_range"),

inventory: Mapped[int] = mapped_column(Integer, server_default="0")
demand_score: Mapped[Decimal] = mapped_column(Numeric(3, 2), server_default="0.50")
```

| Column | Type | Constraint | Default |
|---|---|---|---|
| `inventory` | `INTEGER NOT NULL` | `ck_products_inventory_non_negative`: `>= 0` | `0` |
| `demand_score` | `NUMERIC(3, 2) NOT NULL` | `ck_products_demand_score_range`: `0 … 1` | `0.50` |

**`server_default`, not `default`:** a Python-side `default=` is only applied by SQLAlchemy when *it* inserts a row. `server_default=` is part of the column definition in PostgreSQL itself — which is what fills the rows that **already exist** when the column is added.

---

## Step 3 — The Second Migration

```bash
alembic revision --autogenerate -m "add inventory and demand score to products"
```

Autogenerate produced:
```python
op.add_column('products', sa.Column('inventory', sa.Integer(), server_default='0', nullable=False))
op.add_column('products', sa.Column('demand_score', sa.Numeric(precision=3, scale=2), server_default='0.50', nullable=False))
```

**It missed both check constraints.** Alembic's autogenerate does not detect `CHECK` constraints at all. Applied as generated, the database would have accepted `inventory = -5` while the model claimed it could not. They were added by hand, with a matching `downgrade()`:

```python
def upgrade() -> None:
    op.add_column(...)   # inventory
    op.add_column(...)   # demand_score
    # Added by hand — Alembic autogenerate does not detect CHECK constraints
    op.create_check_constraint(op.f('ck_products_inventory_non_negative'), 'products', 'inventory >= 0')
    op.create_check_constraint(op.f('ck_products_demand_score_range'), 'products',
                               'demand_score >= 0 AND demand_score <= 1')

def downgrade() -> None:
    op.drop_constraint(op.f('ck_products_demand_score_range'), 'products', type_='check')
    op.drop_constraint(op.f('ck_products_inventory_non_negative'), 'products', type_='check')
    op.drop_column('products', 'demand_score')
    op.drop_column('products', 'inventory')
```

This is the Day 6 lesson made concrete: **autogenerate writes a draft; an engineer reviews it.** The naming convention from `base.py` is what makes `op.f('ck_products_...')` names predictable enough to write by hand.

**Why `NOT NULL` is safe on a table that already has rows:** adding a `NOT NULL` column with no default to a populated table fails — existing rows would have no value. With `server_default`, PostgreSQL fills every existing row as part of the same `ALTER TABLE`.

**Applied and verified:**
```bash
alembic upgrade head      # b2cda35e279c -> b7ea73182097
alembic downgrade -1      # columns and constraints removed, original 7 columns remain
alembic upgrade head      # back again
alembic check             # "No new upgrade operations detected." — models and database match
```

The three existing products received the defaults (`inventory=0`, `demand_score=0.50`), exactly as designed.

**`alembic check`** compares the ORM models with the live database and fails if a migration is missing — a cheap guard against "changed the model, forgot the migration". (It cannot see check constraints either, so it complements the review rather than replacing it.)

---

## Step 4 — Seeding Existing Rows

The seed uses `ON CONFLICT DO NOTHING`, so it would never write the new sample values into the three products that already existed. That is the right default — the seed must not clobber real edits — so overwriting is **opt-in**:

```bash
python -m app.database.seed            # insert missing products only (default, unchanged)
python -m app.database.seed --update   # also reset existing sample products to the seed values
```

```python
statement = insert(ProductORM).values(rows)
if overwrite:
    statement = statement.on_conflict_do_update(
        index_elements=[ProductORM.product_id],
        set_={column: statement.excluded[column] for column in _SEEDED_COLUMNS},
    )
else:
    statement = statement.on_conflict_do_nothing(index_elements=[ProductORM.product_id])
```

`excluded` is PostgreSQL's name for "the row you tried to insert". `--update` only ever touches the sample product IDs; it never deletes anything, and the surrogate `id` and `created_at` are preserved.

**Sample values:**

| SKU | Product | Inventory | Demand score | Story |
|---|---|---|---|---|
| `SKU-12345` | Wireless Headphones | 120 | 0.80 | Popular and well stocked |
| `SKU-67890` | Mechanical Keyboard | 35 | 0.55 | Steady seller, low stock |
| `SKU-11111` | Bluetooth Speaker | 300 | 0.20 | Overstocked and slow — a clearance candidate |

---

## Step 5 — Repository and API

**`PostgresProductRepository._to_domain()`** maps the two new columns. That is the *only* persistence code that changed.

**`PricingResponse`** now includes both fields (an additive change — existing clients are unaffected):
```json
{
  "product_id": "SKU-11111",
  "product_name": "Bluetooth Speaker",
  "cost_price": "900.00",
  "competitor_price": "400.00",
  "inventory": 300,
  "demand_score": "0.20",
  "suggested_price": "945.00",
  "strategy_used": "competitive",
  "margin_percent": "4.76",
  "guardrails_applied": [
    "PriceFloorGuard: price raised from 388.00 to 945.00 (minimum margin of 5% enforced)"
  ],
  "message": "Strategy 'competitive' applied. Suggested price is Rs.545.00 above competitor price."
}
```

**Unchanged:** `PricingService`, all four strategies, both guards, the router, the exception handlers, and every suggested price. No strategy reads the new fields yet — they are data, waiting for a strategy that needs them.

---

## Step 6 — Tests

| Test | Proves |
|---|---|
| `test_demand_signals_default_to_neutral` | Defaults are `0` and `0.50` |
| `test_product_rejects_invalid_demand_signals` | Domain rejects negative stock and scores outside 0..1 |
| `test_product_accepts_demand_score_boundaries` | `0.00` and `1.00` are valid |
| `test_demand_signals_round_trip` | A `Product` saved and reloaded from PostgreSQL is equal in every field |
| `test_seed_overwrite_resets_existing_rows` | Default seed never overwrites; `overwrite=True` does |
| `test_database_rejects_invalid_demand_signals` | Both hand-written check constraints fire, by name |
| `test_existing_rows_get_column_defaults` | A row inserted without the new columns gets `0` / `0.50` — what the migration gave existing rows |
| `test_demand_signals_are_served_from_postgres` | The fields reach the API response from the real database |

**Result: 129 tests passing**, 16 of them against PostgreSQL (without the database: 113 pass, 16 skipped), and zero test rows left in the database.

---

## Interview Questions — Day 6 Follow-up

**Q: How do you add a `NOT NULL` column to a table that already has millions of rows?**
Give it a server-side default so existing rows get a value in the same `ALTER TABLE`. (On very large tables you may instead add it nullable, backfill in batches, then add `NOT NULL` — to avoid long locks. PostgreSQL 11+ makes a constant default nearly instant, because it stores the default in the catalog instead of rewriting every row.)

**Q: What can Alembic autogenerate miss?**
`CHECK` constraints, some server default changes, column and table renames (seen as drop + add — which would *delete data*), and anything not declared in the models. That is why every generated migration is read before it is applied.

**Q: Why should a seed script not overwrite existing rows by default?**
Seeds run repeatedly — on every new laptop, in CI, after resets. If they overwrote by default, re-running one would silently undo real edits. Overwriting is a deliberate, opt-in action.

**Q: Why add columns nothing uses yet?**
Normally you would not (YAGNI). Here they are explicit inputs on the roadmap — demand-aware pricing and ML — and data only becomes useful once it has been collected for a while. The cost is one reviewed migration, and the domain model validates them from day one.

---

---

# Day 7 — Redis Integration, Cache-Aside Pattern & Decorator Pattern

**Goal:** Introduce a Redis-backed caching layer to reduce unnecessary database queries while keeping the business logic completely unchanged.

**Problem with Day 6:**

- Every pricing request queried PostgreSQL.
- Frequently requested products caused repeated, identical database reads.
- There was no caching mechanism.
- The business layer had no performance optimization.

The Repository Pattern already abstracted data access, but every request still became a database lookup — even when the same product was requested again a second later.

**Solution:**

- Redis (in Docker)
- A shared async Redis client
- `ProductCache` — the only code that talks to Redis
- The Cache-Aside pattern
- The Decorator pattern: `CachedProductRepository`
- Dependency Injection integration
- Graceful degradation when Redis fails

This adds a transparent caching layer **without modifying the business layer**. `PricingService`, the strategies, the guards, the router, and `PostgresProductRepository` were not changed.

---

## New File Structure

```
SmartPrice(BackendProject)/
├── docker-compose.yml          <- + Redis 7 service
└── app/
    ├── cache/                  <- NEW (the folder created empty on Day 1)
    │   ├── redis_client.py     <- get_redis_client(), ping_redis(), close_redis_client()
    │   └── product_cache.py    <- ProductCache: keys, JSON, TTL, invalidation, degradation
    ├── repositories/
    │   └── cached.py           <- NEW: CachedProductRepository (Decorator + Cache-Aside)
    ├── api/dependencies.py     <- wraps the PostgreSQL repository in the cache decorator
    ├── database/seed.py        <- --update now also invalidates cached products
    └── main.py                 <- Redis startup check, shutdown cleanup, /health "cache"
```

---

## Step 1 — Redis in Docker

**File:** `docker-compose.yml`

```yaml
redis:
  image: redis:7-alpine
  container_name: smartprice-redis
  command: ["redis-server", "--save", "", "--appendonly", "no",
            "--maxmemory", "64mb", "--maxmemory-policy", "allkeys-lru"]
  ports:
    - "${REDIS_PORT:-6379}:6379"
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
```

- **Redis 7 Alpine** — small image (Redis 7.4.12 at the time of writing).
- **Port 6379** — Redis's standard port; both containers share the Compose network.
- **Configured as a pure cache:**
  - `--save ""` and `--appendonly no` turn persistence off. A cache never needs to survive a restart — PostgreSQL has the real data, and an empty cache simply refills itself.
  - `--maxmemory 64mb --maxmemory-policy allkeys-lru`: when memory is full, Redis evicts the least-recently-used keys instead of rejecting writes. Popular products stay; rarely requested ones make room.

**Verified connectivity:**
```
docker exec smartprice-redis redis-cli PING
   ↓
PONG
```

---

## Step 2 — Redis Client

**File:** `app/cache/redis_client.py`

```python
@lru_cache
def get_redis_client() -> Redis:
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.redis_socket_timeout,          # 0.5 s
        socket_connect_timeout=settings.redis_socket_timeout,
        retry=Retry(NoBackoff(), retries=0),
    )
```

```
get_redis_client()
        │
        ▼
  Redis client  ──▶  connection pool  ──▶  Redis server
```

| Choice | Why |
|---|---|
| `redis.asyncio.Redis` | The repository interface is `async` (Day 5). A synchronous client would block the event loop on every cache lookup. |
| `@lru_cache` | Singleton — the client is created once and reused, like `get_settings()`. |
| Connection pooling | Built into the client: requests reuse open connections instead of reconnecting. Creating the client does not connect. |
| `decode_responses=True` | Redis returns `str` instead of `bytes`, so cached JSON can be parsed directly. |
| 0.5 s timeouts, no retries | **Fail fast.** redis-py retries failed connections by default. For a cache, waiting is worse than skipping it — a miss costs one fast PostgreSQL query. |

**Shutdown:** `close_redis_client()` closes the pool and clears the `lru_cache`, so the next application instance (or test) gets a fresh client bound to its own event loop.

---

## Step 3 — ProductCache

**File:** `app/cache/product_cache.py`

A dedicated class that owns **every** Redis interaction for products:

| Responsibility | Implementation |
|---|---|
| Key generation | `product:SKU-12345` |
| Serialization | `Product` → `asdict()` → JSON |
| Deserialization | JSON → `Decimal` conversion → `Product` |
| Expiration | `SET ... EX 300` (`CACHE_TTL_SECONDS`, which existed in `Settings` since Day 1) |
| Invalidation | `invalidate(product_id)` → `DEL product:...` |
| Graceful degradation | Redis errors are caught here; callers never see one |

### Serialization

```python
@staticmethod
def _serialize(product: Product) -> str:
    return json.dumps(asdict(product), default=str)
```

```
Product → asdict() → {"cost_price": Decimal("500.00"), ...} → JSON → Redis
```

`Decimal` is not JSON-serializable; `default=str` turns it into `"500.00"` — exact, and with its two decimal places intact. Converting to `float` would bring back the exact problem Day 2 eliminated.

What lands in Redis:
```json
{"product_id": "SKU-12345", "name": "Wireless Headphones", "cost_price": "500.00",
 "competitor_price": "999.00", "inventory": 120, "demand_score": "0.80"}
```

### Deserialization

```python
@staticmethod
def _deserialize(raw: str) -> Product:
    data = json.loads(raw)
    return Product(
        product_id=data["product_id"],
        name=data["name"],
        cost_price=Decimal(data["cost_price"]),
        competitor_price=Decimal(data["competitor_price"]),
        inventory=int(data["inventory"]),
        demand_score=Decimal(data["demand_score"]),
    )
```

```
Redis → JSON → Decimal conversion → Product
```

Restoring `Decimal` before building the domain model guarantees **type consistency**: a product from Redis is *equal* to the same product from PostgreSQL, so a strategy can never receive a string or a float for money. Building through `Product(...)` also re-runs the domain invariants (`__post_init__`) — a cached value that is somehow invalid is rejected exactly like a bad database row.

### Unreadable entries

If an entry cannot be read back — corrupt JSON, a missing field from an older shape of `Product`, an invalid value — `get()` treats it as a **miss** and deletes the key, so the next request rebuilds it from PostgreSQL. This matters on deploys: when `Product` gains a field (as it did in the Day 6 follow-up), old cache entries heal themselves instead of breaking requests.

---

## Step 4 — Decorator Pattern: CachedProductRepository

**File:** `app/repositories/cached.py`

```python
class CachedProductRepository(ProductRepository):
    def __init__(self, inner: ProductRepository, cache: ProductCache) -> None:
        self._inner = inner
        self._cache = cache

    async def get_by_id(self, product_id: str) -> Product:
        cached = await self._cache.get(product_id)
        if cached is not None:
            logger.info("Cache HIT | product_id=%s", product_id)
            return cached

        logger.info("Cache MISS | product_id=%s", product_id)
        product = await self._inner.get_by_id(product_id)   # may raise ProductNotFoundError

        if await self._cache.set(product):
            logger.info("Product cached | product_id=%s", product_id)
        return product
```

```
PricingService
        │
        ▼
ProductRepository              <- the service still only knows this
        │
        ▼
CachedProductRepository        <- IS a ProductRepository, and HAS a ProductRepository
   ┌────┴─────────┐
   ▼              ▼
ProductCache   PostgresProductRepository
   │              │
   ▼              ▼
 Redis        PostgreSQL
```

**What makes it a Decorator:** it implements the same interface as the object it wraps, and adds behaviour around the call — before (check the cache) and after (fill the cache). Callers cannot tell the difference.

**Why not put caching inside `PostgresProductRepository`?** Then the PostgreSQL class would have two jobs, every future repository would need its own copy of the caching code, and tests could not run the database code without Redis. The decorator wraps *any* `ProductRepository` — tests wrap an in-memory one.

### What is not cached: "not found"

`ProductNotFoundError` propagates untouched and nothing is stored. Two reasons:
- A product created a second later must be visible immediately, not after a cached "not found" expires.
- Otherwise anyone could fill Redis by requesting millions of made-up IDs.

---

## Step 5 — Cache-Aside Pattern

```
Request
   │
   ▼
Redis: GET product:SKU-12345
   │
Cache hit? ── Yes ──▶ return Product                         (no database query)
   │
   No
   │
   ▼
PostgreSQL: SELECT ... WHERE product_id = 'SKU-12345'
   │
   ▼
Redis: SET product:SKU-12345 <json> EX 300
   │
   ▼
return Product
```

- **Lazy:** products are loaded into Redis only after a miss. Products nobody requests are never cached.
- **PostgreSQL remains the single source of truth.** Redis only ever holds copies; losing all of Redis loses nothing.
- **TTL bounds staleness:** if a row changes in PostgreSQL without an explicit invalidation, the cached copy is at most 300 seconds old.

### Invalidation

```python
await cache.invalidate("SKU-12345")    # DEL product:SKU-12345
```

`python -m app.database.seed --update` rewrites the sample products in PostgreSQL, so it now invalidates their cache entries too. Verified: a deliberately stale `product:SKU-12345` key was gone after `--update`.

**The rule for the future:** any code that changes a product row must invalidate that product's cache entry *after* the database commit. (Invalidating *before* the commit leaves a window in which another request can re-cache the old row.)

---

## Step 6 — Dependency Injection

**File:** `app/api/dependencies.py`

```python
@lru_cache
def get_product_cache() -> ProductCache:
    return ProductCache(
        get_redis_client(),
        ttl_seconds=settings.cache_ttl_seconds,
        retry_seconds=settings.cache_retry_seconds,
    )

ProductCacheDep = Annotated[ProductCache, Depends(get_product_cache)]


def get_product_repository(session: DatabaseSessionDep, cache: ProductCacheDep) -> ProductRepository:
    return CachedProductRepository(inner=PostgresProductRepository(session), cache=cache)
```

The dependency graph now assembles:
```
Redis client (singleton)
        │
        ▼
ProductCache (singleton)          get_db() → AsyncSession (per request)
        │                                  │
        │                                  ▼
        │                       PostgresProductRepository
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
           CachedProductRepository
                       │
                       ▼
                PricingService      <- unchanged
```

**One function changed.** `get_product_repository()` went from `PostgresProductRepository(session)` to wrapping it. That is the Day 5 promise, kept for the second time.

**Why is `ProductCache` a singleton, when the repository is per request?** It holds no per-request state — only the shared Redis client and the circuit-breaker state (Step 8). If Redis goes down, *every* request should skip it, not each one rediscover the outage.

### Router

Routers still never create repositories or services — they only declare `service: PricingServiceDep`. (That refactoring happened on Day 5; Day 7 did not touch `routes/pricing.py`.)

```
Router → Depends() → PricingService
```

All object construction stays centralized in `api/dependencies.py`.

---

## Step 7 — Logging

The decorator logs every cache decision:
```
INFO | app.repositories.cached | Cache MISS | product_id=SKU-12345
INFO | app.repositories.cached | Product cached | product_id=SKU-12345
INFO | app.repositories.cached | Cache HIT | product_id=SKU-12345
```

The business layer never logs (or knows) anything about Redis.

**Bug found while verifying:** during a Redis outage the log still said `Product cached`, although nothing had been stored. `ProductCache.set()` now returns whether the write succeeded, and the decorator logs `Product cached` only when it really happened. A misleading log line is a bug: it sends whoever is debugging an outage in the wrong direction.

---

## Step 8 — Graceful Degradation

Redis failures never break a request:

```
Redis failure (connection refused, timeout, ...)
        │
        ▼
ProductCache logs a WARNING
        │
        ▼
get() returns None (a miss) · set()/invalidate() do nothing
        │
        ▼
CachedProductRepository falls through to PostgreSQL
        │
        ▼
HTTP 200 — same price, just without the speed-up
```

**Circuit breaker.** After one failure, `ProductCache` stops contacting Redis for `CACHE_RETRY_SECONDS` (30 s), then tries again:

```python
def _available(self) -> bool:
    return time.monotonic() >= self._skip_until

def _trip(self, exc: Exception) -> None:
    self._skip_until = time.monotonic() + self._retry_seconds
    logger.warning("Redis unavailable - serving from the database for %ds | ...")
```

Without it, *every* request during an outage would first wait for a Redis timeout, and the log would fill with one warning per request. With it: one warning, then no Redis calls at all until the window passes.

**`/health` reports Redis separately:**
```json
{ "status": "ok", "database": "ok", "cache": "unavailable" }
```
`status` stays `"ok"` — the application still works without Redis. Only the database (the source of truth) can make it `"degraded"`.

**Startup** logs `Redis connection OK - product cache enabled`, or a *warning* (not an error) when Redis is down: a missing cache is a performance problem, not an outage.

**Verified on the real server** (uvicorn, PostgreSQL and Redis in Docker):

| Step | Result | Time |
|---|---|---|
| 1. First request (cold cache) | 200, `969.03` — MISS, read from PostgreSQL, cached | ~29–36 ms |
| 2. Same request again | 200, `969.03` — HIT, no database query | ~4–5 ms |
| 3. `docker stop smartprice-redis`, request | 200, `969.03` — served from PostgreSQL | ~10–20 ms |
| 4. `/health` | `"status": "ok"`, `"cache": "unavailable"` | |
| 5. `docker start smartprice-redis`, wait for the retry window | MISS → cached → HIT again | |

(The first request is also slower because it opens the database and Redis connections; the important number is that repeat requests skip PostgreSQL entirely.)

---

## Step 9 — Tests

### Unit tests — no Redis, no database

`conftest.py` provides `FakeRedis`: an in-memory stand-in with `get`/`set`/`delete` that records TTLs and call counts, and can be switched to fail like a server that is down.

| File | Covers |
|---|---|
| `test_product_cache.py` | Key format; round trip restores exact `Decimal`/`int` types; money stored as `"500.00"`; TTL is set; invalidation; unreadable entries become misses and are deleted; Redis failure behaves like an empty cache; the circuit breaker skips Redis during the retry window and retries after it |
| `test_cached_repository.py` | Miss → inner repository → cached; hits never touch the inner repository (3 requests, 1 database call); not-found is raised and **not** cached; Redis down falls back; the MISS / cached / HIT log sequence; no false "Product cached" during an outage; a cached product prices exactly like one from the database |

### Integration tests — the real Redis (`@pytest.mark.redis`)

`test_redis_cache.py` checks the round trip, the real TTL (`0 < TTL <= 300`), real deletion, and that an unreachable Redis (`redis://127.0.0.1:1`) degrades to a miss. Every test deletes its keys afterwards — verified: zero `product:TEST-*` keys left.

End to end through the real application (`test_pricing_api.py`): starting from a cold key, two identical requests produce exactly `Cache MISS → Product cached → Cache HIT`, the key exists in Redis with a TTL, and both responses are identical.

### Architecture tests

- **Cache ignorance:** the Day 6 persistence rule now also forbids `redis` and `app.cache`. Only `repositories/postgres.py` and `repositories/cached.py` — the two adapters — may import them.
- **ASCII log messages (new):** verification uncovered that log messages containing `—` or `≥` crash on Windows when output is redirected (the console encoding is cp1252): Python prints `--- Logging error ---` and the line is lost. All such messages were changed to plain ASCII, and a test now parses every `logger.*()` call in `app/` and fails on non-ASCII text.

### Results

```
PostgreSQL + Redis running:   200 passed
Redis stopped:                194 passed, 6 skipped
```

With Redis stopped, the end-to-end PostgreSQL tests still pass — graceful degradation is proven by the suite, not just claimed.

---

## Day 7 — Request Flow

```
Client
    │
    ▼
PricingRequest  (Pydantic validation)
    │
    ▼
Router ── Depends() ──▶ PricingService(CachedProductRepository(PostgresProductRepository, ProductCache))
    │
    ▼
PricingService.calculate_price()
    │
    ▼
CachedProductRepository.get_by_id()
    │
    ▼
Redis
    │
    ├────────────────────┐
    │                    │
Cache hit           Cache miss / Redis down
    │                    │
    │                    ▼
    │               PostgreSQL
    │                    │
    │                    ▼
    │             Domain Product
    │                    │
    │                    ▼
    │          Cache Product (if Redis is up)
    │                    │
    └─────────┬──────────┘
              ▼
       Pricing Strategy
              │
              ▼
         Guard Chain
              │
              ▼
        PricingResponse
```

---

## Day 7 — Final Architecture

```
                    Client
                       │
                       ▼
                 FastAPI Router ◀──────── Global Exception Handlers
                       │
            (Dependency Injection)
                       │
                       ▼
               PricingService
                       │
                       ▼
        CachedProductRepository   (Decorator)
               │              │
               ▼              ▼
        ProductCache   PostgresProductRepository
               │              │
               ▼              ▼
            Redis        PostgreSQL
         (copies, TTL)  (source of truth)
```

---

## Day 7 — Design Principles Applied

### Decorator Pattern

Caching is added by wrapping an existing repository instead of modifying it.

```
          ProductRepository
                  ▲
       ┌──────────┴───────────┐
       │                      │
PostgresProductRepository   CachedProductRepository ──wraps──▶ any ProductRepository
```

The decorator transparently extends repository behaviour, and can wrap any implementation — PostgreSQL in production, an in-memory repository in tests.

### Cache-Aside Pattern

The application checks Redis first. Only cache misses query the database, and their results are cached for the next request. PostgreSQL stays the single source of truth.

### Dependency Inversion Principle (DIP)

`PricingService` continues to depend only on the `ProductRepository` abstraction. Introducing Redis required no change to the business logic.

### Single Responsibility Principle (SRP)

| Component | Its only responsibility |
|---|---|
| Router | HTTP |
| Dependency providers | Wiring the object graph |
| `PricingService` | Pricing orchestration |
| `CachedProductRepository` | Cache-aside orchestration |
| `ProductCache` | Redis operations: keys, JSON, TTL, invalidation, failures |
| `PostgresProductRepository` | Database access |
| Strategy | Price calculation |
| Guard chain | Business validation |

### Open/Closed Principle (OCP)

Caching was introduced by **adding** a class. Neither `PricingService` nor `PostgresProductRepository` changed. Any future repository can be wrapped by the same decorator.

### Separation of Concerns

- **API layer** → routers, schemas, dependency providers
- **Business layer** → `PricingService`, strategies, guards
- **Domain layer** → `Product`
- **Repository layer** → cached and PostgreSQL repositories
- **Caching layer** → `ProductCache`
- **Persistence layer** → PostgreSQL
- **Infrastructure** → Redis, Docker

### Persistence & Cache Ignorance

The domain model (`Product`) knows nothing about Redis, SQLAlchemy, PostgreSQL, FastAPI or Pydantic. Business logic is independent of both persistence and caching technology — enforced by `test_architecture.py`.

---

## Notes — Additions Beyond the Original Plan

1. **Async Redis client** (`redis.asyncio`) — the repository interface is async; a sync client would block the event loop.
2. **Fail-fast client + circuit breaker** — so a Redis outage costs one warning, not a timeout on every request.
3. **"Not found" is never cached** — new products appear immediately; no cache flooding with fake IDs.
4. **Self-healing cache entries** — unreadable or outdated JSON is treated as a miss and deleted.
5. **Seed invalidation** — `seed --update` clears the cache entries of the rows it rewrites.
6. **Redis configured as a cache** — no persistence, 64 MB limit, LRU eviction.
7. **`/health` reports the cache** without marking the app degraded.
8. **Two bugs fixed during verification** — a false `Product cached` log line during outages, and non-ASCII log messages that are silently lost on Windows.

---

## Interview Questions — Day 7

**Q: What is the Cache-Aside pattern?**
The application owns the caching logic: read the cache; on a miss, read the database and write the result to the cache. The cache never talks to the database itself. Alternatives are read-through (the cache loads data itself) and write-through (every database write also writes the cache).

**Q: What is the Decorator pattern, and why use it for caching?**
An object that implements an interface and wraps another object of the same interface, adding behaviour around its calls. Caching becomes a separate, removable, testable layer — the database repository stays focused on SQL, and the same decorator can wrap any repository.

**Q: How do you keep the cache consistent with the database?**
Three tools here: a TTL bounds how stale an entry can be; explicit invalidation after a write removes it immediately; and the database stays the source of truth, so a cache can always be thrown away. Invalidate *after* the commit, otherwise a concurrent request can re-cache the old value.

**Q: What should happen when Redis goes down?**
Nothing visible to the client. A cache is an optimization: catch the error, serve from the database, log it — and stop retrying on every request (a circuit breaker), or the outage turns into added latency for everyone.

**Q: Why not cache "product not found"?**
It would hide newly created products until the entry expired, and it lets anyone fill the cache with junk keys. Negative caching can be useful under heavy abuse, but then with a very short TTL.

**Q: Why store money in Redis as strings?**
JSON numbers are binary floats in most parsers. `"500.00"` converted back with `Decimal("500.00")` is exact; `500.0` is not guaranteed to be.

**Q: What is a cache stampede?**
When a popular key expires, many concurrent requests miss at the same moment and all hit the database. Mitigations: a lock so only one request rebuilds the entry, random jitter on TTLs, or refreshing popular keys shortly before they expire. Not needed at SmartPrice's current scale — but it is the next problem caching brings.

**Q: Why turn off Redis persistence?**
The data is a disposable copy. Persistence would cost disk I/O and could restore stale entries after a restart. An empty cache refills itself from PostgreSQL.

---

## Looking Ahead — Day 8

- Persist price quotes (a `price_quotes` table and migration) so `GET /api/v1/pricing/` returns real history instead of `[]`.
- Anything that updates a product row must call `ProductCache.invalidate()` after its commit.

---

---

# Day 8 — Real Sales Data (ML Phase A)

**Goal:** Give the ML work real data to learn from. Before this day the project had **no machine learning at all** — no sales history, no model, no training code. `app/ml/` was an empty folder and `demand_score` was a hand-typed number. Every ML phase that follows (forecasting, anomaly detection, MLflow, explainability) needs a demand history first.

**Decision: real data, not synthetic.** A synthetic generator is easy, but an interviewer can fairly say "your model just learned the formula you wrote". Real transactions bring real problems — closed days, bulk orders, duplicates, returns — and handling them is exactly what an ML engineer is asked about.

---

## The Dataset

**UCI Online Retail II** — https://archive.ics.uci.edu/dataset/502/online+retail+ii

Real transactions of a UK online gift wholesaler, **1 Dec 2009 – 9 Dec 2011**.

| | |
|---|---|
| Download | `online_retail_ii.zip`, 45,622,418 bytes (one `.xlsx` with two sheets) |
| SHA-256 (zip) | `572e3627...8e67bfb` |
| Raw rows | **1,067,371** transaction lines |
| Columns | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |

The raw zip lives in `data/raw/` and is **git-ignored** (45 MB, re-downloadable). The processed output (~410 KB) **is committed**: it is small, hashed, and it is what training and the database seed read.

---

## What Profiling the Raw Data Revealed

Profiling came before any pipeline code. Every finding became a rule:

| Finding | Count | Consequence |
|---|---|---|
| Exact duplicate rows | **34,335** | The two Excel sheets overlap on 1–9 Dec 2010 (45,046 rows in that window) → drop exact duplicates |
| Cancellation lines (invoice starts with `C`) | 19,494 | Not demand → drop |
| Quantity ≤ 0 (returns, adjustments) | 22,950 | Not demand → drop |
| Price ≤ 0 (free samples, write-offs) | 6,207 | Not demand → drop |
| Non-product codes (`POST`, `M`, `D`, `BANK CHARGES`, `AMAZONFEE`, `DCGS…`) | — | Postage, fees, discounts → keep only `^\d{5}[A-Za-z]?$` codes |
| Lower-case code variants (`85123a`) | — | Same product → upper-case all codes |
| **Trading days by weekday** | Mon 94, Tue 104, Wed 104, Thu 103, Fri 99, **Sat 1**, Sun 99 | The shop does not trade on Saturdays |
| Closed non-Saturday days | Christmas breaks, UK bank holidays | Zero demand on these days is *planned*, not a demand signal → calendar features in Phase 1 |
| Daily demand per product | median ~25–120, max **2,414 – 4,314** | Wholesale bulk orders → heavy tails; matters for metric choice and anomaly detection |

After cleaning: **1,002,004** rows; **604 trading days** out of **739 calendar days**.

---

## Step 1 — The Pipeline

**File:** `app/ml/data/prepare.py` — one command:
```bash
python -m app.ml.data.prepare
```

Five small pure functions, each unit-tested:

```
load_raw()             zip → both Excel sheets → one DataFrame
clean_transactions()   duplicates, cancellations, returns, free items, non-products out
select_top_products()  the 20 products sold on the most DISTINCT DAYS
build_daily_sales()    one row per product per CALENDAR day, zeros filled in
build_catalog()        product name, competitor price, cost (documented assumption)
```

### Why "most distinct days", not "most units"?

A product sold in one huge order has a series that is almost all zeros — there is nothing to learn from it. The chosen 20 sell on **559–603 of the 604 trading days**.

### Why fill in zero-sale days?

```
Without zero rows:  2010-01-04, 2010-01-06, 2010-01-07 ...   "lag 1" of the 6th = the 4th  ✗
With zero rows:     2010-01-04, 2010-01-05 (0), 2010-01-06   "lag 1" of the 6th = the 5th  ✓
```

If missing days are left out, every lag and rolling window silently means "N rows ago" instead of "N days ago". The model would learn from a distorted calendar.

### Price on zero-sale days

`avg_price` is the **quantity-weighted** average price actually paid that day: (2 × 1.00 + 2 × 2.00) / 4 = 1.50. On a zero-sale day nobody paid anything, so the price is **left empty**, not invented. The Phase 1 feature pipeline decides how to fill it, using only past data.

### Documented assumptions

The dataset has no costs and no competitor prices, which the pricing engine needs:

| Field | Rule | Why |
|---|---|---|
| `competitor_price` | the product's **median** selling price | a robust "market price" proxy (median ignores bulk-discount outliers) |
| `cost_price` | **60%** of that median (`COST_RATIO`) | a typical ~40% wholesale gross margin |
| `inventory` | not set (default 0) | the dataset has no stock levels — and inventing them would leak information into the model |

These are labelled as assumptions in the code, in `metadata.json`, and here.

### Output

| File | Content |
|---|---|
| `data/processed/daily_sales.csv` | **14,780 rows** = 20 products × 739 days; columns `product_id, sale_date, units_sold, orders, avg_price` |
| `data/processed/products.csv` | 20 products — e.g. `85123A, White Hanging Heart T-Light Holder, 1.77, 2.95` |
| `data/processed/metadata.json` | source, row counts, date range, **SHA-256 of the raw zip and of `daily_sales.csv`** |

**3,295** of the 14,780 product-days have zero sales (22%). That is why Phase 1 uses **WAPE** rather than MAPE: MAPE divides by the actual value, which is undefined when demand is 0.

The `daily_sales.csv` hash (`56fdd5eb…`) is the **dataset version**: Phase 3 logs it with every training run, so any model can be traced back to the exact data it learned from.

---

## Step 2 — The `sales_history` Table

**File:** `app/database/models/sales_history.py` · **Migration:** `create sales history table`

| Column | Type | Constraint |
|---|---|---|
| `id` | `INTEGER` | primary key |
| `product_id` | `VARCHAR(64)` | **foreign key → `products.product_id`**, `ON DELETE CASCADE` |
| `sale_date` | `DATE` | **unique with `product_id`** (`uq_sales_history_product_id`) |
| `units_sold` | `INTEGER` | `>= 0` — the forecasting target |
| `orders` | `INTEGER` | `>= 0` — distinct invoices that day |
| `avg_price` | `NUMERIC(12, 4)` | `NULL` or `> 0` — `NULL` exactly on zero-sale days |

- The **unique (product_id, sale_date)** constraint makes duplicates impossible and is the index every "history of one product" query uses.
- The **foreign key** means a sale can never point at a product that does not exist.
- Autogenerate first named the unique constraint `product_day` (an explicit name bypasses the naming convention). The model was changed so the convention names it — consistent with every other constraint.
- Applied, rolled back, re-applied; `alembic check` reports no drift.

---

## Step 3 — Seeding the Real Data

`python -m app.database.seed` now loads, in order:
1. the 3 hand-written demo products (`SKU-12345`, …) — kept so every earlier DEVLOG scenario still works;
2. the **20 real products** from `products.csv`;
3. their **14,780 daily sales rows** (products first — the foreign key requires it).

```
First run:   products inserted=20 | already present=3     Sales history | rows inserted=14780
Second run:  products inserted=0  | already present=23    Sales history | rows inserted=0
```

Still idempotent (`ON CONFLICT DO NOTHING` on the unique constraint). Rows are inserted in **batches of 5,000**: asyncpg allows at most 32,767 bind parameters per statement, and 14,780 rows × 5 columns would exceed it.

If `data/processed/` is missing, the seed warns and loads only the demo products — a fresh deploy never fails because of it.

Verified in PostgreSQL: 14,780 rows, 20 products, 2009-12-01 → 2011-12-09, 3,295 zero days, 3,295 NULL prices.

---

## Step 4 — Configuration

Paths are settings, not hard-coded strings:

```python
raw_data_path: str = "data/raw/online_retail_ii.zip"
processed_data_dir: str = "data/processed"

def resolve_path(path: str) -> Path:
    """Absolute paths are kept; relative ones are taken from the project root."""
```

`resolve_path()` makes commands work from any working directory.

**`requirements.txt` is now pinned** (`fastapi==0.141.1`, `pandas==3.0.5`, …): the same environment on every machine and every deploy. `openpyxl==3.1.5` was added to read the `.xlsx`.

---

## Step 5 — Tests

| File | Covers |
|---|---|
| `test_data_prepare.py` | Each cleaning rule on a tiny hand-built frame: cancellations, returns, free items, postage/fee/gift-set codes, overlap duplicates, lower-case codes; "most days" beats "most units"; deterministic tie-break; one row per calendar day with zeros; quantity-weighted price; no price on zero days; the 60% cost rule and name cleaning |
| `test_processed_data.py` | The committed files: hash matches `metadata.json`; every product has every calendar day; no negative demand; price present exactly on sale days; catalog matches sales |
| `test_sales_history_seed.py` | Idempotent seeding; foreign key rejects unknown products; check constraint rejects negative demand; the full 14,780-row dataset loads (all rolled back) |
| `test_architecture.py` | `app/ml/` added to the business layer: it may not import FastAPI, the API layer, SQLAlchemy, Redis or the cache |

The `db_session` fixture (a transaction rolled back after every test) moved to `conftest.py` so all database tests share it.

**Result: 234 tests passing**, zero test rows left in the database.

---

## Interview Questions — Day 8

**Q: Why did you choose this dataset?**
It is real, public, and has what demand forecasting needs: dates, quantities and prices per product over two years — enough for weekly seasonality and a held-out test period. Real data also forced real cleaning decisions.

**Q: Why remove duplicates? Couldn't two identical lines be two real purchases?**
In general, yes. Here, profiling showed the duplicates concentrate in 1–9 Dec 2010, where the two sheets overlap — an export artefact, not customer behaviour.

**Q: Why fill missing days with zeros instead of dropping them?**
Lag and rolling features count rows. Without the zero rows, "7 rows ago" stops meaning "7 days ago". Zeros are also true information: nothing was sold.

**Q: Why WAPE and not MAPE?**
22% of product-days have zero sales. MAPE divides each error by the actual value — undefined at zero, and exploding near it. WAPE = Σ|error| / Σ|actual| divides once, by total demand.

**Q: The dataset has no costs. Isn't the pricing part made up?**
The costs and competitor prices are documented assumptions (median price; cost = 60% of it), clearly labelled. The demand data — what the ML learns from — is entirely real.

**Q: How would you know which data a model was trained on?**
The SHA-256 of `daily_sales.csv` is recorded in `metadata.json` and will be logged with every training run (Phase 3).

---

---

# Day 9 — Demand Forecasting & Honest Evaluation (ML Phase 1)

**Goal:** Build a next-day demand forecast for each product and answer the two questions every interviewer asks: **"Why this model?"** and **"How do you know it works?"**

The answer has to come from a protocol that cannot fool itself: leakage-safe features, evaluation in time order, baselines to beat, and a test set touched only once.

```bash
python -m app.ml.forecasting.evaluate     # ~6 seconds; writes reports/forecast_comparison.md + plot
```

---

## New Files

```
app/ml/
├── metrics.py                  MAE, RMSE, WAPE — written by hand
├── features.py                 leakage-safe feature table
└── forecasting/
    ├── models.py               5 candidates with the same fit/predict interface
    └── evaluate.py             splits, walk-forward validation, test scoring, reports
reports/
├── forecast_comparison.md      generated results table
└── forecast_actual_vs_predicted.png
tests/
├── test_metrics.py             metrics against hand-worked examples
├── test_features.py            leakage tests + feature formulas
└── test_forecast_evaluation.py split/fold properties, selection rule
```

---

## Step 1 — The Problem

**Target:** `units_sold` for one product on day *t*.
**Forecast made:** at the end of day *t − 1* (next-day forecast).
**One global model** for all 20 products, with `product_id` as a feature: each product alone has ~700 days, but together the model sees 14,220 rows and can learn shared patterns (weekly cycle, Christmas) once.

---

## Step 2 — Metrics

**File:** `app/ml/metrics.py`

| Metric | Formula | Reads as |
|---|---|---|
| MAE | mean(\|actual − predicted\|) | average miss, in units |
| RMSE | √mean((actual − predicted)²) | like MAE, but big misses count more |
| **WAPE** | Σ\|actual − predicted\| / Σ\|actual\| | total miss as a share of total demand |

**WAPE, not MAPE:** 22% of product-days have zero sales. MAPE divides every row by its own actual value — division by zero on those days, and explosions on 1-unit days. WAPE divides once, by total demand. **WAPE is the selection metric.**

Unit tests check each one against hand-worked numbers, e.g. actual `[1, 2, 3]`, predicted `[2, 2, 1]`: errors 1, 0, 2 → MAE 1.0, RMSE √(5/3), WAPE 3/6 = 0.5.

---

## Step 3 — Leakage-Safe Features

**File:** `app/ml/features.py`

**The rule:** the row for day *t* may only use what is known at the **end of day t − 1**.

| Group | Features | Leakage guard |
|---|---|---|
| Lags | `lag_1`, `lag_7`, `lag_14` | `.shift(k)` per product |
| Rolling stats | `roll_mean_7`, `roll_std_7`, `roll_mean_28`, `roll_std_28` | computed on `.shift(1)` — window ends **yesterday** |
| Calendar | `day_of_week`, `month`, `is_holiday` (English bank holidays), `is_christmas_break` (24 Dec – 3 Jan) | known in advance |
| Price | `last_price`, `price_vs_28d` | last price paid **before** day *t*, carried forward over zero-sale days |
| Category | `product_id` | — |

```python
units.rolling(7).mean()           # window = t-6 .. t    ← includes today's answer!
units.shift(1).rolling(7).mean()  # window = t-7 .. t-1  ← past only
```

- **All computed per product** (`groupby`), so one product's history never enters another's features.
- **Why not today's `avg_price`?** It is an *outcome* of day *t* — who bought, at what bulk discount. Only the last price paid before *t* is known in advance.
- **Why `is_christmas_break`?** The shop is shut 24 Dec – 3 Jan, but not every one of those days is a bank holiday (e.g. 30 Dec). The `holidays` package calendar matched every closure found in the Phase A profiling (Good Friday, Easter Monday, May Day, Spring and Late Summer bank holidays).
- **Not used — inventory:** the dataset has no stock history; filling it from today's sales would leak the answer.
- The first 28 days of each product are dropped (no full 28-day window yet): **14,220** rows remain.

### How leakage is tested

The strongest test does not re-check formulas — it **changes the future and asserts the past does not move**:

```python
future_changed.loc[is_future, "units_sold"] = 99_999
future_changed.loc[is_future, "avg_price"] = 500.0
pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])   # features up to the cutoff are identical
```

Plus: today's own sales and price never change today's features; changing product B never changes product A; lags and rolling windows match hand calculations.

**The tests were proven to catch a real leak:** in a scratch copy, `shift(1)` was changed to `shift(0)` for the rolling windows — the most common leakage bug — and **6 tests failed**.

---

## Step 4 — The Candidates

**File:** `app/ml/forecasting/models.py` — all share `fit(X, y)` / `predict(X)`.

| # | Model | Idea |
|---|---|---|
| 1 | Naive | tomorrow = today (`lag_1`) |
| 2 | Seasonal naive | tomorrow = same weekday last week (`lag_7`) |
| 3 | Moving average | tomorrow = mean of the last 7 days |
| 4 | **Ridge regression** | linear model; one-hot product/weekday/month, standardised numbers, `alpha=1.0` |
| 5 | **LightGBM** | gradient-boosted trees, 400 trees, learning rate 0.05, `random_state=42` |

**Why baselines?** A model is only worth deploying if it beats a rule that costs nothing.

**Why LightGBM over XGBoost?** Native categorical support for `product_id` (no one-hot), fast on CPU, and it can return per-feature SHAP contributions itself (`pred_contrib=True`) — which lets the API explain predictions without the heavy `shap` package (Phase 4).

**Hyperparameters are common defaults, not tuned** — and never on the test set.

**Predictions are clipped at 0** — nobody sells −3 units.

### Decision: train on log(1 + units)

Demand is heavy-tailed (median ~30, bulk orders above 2,000). With squared-error training on raw units, those rare spikes dominate the fit. Both learned models are trained on `log1p(units)` and mapped back with `expm1`. `log1p`, not `log`, because demand can be 0.

**Checked on the validation folds only (never the test set):**

| Model | Target | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| LightGBM | raw units | 62.07 | 137.13 | 1.129 |
| LightGBM | **log1p** | **42.45** | 129.83 | **0.767** |
| Ridge | raw units | 52.50 | **127.64** | 0.955 |
| Ridge | **log1p** | **42.43** | 132.75 | **0.763** |

The log target cuts WAPE by 20–32%. The honest trade-off: Ridge's RMSE gets slightly *worse*. Expected — a log target aims at the typical day, not the spike-inflated average, and RMSE is the metric most sensitive to spikes. Since WAPE is the business metric, log1p stays.

---

## Step 5 — The Evaluation Protocol

**File:** `app/ml/forecasting/evaluate.py`

```
                              development period                         |   test period
  2009-12-29 ........................................................ 2011-09-09 | 2011-09-10 .. 2011-12-09
  [ fold 1 train ][val]                                                 |
  [ fold 2 train      ][val]                                            |   91 days,
  [ fold 3 train           ][val]       walk-forward validation         |   scored ONCE,
  [ fold 4 train                ][val]  (model selection only here)     |   after selection
  [ fold 5 train                     ][val]                             |
```

1. **Chronological hold-out:** the last **91 days** (10 Sep – 9 Dec 2011, 1,820 rows) are the test set.
2. **Walk-forward validation** with `TimeSeriesSplit` (5 folds × 60 days) on the development period (12,400 rows). Each fold trains on everything *before* its window.
3. **Selection:** lowest mean validation WAPE.
4. **Then** every candidate is refit on the full development period and scored on the test set **once**.

### Why not a random split?

A random split puts December 2011 in training and November 2011 in test: the model "predicts" November having already seen what came after it. That information does not exist at forecast time, so the score is optimistic and will not hold in production. Splitting by date keeps every evaluation in the real direction: **past → future**.

**Folds split on dates, not rows:** all 20 products of a day land on the same side — the test asserts every validation date has all 20 products.

---

## Step 6 — Results (real runs)

### Walk-forward validation — used to choose

| Model | MAE | RMSE | WAPE (mean ± std over 5 folds) |
|---|---:|---:|---:|
| Naive (yesterday) | 68.55 | 182.14 | 1.238 ± 0.053 |
| Seasonal naive (same weekday last week) | 64.59 | 181.78 | 1.164 ± 0.029 |
| Moving average (last 7 days) | 58.04 | 138.95 | 1.049 ± 0.045 |
| **Ridge regression (selected)** | 42.43 | 132.75 | **0.763 ± 0.029** |
| LightGBM | 42.45 | 129.83 | 0.767 ± 0.033 |

### Test set — held out until the end

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive (yesterday) | 84.54 | 208.15 | 1.177 |
| Seasonal naive (same weekday last week) | 71.09 | 182.88 | 0.990 |
| Moving average (last 7 days) | 66.65 | 154.80 | 0.928 |
| **Ridge regression (selected)** | 50.68 | 151.42 | **0.706** |
| LightGBM | 50.59 | 151.77 | 0.705 |

**Selected: Ridge regression.** Test WAPE **0.706** vs **0.990** for seasonal naive — **28.7% lower error**. MAE falls from 71.09 to 50.68 units per product-day.

### Reading the results honestly

- **Ridge and LightGBM tie.** Their validation WAPE differs by 0.004 — far inside one standard deviation (~0.03). The fixed rule picked Ridge; with a tie, the simpler, faster, fully interpretable model is the better engineering choice anyway. Trees did not find interactions worth their complexity on these features.
- **The test agrees with validation:** the ranking is the same, and the selected model's test WAPE (0.706) is close to its validation WAPE (0.763). No sign the selection overfitted.
- **WAPE ~0.7 is still high — and the plot shows why.** This is a wholesaler: on a typical day a product sells 25–120 units, then a single bulk order brings 1,000–3,000. No history-based feature can know a bulk order is coming. Ridge tracks the underlying level and the Saturday/holiday zeros; it does not chase spikes.
- **The seasonal-naive failure is visible in the plot:** it repeats every spike exactly 7 days late — double the error, once for missing the spike and once for predicting a phantom one.
- **Unpredictable spikes are the motivation for Phase 2:** if they cannot be forecast, they must at least be *detected*, so pricing does not overreact to them.

![Actual vs predicted](reports/forecast_actual_vs_predicted.png)

---

## Step 7 — Tests

| File | Covers |
|---|---|
| `test_metrics.py` | MAE / RMSE / WAPE on hand-worked examples; RMSE punishes one big miss more than MAE; WAPE with zero-demand days; WAPE undefined when nothing sold; invalid input rejected |
| `test_features.py` | Changing the future never changes past features (3 cutoffs); today's own sales and price are never features; no cross-product leakage; lags, rolling windows and last price by hand; calendar flags; warm-up rows dropped |
| `test_forecast_evaluation.py` | Test set = last 91 days, no overlap; every fold trains strictly on the past, splits whole days, expands forward; predictions never negative; **selected model = lowest mean validation WAPE** regardless of test results; report generated |

**Result: 275 tests passing.**

`requirements.txt` gained `lightgbm==4.7.0`, `holidays==0.108`, `matplotlib==3.11.2`.

---

## Interview Questions — Day 9

**Q: Why didn't you use a random train/test split?**
Time-series rows are ordered. A random split trains on the future to predict the past, which inflates scores. I hold out the last 91 days and validate walk-forward, always training on the past only.

**Q: How do you know your features don't leak?**
Every history feature is built from `.shift(1)` per product, and a test changes all future values and asserts past features are byte-for-byte identical. I verified the test catches the classic bug — removing the shift made 6 tests fail.

**Q: Why did a linear model win?**
It didn't really "win" — it tied with LightGBM within the fold-to-fold noise, and my rule selects by mean validation WAPE. A tie favours the simpler model: faster, easier to explain, fewer ways to fail.

**Q: Your WAPE is 0.71. Is that good?**
It is 29% better than the best naive baseline on unseen data. In absolute terms it is high because a wholesaler's demand is dominated by unpredictable bulk orders. Better accuracy would need information about those orders (customer pipelines, quotes), not a fancier model.

**Q: Why log-transform the target?**
Heavy tails. Squared error on raw units lets rare 3,000-unit days dominate. On log scale they matter proportionally. I checked it on the validation folds: WAPE dropped from 0.955 to 0.763 for Ridge, 1.129 to 0.767 for LightGBM.

**Q: Did you tune on the test set?**
No. Selection uses only the walk-forward folds; the test set was scored once at the end, and the test asserts the selection does not depend on it.

---

---

# Day 10 — Anomaly Detection & Labelled Evaluation (ML Phase 2)

**Goal:** Detect unusual demand days — and *measure* how well detection works, with precision, recall and F1.

Phase 1 showed that one-off bulk orders cannot be forecast. If they cannot be predicted, the system should at least **notice** unusual days, so pricing does not overreact to them and broken data feeds are caught.

```bash
python -m app.ml.anomaly.evaluate     # ~4 seconds; writes reports/anomaly_comparison.md + plot
```

---

## New Files

```
app/ml/anomaly/
├── calendar.py       which days the shop is open (closed days are never "anomalies")
├── inject.py         plants labelled spikes, drops and flatlines into real data
├── detectors.py      rolling z-score, IQR fences, Isolation Forest, z-score + repeated-value rule
└── evaluate.py       thresholds on validation, scored once on test, report + plot
reports/
├── anomaly_comparison.md
└── anomaly_example.png
tests/test_anomaly_detection.py
```

---

## Step 1 — The Labelling Problem

The dataset has **no anomaly labels**. Without labels there is no precision or recall — only opinions.

**Standard fix: inject known anomalies into real held-out data.** We know exactly where and what they are, so every detector can be scored.

| Type | What is planted | Real-world cause it imitates |
|---|---|---|
| **Spike** | demand = 4–8 × the product's normal level | pricing error, bot orders, a duplicated import |
| **Drop** | demand = 0 on a normally busy open day (normal ≥ 10 units) | checkout outage, a missing data load |
| **Flatline** | the first day's value repeated for 4 more days | a stuck feed re-sending yesterday's number |

- "Normal level" = median of the product's previous 28 **open** days.
- Injection is **seeded** — the labelled set is identical on every run.
- Flatline: the first day is genuine; only the 4 repeats are labelled.

**Honest limitation:** the real data already contains natural extremes (bulk orders) with no label. Flagging one counts as a false positive, so **reported precision is a lower bound**.

### Only open days are judged

The shop never trades on Saturdays and closes on bank holidays and 24 Dec – 3 Jan. A zero on those days is *planned*. `calendar.py` builds an open-day mask from the calendar alone (no sales data), and every detector only judges open days.

---

## Step 2 — The Detectors

**File:** `app/ml/anomaly/detectors.py`. Each returns a **score** (higher = more unusual) for a day that has just happened.

**Shared reference:** the product's previous 28 **open** days, on a **log1p scale**:
- previous days only — an anomaly must not inflate its own baseline;
- open days only — planned zeros are not "normal demand";
- log scale — demand is heavy-tailed; 10 → 40 units should count like 100 → 400.

| Detector | Score | Classic rule of thumb |
|---|---|---|
| **Rolling z-score** | \|today − mean₂₈\| / std₂₈ | flag if \|z\| > 3 |
| **IQR fences** (Tukey) | distance outside [Q1, Q3] of the last 28 days, in IQRs | flag if > 1.5 IQR |
| **Isolation Forest** | how *few* random splits isolate the day (6 features) | — |
| **z-score + repeated-value rule** | \|z\|, or "flag" if the last 4 open days are identical | — |

**Why IQR as well as z-score?** Medians and quartiles ignore extreme values, so one past bulk order does not widen the "normal" range the way it inflates a standard deviation.

**How Isolation Forest works:** random trees split on random features at random values. An unusual point sits alone in an empty region and is isolated in **few** splits; a normal point needs many. Short average path length = anomaly. It is **unsupervised** — trained on the history before each window, it never sees a label. Features: today's log demand, `z_28`, change vs the same weekday last week, change vs the last-7-day mean, `flat_range_4` (max − min over the last 4 open days), weekday. `n_estimators=200`, `random_state=42`.

---

## Step 3 — The Protocol

```
 ... history ... | validation (91 days)          | test (last 91 days)
                 | anomalies injected, seed 1     | anomalies injected, different seed
                 | -> choose each threshold here  | -> precision / recall / F1, once
```

1. Inject labelled anomalies into the validation window (one copy of the data) and, separately, the test window (another copy, another seed).
2. Isolation Forest is trained on the history **before** each window.
3. **Threshold = the score cut-off with the best F1 on validation**, then frozen.
4. The frozen threshold is applied to the test window, once.

### Why choose the threshold by F1?

```
lower threshold  → more flags → higher recall, lower precision  (alert fatigue)
higher threshold → fewer flags → higher precision, lower recall (missed problems)
```

F1 — the harmonic mean of precision and recall — picks the balance point when both kinds of error matter equally. Choosing it on validation and freezing it before the test is what keeps the numbers honest. The chosen thresholds land near the textbook values: **\|z\| ≥ 2.02** (rule of thumb: 3) and **1.54 IQR** (rule of thumb: 1.5).

---

## Step 4 — What Actually Happened (the honest version)

### First run

Three detectors. Two numbers did not make sense: **flatline recall 0.00 for every detector**, and **spike recall only ~0.5** for 4–8× spikes.

Seeing those test numbers once was unavoidable. Everything after this was diagnosed on the **validation window only**:

| Finding (validation) | Value |
|---|---|
| Median \|z\| of injected spikes | **1.71** |
| \|z\| of normal days: 95th / 99th percentile | 2.10 / 2.89 |
| Injected flatline rows where the last 4 open days are identical | 7 of 15 |
| Normal rows where that happens | **0 of 1,465** |
| Isolation Forest score: flatline median vs. normal 95th percentile | 0.484 vs 0.573 |

**Spikes are hard because this business is noisy.** A wholesaler's normal day-to-day spread on a log scale is so wide that a 4–8× day is often within normal range — a normal bulk order looks just like it. The spike definition was fixed *before* any results, so it was **kept** — making spikes bigger to get better numbers would be cheating.

**Flatlines have a near-perfect signal that no detector used.** "4 identical open days" happens on half the flatline rows and on zero normal rows. But a stuck feed repeats a *normal-looking* number, so a "how far from normal is today?" score cannot see it, and Isolation Forest treats it as one weak feature among six.

### The fix — and test-set hygiene

A fourth detector: **rolling z-score + a repeated-value rule** — flag if |z| ≥ threshold **or** the last 4 open days are identical. Justified by validation evidence alone.

Because the first test draw (seed 2) had been seen, **the final numbers use a fresh draw of test anomalies (seed 3)** that the final design was never tuned on. The report states this.

### A bug caught by a unit test

The first version of the rule checked "rolling standard deviation < 1e-9". A unit test with a stuck value failed: pandas computes rolling std with running sums, and identical values produced **2.8 × 10⁻⁸**, not 0. The rule was silently missing some flatlines. Fix: check the **range (max − min) of integer units**, which is exactly 0. Isolation Forest uses the same corrected feature — which changed its results (F1 0.454 → 0.400): a reminder that model results depend on small feature details.

---

## Step 5 — Results (test window, fresh seed 3)

Validation: 2011-06-11 – 2011-09-09, 1,540 open product-days, injected 30 spikes, 30 drops, 15 flatline days.
Test: 2011-09-10 – 2011-12-09, 1,560 open product-days, injected 30 spikes, 30 drops, 15 flatline days.

| Detector | Threshold | Flags | Precision | Recall | F1 | Spike recall | Drop recall | Flatline recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rolling z-score | 2.019 | 107 | 0.364 | 0.520 | 0.429 | 0.43 | 0.77 | 0.20 |
| IQR fences | 1.539 | 101 | 0.317 | 0.427 | 0.364 | 0.27 | 0.67 | 0.27 |
| Isolation Forest | 0.556 | 135 | 0.311 | 0.560 | 0.400 | **0.53** | **0.80** | 0.13 |
| **z-score + repeated-value rule** | 2.019 | 114 | **0.395** | **0.600** | **0.476** | 0.43 | 0.77 | **0.60** |

### Reading the results

- **Best overall: z-score + repeated-value rule** (F1 0.476). A one-line rule tripled flatline recall (0.20 → 0.60) at the cost of 7 extra flags.
- **Isolation Forest is best at spikes and drops**, but not by enough to win overall — it needs more flags (135) to get there, and it misses flatlines.
- **IQR is worst** here: its robust fences adapt *less* to this data's genuine volatility, so it flags normal busy days and misses real spikes.
- **Precision ~0.3–0.4 is a lower bound.** The example plot shows the best detector flagging a real 1,270-unit bulk order on 3 Nov 2011 — a genuine extreme with no label, counted as a false positive.
- **Drops are the easiest anomaly** (77–80% recall): a zero on a busy day is far from normal on a log scale.
- **The flatline rule fires only from the 4th identical day.** Detecting "nothing changed" always needs time to pass — detection delay is inherent to this anomaly type.

![Example](reports/anomaly_example.png)

---

## Step 6 — Tests

`tests/test_anomaly_detection.py` — 15 tests:

| Area | Covers |
|---|---|
| Calendar | Saturday, Good Friday and the Christmas break are closed; an ordinary Monday is open |
| Injection | Exact counts per type; all inside the window; no duplicates; spikes ≥ 3× the original; drops are zeros on busy days; flatlines repeat one value; seeded = reproducible (and different seeds differ); untouched rows unchanged |
| Detectors | Only open days judged; the reference window never uses the future; z-score and IQR score a planted spike and drop above the 99th percentile of normal days; Isolation Forest scores an extreme day above the 99th percentile; the repeated-value rule catches a stuck feed that a plain z-score scores below 2 |
| Thresholds | Picks the F1-maximising cut-off on a hand-worked example; separates perfectly separable scores |
| Protocol | Validation precedes test and uses a different seed; the full evaluation reports all four detectors with valid metrics |

**Result: 305 tests passing.**

---

## Interview Questions — Day 10

**Q: How did you evaluate anomaly detection without labels?**
I injected labelled anomalies — spikes, drops, flatlines — into real held-out data, with a fixed seed. That gives ground truth for precision, recall and F1. The caveat: real unlabelled extremes count as false positives, so precision is a lower bound.

**Q: How did you choose the threshold?**
On a separate validation window with its own injected anomalies, I picked the cut-off that maximised F1, then froze it before scoring the test window. The chosen |z| threshold, 2.02, is close to the classic 2–3 range.

**Q: Why didn't Isolation Forest win?**
It was best at spikes and drops, but it treats "the value hasn't changed" as one weak feature among six, so it barely catches flatlines. A simple explicit rule captured that pattern far better. More complex is not automatically better — the evaluation shows where each method is strong.

**Q: You changed the detector after seeing test results. Isn't that cheating?**
I saw one test draw, then diagnosed only on validation, and re-drew a fresh set of test anomalies for the final numbers. It is documented in the report.

**Q: Why are spikes only ~50% detected?**
Because in wholesale data a 4–8× day is often normal — bulk orders. Statistically, the injected spikes overlap with normal variation (median |z| 1.71 vs. a 95th percentile of 2.10 on normal days). Making the spikes bigger would have improved the score but not the system.

**Q: How did a unit test change your results?**
Rolling standard deviation of identical values came out as 2.8e-8, not 0, so "std < 1e-9" missed flatlines. Switching to an exact max − min check fixed it — floating-point equality is a classic trap.

---

---

# Day 11 — Experiment Tracking, Versioned Models & Model Loading (ML Phase 3)

**Goal:** Make training reproducible and traceable, and connect the trained model to the API.

- **One command** trains, evaluates, versions and tracks a model.
- **MLflow** records every run: parameters, metrics, the feature list, the dataset hash, reports and the model.
- The API loads **the model version named in config** — never a hard-coded path.

```bash
python -m app.ml.train                                   # ~10 s -> models/v1, v2, ...
mlflow ui --backend-store-uri sqlite:///mlflow.db        # browse every run
```

---

## New Files

```
app/ml/
├── model_store.py        save / load versioned models (imported by the API — no training libraries)
└── train.py              the one-command pipeline (MLflow, evaluation, final fit, versioning)
app/api/routes/model.py   GET /api/v1/models/current
app/schemas/model.py      ModelInfoResponse
models/v1/                model.joblib (5 KB) + metadata.json   <- committed: what the API serves
requirements-dev.txt      training, evaluation, tracking and test tools
tests/test_model_store.py, test_model_api.py, test_training.py
```

---

## Step 1 — MLflow, Without a Server

**Requirement:** a local, file-based tracking store — no tracking server.

**What happened:** MLflow 3.18 now *refuses* the classic `./mlruns` folder store (`"The filesystem tracking backend is in maintenance mode"`), unless an opt-out environment variable is set. Its recommended replacement is still a **single local file with no server**: a SQLite database.

```
sqlite:///mlflow.db      <- runs, parameters, metrics, tags (one file)
mlartifacts/             <- artifacts: reports, plots, model files
```

Both are git-ignored — they are a developer's run history. Before installing, a `pip install --dry-run mlflow` confirmed MLflow 3.18.0 would **not downgrade** pandas 3, numpy, scikit-learn, SQLAlchemy, Alembic or FastAPI — it only adds packages.

Settings (overridable by environment variables):
```python
mlflow_tracking_uri: str = "sqlite:///mlflow.db"
mlflow_artifact_dir: str = "mlartifacts"
mlflow_experiment: str = "smartprice-demand"
```

`train.py` pins the relative SQLite path to the project root, so the same database is used no matter which directory the command runs from.

---

## Step 2 — One Command: `python -m app.ml.train`

```
1. Load data/processed/daily_sales.csv  →  SHA-256 hash (the dataset version)
2. Forecasting evaluation (Day 9)       →  walk-forward selection + one test score
3. Anomaly evaluation (Day 10)          →  thresholds on validation + one test score
4. Refit the SELECTED model on ALL data
5. Save models/<next version>/          →  model.joblib + metadata.json
6. Log everything to MLflow
```

**Why refit on all data at step 4?** The test score from step 2 is the honest estimate of performance. The deployed model should also learn from the most recent 91 days — they are the most relevant for tomorrow. This is the standard "evaluate, then retrain on everything" practice.

### What one run logs to MLflow

| Kind | Examples |
|---|---|
| **Tags** | `model_version=v1`, `selected_model=Ridge regression`, `dataset_sha256=56fdd5eb…` |
| **Parameters** (20) | `selection_metric=wape`, `test_days=91`, `cv_folds=5`, `cv_validation_days=60`, `target_transform=log1p`, `random_state=42`, `n_features=14`, `training_rows=14220`, `anomaly_detector`, `anomaly_threshold=2.0192`, `model__alpha=1.0`, … |
| **Metrics** (21) | `validation_wape=0.763`, `test_wape=0.706`, `test_mae=50.68`, `test_rmse=151.42`, every candidate's validation and test WAPE, `anomaly_precision/recall/f1` |
| **Artifacts** | `features.json`, `reports/` (both tables, both plots), `model/v1/model.joblib`, `model/v1/metadata.json` |

**A cleanup worth knowing:** the first run logged 39 parameters — scikit-learn's `get_params(deep=True)` includes every pipeline internal (`model__memory`, `model__preprocess__categories__drop`…). Only the final estimator's own hyperparameters are logged now. A run log is for people; noise hides what matters.

---

## Step 3 — Versioned Model Store

**File:** `app/ml/model_store.py`

```
models/
└── v1/
    ├── model.joblib       the fitted scikit-learn pipeline (5,310 bytes)
    └── metadata.json      what it is, what it was trained on, how well it did
```

| Rule | Why |
|---|---|
| Every training run writes a **new** folder (`v1`, `v2`, …) | Old models stay available for comparison and rollback |
| `save_bundle()` **refuses to overwrite** an existing version | A version name must always mean the same model |
| `load_bundle()` **refuses a model trained on different features** | A feature mismatch would give silently wrong predictions; failing loudly is safer |
| `load_bundle()` **warns** if the scikit-learn version differs | A pickled model is only guaranteed to load under the library version that wrote it |

`metadata.json` records: version, creation time, model name, target and its `log1p` transform, the **feature list**, the **dataset path, SHA-256, rows and date range**, the selection protocol, validation and test metrics, every candidate's test WAPE, the anomaly detector and its frozen threshold, the model's hyperparameters, **library versions** (Python, scikit-learn, pandas, numpy) and the **MLflow run id** — from any deployed model you can find the exact run, and from the run the exact data.

**`models/` is committed** — it is what the Docker image will serve. The MLflow database is not.

---

## Step 4 — The API Loads the Configured Version

```python
# settings.py
model_dir: str = "models"
model_version: str = "v1"          # MODEL_VERSION=v2 deploys v2; MODEL_VERSION=v1 rolls back
```

The old, never-used `model_path` and `prediction_threshold` settings (from Day 1) were removed.

**Loaded once, at startup** (`main.lifespan`), and kept on `app.state` — reading a model from disk on every request would be slow:

```
INFO | app.main | Forecast model v1 loaded | Ridge regression | test WAPE 0.706
```

**If the model is missing**, the app still starts. Pricing does not need the model and keeps working; ML endpoints answer **503** with the standard error body:

```json
{ "detail": "Forecast model 'v999' is not available", "code": "model_unavailable" }
```

`ModelUnavailableError` is a new business exception; the Day 5 status-code table gained one line (`→ 503`). 503 rather than 500: it is a known, recoverable server-side state, not a crash.

**New endpoint — `GET /api/v1/models/current`:** which version is serving, its features, dataset hash, training range, validation and test metrics, anomaly detector and threshold, MLflow run id.

**`/health` now also reports** `"model": "v1"` (or `"unavailable"`).

---

## Step 5 — Runtime vs. Development Dependencies

| File | Contents | Used by |
|---|---|---|
| `requirements.txt` | FastAPI, SQLAlchemy, Redis, numpy, pandas, **scikit-learn, joblib, holidays** | the API / Docker image |
| `requirements-dev.txt` | `-r requirements.txt` + openpyxl, LightGBM, matplotlib, **MLflow**, pytest | training, evaluation, tests |

MLflow alone brings Flask, pyarrow, aiohttp and more — none of it belongs in a 512 MB free-tier container. The selected Ridge model needs only scikit-learn to load. (If a future version selects LightGBM, `lightgbm` must move into `requirements.txt` — noted in the file.)

**Enforced by a test:** a fresh Python process imports `app.main`, loads the deployed model, and fails if `mlflow`, `lightgbm`, `matplotlib`, `openpyxl` or `pytest` were imported. Proven to work: adding `import matplotlib` to `model_store.py` in a scratch copy made it fail with `API startup imported training tools: IMPORTED=matplotlib`.

---

## Step 6 — Tests

| File | Covers |
|---|---|
| `test_model_store.py` | Save/load round trip; versions count up (`v1`, `v3` → `v4`, non-version folders ignored); versions are immutable; missing version → `ModelNotAvailableError`; wrong feature list refused; scikit-learn mismatch logged; default version comes from config; **the committed model loads, predicts finite non-negative demand, and its dataset hash matches the processed data** |
| `test_model_api.py` | `/models/current` describes the served model; a missing version gives **503 while pricing still returns 200**; `/health` reports the loaded or missing model |
| `test_training.py` | **The full pipeline, twice, into a temporary folder:** versions `v1` then `v2`; same data → same selection and hash; all four reports written; the MLflow run has the dataset-hash tag, the selected-model parameter, `test_wape`, `features.json`, `reports/` and both model files |
| `test_architecture.py` | API startup never imports training tools |

**Result: 327 tests passing.** The training test redirects models, reports and MLflow to a temporary folder, so running the suite never creates real model versions.

---

## Interview Questions — Day 11

**Q: How do you reproduce a model?**
Every model's `metadata.json` has the SHA-256 of its training data, the feature list, hyperparameters, library versions and the MLflow run id. Same data hash + same code + same seed → same model. Training is one command.

**Q: How do you deploy a new model, or roll back?**
Train → a new immutable `models/vN`. Set `MODEL_VERSION=vN` and restart. Rolling back is setting the old version — no code change, no rebuild of the model.

**Q: Why not load the model inside each request?**
Disk I/O and unpickling on every request would add latency. It is loaded once at startup and held in memory.

**Q: What happens if the model file is missing in production?**
The app starts anyway; pricing works; ML endpoints return 503 `model_unavailable`, `/health` shows `"model": "unavailable"`, and the startup log says why.

**Q: Why isn't MLflow in the production image?**
The API only needs to *load* a scikit-learn model. MLflow is a training-time tool with heavy dependencies. Keeping them apart keeps the image small — and a test guarantees the API never imports it.

**Q: Why retrain on all data after evaluating?**
The held-out score is the performance estimate; the most recent data is the most useful for predicting tomorrow. Evaluate first, then refit the same configuration on everything.

---

---

# Day 12 — Explainability & ML-Driven Pricing (ML Phase 4)

**Goal:** Answer "why did the model say that?" — globally (which features matter) and for every single forecast — and connect the forecast to the price, so the recommendation can be explained end to end.

```bash
python -m app.ml.explain_report            # feature importance + SHAP plot (also run by app.ml.train)
GET /api/v1/forecast/85123A                # next-day forecast + top 5 reasons + recent anomalies
GET /api/v1/explain/85123A                 # why this forecast AND why this price, in plain language
POST /api/v1/pricing/ {"product_id": "85123A", "strategy": "demand_based"}
```

---

## New Files

```
app/ml/explain.py                     exact per-feature contributions + plain-language descriptions (API-safe)
app/ml/explain_report.py              global importance + SHAP summary (offline, uses `shap`)
app/ml/forecasting/predict.py         predict_demand(), split out so the API never imports LightGBM
app/models/forecast.py                DailySales, ForecastFactor, AnomalyFlag, DemandForecast
app/repositories/base.py              + SalesHistoryRepository (abstraction)
app/repositories/postgres.py          + PostgresSalesHistoryRepository
app/repositories/mock.py              + InMemorySalesHistoryRepository
app/repositories/demand_aware.py      DemandAwareProductRepository (decorator: demand_score from the forecast)
app/services/forecast.py              DemandForecastService
app/services/explanation.py           ExplanationService (forecast + price -> plain language)
app/strategies/demand_based.py        DemandBasedPricingStrategy
app/api/routes/forecast.py            GET /forecast/{id}, GET /explain/{id}
reports/feature_importance.png, shap_summary_lightgbm.png, explainability.md
```

---

## Step 1 — Exact Contributions

Both forecasters predict log(1 + units). On that scale every prediction splits **exactly** into a base value plus one contribution per feature:

```
log1p(forecast) = base + Σ contribution(feature)
```

| Model | How contributions are computed | Exactness |
|---|---|---|
| **Ridge** (deployed) | coefficient × transformed value; one-hot columns summed back to their feature | the formula itself — this *is* the SHAP value for a linear model |
| **LightGBM** | `predict(..., pred_contrib=True)` — LightGBM's built-in TreeSHAP | verified **identical** to the `shap` package (max difference 0.0) |

Measured on real rows: `base + Σ contributions` reproduces the model's raw prediction to **1.1 × 10⁻¹⁵** (Ridge) and **1.5 × 10⁻¹⁴** (LightGBM) — floating-point rounding, nothing more.

**Reading a contribution:** the scale is log(1 + units), so `exp(contribution)` is a multiplier: +0.30 → ×1.35 ("raises the forecast ~35%"), −0.41 → ×0.67.

**Plain language:** `describe()` turns each feature into a sentence — `lag_7` → *"Sales on the same weekday last week: 41 units"*, `month` → *"Month: December"*.

**Why the API does not use the `shap` package:** for the deployed linear model the exact answer is one multiplication; for LightGBM, LightGBM computes TreeSHAP itself. `shap` (and matplotlib) stay development-only — `shap==0.54.1` is in `requirements-dev.txt`, and the import guard now also forbids `shap` in the API.

---

## Step 2 — Global Importance & SHAP Summary

**File:** `app/ml/explain_report.py` — runs inside `python -m app.ml.train`, so the plots are logged to MLflow with every run.

**Global importance = mean |contribution|** over all 14,220 rows — literally the average of the per-forecast explanations. Chosen over LightGBM's split/gain importance, which describes how the trees were *built* (and favours high-cardinality features), not how much a feature *moves predictions*. Gain is still reported for reference.

| Feature | Ridge (deployed) | LightGBM | LightGBM gain share |
|---|---:|---:|---:|
| `day_of_week` | 0.845 | 0.513 | 19.5% |
| `roll_mean_28` | 0.279 | 0.308 | 10.9% |
| `product_id` | 0.196 | 0.154 | 3.4% |
| `is_holiday` | 0.116 | 0.126 | 7.7% |
| `roll_mean_7` | 0.113 | 0.094 | 3.9% |
| `month` | 0.106 | 0.079 | 2.6% |
| `lag_7` | 0.029 | **0.404** | **32.5%** |

**Agreement:** both models' top drivers are the weekday (closed Saturdays), the product's recent 28-day level, and the product itself.

**The interesting difference — `lag_7`:** LightGBM relies on it heavily, Ridge hardly at all. The SHAP summary shows why: *low* `lag_7` (zero sales on the same weekday last week) pushes LightGBM's forecast down by up to ~1.8 on the log scale. The tree learned **"if last week's same day was zero, the shop is probably closed again"** — a second route to information `day_of_week` already carries. Ridge gets it directly from the one-hot weekday. Same knowledge, different encoding.

![Feature importance](reports/feature_importance.png)
![SHAP summary](reports/shap_summary_lightgbm.png)

---

## Step 3 — The Live Forecast

**File:** `app/services/forecast.py`

```
sales_history (DB, last 90 days) ─► build_features()  ─► deployed model ─► forecast
                                    (SAME function            │
                                     as training)             ├─► exact contributions ─► top 5 reasons
                                                              └─► deployed anomaly detector ─► recent flags
```

- **No training/serving skew:** the API builds features with the *same* `build_features()` used in training.
- **Sales history is a repository** (`SalesHistoryRepository`): PostgreSQL in production, an in-memory list in tests — the Day 5 pattern again.
- **Which day?** The data ends Friday 9 Dec 2011. The next day is a Saturday, when the shop is always closed — forecasting it would just say "0". The service forecasts the **next open day** (Sunday 11 Dec). The Saturday in between is a *planned* closure, so it is filled in as 0 — exactly what every past Saturday shows. Tested: Friday → Sunday, Thursday before Good Friday → Easter Sunday.
- **Recent anomalies:** the detector selected on Day 10 (z-score + repeated-value rule) runs with its **frozen threshold from `metadata.json`** over the last 14 days. On real data it flags product 22197's **3,111-unit bulk order on 8 Dec 2011**.

### From forecast to demand score — and an honest bias check

The pricing engine already has a 0–1 `demand_score` (Day 6 follow-up). The forecast now sets it:

```
demand_score = clip(forecast / typical, 0, 2) / 2       0.50 = a normal day
```

**What "typical" means took three attempts — each one taught something:**

1. *Mean of all recent open days* → every score 0.04–0.40. Wrong comparison: the forecast is for a **Sunday**, this shop's quietest trading day, and a mean is inflated by bulk orders.
2. *Median of the same weekday, last 4 weeks (actual sales)* → still mostly low. A check showed why: **the model is systematically cautious.** Out of sample it predicts **58% of total units** (median prediction 0.86× the median actual). That is the expected price of training on log(units) for WAPE — it predicts a typical day, not the bulk orders. Comparing it with raw actuals makes every day look quiet.
3. ✅ ***The model's own forecasts for the same weekday over the last 4 weeks.*** Both sides of the ratio share the same caution, so it cancels out: 0.50 now means "a normal Sunday by this model's standards". The actual recent median is still returned (`recent_actual_units`) for context.

Even then, all 20 products score below 0.50 (0.17–0.40) for 11 Dec 2011. **Checked against the data, not assumed:** in December 2010, mid-December Sundays sold **20%** of November's Sunday volume (184 vs. 918 units per Sunday) — wholesale customers had already stocked up for Christmas. The model learned a real seasonal slowdown (*"Month: December — lowers the forecast ×0.67"*).

---

## Step 4 — ML Reaches Pricing (without changing pricing)

### A new strategy — `demand_based`

```python
price = competitor_price × (0.90 + 0.20 × demand_score)
#  score 0.00 -> 10% below competitor · 0.50 -> match · 1.00 -> 10% above
```

- Registered in the Factory with one line; a new enum value. Open/Closed, again.
- **Why only ±10%?** The forecast is uncertain (test WAPE ~0.7). A bounded, linear nudge means a wrong forecast can only move the price a little — and the guardrails still run after it. Tested: low demand on a product near its cost is still raised to the cost floor.
- The strategy stays a pure formula on a `Product` — it knows nothing about models or history.

### A third repository decorator — `DemandAwareProductRepository`

```
PricingService
      │
DemandAwareProductRepository   <- demand_score = forecast            (Phase 4)
      │
CachedProductRepository        <- Redis                              (Day 7)
      │
PostgresProductRepository      <- PostgreSQL                         (Day 6)
```

This is how the model reaches pricing **without touching `PricingService`, any strategy, or the guardrails**: they still receive a plain `Product`; its `demand_score` now comes from the model instead of a stored placeholder.

- It sits **outside** the cache: Redis caches the product row; the forecast-based score is added per request, so it is never stale.
- **It never breaks pricing:** no history (the demo SKUs) or no model → the stored score is kept. Tested.
- **A bug the Day 7 cache test caught:** the first version loaded the product **twice** per pricing request — once in the decorator, again in the forecast service's existence check. The cache-event test saw two extra `Cache HIT`s. `demand_score()` now skips the redundant check (the product was loaded a moment earlier).

---

## Step 5 — The Endpoints

**`GET /api/v1/forecast/{product_id}`** — forecast date, predicted units, typical units, demand score, model version, the forecast before any feature (`base_units`), the **top 5 factors** (description, raises/lowers, multiplier, exact contribution), and recent anomaly flags.

**`GET /api/v1/explain/{product_id}?strategy=demand_based`** — the forecast, the price breakdown (strategy, formula, demand factor, guardrails, margin), and a **plain-language summary**. Real output for product 85123A:

> Next open day (Sun 11 Dec 2011): about 26 units of White Hanging Heart T-Light Holder expected, 42% below a typical Sunday (45) - demand score 0.29 (0.50 = normal).
> Main reasons: Average daily sales over the last 28 days: 132.1 units (raises it x1.70); Month: December (lowers it x0.67); This product's usual demand level (85123A) (raises it x1.46).
> Price: competitor 2.95 x 0.958 (0.90 + 0.20 x demand score 0.29).
> No guardrail needed to change it.
> Recommended price: 2.83 (37.46% margin).
> Unusual days in the last 2 weeks: 27 Nov (8 units: unusually low demand), 04 Dec (8 units: unusually low demand), 09 Dec (4 units: unusually low demand).

| Situation | Response |
|---|---|
| Unknown product | **404** `product_not_found` |
| Product without enough history (the demo SKUs) | **422** `no_sales_history` (new business exception) |
| No model loaded | **503** `model_unavailable` — while `POST /pricing` still answers 200 with the stored score |

---

## Step 6 — Tests

| File | Covers |
|---|---|
| `test_explain.py` | Contributions add up **exactly** to the raw prediction (Ridge and LightGBM); a Ridge category's contribution equals its coefficient; a numeric one equals coefficient × scaled value; unsupported models refused; every feature has a sentence; multiplier wording |
| `test_forecast_service.py` | Demand-score formula at 0, 0.5, 0.75, 1, the cap and the no-reference case; next open day (Fri → Sun, Good Friday skipped); planned closures filled with 0 and the target never used as a feature; forecast equals the model called directly; top factors are the largest and their multipliers equal exp(contribution); "typical" is the model's own median for the last 4 Sundays; a planted spike is flagged; no-history / unknown / too-short history handled |
| `test_demand_pricing.py` | Demand factor at each score; the docstring example (2.95, score 0.29 → 2.83); Factory registration; **guardrails still protect demand-based prices**; the decorator fills in the score, keeps the stored one without a forecast, never mutates the stored product |
| `test_forecast_api.py` | Both endpoints through the **real** dependency graph (only the data sources in memory); explanation consistent with the formula; `/pricing` uses the same score as `/forecast`; 422 / 404 / 503 paths; pricing survives a missing model |
| PostgreSQL | `PostgresSalesHistoryRepository.recent()` returns the newest days, oldest first; forecast, explain and demand-based pricing end to end on the real database |

**Result: 397 tests passing.**

---

## Interview Questions — Day 12

**Q: How do you explain an individual prediction?**
For the deployed Ridge model, on the log scale the prediction *is* intercept + Σ coefficient × value — I group one-hot columns back to their feature. For LightGBM, TreeSHAP via `pred_contrib`. Both add up exactly to the prediction; a test checks it to 1e-9.

**Q: Why mean |SHAP| instead of LightGBM's feature importance?**
Gain and split counts describe tree construction and are biased toward high-cardinality features. Mean |SHAP| measures the effect on the output, and is comparable across model types.

**Q: Your two models disagree on `lag_7`. Which is right?**
Both. LightGBM uses "zero sales last week on this day" as a proxy for "the shop is closed"; Ridge reads the same fact from the weekday one-hot. Explainability showed the same knowledge encoded two ways.

**Q: How does the model influence the price?**
Through a decorator that sets the product's demand score from the forecast, and a bounded strategy: competitor × (0.90 + 0.20 × score). The pricing service, the other strategies and the guardrails are untouched, and pricing falls back to the stored score if the model cannot answer.

**Q: Why compare the forecast with the model's own past forecasts?**
Because the model is systematically cautious (it predicts 58% of actual volume out of sample — the cost of a log target). Comparing it with raw sales would call every day quiet. Comparing it with itself cancels the bias.

**Q: How did you check the low demand scores weren't a bug?**
Against the data: in December 2010, mid-December Sunday demand was 20% of November's. The model learned a real seasonal slowdown.

---

---

# Day 13 — Test Audit & Hardening (ML Phase 5)

**Goal:** Prove the ML system is safe to change. The brief asked for tests of five things — leakage, metrics, model loading, endpoint regressions, and Redis being down. Most already existed from earlier days, so this day started with an **audit**: map each requirement to the tests that cover it, then close the real gaps.

```bash
pytest                                  # everything (PostgreSQL + Redis tests skip if they are not running)
pytest --cov=app --cov-report=term      # with line coverage
```

---

## Step 1 — The Audit

| Requirement | Already covered by (before today) |
|---|---|
| **Leakage** — features never use future data | `test_features.py`: change every future value → past features identical (3 cutoffs); today's own sales/price never a feature; no cross-product leakage. Proven on Day 9 to catch a real leak (6 failures when `shift(1)` was removed). |
| **Metric functions** on known examples | `test_metrics.py`: 9 hand-worked cases |
| **Model-loading path** | `test_model_store.py` (8) + `test_model_api.py` (4) |
| **Existing endpoints** still working | `test_pricing_api.py` (31): every DEVLOG scenario since Day 2 |
| **Redis unavailable** | `test_cached_repository.py`, `test_product_cache.py`, `test_redis_cache.py`; the full suite run with Redis stopped (Day 7) |

**Four real gaps found:**

1. **Serving-time leakage / skew** — nothing checked that the *live* forecast sees exactly the features training would have built.
2. **A damaged model file crashed the API.** `load_bundle()` turned a *missing* file into a clean error, but a *corrupt* `model.joblib` raised a raw unpickling error that startup did not catch.
3. **No route inventory** — a deleted or renamed endpoint would only be noticed if some test happened to call it.
4. **The Day 12 ML endpoints were never tested with Redis down.**

---

## Step 2 — Bug: a Corrupt Model Crashed Startup

**Reproduced first:** a `models/v1` folder with valid metadata and a `model.joblib` containing the bytes `not a pickle`:

```
STARTUP CRASHED: KeyError 110
```

A meaningless error, and the whole API down — including pricing, which does not even need the model.

**Fix** (`app/ml/model_store.py`): any exception while unpickling — and unreadable metadata JSON — becomes `ModelNotAvailableError`, which startup already handles:

```python
try:
    model = joblib.load(model_path)
except Exception as exc:
    raise ModelNotAvailableError(
        f"Model version '{version}' could not be loaded ({type(exc).__name__}: {exc}) - retrain or pick another version"
    ) from exc
```

Catching *every* exception is normally a smell; here it is deliberate: a corrupted pickle can raise `KeyError`, `EOFError`, `UnpicklingError`, `ValueError`… and every one of them means the same thing — "this model cannot be used".

**After:**
```
WARNING | app.main | No forecast model loaded - Model version 'v1' could not be loaded (KeyError: 110) - retrain or pick another version
health: "model": "unavailable"  ·  /models/current: 503  ·  /pricing: 200
```

**Tests:** a corrupted, an empty and a JSON-damaged file each raise `ModelNotAvailableError`; the API starts with a corrupted model, returns 503 for the model endpoint and 200 for pricing.

---

## Step 3 — Training/Serving Parity (No Future Data at Serving Time)

**File:** `tests/test_serving_parity.py`

Two independent paths to the same forecast must agree:

```
OFFLINE  build_features() on the FULL dataset — every day, including all days AFTER
         the forecast date — then take that date's row and predict.
         (Exactly how the model was trained and evaluated.)

ONLINE   DemandForecastService given history only UP TO the day before the
         forecast date — how the API serves a forecast.
```

Five cases across products, months and weekdays — a Tuesday in March, a low-volume product, a Monday in the first year, and **two Sundays**, where the service must insert the planned Saturday closure itself. **All five match** (to the service's 0.1-unit rounding).

What a match proves:
- **No training/serving skew** — the API computes features exactly as training did.
- **No future leakage at serving time** — the online path never had any day after the forecast date, yet produced the same number.

**Proven to catch a real serving bug:** in a scratch copy, the service stopped inserting the closed Saturday before a Sunday forecast — a plausible mistake. **Exactly the two Sunday cases failed.**

A second test wraps the history repository and asserts the service never receives a row dated after its cutoff.

---

## Step 4 — API Regression

**File:** `tests/test_api_regression.py`

- **Route inventory:** the exact set of (method, path) pairs is frozen — `GET /health`, `GET`/`POST /api/v1/pricing/`, `GET /api/v1/models/current`, `GET /api/v1/forecast/{id}`, `GET /api/v1/explain/{id}`. Removing or renaming one fails; adding one also fails until the inventory is updated — a deliberate, reviewed change.
- **Response contracts:** every field a client reads is listed per endpoint, and checked as "at least these" — adding a field is fine, removing or renaming one fails. Pricing is checked for **all five strategies**.
- **Error contract:** 404 / 422 / custom errors keep the `{"detail", "code"}` shape.

**A FastAPI detail found on the way:** the first version listed routes from `app.routes` and saw only `/health`. FastAPI 0.141 wraps included routers in a private `_IncludedRouter` class. The inventory now reads the **OpenAPI schema** — the public contract clients actually see — instead of framework internals.

---

## Step 5 — Redis Down, Across the Whole API

**File:** `tests/test_redis_outage.py`

Redis is pointed at `redis://127.0.0.1:1/0` — nothing listens there, which is exactly what an outage looks like to the app. The **real** dependency graph is used (real `ProductCache`, cache decorator, demand-aware repository, forecast service, pricing); only PostgreSQL is replaced by in-memory data.

| Check | Result |
|---|---|
| `/pricing` (demand_based), `/forecast`, `/explain`, `/models/current` | all **200** |
| Pricing response with Redis down vs. Redis up | **identical** |
| `/health` | `"status": "ok"`, `"cache": "unavailable"` |
| 5 requests in a row | **one** "Redis unavailable" warning — the circuit breaker (Day 7) skips Redis after the first failure |

---

## Step 6 — Coverage and Numbers

`pytest-cov==7.1.0` added to `requirements-dev.txt`.

| | |
|---|---|
| Tests | **428 passing** |
| Line coverage of `app/` | **94%** (1,875 statements, 113 not run) |
| With PostgreSQL **and** Redis stopped | **400 passed, 28 skipped** — a fresh checkout runs almost everything with no setup |
| Full suite time | ~49 s (with coverage) |

The files below 85% are honest gaps, not hidden ones: the command-line `main()` functions of `app/database/seed.py` (64%) and `app/ml/data/prepare.py` (66% — it reads the 45 MB raw file), plus database ping/dispose helpers (`session.py`, 81%). They are exercised by running the commands themselves, as every phase did.

---

## Interview Questions — Day 13

**Q: How do you test for data leakage?**
Two ways. Offline: change all future values and assert past features are identical. Online: the live service, given history only up to day D, must produce the same forecast as the training pipeline run on the full dataset. Both were proven to catch a real bug.

**Q: What is training/serving skew, and how do you prevent it?**
When production computes features differently from training, the model gets inputs it never saw. Prevention: one `build_features()` function for both. Detection: a parity test comparing the two paths on real past dates.

**Q: What happens if someone deploys a corrupted model file?**
The app starts, logs exactly which version failed and why, reports `"model": "unavailable"` on `/health`, returns 503 on ML endpoints — and pricing keeps working. Before today it crashed at startup; a test now guards it.

**Q: How do you stop API changes from breaking clients?**
A frozen route inventory from the OpenAPI schema, and per-endpoint field contracts checked as "at least these" — additive changes pass, removals and renames fail.

**Q: What is your test coverage?**
94% of lines in `app/`, measured with pytest-cov. The uncovered lines are mostly command-line entry points that are run manually. Coverage is a floor, not proof — the leakage and parity tests matter more, which is why each was shown to fail on an injected bug.

---

---

# Day 14 — Deployment Readiness & Live Demo (ML Phase 6)

**Goal:** Make SmartPrice deployable on free tiers — **Render** (web services) + **Neon** or **Supabase** (PostgreSQL) + **Upstash** (Redis) — and give interviewers something to click: a Streamlit demo.

**What was verified locally, and what was not:** the production image was built and run against a brand-new empty database; the demo was run against the live API. The actual cloud deploy needs the owner's Render / Neon / Upstash accounts, so it was **not** performed — the steps are at the end of this entry.

---

## New Files

```
Dockerfile                 slim production image (runtime requirements only, non-root)
.dockerignore              keeps secrets, the raw data, tests and MLflow out of the image
scripts/start.sh           migrations -> idempotent seed -> uvicorn on $PORT
.gitattributes             *.sh always LF (a CRLF script fails inside Linux)
render.yaml                Render Blueprint: API (Docker) + demo (Python), both free
demo/app.py                the Streamlit demo
demo/requirements.txt      streamlit + requests only
app/api/routes/products.py GET /api/v1/products/ (the demo's product picker)
app/schemas/product.py
docker-compose.yml         + optional `api` service (profile "app")
```

---

## Step 1 — Configuration for Cloud Providers

### Database URL: accept what the provider gives

Neon and Supabase hand out:
```
postgresql://user:pw@ep-cool-name.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```
The app's async driver needs:
```
postgresql+asyncpg://user:pw@ep-cool-name.eu-central-1.aws.neon.tech/neondb?ssl=require
```

Rather than ask anyone to hand-edit a secret, a Pydantic validator on `Settings.database_url` normalises it:

| Input | Becomes | Why |
|---|---|---|
| `postgres://` or `postgresql://` | `postgresql+asyncpg://` | tells SQLAlchemy to use the async driver |
| `sslmode=require` | `ssl=require` | asyncpg's name for the same option |
| `channel_binding=require` | *(dropped)* | a libpq-only option asyncpg rejects |
| already `postgresql+asyncpg://...` | unchanged | local Docker keeps working |

Because it lives in `Settings`, Alembic (`alembic/env.py`) and the app get the same normalised URL.

**Tested for real, not just as string rewriting:** the local Docker PostgreSQL has no TLS, so a correctly passed `ssl=require` **must be refused** — and the test asserts exactly that. Passwords with URL-encoded characters (`p%40ss%23word`) and unrelated options survive.

**Neon note:** use the **direct** connection string (host without `-pooler`). Pooled connections go through PgBouncer in transaction mode, which conflicts with asyncpg's prepared-statement cache.

### Environments: one variable decides which database

| Environment | PostgreSQL | Redis | Where `DATABASE_URL` / `REDIS_URL` live |
|---|---|---|---|
| Local development + tests | Docker (`postgres:17`) | Docker (`redis:7-alpine`) | `.env` (git-ignored) |
| Production (deployed) | **Neon** (managed, free tier) | **Upstash** (free tier) | Render's environment variables (`sync: false` in `render.yaml`) |

The code never knows which one it talks to — the twelve-factor "config in the environment" rule. All services stay on **free tiers**: this is a portfolio project, but built the way a production system would be.

**A trap found while setting this up:** a Neon URL was appended to `.env` as a *second* `DATABASE_URL` line. With duplicate keys the **last one silently wins**, so `alembic upgrade head` targeted Neon instead of the local database. Rule: **exactly one `DATABASE_URL` in `.env`** (local). To run a one-off command against Neon, override it for that command only, e.g. in PowerShell:

```powershell
$env:DATABASE_URL = "<neon connection string>"; alembic upgrade head; python -m app.database.seed; Remove-Item Env:DATABASE_URL
```

(A real environment variable takes precedence over `.env`.)

**Network gotcha:** that same run failed with `getaddrinfo failed`. The cause was not the code: the network's DNS server (a phone hotspot) answered "DNS operation refused" for the Neon hostname, while public resolvers (1.1.1.1, 8.8.8.8) resolved it immediately. Fix: set the machine's DNS to a public resolver, or use another network. A useful reminder that "can't connect to the database" can be a DNS problem before it is a database problem.

### Redis over TLS (Upstash)

Upstash URLs start with `rediss://` (two s = TLS). redis-py switches to an SSL connection by itself — tested: `rediss://` gives an `SSLConnection` pool, `redis://` does not. No code change was needed; the test makes sure it stays that way.

### Currency

The real dataset is a UK wholesaler, but messages still said "Rs." (a Day 2 leftover). New setting `currency_symbol` (default `£`), used in the pricing message and the explanation text: *"Recommended price: £2.83 (37.46% margin)"*.

### `.env.example`

Documents every setting, including how to paste the Neon and Upstash URLs, with placeholders (`change-me`) — a test checks the keys are there and no real secret is.

---

## Step 2 — `GET /api/v1/products/`

The demo needs a product list. `list_all()` was added to the `ProductRepository` contract and implemented by every repository:

| Repository | `list_all()` |
|---|---|
| Postgres | `SELECT ... ORDER BY product_id` |
| Mock | sorted in memory |
| Cached (Redis) | **not cached** — one cheap query, and a cached list would go stale whenever a product is added |
| Demand-aware | passes through — a forecast for every product on every listing would be slow |

The Phase 5 route inventory **failed** until `GET /api/v1/products/` was added to it — exactly the "conscious, reviewed change" it exists to force.

---

## Step 3 — The Production Image

**File:** `Dockerfile`

```dockerfile
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt          # runtime only — no MLflow, LightGBM, SHAP, matplotlib, pytest
COPY alembic.ini . ; COPY alembic ./alembic ; COPY app ./app
COPY models ./models                         # the versioned model the API serves
COPY data/processed ./data/processed         # what the seed loads into a fresh database
RUN useradd --create-home --uid 10001 smartprice
USER smartprice                              # never run as root
HEALTHCHECK ... /health
CMD ["./scripts/start.sh"]
```

- **Dependencies first:** that layer is reused on every rebuild until `requirements.txt` changes.
- **No build stage needed:** every dependency ships as a pre-built wheel.
- **Non-root user**, Docker `HEALTHCHECK` on `/health`.

### Size — measured, and explained

| Layer | Size |
|---|---|
| Debian slim + Python 3.12 | ~134 MB |
| `pip install -r requirements.txt` | 465 MB |
| App, migrations, model, processed data | ~1 MB |
| **Total (uncompressed)** | **774 MB** |

85% of the dependency layer is the scientific stack the model needs: SciPy 109 MB (a hard dependency of scikit-learn), pandas 73 MB, scikit-learn 49 MB, NumPy 43 MB. Nothing optional is in it. On a free tier the limit that matters is **RAM**, not image size — and the running container used **151 MiB** after serving every endpoint (free tier: 512 MB).

### `scripts/start.sh` — a fresh database becomes ready by itself

```sh
alembic upgrade head          # create / update the schema (no-op when current)
python -m app.database.seed   # demo products + sales history (idempotent)
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips="*"
```

- `$PORT` — Render (and most hosts) tell the app which port to listen on.
- `exec` — uvicorn replaces the shell, so it receives stop signals directly and the lifespan closes connections cleanly.
- `--proxy-headers` — behind Render's proxy, the app sees the client's real scheme and address.

### Verified against a brand-new, empty database

An empty `smartprice_fresh` database was created; the image was started with a **provider-style** URL (`postgresql://...?sslmode=disable`) and a non-default `PORT=9000`:

```
Running upgrade  -> b2cda35e279c, create products table
Running upgrade b2cda35e279c -> b7ea73182097, add inventory and demand score to products
Running upgrade b7ea73182097 -> 44e71ee47cb1, create sales history table
Seed complete | products inserted=23 | already present=0
Sales history | rows inserted=14780 | already present=0
Forecast model v1 loaded | Ridge regression | test WAPE 0.706
Uvicorn running on http://0.0.0.0:9000
```

`/health` → database, cache and model all ok; products, forecast, explain and pricing all answered. **Restarted:** no migrations ran and the seed inserted 0 rows — safe to run on every boot.

### A bug found by running the image

The container log showed `SyntaxWarning: invalid escape sequence '\-'` — an ASCII diagram (`\->`) in a docstring. Python 3.12 warns at import; a future version will make it a `SyntaxError`, i.e. the app would stop starting. Fixed, and a new architecture test compiles **every** module with warnings turned into errors. (It failed on the unfixed file before the fix landed — proof it works.)

### `docker compose --profile app up -d --build`

The `api` service runs the same image locally against the Compose PostgreSQL and Redis. It sits behind a profile, so plain `docker compose up -d` still starts only the databases for day-to-day development.

---

## Step 4 — The Streamlit Demo

**File:** `demo/app.py` — a thin client: every number comes from the API over HTTP; it never imports the app's code, so it deploys as its own small service (`streamlit` + `requests`).

```bash
streamlit run demo/app.py                    # SMARTPRICE_API_URL defaults to http://localhost:8000
```

What an interviewer sees:
- **Sidebar:** API / database / cache / model status; product picker (from `GET /products`); strategy picker; a "Model" panel (version, training range, held-out test WAPE, anomaly F1, dataset hash).
- **Headline:** forecast for the next open day vs. a typical same weekday, demand score, recommended price vs. competitor, margin.
- **Why:** the API's plain-language summary.
- **What drives this forecast:** the top 5 exact contributions as a bar chart, with the full sentences in a table.
- **How the price was set:** strategy, competitor price, cost, demand factor, guardrails.
- **Unusual days:** the anomaly detector's flags for the last two weeks.
- **Deep links:** `?product=22197` opens a product directly — handy for sharing.

Tried in a browser against the running API:
- **85123A** — 26 units vs. a typical Sunday of 45 (−42%), demand score 0.29, £2.83 (−4.1% vs competitor), 37.5% margin.
- **22197** — the real **3,111-unit bulk order on 8 Dec 2011** listed under "Unusual days".
- **SKU-12345** (no history) — a friendly explanation instead of an error.

**Two fixes from actually looking at it:**
1. Chart labels were cut off ("Average daily sale…") → short names on the chart, full sentences in a table.
2. One factor can multiply the forecast ×27 while others are ×0.3–×1.7; on a percent axis (+2,625%) every other bar vanished. Multipliers are naturally a **log-scale** quantity, so the chart shows the exact log contribution (+0.69 = doubles, −0.69 = halves) with the multiplier beside it.

**Free-tier cold starts:** a sleeping free Render service needs up to about a minute to wake; the demo waits up to 60 s and says so in its spinner.

---

## Step 5 — `render.yaml`

| Service | Runtime | Build / start | Secrets (`sync: false`) |
|---|---|---|---|
| `smartprice-api` | Docker (`./Dockerfile`) | `scripts/start.sh` · health check `/health` | `DATABASE_URL`, `REDIS_URL` |
| `smartprice-demo` | Python 3.12 | `pip install -r demo/requirements.txt` · `streamlit run demo/app.py --server.port $PORT` | `SMARTPRICE_API_URL` |

`sync: false` makes Render ask for the value in its dashboard; it is never stored in the repository. **The file has not been validated by Render itself** (no account was used) — `tests/test_deployment_files.py` checks what can be checked locally: both services on the free plan, the health path, secrets never committed, the demo installs only its own requirements, the start script uses `$PORT` and LF line endings, the Dockerfile installs runtime requirements only and runs as non-root, and `.dockerignore` excludes `.env`, `.venv/`, the raw data, `mlflow.db` and the tests.

---

## Deploying (free tiers) — step by step

1. **Neon** (or Supabase): create a project → copy the **direct** connection string (Neon: host *without* `-pooler`).
2. **Upstash**: create a Redis database → copy the `rediss://default:<token>@<name>.upstash.io:6379` URL.
3. Push the repository to GitHub (`.env` is git-ignored — check before the first push).
4. **Render** → New → **Blueprint** → select the repository. Render reads `render.yaml` and creates both services.
5. Enter the secrets when asked: `DATABASE_URL` (step 1), `REDIS_URL` (step 2). Leave `SMARTPRICE_API_URL` until the API has a URL.
6. Wait for `smartprice-api` to go live — its first boot creates the schema and loads 14,780 rows of sales history. Copy its URL (e.g. `https://smartprice-api.onrender.com`).
7. Set `SMARTPRICE_API_URL` on `smartprice-demo` to that URL → the demo redeploys.
8. Open `https://<api>/health` (expect `database`, `cache`, `model` all ok), then the demo URL.

---

## Step 6 — Tests

| File | Covers |
|---|---|
| `test_cloud_config.py` | Neon / Render / local URLs normalised (incl. encoded passwords); `Settings` applies it; **`ssl=require` really reaches the driver** (refused by the non-TLS local database); `rediss://` → SSL connection, `redis://` → plain; `.env.example` complete with placeholders only |
| `test_deployment_files.py` | Blueprint services, health path, no committed secrets, demo requirements, `$PORT` + LF start script, runtime-only non-root Dockerfile, `.dockerignore` |
| `test_api_regression.py` | `/products/` added to the inventory; 23 products, the contract fields, ordered by id |
| `test_postgres_repository.py` | `list_all()` returns domain products in order |
| `test_architecture.py` | every module compiles with warnings as errors |

**Result: 522 tests passing.** New pinned dev tools: `streamlit==1.66.1` (demo), `pyyaml==6.0.3` (parse `render.yaml` in tests). Both were checked with `pip install --dry-run` first — neither changes pandas, NumPy or scikit-learn.

---

## Interview Questions — Day 14

**Q: How does a fresh deployment get a schema and data?**
The container's start script runs `alembic upgrade head`, then an idempotent seed, then the server. Verified against an empty database: 3 migrations, 23 products, 14,780 sales rows — and a restart changes nothing.

**Q: Why is the image 774 MB if it is "slim"?**
The base is slim; 85% of the dependency layer is SciPy, pandas, scikit-learn and NumPy, which the model needs. Training tools are excluded. On a free tier, RAM is the constraint — the container ran at 151 MiB.

**Q: How do you handle the different database URL formats of cloud providers?**
A validator in the settings converts provider URLs (`postgresql://...?sslmode=require&channel_binding=...`) into the async driver's format. A test proves the TLS option really reaches the driver, by checking that a non-TLS database refuses it.

**Q: How are secrets handled?**
Only in environment variables. `.env` is git-ignored, `.env.example` has placeholders, and `render.yaml` marks every credential `sync: false` — a test fails if one is ever given a value.

**Q: Why is the demo a separate service?**
It is a client of the API, like any other. Keeping it separate keeps the API image small, and it means the demo exercises the real public API — what you see is what any client gets.

---

---

# Day 15 — Documentation & the Final Numbers (ML Phase 7)

**Goal:** Make the project understandable in two minutes (README) and defensible in an hour (this log) — with every number traceable to a command that produced it.

---

## Step 1 — README.md

**File:** `README.md` (new — the project had none).

| Section | Contents |
|---|---|
| Pitch | one sentence, plus placeholders for the live demo, API docs and demo-video links (filled in after deploying) |
| Results at a glance | the headline numbers, each from a real run |
| Architecture | a **Mermaid** diagram: clients → routers → DI → services → repository decorators → PostgreSQL / Redis / model; the offline training path → MLflow |
| ML approach | data and its assumptions, leakage-safe features, walk-forward evaluation, candidates, metrics, anomaly protocol, explainability, MLOps |
| Results | the full forecast (validation + test), anomaly and feature-importance tables, copied from `reports/` |
| Honest limitations | why WAPE stays ~0.7, the model's systematic caution, the price assumptions |
| API | every endpoint and the error contract |
| Run / retrain / test / deploy | copy-paste commands; the 8-step free-tier deploy |
| Structure & patterns | where everything lives; which pattern each part uses |

### Every number has a source

| Claim in the README | Source |
|---|---|
| Test WAPE 0.706 vs 0.990, 28.7% lower; validation 0.763 ± 0.029 | `reports/forecast_comparison.md`, `models/v1/metadata.json` |
| Anomaly F1 0.476 (P 0.395, R 0.600) and the detector table | `reports/anomaly_comparison.md` |
| 1,067,371 → 1,002,004 rows; 14,780 product-days; 22% zero days | `data/processed/metadata.json` (3,295 / 14,780 zero rows) |
| Feature importance table | `reports/explainability.md` |
| Contributions exact within 1e-14 | Day 12 measurement (Ridge 1.1e-15, LightGBM 1.5e-14) |
| 522 tests, 94% coverage | `pytest --cov=app`, run for this entry: **522 passed**, `TOTAL 1929 statements, 114 missed, 94%` |
| 151 MiB RAM; fresh-database bootstrap | Day 14 container run (`docker stats`, container log) |
| Model predicts ~58% of test volume | Day 12 out-of-sample check |

One claim was corrected while writing: a first draft said contributions add up "to 1e-15". That holds for Ridge (1.1e-15) but not LightGBM (1.5e-14), so the README says "within 1e-14".

---

## Step 2 — This Log, Brought Up to Date

- **Project Overview** rewritten to describe the system as it is now, with a note that Days 1-7 contain no machine learning: the early entries' "ML-powered" wording described the goal, not the code. (Flagged as an overclaim in the Day 8 audit; now explained instead of hidden.)
- **Tech Stack** gained LightGBM, SHAP, matplotlib, MLflow, Streamlit, Docker/Render/Neon/Upstash and pytest-cov, each with the day it arrived.
- **Running the Application** now covers install, Docker, migrations, seed, training, the API, the demo, the Docker-as-deployed option and the tests.

---

## Step 3 — The Whole ML Upgrade, in One Table

| Phase | Day | What was added | Key real result |
|---|---|---|---|
| A — Data | 8 | UCI Online Retail II pipeline, `sales_history` table, hashed dataset | 1,067,371 → 1,002,004 rows; 20 products × 739 days |
| 1 — Forecasting | 9 | leakage-safe features, 5 models, walk-forward validation, WAPE | test WAPE 0.706 vs 0.990 baseline (−28.7%) |
| 2 — Anomalies | 10 | injected labels, 4 detectors, F1-chosen thresholds | best F1 0.476; flatline recall 0.07 → 0.60 with a one-line rule |
| 3 — MLOps | 11 | one-command training, MLflow, versioned models, config-selected model | every run traceable to its dataset hash |
| 4 — Explainability | 12 | exact contributions, SHAP summary, `/forecast`, `/explain`, `demand_based` pricing | contributions exact within 1e-14 |
| 5 — Tests | 13 | audit, parity, API contracts, Redis outage | startup crash on a corrupt model found and fixed |
| 6 — Deployment | 14 | Dockerfile, cloud config, `render.yaml`, Streamlit demo | fresh database bootstrapped in one boot; 151 MiB RAM |
| 7 — Docs | 15 | README, updated log | 522 tests, 94% coverage |

**Bugs found by verification during the upgrade** — each one by running or testing something, not by reading code:

| Day | Bug | Found by |
|---|---|---|
| 10 | rolling std of identical values = 2.8e-8, not 0 → the flatline rule silently missed cases | a unit test with a stuck value |
| 11 | MLflow logged 39 noisy parameters (pipeline internals) | reading the first run's log |
| 12 | every pricing request loaded the product twice | the Day 7 cache-event test |
| 12 | demand scores biased low (a wrong "typical" baseline, then the model's own caution) | checking scores against the data |
| 13 | a corrupt model file crashed API startup | the test audit, then a reproduction |
| 14 | an invalid escape sequence in a docstring (a future `SyntaxError`) | running the Docker image |
| 14 | demo chart unreadable (clipped labels; a ×27 factor flattening the rest) | looking at the demo in a browser |

**What is not done:** the cloud deploy itself (it needs the owner's Render, Neon and Upstash accounts). Everything it needs is in the repository and verified locally; the README and Day 14 give the steps.

---

## Interview Questions — Day 15

**Q: Walk me through the project in two minutes.**
A FastAPI pricing backend built in layers with classic patterns (Strategy, Factory, Chain of Responsibility, Repository, Decorator, DI), on PostgreSQL and Redis. On top, an ML layer trained on real UCI retail data: a next-day demand forecast selected by walk-forward validation (28.7% better than the seasonal-naive baseline on held-out data), anomaly detection evaluated on injected labels, MLflow tracking with versioned models, and exact per-feature explanations. The forecast feeds a bounded demand-based price through a repository decorator, so the pricing core never changed. 522 tests, 94% coverage, a Docker image that bootstraps its own database, and a Streamlit demo.

**Q: What would you do next?**
Get information about bulk orders (customer pipelines, quotes) — that, not a fancier model, is what limits accuracy. Then deploy, add monitoring of live forecast error against actual sales, and retrain on a schedule when the data drifts.

**Q: What are you most careful about in this project?**
Not fooling myself: time-ordered evaluation, a test set used once, leakage and parity tests that were each shown to fail on an injected bug, and every reported number traceable to the command that produced it.

---

---

# Day 16 — Running It Day to Day, Understanding It, and Hosting It

**Goal:** Everything that happened after the project was "done": making it easy to start, answering the questions an interviewer (or the owner) asks about how it works, and preparing the hosting plan. This entry closes the gaps so the DEVLOG covers the whole project.

---

## Step 1 — One-Command Start and Stop

Starting the demo took five commands in two terminals. Now it is one.

| File | What it does |
|---|---|
| `scripts/start-demo.ps1` | starts Docker Desktop if needed (waits for it) → `docker compose up -d` (waits until PostgreSQL and Redis are healthy) → `alembic upgrade head` → `python -m app.database.seed` → opens the **API** (port 8000) and the **demo** (port 8501) in their own windows → opens the browser |
| `scripts/stop-demo.ps1` | stops whatever listens on ports 8000 and 8501 (the whole process tree — `uvicorn --reload` has a child process), then `docker compose stop` — **data is kept** |
| `start-demo.bat` / `stop-demo.bat` | double-click wrappers (`powershell -ExecutionPolicy Bypass -File ...`) |
| `.vscode/tasks.json` | **Ctrl+Shift+P → Tasks: Run Task → Start / Stop SmartPrice demo** inside Antigravity / VS Code |

Details worth knowing:
- The scripts use `.venv\Scripts\python.exe` directly, so the virtual environment does not need activating.
- Already running? The start script checks the ports and does not start a second copy.
- Alembic and the seed run every time because they are **no-ops** when the database is current (verified: `products inserted=0 | already present=23`, `rows inserted=0 | already present=14780`).
- **Double-clicking a `.bat` in the editor's file tree only opens it** — it runs from Windows File Explorer, from the terminal as `.\start-demo.bat`, or via the task.
- `.gitignore` excluded all of `.vscode/`, which would have kept `tasks.json` out of git — now `.vscode/*` is ignored except `!.vscode/tasks.json`.

**Bug found by running it:** the first version had `$ErrorActionPreference = "Stop"`. In Windows PowerShell 5.1 that turns *any* text a native program writes to stderr into a fatal error — and Docker prints a harmless `No blkio throttle` warning there; Alembic logs to stderr too. Fix: `"Continue"`, and every step checks `$LASTEXITCODE` (the real success signal). Tested: Docker detected, both containers healthy, migrations and seed no-ops, API started, `/health` → database, cache and model all `ok`.

### Two first-run errors, explained
| What you see | Why | Fix |
|---|---|---|
| `{"detail":"Not Found"}` at `http://localhost:8000/` | there is no page at `/` — the API **is** running | open `/docs`, `/health` or `/api/v1/...` |
| `'streamlit' is not recognized` | that terminal has not activated `.venv` | `.venv\Scripts\activate`, or `.venv\Scripts\python.exe -m streamlit run demo/app.py` |

---

## Step 2 — Local Database vs Neon in `.env`

A Neon project was created, and its URL briefly replaced the local one in `.env`. It was switched back (in a separate session):
- the **active** `DATABASE_URL` points at the local Docker PostgreSQL (`localhost:5432/smartprice`);
- the Neon URL is kept **commented out**, with a note that in production it belongs in **Render's environment variables**, not in `.env`.

The `DATABASE_URL_POOLED` and `NEON_REALTIME_*` entries in `.env` are not read by the app (it only reads `DATABASE_URL`). For deployment use Neon's **direct** connection string, not the pooled one (Day 14).

---

## Step 3 — Neon Agent Skills

`npx neon skills -y` installs Neon's agent skills for Claude Code. It needs **Node.js ≥ 22.20**; the machine has 20.19.5.
- The `neon` npm package was checked first: it is Neon's official CLI (`github.com/neondatabase/neon-pkgs`).
- Run with a **temporary Node 22** from the npx cache, without touching the installed Node 20. Running it nested inside `npx -p ... -c` failed: the outer npx leaked `npm_config_call` / `npm_config_package` into the CLI's own inner `npx` call. Running the inner command directly (Node 22 first on `PATH`, those variables removed) worked.
- Result: 7 skills in `.claude/skills/` — `neon`, `neon-postgres`, `neon-auth`, `neon-functions`, `neon-object-storage`, `neon-realtime`, `neon-ai-gateway`. They load when a new Claude Code session starts in this folder. They are tooling for working on the project, not part of the app.

---

## Step 4 — Understanding the System (Q&A Summary)

A plain-language summary of the questions asked after completion (it was first written as `docs/takeaways-running-and-infrastructure.md`, later removed — this section keeps the content):

### Swagger and Streamlit
- `/docs` is **generated by FastAPI** from the route decorators (`@router.post(...)`) and the Pydantic schemas; the "Schemas" section is the shape of every request and response.
- Streamlit is a **thin client**: `demo/app.py` calls the API over HTTP (`requests.get(".../api/v1/explain/85123A")`) and draws it with one-line widgets (`st.metric`, `st.bar_chart`). No HTML/CSS/JS is written — only Python layout code.

### Docker, PostgreSQL, Redis
- **Image** (frozen program) → **container** (running copy) → **port mapping** (`localhost:5432` → the container) → **volume** (`smartprice_pgdata`, where the data survives restarts).
- The app's code knows it talks to **PostgreSQL and Redis**; **where** they run is only in `.env` (or the host's settings); **Docker** appears only in `docker-compose.yml` and `Dockerfile`. Business logic (pricing, forecasting) knows about neither — repositories hide them, and architecture tests enforce it.
- **Why PostgreSQL and not a CSV?** Training *does* read a CSV (a versioned, hashed snapshot). The live API needs fast lookups, safe concurrent updates, validation rules, transactions and storage that survives redeploys. *"Training reads a CSV snapshot; the API reads PostgreSQL. Same data, two jobs."*

### Offline vs online ML
| | Offline (training) | Online (serving) |
|---|---|---|
| When | `python -m app.ml.train` | every API request |
| Does | clean data, build features, compare models, **save** the best | **load** it once, build features for one product, **predict** |
| Data | full history (14,220 training rows) | last 90 days of one product, from PostgreSQL |

### What the model predicts — and what it does not
Ridge predicts **tomorrow's units sold** (on the log1p scale), **not the price**. The price comes afterwards: forecast → demand score → `demand_based` strategy → guardrails.

The deployed model's **learned weights** (numeric features standardised; on the log scale, `exp(weight)` = the multiplier):

| Feature | Weight | Meaning |
|---|---:|---|
| `roll_mean_28` | +0.383 | the strongest numeric driver — a product's recent level (×1.47 per standard deviation) |
| `roll_mean_7` | +0.172 | the last week's level |
| `lag_7` | +0.059 | same weekday last week |
| `lag_1` | +0.022 | small: the rolling averages already carry yesterday's information |
| `roll_std_7` / `roll_std_28` | −0.142 / −0.159 | jumpy sales → a more cautious forecast |
| `is_holiday` | −0.368 | holidays reduce demand |
| weekday: Saturday | −2.81 | ×0.06 — the shop is closed |
| intercept | 2.832 | the starting point |

Worked example (85123A, data to 9 Dec 2011): `2.832 + 0.565 (28-day avg) + 0.422 (Friday) − 0.407 (December) + 0.379 (product) + ... = 4.022` → `e^4.022 − 1 = 54.8` units. (Only 4 actually sold that day — the forecast is good on average, not on every day.)

### Points that are easy to get wrong
- **WAPE is an error, so lower is better:** 0.706 (Ridge) beats 0.990 (seasonal naive). `(0.990 − 0.706) / 0.990 = 28.7%` less error.
- **The ±10% in `demand_based` is a design choice, not learned.** It is a cautious bound because forecast error is large (WAPE ~0.7); with real business data it would be set from price-elasticity experiments (A/B tests).
- **The deployed anomaly detector is statistics plus a rule** (rolling z-score with |z| ≥ 2.019 chosen by F1 on validation, plus "4 identical open days"). Isolation Forest — the ML option — was evaluated and lost (F1 0.400 vs 0.476).
- **ML is bounded:** 4 of the 5 strategies, the guardrails and the caching do not use ML at all.

### Hosting concepts
- **PostgreSQL / Redis are software**; postgresql.org and redis.io give you the program, not a running database. **Neon** (managed PostgreSQL) and **Upstash** (managed Redis) run the real software in the cloud and give you a URL.
- **Render** is a hosting service for long-running servers and Docker containers — it runs the API and the Streamlit demo. **Vercel** is built for front-end sites and short serverless functions: a poor fit for a FastAPI server that keeps a model in memory, a 774 MB image, or Streamlit. A common later combination: React on Vercel + API on Render.

---

## Step 5 — Hosting Plan and Cost

### Free-tier limits (checked on 11 Nov 2026 — they change; re-check before relying on them)

| Service | Role | Free tier | What matters here |
|---|---|---|---|
| **Render** | runs the API (Docker) and the demo (Python) | **750 instance hours per month per workspace**, shared by all free services; spins down after **15 min** without traffic; waking takes **~1 min**; ~512 MB RAM | two services sleeping when idle stay within 750 h; the API uses ~151 MiB |
| **Neon** | PostgreSQL | **1 GB storage** and **100 CU-hours** of compute per project per month; scales to zero after 5 min idle; **no overage charges on the free plan** | the data is a few MB |
| **Upstash** | Redis cache (optional) | **500K commands per month**, ~250 MB data, 10 GB bandwidth, 1 free database | a demo uses a tiny fraction; adding a payment method moves the account off the free plan |

**Total cost: ₹0 / $0**, with no credit card needed for any of the three at the time of checking. The trade-off of free: the first request after ~15 minutes of quiet takes about a minute (Render wakes up, Neon wakes up).

**If it must be always on:** Render's paid instances cost a few dollars per service per month (check render.com/pricing), and Neon/Upstash usage-based plans start at low amounts. Not needed for a portfolio demo.

### Before the first deploy
1. **Turn the folder into a git repository** — it is not one yet: `git init`, then `git status` and confirm **`.env` is not listed** (it contains passwords; `.gitignore` excludes it).
2. Confirm `models/v1/` and `data/processed/` **are** included — the Docker image needs them (the model to serve, the data the seed loads).
3. First commit, create a GitHub repository, push.

### Steps
1. **Neon:** in the existing project → Connect → turn **connection pooling off** → copy the direct string (`postgresql://...?sslmode=require...`).
2. **Upstash** (optional): create a Redis database → copy the `rediss://default:<token>@<name>.upstash.io:6379` URL.
3. **Render:** sign in with GitHub → **New → Blueprint** → choose the repository. Render reads `render.yaml` and proposes `smartprice-api` (Docker) and `smartprice-demo` (Python).
4. Enter **`DATABASE_URL`** (step 1) and **`REDIS_URL`** (step 2, or skip). Leave `SMARTPRICE_API_URL` for now. Apply.
5. Wait for `smartprice-api` to go live. Its first boot runs the 3 migrations and loads 23 products + 14,780 sales rows into Neon. Copy its URL.
6. Set **`SMARTPRICE_API_URL`** on `smartprice-demo` to that URL → it redeploys.
7. Check `https://<api>/health` → `database`, `cache`, `model` all `ok`; then open the demo.
8. Put the demo and API-docs links into the README placeholders.

**After that:** every `git push` redeploys automatically. To deploy a newer model, train locally (`python -m app.ml.train` → `models/v2`), commit it, and change `MODEL_VERSION` on Render.

---

## What Is Still Open

| Item | Status |
|---|---|
| Cloud deploy | prepared and verified locally; needs the owner's Render / Neon / Upstash accounts |
| README placeholders | live demo, API docs and video links — fill in after deploying |
| Git repository | not created yet |
| `LEARNING.md` | requested in the master prompt; deferred by the owner. The material exists across Days 8–16 |

## Interview Questions — Day 16

**Q: How do you run the project?**
`.\start-demo.bat` — it starts Docker, the databases, applies migrations, seeds idempotently, and opens the API and the demo. Or the five commands it wraps.

**Q: Where would you host it, and what does it cost?**
Render for the API container and the demo, Neon for PostgreSQL, Upstash for Redis — all free tiers, $0. The trade-off is cold starts after idle periods; always-on would cost a few dollars a month per service.

**Q: Why not Vercel?**
It is built for front ends and short serverless functions. The API is a long-running server that keeps a model in memory, ships as a large Docker image, and the demo is Streamlit — Render runs both as normal services.

**Q: Does your ML model predict the price?**
No — it predicts tomorrow's demand. A bounded strategy turns demand into at most a ±10% adjustment of the competitor's price, and rule-based guardrails have the final say.

---

# Day 17 — Live in Production: How the Hosting Works, and Two Debugging Stories

**Date:** 11 Nov 2026

**What happened today:**
- **Clean GitHub history.** The repository was recreated as one commit authored only by the owner: https://github.com/marshallmathers0/SmartPrice
- **Deployed on free tiers:** Render + Neon + Upstash.
  - Demo: https://smartprice-demo.onrender.com
  - API docs: https://smartprice-api-n7bp.onrender.com/docs
  - Health: https://smartprice-api-n7bp.onrender.com/health
- **Verified live:**
  - `/health` reports database `ok`, cache `ok` and model `v1`.
  - Products, forecast and explain all return 200.
  - POST `/pricing` returns 200 for all 15 product × strategy combinations tried.
  - The demo loads, with its status line showing API, database, cache and model all ok.
- **README:** its top now links the demo, Swagger and the health check.

This closes two "still open" items from Day 16: the git repository and the cloud deploy.

---

## Part 1 — How Render, Neon and Upstash Work Together

### The three services

| Service | What it is | What it runs for SmartPrice |
|---|---|---|
| **Render** | A platform that builds code from GitHub and runs it as web services | `smartprice-api` (the Docker image) and `smartprice-demo` (Streamlit) |
| **Neon** | Managed, serverless **PostgreSQL** | The `products` and `sales_history` tables |
| **Upstash** | Managed, serverless **Redis** | The product cache (`product:<id>` keys, 5-minute TTL) |

Render is where the **code** runs. Neon and Upstash are where the **data** lives. Nothing on Render stores data, because a free Render service has no persistent disk: every restart starts from a clean container. That is why the database and cache live outside Render.

### The picture

```
   Browser (anyone)
      │ HTTPS
      ▼
┌──────────────────────── Render ────────────────────────┐
│  smartprice-demo (Streamlit, Python)                   │
│     reads SMARTPRICE_API_URL ──┐                       │
│                                │ HTTPS (server-side)   │
│                                ▼                       │
│  smartprice-api (Docker: FastAPI + Ridge model)        │
│     reads DATABASE_URL ─────┐        reads REDIS_URL ──┼──┐
└─────────────────────────────┼──────────────────────────┘  │
                              │ TCP 5432 + TLS              │ TCP 6379 + TLS
                              ▼                             ▼
                   Neon (PostgreSQL)              Upstash (Redis)
                   ep-xxx.neon.tech               xxx.upstash.io
```

### "Pasting a URL" connects them: what that actually means

A connection string is a whole set of connection instructions packed into one line:

```
postgresql://neondb_owner:SECRET@ep-xxx.eu-central-1.aws.neon.tech/neondb?sslmode=require
└── scheme ─┘└─── user ───┘└pass┘└────────────── host ──────────────┘└─db─┘└─ options ─┘

rediss://default:SECRET@xxx.upstash.io:6379
└scheme┘└─user─┘└pass┘└───── host ───┘└port┘      rediss = Redis over TLS (two s's)
```

- The **scheme** chooses the protocol and driver.
- The **user and password** prove who we are.
- The **host and port** say where the server is.
- The **database name** picks the database.
- The **options** set the security (`sslmode=require` means encrypted or nothing).

When this text is pasted into Render's **Environment** tab, the following happens, step by step:

1. **Render stores it as an environment variable** and injects it into the container's process when it starts. The value is never in the code or on GitHub: `render.yaml` marks it `sync: false`, so Render asks for it in the dashboard.

2. **pydantic-settings reads it at startup.** `Settings` (in `app/config/settings.py`) maps the environment variable `DATABASE_URL` to the field `database_url`. The same code reads `.env` locally and the real environment on Render, with no `if production:` branches.

3. **A validator translates the URL into what the async driver needs.** Neon gives `postgresql://...?sslmode=require&channel_binding=require`. SQLAlchemy's async engine needs `postgresql+asyncpg://...?ssl=require`. `normalise_database_url()` does three things:
   - rewrites the scheme;
   - renames `sslmode` to `ssl`;
   - drops `channel_binding`, an option only libpq understands, which asyncpg would reject.

   A local Docker URL passes through unchanged.

4. **The container's entry point uses it before the server starts.** `scripts/start.sh` runs these in order:
   - `alembic upgrade head`, which connects to Neon and creates or updates the tables;
   - the idempotent seed, which loads the product catalog and 14,780 sales rows only if they are missing;
   - `uvicorn`.

   So a brand-new, empty Neon database becomes ready on the first boot.

5. **SQLAlchemy opens a connection pool to Neon.** asyncpg looks up `ep-xxx...neon.tech` in DNS and opens a TCP connection on port 5432. It upgrades the connection to TLS, then logs in with the user and password.
   - Neon's proxy reads the hostname from the TLS handshake (SNI) to know which project's compute to route to.
   - If the compute was **scaled to zero** (idle for 5 minutes), Neon wakes it in a few hundred milliseconds.
   - The pool keeps connections open and lends them to requests. `pool_pre_ping` replaces any connection Neon closed while it was idle.

6. **The Redis client works the same way.** `Redis.from_url(REDIS_URL)` parses the URL, opens TCP to `xxx.upstash.io:6379`, upgrades to TLS (because of `rediss://`) and sends `AUTH default <password>`. Creating the client does not connect; the first command does. Short timeouts plus a circuit breaker mean a slow or broken Redis never blocks a request, and the API falls back to PostgreSQL.

7. **The demo finds the API through the same mechanism.** `SMARTPRICE_API_URL` is just the API's public HTTPS address. The Streamlit Python code runs **on Render's server, not in the visitor's browser**, so it calls the API server-to-server with `requests`, and CORS is not involved. The browser only ever talks to the demo.

8. **Render routes the public traffic.** Render's load balancer terminates HTTPS for `smartprice-api-n7bp.onrender.com` and forwards plain HTTP to the container on `$PORT`. That is why `start.sh` passes `--proxy-headers` to uvicorn: it trusts Render's `X-Forwarded-*` headers, so the app knows the original request was HTTPS.
   - Render checks `GET /health` before switching traffic to a new deploy.
   - A deploy that never becomes healthy never replaces the working one.

9. **Every `git push` to `main` redeploys.** Render watches the GitHub repository (`autoDeploy: true`), rebuilds the image and repeats steps 1–8.

### Why the direct Neon string, not the pooled one
Neon offers two hosts: `ep-xxx` (direct) and `ep-xxx-pooler` (through PgBouncer). The pooler in transaction mode can hand each statement to a different server connection. asyncpg caches **prepared statements** per connection, so a statement prepared on one connection may be missing on the next, and requests fail. The app already pools its own connections in SQLAlchemy, so the direct host is both correct and enough.

### Free-tier behaviour to expect
- **Render:** a service sleeps after 15 minutes with no traffic, and the next request waits about a minute while it wakes. The 750 free hours a month are shared across the account.
- **Neon:** compute scales to zero after 5 idle minutes and wakes in under a second. 1 GB of storage.
- **Upstash:** 500K commands a month.

The total cost is $0.

---

## Part 2 — Debugging Stories (Interview Format: Situation → Investigation → Fix → Lesson)

### Backend story — "The deploy said healthy, but the product page returned 500"

**Situation.** I deployed SmartPrice to Render with Neon for PostgreSQL and Upstash for Redis. The health check returned `"status": "ok"` and `"database": "ok"`, but `"cache": "unavailable"`. The Streamlit demo showed only **"Not Found"**, and `GET /api/v1/products/` returned **500 Internal server error**. That was the demo's first call, and the API's first impression.

**Investigation: one symptom at a time, cheapest check first.**

1. **The demo's "Not Found".** The demo's error message prints the exact URL it calls, and that was the clue. When the Blueprint was created, the API's URL didn't exist yet, so I had entered the *expected* name, `smartprice-api.onrender.com`. Render had added a suffix because the name was taken, so the real URL was `smartprice-api-n7bp.onrender.com`. I called the guessed host with `curl` and got **404**: someone else's service, or none. **Root cause 1:** a wrong environment variable, not a code bug.
   - While fixing it, the new URL was pasted *after* the old one instead of replacing it, giving `https://smartprice-api.onrender.comhttps://...`.
   - The error message showed this immediately. Printing the URL in the error paid off twice.

2. **The API's 500 on products.** I narrowed it down by comparing endpoints:
   - `/api/v1/models/current` returned **200**, so the app, the model and the routing were fine.
   - `/api/v1/products/` returned **500**.

   The difference between them is the dependency chain. The products route goes through `CachedProductRepository`, which builds the Redis client. The model route never touches Redis. Combined with `"cache": "unavailable"`, that pointed at Redis.

3. **"But Redis failures are supposed to be harmless."** The cache was designed so that a Redis *outage* falls back to PostgreSQL, and tests prove that for every endpoint. So why a 500? Reading the code with that question in mind, I found that the outage handling catches **runtime** errors: `RedisError` and `OSError` from a server that is down or slow. A URL that **isn't a Redis URL at all** fails earlier, when the client is created, inside the dependency, which is outside that safety net.
   - I reproduced it locally in one line: `Redis.from_url("https://xxx.upstash.io")` raises `ValueError: Redis URL must specify one of the following schemes (redis://, rediss://, unix://)`.
   - Copying Upstash's REST URL (`https://...`) or its `redis-cli --tls -u ...` command instead of the `rediss://` string produces exactly that.
   - `/health` hid it. Its Redis ping catches **every** exception and reports `"unavailable"`, so health looked like an ordinary outage while the request path crashed.

**Fix.**
- I set `REDIS_URL` to Upstash's TCP connection string (`rediss://default:<token>@xxx.upstash.io:6379`).
- I set `SMARTPRICE_API_URL` to exactly `https://smartprice-api-n7bp.onrender.com`.
- After the redeploy, `/health` showed `"cache": "ok"`, products returned 200, and all 15 product × strategy pricing calls returned 200. The demo rendered a full forecast with its explanation.
- The 500 that appeared once more afterwards came from requests made while the redeploy was still rolling out. Repeating the calls after it finished gave 200 every time.

**Lessons.**
- **Narrow down by comparing endpoints.** One route working and one failing tells you which dependency to suspect, before reading any logs.
- **Error messages should include the input that failed.** "Cannot reach the API at `<url>`" diagnosed two configuration mistakes in seconds.
- **A health check must fail the same way the real path fails.** Here it was more forgiving than the request path, which turned a crash into what looked like an outage.
- **Graceful degradation has to cover misconfiguration, not just outages.** The improvement this points to: validate `REDIS_URL` at startup and fall back to "no cache" with a loud warning, instead of a 500 on every product request.
- **Most production incidents are configuration, not code.** Code had 522 tests. The two environment variables had none.

### AI/ML story — "Every detector scored 0.00 on flatlines, and the fix exposed a floating-point bug"

**Situation.** I was evaluating anomaly detectors on real sales data. Real data has no anomaly labels, so I **injected** labelled anomalies into the test window:
- 30 spikes (4–8× normal);
- 30 drops;
- 15 "flatlines", where a broken data feed repeats the same number for days.

After the first run, the two numbers that stood out were **flatline recall 0.00 for every detector**, and spike recall of only about 0.5.

**Investigation: on validation data only.** I had already seen the test numbers once, so tuning on them would have been cheating. Every diagnosis from here used the **validation window**.

1. **Spikes.** The median |z-score| of an injected spike was **1.71**, while the 95th percentile for *normal* days was **2.10**. A wholesaler's ordinary bulk orders make a 4–8× day look normal. The spike definition had been fixed before any results, so I **kept it**: making spikes bigger to get a better score would have been cheating. I reported it as a limitation instead.

2. **Flatlines.** A stuck feed repeats a *normal-looking* number. Any "how far is today from normal?" score, whether z-score, IQR or Isolation Forest, cannot see it by design. So I looked for a signal that *does* separate them. "The last 4 open days are identical" was true for **7 of 15** flatline rows and **0 of 1,465** normal rows: a near-perfect signal that no detector was using.

3. **The fix.** I added a fourth detector, **rolling z-score OR "last 4 open days identical"**, justified by validation evidence alone.

**The second bug, caught by a unit test.** The first version of the "identical" rule checked `rolling std < 1e-9`. A unit test fed it a stuck value and **failed**: pandas computes rolling standard deviation with running sums, and identical values came out as **2.8 × 10⁻⁸**, not 0. The rule was silently missing flatlines. I replaced it with the **range (max − min) of integer unit counts**, which is exactly 0 when the values are identical. No tolerance to tune, and no floating-point residue.

**Fix and test-set hygiene.**
- I re-ran the evaluation on a **fresh random draw of test anomalies (seed 3)** that the new design was never tuned on, and stated this in the report.
- **Result:** F1 rose from 0.429 to **0.476**, and flatline recall from 0.20 to **0.60**.
- Isolation Forest used the same corrected feature, and its F1 *changed* (0.454 → 0.400). That was a reminder that small feature details move model results.

**Lessons.**
- **Read the zeros.** A metric of exactly 0.00 usually means a detector *cannot see* that kind of anomaly, not that it saw it badly.
- **Diagnose on validation and score on test once.** Once test data has been seen, re-draw it before reporting.
- **Don't move the goalposts.** Making spikes easier to detect would have improved the number and made it meaningless.
- **Floating-point equality is a trap.** For "are these identical?", compare exact integers rather than a float statistic against a tolerance.
- **Unit tests with hand-built inputs catch what aggregate metrics hide.** The metric only said "recall is a bit low"; the test said exactly why.

---

## Interview Questions — Day 17

**Q: How does your API know where the database is?**
From the `DATABASE_URL` environment variable, which Render injects into the container. pydantic-settings reads it, a validator converts Neon's string to the asyncpg format, Alembic migrates the database on boot, and SQLAlchemy keeps a TLS connection pool to Neon. Nothing about the environment is hard-coded.

**Q: Why is the database not on Render?**
A free Render service has no persistent disk, and every deploy starts a fresh container. State belongs in managed services built for it: Neon for PostgreSQL, Upstash for Redis. The compute is stateless and can be replaced at any time.

**Q: Tell me about a production issue you debugged.**
The backend story above: a healthy-looking deploy where products returned 500. I narrowed it down by comparing a working endpoint with a failing one, reproduced the cause in one line locally, and found that my health check was more forgiving than the real request path.

**Q: Tell me about an ML result that surprised you.**
The AI/ML story above: 0.00 flatline recall for every detector. It led to a new detector feature found on validation data, a floating-point bug caught by a unit test, and a fresh test draw so the reported number stayed honest.
