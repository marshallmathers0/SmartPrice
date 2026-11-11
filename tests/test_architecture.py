"""
Day 5 — Architecture rules.

The business layer must never depend on the web framework or the API layer.
These tests read every business module's imports, so the moment someone adds
`from fastapi import HTTPException` to a service, the suite fails.
"""
import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

BUSINESS_PACKAGES = [
    "models",
    "exceptions",
    "repositories",
    "services",
    "strategies",
    "guardrails",
    "factories",
    "ml",           # Phase A: ML code must not depend on the web layer or the database either
]

# Web framework + our own HTTP layer
FORBIDDEN_IMPORTS = ["fastapi", "starlette", "app.api", "app.schemas"]

BUSINESS_MODULES = sorted(
    path for package in BUSINESS_PACKAGES for path in (APP_DIR / package).rglob("*.py")
)


def imported_modules(path: Path) -> set[str]:
    """Every module name a file imports (import x / from x import y)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def is_forbidden(module: str, banned_modules: list[str]) -> bool:
    return any(module == banned or module.startswith(banned + ".") for banned in banned_modules)


def test_business_modules_were_found():
    # Guards against the parametrized test below silently checking nothing
    assert len(BUSINESS_MODULES) >= 20


@pytest.mark.parametrize(
    "path",
    BUSINESS_MODULES,
    ids=[path.relative_to(APP_DIR).as_posix() for path in BUSINESS_MODULES],
)
def test_business_layer_does_not_import_web_or_api_layer(path):
    violations = sorted(module for module in imported_modules(path) if is_forbidden(module, FORBIDDEN_IMPORTS))

    assert violations == [], f"{path.relative_to(APP_DIR)} imports {violations}"


# ------------------------------------------------------------------
# Day 6 — Persistence Ignorance
# ------------------------------------------------------------------
# Only concrete repositories may touch SQLAlchemy or the database package.
# The domain model, the service, strategies, guards — and even the
# ProductRepository abstraction — must not know a database exists.
# ------------------------------------------------------------------
#
# Day 7: the same rule covers caching — only the caching decorator may
# know that a cache (Redis) exists.
PERSISTENCE_IMPORTS = ["sqlalchemy", "asyncpg", "app.database", "redis", "app.cache"]

DATABASE_ADAPTERS = {
    APP_DIR / "repositories" / "postgres.py",   # talks to PostgreSQL
    APP_DIR / "repositories" / "cached.py",     # talks to the product cache
}

PERSISTENCE_IGNORANT_MODULES = [path for path in BUSINESS_MODULES if path not in DATABASE_ADAPTERS]


@pytest.mark.parametrize(
    "path",
    PERSISTENCE_IGNORANT_MODULES,
    ids=[path.relative_to(APP_DIR).as_posix() for path in PERSISTENCE_IGNORANT_MODULES],
)
def test_business_layer_does_not_import_persistence(path):
    violations = sorted(module for module in imported_modules(path) if is_forbidden(module, PERSISTENCE_IMPORTS))

    assert violations == [], f"{path.relative_to(APP_DIR)} imports {violations}"


# ------------------------------------------------------------------
# Log messages must be ASCII
# ------------------------------------------------------------------
# On Windows, a console or redirected stdout often uses the cp1252
# encoding; a log message containing "—" or "≥" then fails to print
# ("--- Logging error ---") and the line is lost.
# ------------------------------------------------------------------
ALL_MODULES = sorted(APP_DIR.rglob("*.py"))


def log_messages(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, node.args[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.endswith("logger")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


@pytest.mark.parametrize("path", ALL_MODULES, ids=[p.relative_to(APP_DIR).as_posix() for p in ALL_MODULES])
def test_log_messages_are_ascii(path):
    bad = [f"line {line}" for line, message in log_messages(path) if not message.isascii()]

    assert bad == [], f"{path.relative_to(APP_DIR)}: non-ASCII log message at {bad}"


# ------------------------------------------------------------------
# Phase 3 — training tools never load in the API
# ------------------------------------------------------------------
# requirements.txt (the production image) does not contain MLflow,
# LightGBM, matplotlib or openpyxl. If starting the API — including
# loading the deployed model — imported any of them, production would crash.
# A fresh interpreter is used so other tests' imports cannot hide a problem.
# ------------------------------------------------------------------
TRAINING_ONLY = ("mlflow", "lightgbm", "matplotlib", "openpyxl", "shap", "pytest")


def test_api_startup_never_imports_training_tools():
    import subprocess
    import sys

    probe = (
        "import sys, app.main\n"
        "from app.ml.model_store import ModelNotAvailableError, load_bundle\n"
        "try:\n"
        "    load_bundle()\n"
        "except ModelNotAvailableError:\n"
        "    pass\n"
        f"print('IMPORTED=' + ','.join(m for m in {TRAINING_ONLY!r} if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=APP_DIR.parent, timeout=120)

    assert result.returncode == 0, result.stderr
    imported = next(line for line in result.stdout.splitlines() if line.startswith("IMPORTED="))
    assert imported == "IMPORTED=", f"API startup imported training tools: {imported}"


# ------------------------------------------------------------------
# Phase 6 — every module compiles without warnings
# ------------------------------------------------------------------
# Found by running the Docker image: a docstring diagram contained "\-",
# an invalid escape sequence. Python 3.12 warns at import time; future
# versions will make it a SyntaxError — i.e. the app would stop starting.
# ------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL_MODULES, ids=[p.relative_to(APP_DIR).as_posix() for p in ALL_MODULES])
def test_module_compiles_without_warnings(path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # turn SyntaxWarning (invalid escapes etc.) into a failure
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
