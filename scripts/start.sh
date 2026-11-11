#!/bin/sh
# Container entry point: make a fresh database ready, then serve.
#
#   1. alembic upgrade head        create / update the schema (no-op if already current)
#   2. python -m app.database.seed load demo products + sales history (idempotent: skips existing rows)
#   3. uvicorn                     serve on $PORT (Render and most hosts set it; default 8000)
#
# exec replaces this shell with uvicorn, so the server receives stop signals directly
# and shuts down gracefully (the lifespan closes database and Redis connections).
set -e

echo "Applying database migrations..."
alembic upgrade head

echo "Seeding demo data (idempotent)..."
python -m app.database.seed

echo "Starting SmartPrice API on port ${PORT:-8000}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips="*"
