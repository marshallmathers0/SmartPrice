# SmartPrice API — production image.
#
#   docker build -t smartprice-api .
#   docker run -p 8000:8000 -e DATABASE_URL=... -e REDIS_URL=... smartprice-api
#
# Slim on purpose (free hosting tiers have ~512 MB of RAM):
#   - python:3.12-slim base (Debian without compilers or docs)
#   - only requirements.txt — the API's runtime needs. Training tools
#     (MLflow, LightGBM, SHAP, matplotlib, pytest) are in requirements-dev.txt
#     and never enter this image.
#   - every dependency ships as a pre-built wheel, so no build stage is needed
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Dependencies first: this layer is reused on every rebuild until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code, migrations, the versioned model and the processed demo data (for the seed)
COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY models ./models
COPY data/processed ./data/processed
COPY scripts/start.sh ./scripts/start.sh

# Never run as root inside the container
RUN useradd --create-home --uid 10001 smartprice && chmod +x scripts/start.sh
USER smartprice

EXPOSE 8000

# Docker-level health check (Render uses its own check on /health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# Migrations -> seed -> server (see scripts/start.sh)
CMD ["./scripts/start.sh"]
