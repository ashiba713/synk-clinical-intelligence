# SYNK - CPU research image.
#
# Build:  docker build -t synk:latest .
# Run:    docker compose up    (API on :8000, dashboard on :8501)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SYNK_ENV=development

WORKDIR /app

# System deps (lightweight: CPU torch only needs libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Project code.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY configs/ ./configs/
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e .

# Generate demo data + a tiny model at build time so the containers are
# immediately usable; runtime volumes override artifacts when provided.
COPY tests/ ./tests/
RUN python -m scripts.cli generate-data --n-patients 40

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["python", "-m", "scripts.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
