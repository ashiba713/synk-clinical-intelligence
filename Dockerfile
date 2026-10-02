# SYNK - Streamlit dashboard for Render

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SYNK_ENV=development \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Render-specific dependencies
COPY requirements-render.txt ./
RUN pip install --no-cache-dir -r requirements-render.txt

# CPU-only PyTorch — avoids CUDA/NVIDIA packages
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY pyproject.toml ./

RUN pip install --no-cache-dir --no-deps -e .

COPY tests/ ./tests/

RUN python -m scripts.cli generate-data --n-patients 40

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false"]