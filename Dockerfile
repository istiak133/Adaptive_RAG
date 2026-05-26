# =====================================================================
# Adaptive Document Preparation System — application image
# =====================================================================
# Two-stage build keeps the runtime image small while letting wheel
# compilation (psycopg2, tokenizers, etc.) happen in a fat builder.
# =====================================================================

# ── Stage 1: builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps for psycopg2, sentence-transformers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --user -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/app"

# Runtime libs only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder
COPY --from=builder /root/.local /root/.local

WORKDIR /app

# Application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY config.yaml ./
COPY data/ ./data/

# Folders the app expects to exist at runtime
RUN mkdir -p /app/kb/chromadb /app/logs /app/outputs

EXPOSE 8000

# Default to API server; override with `docker compose run app <cmd>`
# for one-shot CLI invocations.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
