"""FastAPI application entry point.

Run with:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000/docs       — interactive Swagger UI
    http://localhost:8000/api/v1/health  — health check
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings


app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    docs_url=settings.api.docs_url,
    redoc_url=settings.api.redoc_url,
    openapi_url=settings.api.openapi_url,
    description=(
        "Backend for an LLM-driven MCQ preparation system with "
        "history-adaptive question generation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app.name,
        "version": settings.app.version,
        "environment": settings.app.environment,
    }
