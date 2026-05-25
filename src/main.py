"""FastAPI application entry point.

Run with:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000/docs       — interactive Swagger UI
    http://localhost:8000/api/v1/...  — endpoints
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import admin, mastery, prep, scenarios, sections, sessions
from src.config import settings


API_PREFIX = "/api/v1"


app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    docs_url=settings.api.docs_url,
    redoc_url=settings.api.redoc_url,
    openapi_url=settings.api.openapi_url,
    description=(
        "Backend for an LLM-driven MCQ preparation system with "
        "history-adaptive question generation. "
        "Built on FastAPI + LangGraph + LangChain + ChromaDB + PostgreSQL."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ──────────────────────────────────────────────────────────


app.include_router(sections.router, prefix=API_PREFIX)
app.include_router(prep.router, prefix=API_PREFIX)
app.include_router(sessions.router, prefix=API_PREFIX)
app.include_router(mastery.router, prefix=API_PREFIX)
app.include_router(scenarios.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)


# ── Health + global error handler ────────────────────────────────────


@app.get(f"{API_PREFIX}/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app.name,
        "version": settings.app.version,
        "environment": settings.app.environment,
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "path": str(request.url.path),
        },
    )
