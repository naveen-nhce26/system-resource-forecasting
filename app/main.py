"""
FastAPI application entrypoint (Phase 1).
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)


def create_app() -> FastAPI:
    """
    FastAPI application factory.
    """

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Production-grade backend for state-wise sales forecasting.",
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        start = time.time()
        status_code: int | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed_ms = (time.time() - start) * 1000.0
            logger.info(
                "request method=%s path=%s status=%s elapsed_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # Normalize errors to API-safe JSON shape
        logger.warning("HTTP error on %s %s: %s", request.method, request.url.path, exc.detail)
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )

    @app.get("/health", tags=["system"])
    def health() -> dict:
        """
        Health check endpoint.
        """

        return {"status": "healthy"}

    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()

