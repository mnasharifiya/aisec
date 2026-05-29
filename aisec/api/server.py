"""
AISec REST API server.

Starts a FastAPI application that exposes the AISec analysis
engine over HTTP. Enterprise teams can integrate AISec into
any language or platform without using the Python SDK.

Usage:
    aisec serve --host 0.0.0.0 --port 8000
    aisec serve --log-path /var/log/aisec/audit.jsonl

Docker:
    docker run -p 8000:8000 aisec serve

API documentation available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
    http://localhost:8000/openapi.json
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aisec.api.routes import router
from aisec.core.engine import AnalysisEngine
from aisec.storage.audit import DEFAULT_LOG_PATH
from aisec.utils.logger import configure_logging, get_logger

log = get_logger("aisec.api.server")


# ── Application factory ───────────────────────────────────────────────────────


def create_app(
    log_path: Path = DEFAULT_LOG_PATH,
    enable_cors: bool = True,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create and configure the AISec FastAPI application.

    Args:
        log_path:        Path to the audit log file.
        enable_cors:     Enable CORS middleware.
        allowed_origins: List of allowed CORS origins.
                         Defaults to ["*"] if None — restrict in production.

    Returns:
        Configured FastAPI application.
    """
    configure_logging(level="INFO", output="stdout")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Startup and shutdown lifecycle manager."""
        # Startup
        log.info(
            "aisec_api_starting",
            log_path=str(log_path),
            version="1.2.0",
        )

        app.state.engine = AnalysisEngine(log_path=log_path)

        ok, errors = app.state.engine.verify_audit_chain()
        if not ok:
            log.warning(
                "audit_chain_broken_at_startup",
                error_count=len(errors),
                first_error=errors[0] if errors else "unknown",
            )
        else:
            log.info(
                "audit_chain_verified",
                entry_count=app.state.engine.audit_count(),
            )

        log.info("aisec_api_ready")
        yield

        # Shutdown
        log.info("aisec_api_shutting_down")

    app = FastAPI(
        title="AISec — AI Runtime Security API",
        description=(
            "Enterprise REST API for runtime security monitoring "
            "of autonomous AI agents.\n\n"
            "**AISec** intercepts AI agent actions, scores them for risk, "
            "enforces policy decisions, and maintains a tamper-evident audit trail.\n\n"
            "Built on: *A Layered Cybersecurity Framework for Enforcing "
            "Human Control over Advanced Autonomous Systems* — "
            "Muhammad Muttaka, AITU 2025."
        ),
        version="1.2.0",
        contact={
            "name": "Muhammad Muttaka",
            "email": "255902@astanait.edu.kz",
            "url": "https://github.com/MNasharifiya/aisec",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    if enable_cors:
        origins = allowed_origins or ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.error(
            "unhandled_exception",
            exc_type=type(exc).__name__,
            path=str(request.url.path),
            detail=str(exc)[:200],
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred. Action blocked by default.",
                "code": 500,
            },
        )

    # Include API routes
    app.include_router(router, prefix="/api/v1")

    return app
