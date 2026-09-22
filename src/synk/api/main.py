"""SYNK FastAPI application.

Creates the app, registers routers, installs uniform error handling and CORS,
and mounts interactive docs at /docs.  All clinical-safety disclaimers are
attached at the schema level; failures never leak stack traces to clients.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from synk.api.routers import core, research
from synk.api.schemas import ErrorResponse
from synk.api.state import init_state
from synk.config.settings import SYNKConfig
from synk.utils.logging import configure_logging

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
**SYNK** - Clinical Intelligence Before Critical Moments.

An explainable multimodal AI system for early sepsis risk stratification using
ICU physiological time-series and clinical documentation.

> **Research prototype - not for clinical diagnosis or treatment.**
> All served data is synthetic demonstration data unless a researcher has
> explicitly configured an authorised local dataset.
"""


def create_app(config: SYNKConfig | None = None) -> FastAPI:
    configure_logging((config.log_level if config else "INFO").upper())

    app = FastAPI(
        title="SYNK API",
        description=API_DESCRIPTION,
        version="0.1.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    cfg = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins if cfg else ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        state = init_state(cfg)
        logger.info(
            "SYNK API ready | engine=%s | synthetic data=%s",
            state.service.engine_id if state.service else "lazy",
            state.config.data.dataset_id,
        )

    # ---- uniform error handling -----------------------------------------
    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content=ErrorResponse(error="invalid_input", detail=str(exc)).model_dump())

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Log the traceback server-side; return a safe message to the client.
        logger.error("Unhandled API error: %s\n%s", exc, traceback.format_exc())
        return JSONResponse(status_code=500, content=ErrorResponse(error="internal_error",
                                                                   detail="Unexpected server error.").model_dump())

    app.include_router(core.router)
    app.include_router(research.router)

    @app.get("/", tags=["system"])
    def root() -> dict[str, Any]:
        return {
            "name": "SYNK API",
            "tagline": "Clinical Intelligence Before Critical Moments.",
            "docs": "/docs",
            "disclaimer": "Research prototype - not for clinical diagnosis or treatment.",
        }

    return app


app = None  # created via create_app(); `python -m scripts.cli serve` builds it
