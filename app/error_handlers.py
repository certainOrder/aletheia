"""Centralized exception handlers for FastAPI.

Handlers produce consistent JSON with fields: error, detail, status, request_id.
They avoid leaking internals while remaining developer-friendly in logs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging_utils import get_request_id

logger = logging.getLogger("app.errors")


def _payload(
    message: str,
    status: int,
    detail: Any | None = None,
    request: Request | None = None,
) -> dict[str, Any]:
    return {
        "error": message,
        "detail": detail,
        "status": status,
        "request_id": (
            (
                request.state.request_id
                if request is not None and hasattr(request.state, "request_id")
                else None
            )
            or (request.headers.get("X-Request-ID") if request is not None else None)
            or get_request_id()
        ),
    }


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    logger.info(
        "http_exception",
        extra={"status": exc.status_code, "detail": exc.detail},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload("HTTPException", exc.status_code, exc.detail, request),
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("validation_error", extra={"errors": exc.errors()})
    return JSONResponse(
        status_code=422,
        content=_payload("ValidationError", 422, exc.errors(), request),
    )


async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    # Log exception with stack for observability; return sanitized message
    logger.exception("unhandled_exception")
    return JSONResponse(
        status_code=500,
        content=_payload(
            "InternalServerError",
            500,
            "An unexpected error occurred.",
            request,
        ),
    )
