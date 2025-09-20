"""Structured logging and request ID propagation utilities.

This module configures JSON-style logging and provides a middleware to attach a
request ID (trace ID) to each request, making logs and error responses easier to
correlate. Logging uses Python's stdlib logging to keep dependencies minimal.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

_request_id_ctx_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Return the current request ID from context, if any."""

    return _request_id_ctx_var.get()


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with level, message, time, and request_id."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - succinct
        payload: dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "time": int(time.time()),
            "logger": record.name,
        }
        # Optional extras
        req_id = get_request_id()
        if req_id:
            payload["request_id"] = req_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a JSON formatter at the provided level."""

    root = logging.getLogger()
    # Avoid duplicate handlers if called multiple times
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to each request and log basic request/response info."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        req_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        token = _request_id_ctx_var.set(req_id)
        # Store on request.state for access in exception handlers (state is dynamic)
        try:
            setattr(request.state, "request_id", req_id)
        except Exception:
            pass
        start = time.time()
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx_var.reset(token)
        duration_ms = int((time.time() - start) * 1000)
        # Add header to response for client correlation
        response.headers[self.header_name] = req_id
        logging.getLogger("app").info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": getattr(response, "status_code", 0),
                "duration_ms": duration_ms,
            },
        )
        return response
