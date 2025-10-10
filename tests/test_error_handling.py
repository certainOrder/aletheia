from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.testclient import TestClient

from app.main import app

router = APIRouter()


@router.get("/boom")
def boom() -> dict[str, Any]:
    raise HTTPException(status_code=418, detail={"reason": "teapot"})


@router.get("/panic")
def panic() -> dict[str, Any]:  # pragma: no cover - exception path counted elsewhere
    raise RuntimeError("kaboom")


app.include_router(router)


def test_http_exception_shaped_response(client):
    r = client.get("/boom", headers={"X-Request-ID": "abc-123"})
    assert r.status_code == 418
    data = r.json()
    assert data["error"] == "HTTPException"
    assert data["status"] == 418
    assert data["detail"] == {"reason": "teapot"}
    # request id echoed back
    assert data["request_id"] == "abc-123"
    assert r.headers.get("X-Request-ID") == "abc-123"


def test_unhandled_exception_returns_500():
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/panic")
    assert r.status_code == 500
    data = r.json()
    assert data["error"] == "InternalServerError"
    assert data["status"] == 500
    assert isinstance(data["request_id"], str) and data["request_id"]
