"""Pydantic response schemas for API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IndexMemoryResponse(BaseModel):
    """Response for `/index-memory`."""

    id: str = Field(..., description="UUID of the created memory shard")


class IngestResponse(BaseModel):
    """Response for `/ingest` containing IDs of created shards."""

    ids: list[str] = Field(..., description="UUIDs of created memory shards")


class RetrievedContextItem(BaseModel):
    id: str
    content: str
    user_id: str | None = None
    tags: list[str] | None = None
    # New in M3: richer retrieval metadata
    score: float | None = None
    source: str | None = None
    metadata: dict | None = None
    # Back-compat: previously used 'distance'; keep optional for older clients
    distance: float | None = None


class RAGChatResponse(BaseModel):
    """Response for `/rag-chat` containing answer and retrieval context."""

    answer: str
    context: list[RetrievedContextItem]


class ErrorResponse(BaseModel):
    """Standard error envelope returned by exception handlers."""

    error: str
    detail: Any
    status: int
    request_id: str
