"""Chat-related request schemas for the API endpoints.

These Pydantic models define inputs for single-turn and RAG chat flows, and for
indexing content into memory shards.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Single-turn chat prompt request."""

    prompt: str


class RAGChatRequest(BaseModel):
    """RAG chat request containing prompt and retrieval parameters."""

    prompt: str
    user_id: str | None = None
    top_k: int = 5


class IndexMemoryRequest(BaseModel):
    """Index content into the vector store for future retrieval."""

    content: str
    user_id: str | None = None
    tags: list[str] | None = None
