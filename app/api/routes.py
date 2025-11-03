"""Secondary API routes mounted under the `/api` prefix.

Currently provides a health check endpoint for liveness/readiness probes.
Optionally exposes a debug search endpoint for development when enabled
via the `ENABLE_DEBUG_ENDPOINTS` environment variable.
"""

import os

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.db import get_db
from app.utils.embeddings import convert_to_embedding, semantic_search

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy"}


# Opt-in debug search endpoint (disabled by default)
if (os.getenv("ENABLE_DEBUG_ENDPOINTS", "false") or "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}:

    @router.get("/debug/search")
    async def debug_search(
        q: str = Query(..., description="Query text to embed and search"),
        top_k: int = Query(5, ge=1, le=50),
        user_id: str | None = Query(None, description="Optional user scope"),
        db: Session = Depends(get_db),
    ):
        emb = convert_to_embedding(q)
        results = semantic_search(db, emb, user_id=user_id, limit=top_k)
        return {"q": q, "top_k": top_k, "user_id": user_id, "results": results}

    @router.get("/debug/headers")
    async def debug_headers(request: Request):
        from app.logging_utils import mask_headers

        return {"headers": mask_headers(request.headers, max_value_len=1024)}
