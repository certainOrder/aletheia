"""Embedding utilities and simple vector search helpers.

Provides a deterministic local embedding fallback for development and tests.
"""

import hashlib
import logging
import math
import random
from typing import Any, Optional, cast

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import (
    DEV_FALLBACKS,
    EMBEDDING_DIM,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    SIMILARITY_METRIC,
)
from app.db.models import MemoryShard

logger = logging.getLogger("app.flow")


def convert_to_embedding(text_input: str) -> list[float]:
    """
    Convert text to an embedding using OpenAI. If authentication fails or
    OpenAI is unreachable, fall back to a deterministic local embedding so
    development smoke tests can proceed without a real key.
    """
    try:
        logger.info(
            "embed_request",
            extra={
                "text_len": len(text_input or ""),
                "provider": (
                    "openai" if (not DEV_FALLBACKS and OPENAI_API_KEY) else "local_fallback"
                ),
            },
        )
        if not DEV_FALLBACKS:
            client = OpenAI(api_key=OPENAI_API_KEY)
            # The OpenAI SDK types model as a Literal of known models; our config is a str.
            # Cast to Any here to satisfy mypy while keeping runtime flexibility.
            result = client.embeddings.create(
                model=cast(Any, OPENAI_EMBEDDING_MODEL), input=text_input
            )
            emb = result.data[0].embedding  # list[float]
            logger.info("embed_result", extra={"dim": len(emb), "provider": "openai"})
            return emb
        else:
            raise RuntimeError("DEV_FALLBACKS enabled")
    except Exception as e:  # fallback for local/dev without keys
        if not DEV_FALLBACKS:
            logging.warning(
                "Falling back to local deterministic embedding due to embedding provider error: %s",
                e,
            )
        # Deterministic embedding based on SHA256 of input
        h = hashlib.sha256(text_input.encode("utf-8")).hexdigest()
        seed = int(h[:16], 16)
        rng = random.Random(seed)
        # Generate EMBEDDING_DIM floats in [-1.0, 1.0]
        emb = [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]
        logger.info("embed_result", extra={"dim": len(emb), "provider": "local_fallback"})
        return emb


def save_embedding_to_db(
    *,
    db: Session,
    content: str,
    embedding: list[float],
    user_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    source: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    shard = MemoryShard(
        content=content,
        embedding=embedding,
        user_id=user_id,
        tags=tags,
        source=source,
        metadata_json=metadata,
    )
    db.add(shard)
    db.commit()
    db.refresh(shard)
    logger.info(
        "embedding_saved",
        extra={
            "shard_id": str(shard.id),
            "content_len": len(content or ""),
            "tags_count": len(tags or []),
            "has_source": bool(source),
            "metadata_keys": list((metadata or {}).keys()),
        },
    )
    return shard


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors; safe for zero norms."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def semantic_search(
    db: Session, query_embedding: list[float], user_id: Optional[str] = None, limit: int = 5
):
    """Return top-k memory shards ordered by configured similarity metric.
    Uses DB distance for ordering and derives score from that distance when possible.
    """
    metric = (SIMILARITY_METRIC or "cosine").lower()

    # First, try DB-side distance annotation for robust ordering and scoring
    try:
        if metric == "cosine":
            dist_expr = MemoryShard.embedding.cosine_distance(query_embedding)
        else:
            dist_expr = MemoryShard.embedding.l2_distance(query_embedding)

        stmt = select(MemoryShard, dist_expr.label("distance"))
        if user_id:
            stmt = stmt.where(MemoryShard.user_id == user_id)
        stmt = stmt.order_by(dist_expr).limit(limit)

        rows = db.execute(stmt).all()
        # rows are tuples: (MemoryShard, distance)
        results: list[dict[str, Any]] = []
        for shard, dist in rows:
            if metric == "cosine" and dist is not None:
                # Convert distance d in [0, 2] to similarity score s in [-1, 1]; clamp for safety
                try:
                    s = 1.0 - float(dist)
                except Exception:
                    s = None
            else:
                s = None
            results.append(
                {
                    "id": str(shard.id),
                    "content": shard.content,
                    "user_id": str(shard.user_id) if shard.user_id else None,
                    "tags": shard.tags,
                    "source": shard.source,
                    "metadata": shard.metadata_json,
                    "score": s,
                }
            )

        logger.info(
            "semantic_search",
            extra={
                "limit": limit,
                "top_k": limit,
                "user_id": user_id,
                "result_count": len(results),
                "metric": metric,
            },
        )

        # Log compact score summary for observability
        try:
            logger.info(
                "retrieval_scores",
                extra={
                    "metric": metric,
                    "scores": [r.get("score") for r in results],
                    "ids": [r.get("id") for r in results],
                },
            )
            # Debug-level per-item score visibility
            for r in results:
                logger.debug(
                    "retrieval_item",
                    extra={
                        "id": r.get("id"),
                        "score": r.get("score"),
                        "content_len": len(r.get("content") or ""),
                    },
                )
        except Exception:  # pragma: no cover
            pass

        return results
    except Exception:
        # Fallback: fetch instances and compute local cosine scores for ordering.
        # Support both real SQLAlchemy Session and the DummySession used in tests.
        shards: list[MemoryShard]
        try:
            if hasattr(db, "execute"):
                stmt2 = select(MemoryShard)
                if user_id:
                    stmt2 = stmt2.where(MemoryShard.user_id == user_id)
                shards = list(db.execute(stmt2.limit(limit * 4)).scalars().all())
            elif hasattr(db, "query"):
                # Duck-typed path for DummySession in tests (treat db as Any for mypy)
                db_any: Any = db
                q = db_any.query(MemoryShard)
                if user_id:
                    q = q.filter(MemoryShard.user_id == user_id)
                if hasattr(q, "limit"):
                    q = q.limit(limit * 4)
                shards = list(q.all())
            else:  # pragma: no cover - extremely defensive
                shards = []
        except Exception:
            shards = []
        scored: list[tuple[Optional[float], MemoryShard]] = []
        for shard in shards:
            if metric == "cosine":
                try:
                    sc = float(_cosine_similarity(query_embedding, shard.embedding))
                except Exception:
                    sc = None
            else:
                sc = None
            scored.append((sc, shard))

        # Sort by score descending where available
        if any(sc is not None for sc, _ in scored):
            scored.sort(key=lambda t: (t[0] is None, -(t[0] or 0.0)))
        scored = scored[:limit]

        results_out: list[dict[str, Any]] = []
        for sc, shard in scored:
            results_out.append(
                {
                    "id": str(shard.id),
                    "content": shard.content,
                    "user_id": str(shard.user_id) if shard.user_id else None,
                    "tags": shard.tags,
                    "source": shard.source,
                    "metadata": shard.metadata_json,
                    "score": sc,
                }
            )

        logger.info(
            "semantic_search_fallback",
            extra={
                "limit": limit,
                "top_k": limit,
                "user_id": user_id,
                "result_count": len(results_out),
                "metric": metric,
            },
        )
        try:
            logger.info(
                "retrieval_scores",
                extra={
                    "metric": metric,
                    "scores": [r.get("score") for r in results_out],
                    "ids": [r.get("id") for r in results_out],
                },
            )
            for r in results_out:
                logger.debug(
                    "retrieval_item",
                    extra={
                        "id": r.get("id"),
                        "score": r.get("score"),
                        "content_len": len(r.get("content") or ""),
                    },
                )
        except Exception:  # pragma: no cover
            pass
        return results_out
