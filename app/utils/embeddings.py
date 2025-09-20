import hashlib
import logging
import random
from typing import Any, Optional, cast

from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import DEV_FALLBACKS, EMBEDDING_DIM, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from app.db.models import MemoryShard


def convert_to_embedding(text_input: str) -> list[float]:
    """
    Convert text to an embedding using OpenAI. If authentication fails or
    OpenAI is unreachable, fall back to a deterministic local embedding so
    development smoke tests can proceed without a real key.
    """
    try:
        if not DEV_FALLBACKS:
            client = OpenAI(api_key=OPENAI_API_KEY)
            # The OpenAI SDK types model as a Literal of known models; our config is a str.
            # Cast to Any here to satisfy mypy while keeping runtime flexibility.
            result = client.embeddings.create(
                model=cast(Any, OPENAI_EMBEDDING_MODEL), input=text_input
            )
            return result.data[0].embedding  # list[float]
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
        return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


def save_embedding_to_db(
    *,
    db: Session,
    content: str,
    embedding: list[float],
    user_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
):
    shard = MemoryShard(
        content=content,
        embedding=embedding,
        user_id=user_id,
        tags=tags,
    )
    db.add(shard)
    db.commit()
    db.refresh(shard)
    return shard


def semantic_search(
    db: Session, query_embedding: list[float], user_id: Optional[str] = None, limit: int = 5
):
    q = db.query(MemoryShard)
    if user_id:
        q = q.filter(MemoryShard.user_id == user_id)
    q = q.order_by(MemoryShard.embedding.l2_distance(query_embedding)).limit(limit)
    rows = q.all()
    return [
        {
            "id": str(r.id),
            "content": r.content,
            "user_id": str(r.user_id) if r.user_id else None,
            "tags": r.tags,
            "distance": None,  # distance not returned by ORM; could add with annotate
        }
        for r in rows
    ]
