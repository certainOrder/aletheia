from typing import List, Optional
from openai import OpenAI
from sqlalchemy.orm import Session
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from app.db.models import MemoryShard


def convert_to_embedding(text_input: str) -> List[float]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    result = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text_input)
    return result.data[0].embedding  # list[float]


def save_embedding_to_db(
    *,
    db: Session,
    content: str,
    embedding: List[float],
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


def semantic_search(db: Session, query_embedding: List[float], user_id: Optional[str] = None, limit: int = 5):
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