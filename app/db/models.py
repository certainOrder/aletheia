"""SQLAlchemy 2.0 typed ORM models for Aletheia.

Defines `MemoryShard`, which stores content, tags, user association, and a pgvector
embedding used for semantic search and RAG context retrieval.
"""

import uuid
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import EMBEDDING_DIM


class Base(DeclarativeBase):
    pass


class MemoryShard(Base):
    __tablename__ = "memory_shards"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_accessed: Mapped[Optional[str]] = mapped_column(DateTime(timezone=True), nullable=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_ids: Mapped[Optional[list[uuid.UUID]]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    importance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    priority_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retention_policy: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # New in M3: content provenance and flexible metadata
    source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - repr aid
        return f"<MemoryShard id={self.id} user_id={self.user_id}>"


class UserProfile(Base):
    """Minimal user profile table to link external users to UUIDs used in shards.

    This enables scoping and satisfies potential DB-level foreign keys. In dev, the table
    will be auto-created via Base.metadata.create_all; in production, prefer migrations.
    """

    __tablename__ = "user_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Original external identifier (e.g., OpenWebUI username or id)
    external_id: Mapped[Optional[str]] = mapped_column(Text, unique=True, nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - repr aid
        return f"<UserProfile id={self.id} external_id={self.external_id}>"
