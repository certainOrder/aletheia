import uuid
from sqlalchemy import Column, Text, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from pgvector.sqlalchemy import Vector
from app.config import EMBEDDING_DIM


Base = declarative_base()


class MemoryShard(Base):
    __tablename__ = "memory_shards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed = Column(DateTime(timezone=True))
    user_id = Column(UUID(as_uuid=True), nullable=True)
    content = Column(Text, nullable=False)
    source_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    importance = Column(Float, nullable=True)
    priority_score = Column(Float, nullable=True)
    retention_policy = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<MemoryShard id={self.id} user_id={self.user_id}>"