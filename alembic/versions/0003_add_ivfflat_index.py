"""add IVFFlat index on memory_shards.embedding

Revision ID: 0003
Revises: 0002
Create Date: 2025-09-20

This migration creates an IVFFlat index (cosine opclass) on memory_shards.embedding.
Creation can be toggled with PGVECTOR_ENABLE_IVFFLAT (default true), and the number
of lists is controlled by PGVECTOR_IVFFLAT_LISTS (default 100).

Note: after index creation, an ANALYZE is executed to help the planner.
"""

from __future__ import annotations

import os

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    enable = (os.getenv("PGVECTOR_ENABLE_IVFFLAT", "true") or "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    lists_str = os.getenv("PGVECTOR_IVFFLAT_LISTS", "100") or "100"
    try:
        lists = int(lists_str)
    except Exception:
        lists = 100

    # Ensure extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    if enable:
        # Observability: emit a lightweight structured event via a SELECT (visible in logs)
        op.execute(
            "SELECT json_build_object("
            "'event','index_build',"
            "'index','ix_memory_shards_embedding_ivfflat_cosine',"
            f"'lists', {lists})"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_memory_shards_embedding_ivfflat_cosine "
            f"ON memory_shards USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
        )
        # Help the planner choose the index
        op.execute(
            "SELECT json_build_object("
            "'event','analyze',"
            "'table','memory_shards',"
            "'index','ix_memory_shards_embedding_ivfflat_cosine')"
        )
        op.execute("ANALYZE memory_shards")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memory_shards_embedding_ivfflat_cosine")
