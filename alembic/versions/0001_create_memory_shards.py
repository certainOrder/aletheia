"""create memory_shards table

Revision ID: 0001
Revises:
Create Date: 2025-09-19

"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "memory_shards",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "source_ids",
            sa.dialects.postgresql.ARRAY(sa.dialects.postgresql.UUID(as_uuid=True)),
            nullable=True,
        ),
        sa.Column("tags", sa.dialects.postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("importance", sa.Float(), nullable=True),
        sa.Column("priority_score", sa.Float(), nullable=True),
        sa.Column("retention_policy", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("memory_shards")
