"""add source/metadata columns and indexes on memory_shards

Revision ID: 0004
Revises: 0003
Create Date: 2025-09-20

This migration adds content provenance and flexible metadata to memory_shards,
and creates helpful indexes for common query patterns.
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add optional content provenance and metadata (idempotent)
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='memory_shards' AND column_name='source'
            ) THEN
                ALTER TABLE memory_shards ADD COLUMN source TEXT;
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='memory_shards' AND column_name='metadata'
            ) THEN
                ALTER TABLE memory_shards ADD COLUMN metadata JSONB;
            END IF;
        END$$;
        """
    )

    # Index on user_id for scoped queries (if not already present)
    op.create_index(
        "ix_memory_shards_user_id",
        "memory_shards",
        ["user_id"],
        unique=False,
        if_not_exists=True,
    )

    # Optional GIN index on metadata for key filtering; use jsonb_path_ops for compact index.
    # SQLAlchemy's create_index with gin ops is a bit clunky; issue raw SQL for clarity
    # and IF NOT EXISTS.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memory_shards_metadata_gin "
        "ON memory_shards USING gin (metadata jsonb_path_ops)"
    )


def downgrade() -> None:
    # Drop indexes first
    op.execute("DROP INDEX IF EXISTS ix_memory_shards_metadata_gin")
    op.drop_index("ix_memory_shards_user_id", table_name="memory_shards", if_exists=True)

    # Drop columns
    op.drop_column("memory_shards", "metadata")
    op.drop_column("memory_shards", "source")
