"""add eng_patterns table for RAG-based pattern management

Revision ID: 0006
Revises: 0005
Create Date: 2025-10-10

This migration adds the eng_patterns table for storing engineering patterns,
coding standards, and best practices with RAG-based retrieval. The table
supports semantic search via pgvector embeddings and efficient tag filtering.

Use cases:
- Store reusable engineering patterns and documentation
- Semantic search to find relevant patterns for code context
- Target specific file patterns (e.g., 'api/*.py', 'routes/*.py')
- Generate AI-assisted context for development workflows
"""

from __future__ import annotations

import os

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create eng_patterns table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS eng_patterns (
            id uuid PRIMARY KEY,
            content text NOT NULL,
            tags text[],
            strategy_type text,
            target_contexts text[],
            last_updated timestamp with time zone DEFAULT now(),
            author text,
            embedding vector(1536),
            metadata jsonb
        )
        """
    )

    # IVFFlat index for vector similarity search (conditional on config)
    # Use environment variable to gate IVFFlat index creation
    use_ivfflat = os.getenv("USE_IVFFLAT_INDEX", "false").lower() == "true"

    if use_ivfflat:
        # Determine lists value from environment or default
        lists = int(os.getenv("IVFFLAT_LISTS", "100"))
        op.execute(
            f"""
            CREATE INDEX IF NOT EXISTS eng_patterns_embedding_idx
            ON eng_patterns USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists})
            """
        )
    else:
        # Basic index for non-IVFFlat environments
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS eng_patterns_embedding_idx
            ON eng_patterns USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )

    # GIN index for efficient tag array searching
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS eng_patterns_tags_idx
        ON eng_patterns USING GIN (tags)
        """
    )

    # Additional indexes for common queries
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS eng_patterns_strategy_type_idx
        ON eng_patterns (strategy_type)
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS eng_patterns_last_updated_idx
        ON eng_patterns (last_updated DESC)
        """
    )


def downgrade() -> None:
    # Drop indexes first
    op.execute("DROP INDEX IF EXISTS eng_patterns_last_updated_idx")
    op.execute("DROP INDEX IF EXISTS eng_patterns_strategy_type_idx")
    op.execute("DROP INDEX IF EXISTS eng_patterns_tags_idx")
    op.execute("DROP INDEX IF EXISTS eng_patterns_embedding_idx")

    # Drop the table
    op.execute("DROP TABLE IF EXISTS eng_patterns")
