"""seed schema milestone 4 (alternative starting point)

Revision ID: f60593c6cba8
Revises: None
Create Date: 2025-10-11

Consolidated seed migration that creates the complete schema up to milestone 4.
This can be used as an alternative starting point for fresh installations.

To use this as your starting point instead of running migrations 0001-0004:
1. alembic upgrade f60593c6cba8   # Apply this seed migration
2. alembic upgrade head            # Apply any migrations after milestone 4

Or continue using the incremental migrations (0001-0004) by ignoring this file:
1. alembic upgrade head            # This will use the original migration path

The seed migration is labeled with 'seed_milestone_4' for easy identification.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = 'f60593c6cba8'
down_revision = None  # This makes it an alternative starting point
branch_labels = ('seed_milestone_4',)  # Label it as a seed migration
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create memory_shards table with complete schema
    op.create_table(
        'memory_shards',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('source_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.Column('importance', sa.Float(), nullable=True),
        sa.Column('priority_score', sa.Float(), nullable=True),
        sa.Column('retention_policy', sa.Text(), nullable=True),
        sa.Column('source', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
    )
    
    # Create indexes (from migrations up to milestone 4)
    op.create_index('ix_memory_shards_user_id', 'memory_shards', ['user_id'])
    op.execute(
        """
        CREATE INDEX ix_memory_shards_embedding_ivfflat_cosine 
        ON memory_shards 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

def downgrade() -> None:
    op.drop_index('ix_memory_shards_embedding_ivfflat_cosine', table_name='memory_shards')
    op.drop_index('ix_memory_shards_user_id', table_name='memory_shards')
    op.drop_table('memory_shards')
    op.execute('DROP EXTENSION IF EXISTS vector')