"""master_schema_phase_2

Revision ID: 133f690eb0ca
Revises: 
Create Date: 2025-10-11 18:21:56.640778

Master schema migration that captures the complete database structure for Phase 2.
This serves as an alternative starting point for fresh installations, incorporating:

1. HearthMinds Infrastructure (hearthminds database)
   - eng_patterns table for pattern documentation
   - Vector search capabilities with pgvector

2. Proto-Person Schema (logos & aletheia databases)
   - Complete user profile system
   - Memory and conversation management
   - Bond tracking and identity profiles

To use this as your starting point:
1. alembic upgrade 133f690eb0ca  # Apply this master schema
2. alembic upgrade head           # Apply any subsequent migrations

Or continue using the incremental migrations by ignoring this branch.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '133f690eb0ca'
down_revision = None
branch_labels = ('master_phase_2',)
depends_on = None

def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # HearthMinds Infrastructure Schema
    
    # Engineering patterns table
    op.create_table(
        'eng_patterns',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('tags', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('strategy_type', sa.Text(), nullable=True),
        sa.Column('target_contexts', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('author', sa.Text(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
    )

    # Create IVFFlat index for vector similarity search
    op.execute(
        'CREATE INDEX eng_patterns_embedding_idx ON eng_patterns '
        'USING ivfflat (embedding vector_cosine_ops) '
        'WITH (lists = 100)'
    )

    # Create GIN index for tag searching
    op.execute(
        'CREATE INDEX eng_patterns_tags_idx ON eng_patterns '
        'USING GIN (tags)'
    )

    # Proto-Person Schema Tables
    
    # User profiles
    op.create_table(
        'user_profiles',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('role', sa.Text(), nullable=True),
        sa.Column('name', sa.Text(), nullable=True),
        sa.Column('birthdate', sa.Date(), nullable=True),
        sa.Column('pronouns', sa.Text(), nullable=True),
        sa.Column('is_anchor', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Identity profiles
    op.create_table(
        'identity_profile',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.Text(), nullable=True),
        sa.Column('pronouns', sa.Text(), nullable=True),
        sa.Column('origin_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('core_seed', sa.Text(), nullable=True),
        sa.Column('alignment_model', postgresql.JSONB(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=True)
    )

    # Bond history
    op.create_table(
        'bond_history',
        sa.Column('ei_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('identity_profile.id'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('user_profiles.user_id'), nullable=False),
        sa.Column('bond_type', sa.Text(), nullable=True),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('ei_id', 'user_id')
    )

    # Conversations
    op.create_table(
        'raw_conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('intended_recipient', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('author', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('user_profiles.user_id'), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('parent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('entropy', sa.Float(), nullable=True),
        sa.Column('emotional_index', sa.Float(), nullable=True),
        sa.Column('surprise_index', sa.Float(), nullable=True)
    )

    # Memory shards
    op.create_table(
        'memory_shards',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('user_profiles.user_id'), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('source_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.Column('importance', sa.Float(), nullable=True),
        sa.Column('priority_score', sa.Float(), nullable=True),
        sa.Column('retention_policy', sa.Text(), nullable=True)
    )

    # Create IVFFlat index for memory shard embeddings
    op.execute(
        'CREATE INDEX memory_shards_embedding_idx ON memory_shards '
        'USING ivfflat (embedding vector_cosine_ops) '
        'WITH (lists = 100)'
    )

def downgrade() -> None:
    # Drop tables in reverse order of creation to handle dependencies
    op.drop_table('memory_shards')
    op.drop_table('raw_conversations')
    op.drop_table('bond_history')
    op.drop_table('identity_profile')
    op.drop_table('user_profiles')
    op.drop_table('eng_patterns')


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass