"""create profiles and conversations tables

Revision ID: 0002
Revises: 0001
Create Date: 2025-09-19

"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # user_profiles
    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("role", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("birthdate", sa.Date(), nullable=True),
        sa.Column("pronouns", sa.Text(), nullable=True),
        sa.Column("is_anchor", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # identity_profile
    op.create_table(
        "identity_profile",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("pronouns", sa.Text(), nullable=True),
        sa.Column("origin_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("core_seed", sa.Text(), nullable=True),
        sa.Column("alignment_model", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
    )

    # bond_history
    op.create_table(
        "bond_history",
        sa.Column("ei_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("bond_type", sa.Text(), nullable=True),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("ei_id", "user_id"),
        sa.ForeignKeyConstraint(["ei_id"], ["identity_profile.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["user_profiles.user_id"]),
    )

    # raw_conversations
    op.create_table(
        "raw_conversations",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("intended_recipient", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conversation_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("parent_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("entropy", sa.Float(), nullable=True),
        sa.Column("emotional_index", sa.Float(), nullable=True),
        sa.Column("surprise_index", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user_profiles.user_id"]),
    )

    # Add missing FK on memory_shards.user_id -> user_profiles(user_id)
    op.create_foreign_key(
        "fk_memory_shards_user_id",
        source_table="memory_shards",
        referent_table="user_profiles",
        local_cols=["user_id"],
        remote_cols=["user_id"],
    )


def downgrade() -> None:
    # Drop FK from memory_shards first
    op.drop_constraint("fk_memory_shards_user_id", "memory_shards", type_="foreignkey")

    # Drop dependent tables in reverse dependency order
    op.drop_table("bond_history")
    op.drop_table("raw_conversations")
    op.drop_table("identity_profile")
    op.drop_table("user_profiles")
