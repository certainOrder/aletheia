"""extend raw_conversations for request/response logging

Revision ID: 0005
Revises: 0004
Create Date: 2025-09-20

This migration augments the existing raw_conversations table to support
OpenAI-compatible chat logging with request/response payloads and metadata,
and adds useful indexes for querying.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # If raw_conversations doesn't exist (older env), create it with full schema
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_conversations (
            id uuid PRIMARY KEY,
            created_at timestamptz DEFAULT now() NOT NULL,
            request_id text,
            user_id text NULL,
            provider text,
            model text,
            messages jsonb,
            response jsonb,
            status_code integer,
            latency_ms integer
        )
        """
    )

    # For environments where raw_conversations exists from 0002, add missing columns
    # Use DO blocks to conditionally add columns only when absent
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='created_at'
            ) THEN
                ALTER TABLE raw_conversations
                ADD COLUMN created_at timestamptz DEFAULT now() NOT NULL;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='request_id'
            ) THEN
                ALTER TABLE raw_conversations ADD COLUMN request_id text;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='provider'
            ) THEN
                ALTER TABLE raw_conversations ADD COLUMN provider text;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='model'
            ) THEN
                ALTER TABLE raw_conversations ADD COLUMN model text;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='messages'
            ) THEN
                ALTER TABLE raw_conversations ADD COLUMN messages jsonb;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='response'
            ) THEN
                ALTER TABLE raw_conversations ADD COLUMN response jsonb;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='status_code'
            ) THEN
                ALTER TABLE raw_conversations ADD COLUMN status_code integer;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='raw_conversations' AND column_name='latency_ms'
            ) THEN
                ALTER TABLE raw_conversations ADD COLUMN latency_ms integer;
            END IF;
        END$$;
        """
    )

    # Indexes for common queries
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_raw_conversations_created_at "
        "ON raw_conversations (created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_raw_conversations_user_id " "ON raw_conversations (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_raw_conversations_request_id "
        "ON raw_conversations (request_id)"
    )


def downgrade() -> None:
    # Drop indexes first
    op.execute("DROP INDEX IF EXISTS ix_raw_conversations_request_id")
    op.execute("DROP INDEX IF EXISTS ix_raw_conversations_user_id")
    op.execute("DROP INDEX IF EXISTS ix_raw_conversations_created_at")

    # Attempt to drop newly added columns; keep table if it pre-existed (0002)
    # Columns in 0002 schema that we should NOT drop include:
    # id, intended_recipient, author, timestamp, conversation_id, user_id (uuid),
    # content, embedding, parent_id, entropy, emotional_index, surprise_index
    # We only drop columns that this migration may have introduced.
    for col in [
        "created_at",
        "request_id",
        "provider",
        "model",
        "messages",
        "response",
        "status_code",
        "latency_ms",
    ]:
        op.execute(
            sa.text(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='raw_conversations' AND column_name=:col
                    ) THEN
                        EXECUTE 'ALTER TABLE raw_conversations DROP COLUMN ' || quote_ident(:col);
                    END IF;
                END$$;
                """
            ).bindparams(col=col)
        )
