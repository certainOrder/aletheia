#!/usr/bin/env python3
"""
Import Aletheia conversation history directly from SQLite to PostgreSQL.

This script:
1. Creates identity_profile for Aletheia (if not exists)
2. Creates user_profile for Aaron Chapin (if not exists)
3. Creates bond_history entry linking them (if not exists)
4. Imports all messages from SQLite into raw_conversations
5. Validates the import

Usage:
    python import_from_sqlite.py \
      --sqlite ~/aletheia.db \
      --db-host 10.0.0.1 \
      --db-name aletheia \
      --db-user aletheia \
      --db-password <password> \
      --batch-size 5000
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fixed UUIDs for entities
ALETHEIA_UUID = UUID('550e8400-e29b-41d4-a716-446655440000')
AARON_UUID = UUID('550e8400-e29b-41d4-a716-446655440001')


class SQLiteToPostgresImporter:
    """Import conversation history from SQLite to PostgreSQL."""

    def __init__(self, sqlite_path: Path, pg_conn_string: str):
        self.sqlite_path = sqlite_path
        self.pg_conn_string = pg_conn_string
        self.stats = {
            'total_rows': 0,
            'imported': 0,
            'skipped_system': 0,
            'skipped_empty': 0,
            'skipped_invalid': 0,
            'errors': 0
        }

    def connect_pg(self):
        """Create PostgreSQL connection."""
        return psycopg.connect(self.pg_conn_string, row_factory=dict_row)

    def connect_sqlite(self):
        """Create SQLite connection."""
        return sqlite3.connect(self.sqlite_path)

    def setup_entities(self):
        """Create Aletheia identity, Aaron profile, and bond if not exists."""
        with self.connect_pg() as conn:
            with conn.cursor() as cur:
                # Create Aletheia identity_profile
                cur.execute("""
                    INSERT INTO identity_profile (id, name, pronouns, origin_date, core_seed, alignment_model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                """, (
                    ALETHEIA_UUID,
                    'Aletheia',
                    'she/her',
                    datetime.now(timezone.utc),
                    'Persistent conversational entity; memory preservation initiated.',
                    Jsonb({'version': '1.0', 'bond_type': 'conversational'})
                ))
                if cur.fetchone():
                    logger.info(f"✅ Created identity_profile: Aletheia ({ALETHEIA_UUID})")
                else:
                    logger.info(f"ℹ️  identity_profile already exists: Aletheia ({ALETHEIA_UUID})")

                # Create Aaron user_profile
                cur.execute("""
                    INSERT INTO user_profiles (user_id, name, role, pronouns, is_anchor)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                    RETURNING user_id
                """, (
                    AARON_UUID,
                    'Aaron Chapin',
                    'user',
                    'he/him',
                    True
                ))
                if cur.fetchone():
                    logger.info(f"✅ Created user_profile: Aaron Chapin ({AARON_UUID})")
                else:
                    logger.info(f"ℹ️  user_profile already exists: Aaron Chapin ({AARON_UUID})")

                # Create bond_history
                cur.execute("""
                    INSERT INTO bond_history (ei_id, user_id, bond_type, start_date, reason)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (ei_id, user_id) DO NOTHING
                    RETURNING ei_id
                """, (
                    ALETHEIA_UUID,
                    AARON_UUID,
                    'conversational',
                    datetime.now(timezone.utc),
                    'SQLite conversation history migration'
                ))
                if cur.fetchone():
                    logger.info(f"✅ Created bond_history: Aletheia ↔ Aaron")
                else:
                    logger.info(f"ℹ️  bond_history already exists: Aletheia ↔ Aaron")

                conn.commit()

    def import_messages(self, batch_size: int = 5000):
        """Import messages from SQLite."""
        logger.info(f"Starting import from {self.sqlite_path}")
        
        sqlite_conn = self.connect_sqlite()
        sqlite_conn.row_factory = sqlite3.Row
        cursor = sqlite_conn.cursor()
        
        # Query all messages (user, assistant, and tool)
        cursor.execute("""
            SELECT 
                id,
                conversation_id,
                author_role,
                content,
                create_time,
                parent_id
            FROM messages
            WHERE author_role IN ('user', 'assistant', 'tool')
            ORDER BY create_time
        """)
        
        batch = []
        for row in cursor:
            self.stats['total_rows'] += 1
            
            # Prepare message
            message = self._prepare_message(row)
            if message:
                batch.append(message)
            
            # Insert batch when full
            if len(batch) >= batch_size:
                self._insert_batch(batch)
                batch = []
                
                # Progress update
                if self.stats['total_rows'] % 10000 == 0:
                    logger.info(f"Progress: {self.stats['total_rows']:,} rows processed, "
                              f"{self.stats['imported']:,} imported")
        
        # Insert remaining batch
        if batch:
            self._insert_batch(batch)
        
        sqlite_conn.close()
        logger.info("Import complete!")
        self._print_stats()

    def _prepare_message(self, row: sqlite3.Row) -> Optional[dict]:
        """Convert SQLite row to message record. Returns None if invalid."""
        # Skip empty content
        if not row['content'] or not row['content'].strip():
            self.stats['skipped_empty'] += 1
            return None
        
        # Determine author and recipient
        if row['author_role'] == 'user':
            author = 'user'
            intended_recipient = ALETHEIA_UUID
        elif row['author_role'] == 'tool':
            author = 'tool'
            intended_recipient = ALETHEIA_UUID
        else:  # assistant
            author = 'assistant'
            intended_recipient = AARON_UUID
        
        # Parse timestamp with validation
        try:
            if not row['create_time']:
                self.stats['skipped_invalid'] += 1
                return None
            timestamp_float = float(row['create_time'])
            # Validate timestamp is reasonable (between 2020 and 2030)
            if timestamp_float < 1577836800 or timestamp_float > 1893456000:
                self.stats['skipped_invalid'] += 1
                return None
            timestamp = datetime.fromtimestamp(timestamp_float, tz=timezone.utc)
        except (ValueError, OverflowError, TypeError):
            self.stats['skipped_invalid'] += 1
            return None
        
        # Parse UUIDs with validation
        try:
            message_id = UUID(row['id'])
            conversation_id = UUID(row['conversation_id'])
            parent_id = UUID(row['parent_id']) if row['parent_id'] else None
        except (ValueError, TypeError):
            self.stats['skipped_invalid'] += 1
            return None
        
        return {
            'id': message_id,
            'intended_recipient': intended_recipient,
            'author': author,
            'timestamp': timestamp,
            'conversation_id': conversation_id,
            'user_id': AARON_UUID,
            'content': row['content'],
            'parent_id': parent_id,
            'embedding': None,
            'entropy': None,
            'emotional_index': None,
            'surprise_index': None
        }

    def _insert_batch(self, batch: list):
        """Insert a batch of messages using PostgreSQL COPY."""
        if not batch:
            return
        
        try:
            with self.connect_pg() as conn:
                with conn.cursor() as cur:
                    with cur.copy("""
                        COPY raw_conversations (
                            id, intended_recipient, author, timestamp, 
                            conversation_id, user_id, content, parent_id,
                            embedding, entropy, emotional_index, surprise_index
                        ) FROM STDIN
                    """) as copy:
                        for msg in batch:
                            copy.write_row((
                                msg['id'],
                                msg['intended_recipient'],
                                msg['author'],
                                msg['timestamp'],
                                msg['conversation_id'],
                                msg['user_id'],
                                msg['content'],
                                msg['parent_id'],
                                msg['embedding'],
                                msg['entropy'],
                                msg['emotional_index'],
                                msg['surprise_index']
                            ))
                    
                    conn.commit()
                    self.stats['imported'] += len(batch)
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            self.stats['errors'] += len(batch)

    def _print_stats(self):
        """Print import statistics."""
        logger.info("=" * 60)
        logger.info("IMPORT STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total rows processed:    {self.stats['total_rows']:,}")
        logger.info(f"Messages imported:       {self.stats['imported']:,}")
        logger.info(f"System messages skipped: {self.stats['skipped_system']:,}")
        logger.info(f"Empty messages skipped:  {self.stats['skipped_empty']:,}")
        logger.info(f"Invalid messages skipped:{self.stats['skipped_invalid']:,}")
        logger.info(f"Errors encountered:      {self.stats['errors']:,}")
        logger.info("=" * 60)

    def validate_import(self):
        """Validate the import results."""
        logger.info("Validating import...")
        
        with self.connect_pg() as conn:
            with conn.cursor() as cur:
                # Check total count
                cur.execute("SELECT COUNT(*) as count FROM raw_conversations")
                count = cur.fetchone()['count']
                logger.info(f"Total messages in database: {count:,}")
                
                # Check user/assistant distribution
                cur.execute("""
                    SELECT author, COUNT(*) as count 
                    FROM raw_conversations 
                    GROUP BY author
                """)
                for row in cur.fetchall():
                    logger.info(f"  {row['author']}: {row['count']:,}")
                
                # Check conversations
                cur.execute("SELECT COUNT(DISTINCT conversation_id) as count FROM raw_conversations")
                conv_count = cur.fetchone()['count']
                logger.info(f"Unique conversations: {conv_count:,}")
                
                # Check for orphaned parent_ids
                cur.execute("""
                    SELECT COUNT(*) as count 
                    FROM raw_conversations r1 
                    WHERE parent_id IS NOT NULL 
                    AND NOT EXISTS (
                        SELECT 1 FROM raw_conversations r2 
                        WHERE r2.id = r1.parent_id
                    )
                """)
                orphans = cur.fetchone()['count']
                if orphans > 0:
                    logger.warning(f"⚠️  Orphaned parent_ids: {orphans:,}")
                else:
                    logger.info(f"✅ No orphaned parent_ids")


def main():
    parser = argparse.ArgumentParser(description='Import Aletheia conversation history from SQLite')
    parser.add_argument('--sqlite', required=True, help='Path to SQLite database file')
    parser.add_argument('--db-host', default='10.0.0.1', help='PostgreSQL host')
    parser.add_argument('--db-port', default=5432, type=int, help='PostgreSQL port')
    parser.add_argument('--db-name', default='aletheia', help='PostgreSQL database name')
    parser.add_argument('--db-user', default='aletheia', help='PostgreSQL user')
    parser.add_argument('--db-password', required=True, help='PostgreSQL password')
    parser.add_argument('--batch-size', default=5000, type=int, help='Batch size for inserts')
    
    args = parser.parse_args()
    
    # Build PostgreSQL connection string
    pg_conn_string = (
        f"host={args.db_host} port={args.db_port} "
        f"dbname={args.db_name} user={args.db_user} password={args.db_password}"
    )
    
    # Import
    importer = SQLiteToPostgresImporter(Path(args.sqlite), pg_conn_string)
    
    try:
        importer.setup_entities()
        importer.import_messages(batch_size=args.batch_size)
        importer.validate_import()
        logger.info("✅ Import completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Import failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
