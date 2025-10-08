-- PostgreSQL DDL for core tables
-- Requires pgvector extension for embedding fields

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE user_profiles (
    user_id uuid PRIMARY KEY,
    role text,
    name text,
    birthdate date,
    pronouns text,
    is_anchor boolean,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

CREATE TABLE identity_profile (
    id uuid PRIMARY KEY,
    name text,
    pronouns text,
    origin_date timestamp with time zone,
    core_seed text,
    alignment_model jsonb,
    embedding vector(1536)
);

CREATE TABLE bond_history (
    ei_id uuid REFERENCES identity_profile(id),
    user_id uuid REFERENCES user_profiles(user_id),
    bond_type text,
    start_date timestamp with time zone,
    end_date timestamp with time zone,
    reason text,
    PRIMARY KEY (ei_id, user_id)
);

CREATE TABLE raw_conversations (
    id uuid PRIMARY KEY,
    intended_recipient uuid,
    author text,
    timestamp timestamp with time zone,
    conversation_id uuid,
    user_id uuid REFERENCES user_profiles(user_id),
    content text,
    embedding vector(1536),
    parent_id uuid,
    entropy float,
    emotional_index float,
    surprise_index float
);

CREATE TABLE memory_shards (
    id uuid PRIMARY KEY,
    timestamp timestamp with time zone,
    last_accessed timestamp with time zone,
    user_id uuid REFERENCES user_profiles(user_id),
    content text,
    source_ids uuid[],
    tags text[],
    embedding vector(1536),
    importance float,
    priority_score float,
    retention_policy text
);

-- New table for engineering pattern RAG implementation
-- This table implements a Retrieval-Augmented Generation (RAG) system for engineering patterns
-- and documentation. It allows for semantic search of code patterns and automated context
-- generation for AI-assisted development.
--
-- Example Usage:
-- Consider a pattern for implementing API error handling. Here's how it would be stored:
--
-- INSERT INTO eng_patterns (
--     id, content, tags, strategy_type, target_contexts, author, metadata
-- ) VALUES (
--     gen_random_uuid(),
--     $$
--     # API Error Handling Pattern
--     
--     When implementing API endpoints, follow these error handling practices:
--     
--     1. Use custom error classes that extend from BaseError
--     2. Include status codes and error types
--     3. Maintain consistent error response structure
--     
--     Example implementation:
--     ```python
--     class APIError(BaseError):
--         def __init__(self, message: str, status_code: int = 500):
--             super().__init__(message)
--             self.status_code = status_code
--     
--     @app.errorhandler(APIError)
--     def handle_api_error(error):
--         return {
--             "error": str(error),
--             "type": error.__class__.__name__,
--             "status": error.status_code
--         }, error.status_code
--     ```
--     
--     This pattern ensures:
--     - Consistent error responses across all endpoints
--     - Easy error tracking and debugging
--     - Clear error inheritance hierarchy
--     $$,
--     ARRAY['error_handling', 'api', 'python', 'best_practice'],
--     'design_pattern',
--     ARRAY['api/*.py', 'routes/*.py'],
--     'primary_dev',
--     '{"complexity": "medium", "required": true}'
-- );
--
-- To retrieve patterns for context generation:
-- ```sql
-- SELECT content, tags
-- FROM eng_patterns
-- WHERE target_contexts @> ARRAY['api/*.py']
-- AND embedding <-> (SELECT embedding FROM query_embedding) < 0.3
-- ORDER BY embedding <-> (SELECT embedding FROM query_embedding)
-- LIMIT 5;
-- ```
--
-- The retrieved content can then be assembled into markdown files for AI context.
--
-- Example Python script for managing context generation:
-- ```python
-- from pathlib import Path
-- import psycopg2
-- from psycopg2.extras import execute_values
-- from typing import List, Dict
-- import openai
-- from rich.console import Console
-- from rich.table import Table
--
-- class ContextManager:
--     def __init__(self, conn_string: str):
--         self.conn = psycopg2.connect(conn_string)
--         self.console = Console()
--
--     def list_available_contexts(self) -> List[Dict[str, any]]:
--         """List all available documentation targets with their pattern counts"""
--         with self.conn.cursor() as cur:
--             cur.execute("""
--                 WITH pattern_counts AS (
--                     SELECT 
--                         unnest(target_contexts) as context,
--                         COUNT(*) as pattern_count,
--                         array_agg(DISTINCT tags) as all_tags
--                     FROM eng_patterns 
--                     GROUP BY unnest(target_contexts)
--                 )
--                 SELECT 
--                     context,
--                     pattern_count,
--                     all_tags
--                 FROM pattern_counts
--                 ORDER BY context
--             """)
--             return [
--                 {
--                     "component": ctx,
--                     "patterns": count,
--                     "tags": sorted(set(sum(tags, [])))  # flatten tag arrays
--                 }
--                 for ctx, count, tags in cur.fetchall()
--             ]
--
--     def generate_context(self, file_pattern: str, query: str = None) -> str:
--         """Generate markdown context for a specific file pattern"""
--         with self.conn.cursor() as cur:
--             if query:
--                 # Get embedding for the query
--                 response = openai.Embedding.create(
--                     input=query,
--                     model="text-embedding-3-small"
--                 )
--                 query_embedding = response['data'][0]['embedding']
--                 
--                 # Semantic search with file pattern filter
--                 cur.execute("""
--                     SELECT content, tags, strategy_type
--                     FROM eng_patterns
--                     WHERE target_contexts @> ARRAY[%s]
--                     ORDER BY embedding <-> %s
--                     LIMIT 5
--                 """, (file_pattern, query_embedding))
--             else:
--                 # Just filter by file pattern
--                 cur.execute("""
--                     SELECT content, tags, strategy_type
--                     FROM eng_patterns
--                     WHERE target_contexts @> ARRAY[%s]
--                 """, (file_pattern,))
--
--             results = cur.fetchall()
--             
--             # Assemble markdown
--             md_content = f"# Context for {file_pattern}\n\n"
--             for content, tags, strategy_type in results:
--                 md_content += f"## {strategy_type.title()}\n"
--                 md_content += f"Tags: {', '.join(tags)}\n\n"
--                 md_content += f"{content}\n\n---\n\n"
--             
--             return md_content
--
-- # Usage example:
-- if __name__ == "__main__":
--     cm = ContextManager("dbname=your_db user=your_user")
--     
--     # List available contexts
--     contexts = cm.list_available_contexts()
--     print("Available contexts by strategy type:")
--     for strategy, patterns in contexts.items():
--         print(f"\n{strategy}:")
--         for pattern in patterns:
--             print(f"  - {pattern}")
--     
--     # Generate context for a specific file
--     file_pattern = input("\nEnter file pattern to generate context for: ")
--     query = input("Enter search query (optional): ")
--     
--     md_content = cm.generate_context(file_pattern, query if query else None)
--     
--     # Save to file
--     output_path = Path(f"context_{file_pattern.replace('*', 'all').replace('/', '_')}.md")
--     output_path.write_text(md_content)
--     print(f"\nContext generated and saved to {output_path}")
-- ```
--
-- This script provides:
-- 1. List all available target contexts grouped by strategy type
-- 2. Generate context files with optional semantic search
-- 3. Save assembled markdown for use with Copilot
-- 
-- Run it to see available patterns and generate context files on demand.

CREATE TABLE eng_patterns (
    id uuid PRIMARY KEY,
    content text NOT NULL, -- The actual pattern/documentation text
    tags text[], -- Array for efficient tag filtering
    strategy_type text, -- To categorize different types of patterns (e.g., 'design_pattern', 'coding_standard', 'integration_guide')
    target_contexts text[] -- ['auth/*.ts', 'middleware/*', 'routes/auth/*']
    last_updated timestamp with time zone DEFAULT now(),
    author text,
    embedding vector(1536), -- For semantic search using pgvector
    metadata jsonb -- JSONB field for flexible additional metadata
);

-- Index for vector similarity search
CREATE INDEX eng_patterns_embedding_idx ON eng_patterns 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for tag searching
CREATE INDEX eng_patterns_tags_idx ON eng_patterns USING GIN (tags);