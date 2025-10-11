-- HearthMinds infrastructure schema
-- Core tables for project-wide functionality

-- Enable vector extension first
CREATE EXTENSION IF NOT EXISTS vector;

-- Engineering patterns table (shared across all AIs)
CREATE TABLE eng_patterns (
    id uuid PRIMARY KEY,
    content text NOT NULL, -- The actual pattern/documentation text
    tags text[], -- Array for efficient tag filtering
    strategy_type text, -- To categorize different types of patterns
    target_contexts text[], -- Array of file patterns where this applies
    last_updated timestamp with time zone DEFAULT now(),
    author text,
    embedding vector(1536), -- For semantic search using pgvector
    metadata jsonb -- Flexible additional metadata
);

-- Create indexes
CREATE INDEX eng_patterns_embedding_idx ON eng_patterns 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX eng_patterns_tags_idx ON eng_patterns USING GIN (tags);

-- Grant permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO logos, aletheia;

-- Initial pattern: Boolean-based validation testing
INSERT INTO eng_patterns (
    id,
    content,
    tags,
    strategy_type,
    target_contexts,
    author,
    metadata
) VALUES (
    gen_random_uuid(),
    $$# Boolean-Based Validation Testing Pattern

A robust pattern for implementing idempotent operations with clear success/failure indicators.

## Core Pattern Structure
```python
def action_x():
    # Perform action
    do_thing()
    # Validate result
    success = validate_thing_done()
    # Log result
    log_result("action_x", success)
    return success
```

## Key Principles
1. **Separation of Concerns**
   - Action: Performs the operation
   - Validation: Checks the result independently
   - Logging: Documents the outcome

2. **Hard Validation**
   - No "soft" checking or retries in validation
   - Validation should be deterministic
   - Each validation checks one specific thing

3. **Clear Boolean Results**
   - Every operation returns True/False
   - No ambiguous states or partial successes
   - Fail fast, fail hard

## Implementation Example
```python
def validate_postgres_stopped(self) -> bool:
    """Check if PostgreSQL service is stopped"""
    result = self.executor.run_command(
        "/usr/bin/systemctl is-active postgresql",
        use_sudo=True
    )
    return result.stdout.strip() in ["inactive", "unknown"]

def stop_postgres(self) -> bool:
    """Stop PostgreSQL service"""
    logger.info("Stopping PostgreSQL service...")
    # Perform action
    result = self.executor.run_command(
        "/usr/bin/systemctl stop postgresql",
        use_sudo=True
    )
    # Validate result
    success = self.validate_postgres_stopped()
    # Log result
    logger.info(f"PostgreSQL service stop {'succeeded' if success else 'failed'}")
    return success
```

## Benefits
1. **Clarity**: Each operation has a clear success/failure state
2. **Testability**: Easy to verify outcomes without mocks
3. **Debugging**: Failures are immediately apparent
4. **Independence**: Validation is separate from action

## When to Use
- System administration tasks
- Database operations
- Infrastructure setup
- Configuration management
- Any idempotent operation

## Anti-Patterns to Avoid
1. Mixing action and validation
2. Retrying in validation methods
3. Returning non-boolean results
4. Skipping validation steps
5. Validating multiple things at once

Remember: If you can't clearly validate the result, rethink the action's scope.$$,
    ARRAY['testing', 'validation', 'best_practice', 'idempotency', 'python'],
    'design_pattern',
    ARRAY['**/*_test.py', '**/*.py', 'app/db/**/*.py'],
    'copilot',
    '{"complexity": "medium", "required": true, "domain": "testing"}'
);
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

CREATE EXTENSION IF NOT EXISTS vector;

-- Engineering patterns table (shared across all AIs)
CREATE TABLE IF NOT EXISTS eng_patterns (
    id uuid PRIMARY KEY,
    content text NOT NULL, -- The actual pattern/documentation text
    tags text[], -- Array for efficient tag filtering
    strategy_type text, -- To categorize different types of patterns (e.g., 'design_pattern', 'coding_standard', 'integration_guide')
    target_contexts text[], -- ['auth/*.ts', 'middleware/*', 'routes/auth/*']
    last_updated timestamp with time zone DEFAULT now(),
    author text,
    embedding vector(1536), -- For semantic search using pgvector
    metadata jsonb -- JSONB field for flexible additional metadata
);

-- Index for vector similarity search
CREATE INDEX IF NOT EXISTS eng_patterns_embedding_idx ON eng_patterns 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for tag searching
CREATE INDEX IF NOT EXISTS eng_patterns_tags_idx ON eng_patterns USING GIN (tags);