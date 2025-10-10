# Database Migrations (Alembic)

This project uses Alembic for schema migrations. In development, the app may still create
extensions/tables for convenience, but production and team workflows should rely on Alembic.

## Setup

- Ensure your virtualenv is active and dev deps are installed:

```bash
make setup
```

- The Alembic config lives in `alembic.ini` and `alembic/`.
- The migration environment reads `DATABASE_URL` from `app.config` to avoid drift.

## Common commands

- Create a new revision (set your message with `m`):

```bash
make migrate-rev m="add new table"
```

- Upgrade to latest:

```bash
make migrate-up
```

- Downgrade one step:

```bash
make migrate-down
```

## Baseline

An initial baseline migration `0001_create_memory_shards.py` creates the `memory_shards` table
and ensures the `vector` extension exists. Adjust the embedding dimension via `EMBEDDING_DIM`
in `.env` and reflect it in new migrations if needed.

## Notes

- Do not commit real secrets. `DATABASE_URL` should come from `.env` (see `.env.example`).
- Prefer migrations over runtime `Base.metadata.create_all()` in production; the app still
  calls it in dev for convenience.
- Keep migrations deterministic and reversible. Avoid data-dependent logic in migrations.

### IVFFlat index (pgvector)

- Migration `0003` creates an IVFFlat index on `memory_shards.embedding` using the cosine opclass.
- Creation is gated by `PGVECTOR_ENABLE_IVFFLAT` (default `true`). The number of lists is controlled by `PGVECTOR_IVFFLAT_LISTS` (default `100`).
- After creating the index or a significant ingest, run `ANALYZE` to ensure good query plans:
Tuning notes
- Consider deferring IVFFlat creation until after initial bulk ingest to reduce build time; toggle via `PGVECTOR_ENABLE_IVFFLAT`.
- Select `PGVECTOR_IVFFLAT_LISTS` based on corpus size: start at 100–200 for small sets; increase (e.g., 512–2048) for larger corpora to improve recall.
- Always `ANALYZE` after big ingests or index builds to update statistics.

```bash
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "ANALYZE memory_shards"
```

### M3: Content metadata and raw conversations (0004, 0005)

- **Migration 0004** adds `source` (TEXT) and `metadata` (JSONB) columns to `memory_shards` for content provenance.
  - `source`: Optional free-form origin (e.g., URL, file path, "wikipedia")
  - `metadata`: Flexible key-value storage (e.g., `{"topic": "health", "verified": true}`)
  - Creates indexes: BTree on `user_id`, GIN on `metadata` (jsonb_path_ops) for key filtering
  - Idempotent: uses DO blocks to check existence before adding columns

- **Migration 0005** creates/augments `raw_conversations` table for request/response logging.
  - Columns: `id`, `created_at`, `request_id`, `user_id`, `provider`, `model`, `messages` (JSONB), `response` (JSONB), `status_code`, `latency_ms`
  - Indexes on `created_at`, `user_id`, `request_id` for efficient querying
  - Idempotent: safe to run on existing populated DBs
  - Every `/v1/chat/completions` call persists a row for observability and replay

Backfill notes:
- Existing shards without `source`/`metadata` will have NULL values (no backfill needed)
- `raw_conversations` logging is forward-only (no historical data backfill)

### Engineering Patterns RAG (0006)

- **Migration 0006** adds `eng_patterns` table for RAG-based engineering pattern management.
  - Stores reusable patterns, coding standards, best practices for AI-assisted development
  - Columns: `id`, `content`, `tags[]`, `strategy_type`, `target_contexts[]`, `last_updated`, `author`, `embedding` (vector 1536), `metadata` (JSONB)
  - Supports semantic search via pgvector embeddings to retrieve relevant patterns for specific file contexts
  - Indexes:
    - IVFFlat on `embedding` for vector cosine similarity (lists=100)
    - GIN on `tags` for efficient tag array filtering
    - BTree on `strategy_type` and `last_updated DESC`
  - Use case: Query patterns by target context (e.g., `api/*.py`) and semantic similarity to generate AI context files
  - Idempotent: safe CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS

## Troubleshooting: existing tables, no Alembic state

If you see errors like `psycopg.errors.DuplicateTable: relation "..." already exists` during
`alembic upgrade`, your database likely has tables created manually or by the app, but the
`alembic_version` table hasn’t been initialized. You can “stamp” the database to the correct
baseline, then upgrade to head.

Quick checklist
- Confirm tables exist:
  - Using Docker:
    ```bash
    docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\\dt"
    ```
  - From host via `localhost` (compose exposes `5432`):
    ```bash
    psql postgresql://postgres:postgres@localhost:5432/aletheia -c "\\dt"
    ```
- If baseline tables exist but Alembic isn’t stamped:
  - Stamp baseline (replace `0001` with your baseline revision if different):
    ```bash
    DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/aletheia \
      ./.venv/bin/alembic stamp 0001
    ```
  - Upgrade to head:
    ```bash
    DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/aletheia \
      ./.venv/bin/alembic upgrade head
    ```

Notes
- Inside containers, use the service hostname `db` in `DATABASE_URL`; from the host, use
  `localhost:5432` as shown above. The `.env` typically sets `DATABASE_URL` with `db` for
  container usage.
- To avoid hostname mismatches entirely, you can exec Alembic inside the API container:
  ```bash
  docker compose exec -T aletheia-api alembic upgrade head
  ```
  Ensure `alembic` is installed in the image or add a layer that installs dev tools.
