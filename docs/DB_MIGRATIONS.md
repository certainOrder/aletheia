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
