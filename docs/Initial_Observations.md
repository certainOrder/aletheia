# Initial Observations and Roadmap

Date: 2025-09-19

## Repository Summary

- Purpose: FastAPI service bridging OpenAI chat and a PostgreSQL database with pgvector for embeddings. Includes a simple static chat UI.
- Entrypoint: `app/main.py` serves static UI, exposes `POST /openai-chat`, basic `GET /`.
- OpenAI integration: `app/services/openai_service.py` uses OpenAI chat completions.
- Database: `app/db/schema.sql` contains substantive DDL with `vector(1536)` columns; ORM is placeholder and mismatched.
- Utilities: `app/utils/embeddings.py` is stubbed.
- Frontend: `app/static/chat.html` minimal chat client hitting `/openai-chat`.
- Requirements: Includes FastAPI, SQLAlchemy, pgvector, OpenAI SDK, dotenv; psycopg duplicated.

## Key Findings

1. README inaccuracies
   - Project name/paths differ from repo (`openai_pgvector_api` vs `aletheia`).
   - Run command suggests `python app/main.py`; correct is using `uvicorn app.main:app --reload`.

2. Router not wired
   - `app/api/routes.py` defines basic routes but isn’t included in `app/main.py`.

3. Environment handling
   - Code loads `aletheia_api.env` from nonstandard path; no `.env.example` exists.

4. Database integration incomplete
   - No SQLAlchemy engine/session wiring.
   - ORM (`app/db/models.py`) does not match `schema.sql` and uses `Float` for embeddings instead of a vector type.
   - No migrations (Alembic). Schema setup is manual via SQL file.

5. Embeddings pipeline missing
   - `app/utils/embeddings.py` contains only stubs; no endpoint to create/search embeddings.

6. Operational hygiene
   - No CORS, structured logging, or consistent error handling.
   - No tests or CI.

## What Works Today

- Start the API with uvicorn (once env has `OPENAI_API_KEY`) and use `/chat` UI to send a prompt to OpenAI via `/openai-chat`.

## Immediate Fixes (Quick Wins)

- Update README for correct name and run steps.
- Add `.env.example` with `OPENAI_API_KEY` and `DATABASE_URL`.
- Clean `requirements.txt`: choose `psycopg[binary]` or `psycopg2-binary` (prefer psycopg3). Optionally pin versions.
- Wire `app/api/routes.py` into `app/main.py` and add CORS middleware.
- Provide an `if __name__ == "__main__":` uvicorn runner for convenience.

## Near-Term Enhancements

- Centralized config in `app/config.py` to read envs once (OpenAI key, DB URL, model names).
- SQLAlchemy setup: engine and session in `app/db/__init__.py`.
- Align ORM models to `schema.sql` and use pgvector’s SQLAlchemy `Vector` type.
- Introduce Alembic migrations and document DB setup.
- Implement basic embeddings endpoints:
  - `POST /embeddings`: create/save embedding for text into `memory_shards`.
  - `POST /search`: cosine distance/IVFFlat or brute-force `embedding <=> :vec` search.

## AI Layer Improvements

- Implement `app/utils/embeddings.py` using OpenAI `text-embedding-3-small`.
- Enhance chat to optionally retrieve top-k memories and ground responses (lightweight RAG).
- Support streaming responses in `/openai-chat` for better UX.

## Tooling & Ops

- Dockerfile and docker-compose for API + Postgres with pgvector.
- Basic tests via pytest; mock OpenAI; integration tests against a test DB.
- Optional lint/type-check (ruff, mypy) and GitHub Actions CI.

## Stretch Goals

- Auth (API keys or OAuth) and per-user scoping for data.
- Observability (metrics, tracing, structured logs, request IDs).
- Background workers for batch embedding generation and cleanup.
- Retention policies and versioned memory shards.
- Frontend UX improvements for chat and search results.

## Acceptance Criteria for Quick-Win PR

- Corrected README and added `.env.example`.
- `uvicorn app.main:app --reload` works; `/health` returns healthy; `/chat` loads.
- Router included; CORS enabled for localhost.
- `requirements.txt` has a single psycopg flavor and installs cleanly.

## Notes and Assumptions

- Assumed PostgreSQL 14+ with `pgvector` extension available.
- Assumed OpenAI SDK v1.x and models: `gpt-4o` for chat, `text-embedding-3-small` for embeddings.
- Embedding dimension 1536 per schema.
