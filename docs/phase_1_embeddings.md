# Phase 1 — OpenWebUI + Embeddings-backed RAG

Date: 2025-09-19

## 1) High-level overview

Goal: Use OpenWebUI as the chat interface while the backend performs server-side RAG. The server embeds the user’s message, searches Postgres/pgvector for relevant context, and augments the Chat request sent to OpenAI.

End-to-end flow:
- OpenWebUI connects to our API using OpenAI-compatible endpoints (`/v1/models`, `/v1/chat/completions`).
- On each chat message, our API:
  1. Generates an embedding for the user’s last message (OpenAI `text-embedding-3-small`).
  2. Queries `memory_shards` with pgvector for nearest matches.
  3. Prepends retrieved context to the Chat messages.
  4. Calls OpenAI Chat (`gpt-4o` by default) and returns a compatible response.

Result: OpenWebUI continues to work as a standard OpenAI client, but answers are improved with your private memory store.


## 2) What was implemented

### OpenAI-compatible interface for OpenWebUI
- `GET /v1/models`: returns the configured model (from `OPENAI_CHAT_MODEL`).
- `POST /v1/chat/completions`: accepts OpenAI-style chat payloads, performs retrieval on the last user message, augments the system/user context, then calls OpenAI Chat and returns a standard response.

### RAG building blocks
- `POST /index-memory`: indexes content into the database by generating an embedding and inserting a `memory_shards` row.
- `POST /rag-chat`: a direct test endpoint for RAG; returns `{ answer, context }` for debugging.

### Database and ORM
- `app/db/__init__.py`: SQLAlchemy engine and `get_db()` session dependency.
- `app/db/models.py`: `MemoryShard` ORM model aligned with `schema.sql` (vector column via `pgvector.sqlalchemy.Vector(EMBEDDING_DIM)`).
- Startup hook attempts to ensure `CREATE EXTENSION IF NOT EXISTS vector` (requires privileges) and calls `Base.metadata.create_all()` for development convenience.

### Embeddings utilities
- `app/utils/embeddings.py`:
  - `convert_to_embedding(text)`: OpenAI embeddings (`OPENAI_EMBEDDING_MODEL`, default `text-embedding-3-small`).
  - `save_embedding_to_db(...)`: persists a `MemoryShard` with `content`, `embedding`, `user_id`, and optional `tags`.
  - `semantic_search(db, query_embedding, user_id?, limit=5)`: nearest-neighbor search using SQLAlchemy/pgvector; currently ordered by L2 distance.

### OpenAI service
- `app/services/openai_service.py`: centralizes OpenAI client creation and exposes `get_response(prompt)` and `chat(messages, model?)` using `OPENAI_CHAT_MODEL`.

### App glue and config
- CORS enabled with `ALLOWED_ORIGINS` for local UIs (OpenWebUI).
- Included `app/api/routes.py` (hello and health routes).
- `app/config.py` centralizes env configuration (`DATABASE_URL`, `OPENAI_API_KEY`, model names, `EMBEDDING_DIM`, CORS origins); loads `.env`.
- `.env.example` added to document required variables.
- `requirements.txt`: cleaned to use psycopg3 (`psycopg[binary]`), FastAPI, SQLAlchemy, pgvector, OpenAI, uvicorn, python-dotenv.


## 3) How the pieces tie together

- OpenWebUI → our API (OpenAI-compatible)
  - Base URL: `http://localhost:8000/v1`
  - API Key in OpenWebUI can be any non-empty string (the server uses your real key from `.env`).
- Our API (FastAPI)
  - Parses messages, embeds the latest user message, runs vector search in Postgres, constructs system/user context, calls OpenAI Chat.
- Postgres + pgvector
  - Stores `memory_shards` with a `vector(1536)` embedding (configurable via `EMBEDDING_DIM`).
  - Retrieval sorts by vector distance.
- OpenAI
  - Chat: `gpt-4o` default (configurable via `OPENAI_CHAT_MODEL`).
  - Embeddings: `text-embedding-3-small` (configurable via `OPENAI_EMBEDDING_MODEL`).


## 4) Setup and usage

### Environment
- Copy and edit `.env`:
  ```bash
  cp .env.example .env
  # set OPENAI_API_KEY and adjust DATABASE_URL if needed
  ```

### Database
- Create DB and enable pgvector (privileged):
  ```bash
  createdb aletheia
  psql -d aletheia -c 'CREATE EXTENSION IF NOT EXISTS vector;'
  ```

### Install and run
- Install deps and launch server:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn app.main:app --reload
  ```

### Index content
- Add content so RAG has something to retrieve:
  ```bash
  curl -X POST http://127.0.0.1:8000/index-memory \
    -H 'Content-Type: application/json' \
    -d '{
      "content": "Aletheia is our FastAPI + pgvector service. Memory shards store text and embeddings.",
      "user_id": "00000000-0000-0000-0000-000000000001",
      "tags": ["docs","overview"]
    }'
  ```

### Connect OpenWebUI
- In OpenWebUI, add a provider:
  - Base URL: `http://127.0.0.1:8000/v1`
  - API Key: any value (e.g., `aletheia-local`)
  - Model: `gpt-4o` (or whatever you set in `.env`)
- Start chatting; the backend injects retrieved context under the hood.


## 5) Checklist — What to do next

### Retrieval quality & data model
- [ ] Decide on distance metric (L2 vs cosine). For OpenAI embeddings, cosine is common.
- [ ] Add an index (IVFFlat) on `embedding` for faster search at scale.
- [ ] Align ORM to all required fields from `schema.sql` (we implemented `memory_shards` first).
- [ ] Add basic filters (e.g., `user_id`, `tags`) in `semantic_search`.
- [ ] Add a batch ingestion endpoint (`/ingest`) for multiple documents.

### API & UX
- [ ] Support streaming Chat completions (SSE) to match OpenAI behavior in OpenWebUI.
- [ ] Return token usage (prompt/completion) when available.
- [ ] Add a debug flag to include retrieved context snippets in responses for inspection.
- [ ] Provide a `DELETE /memory/:id` and `PUT /memory/:id` for lifecycle management.

### Ops & quality
- [ ] Replace `create_all` with Alembic migrations; document DB migration flow.
- [ ] Dockerfile + docker-compose (API + Postgres + pgvector). Preinstall extension.
- [ ] Tests: unit tests for embeddings utils (mock OpenAI) and integration tests for retrieval.
- [ ] Structured logging, request IDs, and better error handling.
- [ ] Basic rate limiting (per-IP or per-key) and request validation.

### Security & multi-tenancy
- [ ] Authentication: API key or OAuth for write/search endpoints.
- [ ] Enforce `user_id` scoping on reads/writes; consider row-level security policies.
- [ ] Tighter CORS configuration per deployment environment.

### Performance
- [ ] Switch to cosine ops if chosen; add proper index with `lists` parameter.
- [ ] Consider caching embeddings for repeated queries.
- [ ] Add pagination/limits and response shaping for large result sets.

### Documentation
- [ ] Update README with new run instructions and OpenWebUI configuration steps.
- [ ] Add an architecture diagram and a short RAG tutorial.


## 6) Acceptance criteria for Phase 1
- OpenWebUI can connect to `http://localhost:8000/v1` and list the model.
- `/v1/chat/completions` returns answers that reflect indexed content when relevant.
- `/index-memory` stores content, and subsequent queries retrieve it.
- Config via `.env` works end-to-end; CORS allows local UIs.


## 7) Notes and assumptions
- Requires PostgreSQL with `pgvector` extension available.
- Startup attempt to install `vector` extension may fail without privileges; manual creation is recommended in dev/prod.
- Embedding dimension defaults to `1536`; keep consistent with chosen embedding model.
