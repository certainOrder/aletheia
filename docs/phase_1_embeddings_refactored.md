
# Define the content of the refactored Markdown document
refactored_content = """\
# Phase 1: OpenWebUI + Embeddings-backed RAG

**Date:** 2025-09-19

---

## 1. High-Level Overview

OpenWebUI connects via OpenAI-compatible endpoints to a FastAPI backend that augments chat requests using semantic search.

### End-to-End Flow
1. OpenWebUI sends a chat message to `/v1/chat/completions`.
2. API generates an embedding for the user’s message using `text-embedding-3-small`.
3. Vector search retrieves relevant memory shards from the Postgres `pgvector` store.
4. Retrieved context is prepended to the chat message.
5. Augmented chat request is sent to OpenAI’s `gpt-4o` (or configurable model).

**Result:** Enhanced chat answers using contextually relevant private memory.

---

## 2. Current Implementation

### ✅ OpenAI-Compatible Interface
- `GET /v1/models`: lists the available model.
- `POST /v1/chat/completions`: supports OpenAI payloads; integrates RAG context.

### ✅ Retrieval Components
- `POST /index-memory`: indexes text + embedding into `memory_shards`.
- `POST /rag-chat`: test-only endpoint returning `{ answer, context }`.

### ✅ Embedding Logic
- `convert_to_embedding(text)`: OpenAI embeddings API.
- `save_embedding_to_db(...)`: creates `MemoryShard` entry.
- `semantic_search(...)`: nearest-neighbor pgvector search (L2 by default).

### ✅ Data & ORM
- `memory_shards` table via SQLAlchemy.
- `vector(1536)` column (configurable via `.env`).
- Schema auto-created in development if permissions allow.

### ✅ Supporting Features
- CORS enabled via `ALLOWED_ORIGINS`.
- Centralized `.env` config (`DATABASE_URL`, `OPENAI_API_KEY`, etc.).
- `requirements.txt` with all core deps (FastAPI, pgvector, etc.).
- Sample `.env.example`.

---

## 3. System Architecture

- **Frontend:** OpenWebUI → uses `/v1` endpoints.
- **Backend:** FastAPI → parses, embeds, searches, calls OpenAI.
- **Database:** Postgres + pgvector → stores and retrieves `MemoryShard` objects.
- **Embedding:** OpenAI `text-embedding-3-small`.
- **Chat Model:** OpenAI `gpt-4o` (configurable).

---

## 4. Setup Instructions

### Environment
```bash
cp .env.example .env
# Set OPENAI_API_KEY and adjust DATABASE_URL
```

### Database
```bash
createdb aletheia
psql -d aletheia -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

### Install & Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Index Content
```bash
curl -X POST http://127.0.0.1:8000/index-memory \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Aletheia is our FastAPI + pgvector service. Memory shards store text and embeddings.",
    "user_id": "00000000-0000-0000-0000-000000000001",
    "tags": ["docs", "overview"]
  }'
```

### Connect OpenWebUI
- Base URL: `http://127.0.0.1:8000/v1`
- API Key: any non-empty value (e.g., `aletheia-local`)
- Model: match your .env setting

---

## 5. To-Do Checklist

### Retrieval & DB
- Decide on cosine vs L2 distance (cosine is standard for OpenAI).
- Add IVFFlat index for scale.
- Complete ORM coverage from schema.sql.
- Filter by user_id and tags.
- Batch ingestion endpoint (`/ingest`).

### API & Features
- Streaming SSE support.
- Return token usage stats.
- Debug mode (return context).
- Lifecycle routes: PUT/DELETE `/memory/:id`.

### DevOps
- Replace `create_all()` with Alembic.
- Dockerize full stack.
- Add tests (unit + integration).
- Structured logging and error handling.
- Rate limiting.

### Security
- API key / OAuth authentication.
- Row-level user_id enforcement.
- Tighten CORS for prod.

### Performance
- Switch to cosine if selected.
- Cache common embeddings.
- Add pagination for large contexts.

### Docs
# Phase 1: OpenWebUI + Embeddings-backed RAG

Date: 2025-09-19

---

## 1) High-Level Overview

OpenWebUI connects via OpenAI-compatible endpoints to a FastAPI backend that augments chat requests using semantic search over a Postgres pgvector store.

End-to-End Flow
1. OpenWebUI calls `POST /v1/chat/completions`.
2. API embeds the user message (`text-embedding-3-small` by default).
3. Vector search retrieves relevant `memory_shards` from Postgres (pgvector).
4. Retrieved context is prepended to the prompt.
5. Augmented chat request is sent to the chat model (`gpt-4o`, configurable) or a dev fallback.

Result: Contextually grounded answers using private memory.

---

## 2) Current Implementation

OpenAI-Compatible Interface
- `GET /v1/models`: Lists the available model.
- `POST /v1/chat/completions`: Accepts OpenAI-style payloads; integrates retrieved context.

Retrieval Components
- `POST /index-memory`: Indexes text + embedding into `memory_shards`.
- `POST /rag-chat`: Test endpoint returning `{ answer, context }`.

Embedding Logic
- `convert_to_embedding(text)`: Uses OpenAI embeddings or a deterministic local fallback (see DEV_FALLBACKS).
- `save_embedding_to_db(...)`: Persists `MemoryShard` rows.
- `semantic_search(...)`: Nearest-neighbor via pgvector (L2 distance currently).

Data & ORM
- `memory_shards` ORM model (SQLAlchemy) with `embedding vector(1536)`.
- Dimension configurable via `.env` (`EMBEDDING_DIM`).
- Dev startup creates extension/table if permitted.

Supporting Features
- CORS via `ALLOWED_ORIGINS`.
- Central `.env` config (`DATABASE_URL`, `OPENAI_*`, `EMBEDDING_DIM`).
- Deterministic dev fallbacks controlled by `DEV_FALLBACKS`.

---

## 3) System Architecture

- Frontend: OpenWebUI → calls `/v1` endpoints.
- Backend: FastAPI → embeds, searches, augments, calls model.
- Database: Postgres + pgvector → `MemoryShard` storage and retrieval.
- Embedding Model: `text-embedding-3-small` (configurable).
- Chat Model: `gpt-4o` (configurable) or dev fallback when enabled.

---

## 4) Setup & Run

See also: `docs/DEV_ENVIRONMENT.md` for full instructions.

Option A — Docker Compose (recommended)
```bash
cp .env.example .env
# For Compose, set DATABASE_URL to use host `db` and enable fallbacks for quick dev:
# DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/aletheia
# POSTGRES_DB=aletheia
# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=postgres
# DEV_FALLBACKS=true

docker compose up -d --build
```

Smoke tests
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

Option B — Direct host run
```bash
createdb aletheia
psql -d aletheia -c 'CREATE EXTENSION IF NOT EXISTS vector;'

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Index Content (either option)
```bash
curl -X POST http://127.0.0.1:8000/index-memory \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Aletheia is our FastAPI + pgvector service. Memory shards store text and embeddings.",
    "user_id": "00000000-0000-0000-0000-000000000001",
    "tags": ["docs", "overview"]
  }'
```

Connect OpenWebUI
- With Docker Compose: preconfigured to use `http://aletheia-api:8000/v1` inside the network.
- Standalone OpenWebUI: set Base URL to `http://127.0.0.1:8000/v1`. Any non-empty API key works when `DEV_FALLBACKS=true`.

---

## 5) To-Do Checklist

Retrieval & DB
- Decide on cosine vs L2 distance (cosine is standard for OpenAI).
- Add IVFFlat index (and/or HNSW) for scale.
- Filter by `user_id` and `tags`; return distance in results.
- Batch ingestion endpoint (`/ingest`).
- Align ORM with full schema.sql coverage.

API & Features
- Streaming SSE support for chat.
- Return token usage stats.
- Debug mode (return retrieved context verbatim).
- CRUD: PUT/DELETE `/memory/:id`.

DevOps
- Replace `create_all()` with Alembic migrations.
- Remove deprecated `version` from docker-compose.yml.
- Add tests (unit + integration) and CI smoke.
- Structured logging and error handling.
- Rate limiting.

Security
- API key / OAuth authentication.
- Enforce row-level `user_id` access.
- Tighten CORS for prod.

Performance
- Switch to cosine if selected; evaluate normalized vectors.
- Cache common embeddings.
- Pagination for large contexts.

Docs
- Update README with Compose quickstart.
- Add architecture diagram + tutorial.

---

## 6) Acceptance Criteria

- OpenWebUI lists model via `/v1/models`.
- Chat completions include relevant private memory.
- `/index-memory` stores content for retrieval.
- `.env` and CORS work across local dev stack.
- Docker Compose stack boots with healthy services; smoke tests pass.

---

## 7) Assumptions

- PostgreSQL has pgvector available.
- Startup schema creation may require elevated permissions in dev only.
- Embedding dimension (default 1536) must match the configured model.

---

## 8) Key Environment Variables

- `DATABASE_URL`: psycopg URI. Use host `db` inside Compose.
- `OPENAI_API_KEY`: required if `DEV_FALLBACKS=false` and you want real calls.
- `DEV_FALLBACKS`: when true, use deterministic local embeddings and chat.
- `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `EMBEDDING_DIM`, `ALLOWED_ORIGINS`.
