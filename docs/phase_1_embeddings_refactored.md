
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
- Update README.
- Add architecture diagram + tutorial.

---

## 6. Acceptance Criteria

- ✅ OpenWebUI lists model via `/v1/models`.
- ✅ Chat completions include relevant private memory.
- ✅ `/index-memory` stores content for retrieval.
- ✅ `.env` and CORS work across local dev stack.

---

## 7. Assumptions

- PostgreSQL server must support pgvector.
- Startup schema creation may require elevated permissions.
- Embedding dimension (default 1536) must match model used.
```