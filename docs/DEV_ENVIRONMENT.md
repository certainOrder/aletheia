# Development Environment

This doc explains how to run Aletheia locally for development, both with Docker Compose and directly on your host, and how to perform quick smoke tests.

## Prerequisites

- Docker Desktop (recommended path) or a working local Python 3.11 environment
- macOS (tested), zsh shell
- Optional: an OpenAI API key if you want to use real API calls instead of the built-in dev fallbacks

## Environment variables

Copy `.env.example` to `.env` and adjust as needed.

- `DATABASE_URL`:
  - Direct (no Docker): `postgresql+psycopg://postgres:postgres@localhost:5432/aletheia`
  - Docker Compose: `postgresql+psycopg://postgres:postgres@db:5432/aletheia`
- `OPENAI_API_KEY`: Only needed if you want real OpenAI calls in dev.
- `OPENAI_CHAT_MODEL`: Defaults to `gpt-4o`.
- `OPENAI_EMBEDDING_MODEL`: Defaults to `text-embedding-3-small`.
- `EMBEDDING_DIM`: Defaults to `1536` and must match the DB vector column dim.
- `ALLOWED_ORIGINS`: Comma-separated list for CORS.
- `DEV_FALLBACKS`: When `true`, the API uses deterministic local fallbacks for embeddings and chat so you can develop without an OpenAI key. Defaults to `false` in `.env.example`; set to `true` for local Docker runs in `.env`.
- `SIMILARITY_METRIC`: Retrieval metric; defaults to `cosine`.
- `CHUNK_SIZE`: Maximum characters per chunk during ingestion; defaults to `800`.
- `CHUNK_OVERLAP`: Overlapping characters between consecutive chunks; defaults to `100`.
 - `PGVECTOR_ENABLE_IVFFLAT`: When `true` (default), Alembic migration `0003` creates an IVFFlat index on `memory_shards.embedding` using the cosine opclass.
 - `PGVECTOR_IVFFLAT_LISTS`: Number of lists for IVFFlat (default `100`). Higher values improve recall at the cost of index size and build time.

### Recommended `.env` for Docker Compose

```
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/aletheia
POSTGRES_DB=aletheia
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DEV_FALLBACKS=true
```

## Run with Docker Compose (recommended)

1) Build and start services:

```bash
docker compose up -d --build
```

Services:
- Postgres with pgvector: `localhost:5432`
- API (FastAPI): `http://localhost:8000`
- OpenWebUI: `http://localhost:3000`

2) Verify health:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

3) Index some content:

```bash
curl -s -X POST http://localhost:8000/index-memory \
  -H 'Content-Type: application/json' \
  -d '{"content":"OpenAI released new embedding models.","metadata":{"source":"smoke-test"}}'
```

4) RAG chat query:

```bash
curl -s -X POST http://localhost:8000/rag-chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"OpenAI released new embedding models.","top_k":3}'
```

5) OpenAI-compatible chat completions (for OpenWebUI):

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"OpenAI released new embedding models."}]}'
```

If `DEV_FALLBACKS=true`, responses are generated locally; otherwise, real OpenAI calls are made (requires `OPENAI_API_KEY`).

### IVFFlat tuning and ANALYZE

- The IVFFlat index is created by migration `0003` when `PGVECTOR_ENABLE_IVFFLAT=true`.
- You can tune the number of lists with `PGVECTOR_IVFFLAT_LISTS` (e.g., 100–2048 depending on corpus size).
- After index creation or large ingests, run `ANALYZE` to help the planner choose the index:

```bash
docker compose exec db bash -lc "psql -U $POSTGRES_USER -d $POSTGRES_DB -c 'ANALYZE memory_shards'"
```

---

## Observability & Logs

Use the API container logs to trace the full flow (request → embed → retrieval → OpenAI → response).

Tail logs:

```bash
docker compose logs -f aletheia-api
```

Key log events (JSON formatted):
- `http_request` — request/response line with `method`, `path`, `status_code`, `duration_ms`, `request_id`.
- `route_v1_chat_completions` — OpenAI-compatible chat entry with `messages_count`, `model`.
- `retrieval_begin` — when retrieval is triggered with `query_len`.
- `embed_request` / `embed_result` — embedding provider and dimension (no content logged).
- `semantic_search` — retrieval stats: `limit`, `user_id`, `result_count`.
- `chat_request` / `chat_result` — provider used and result choice count.
- `completion_result` — final completion with `model`, `content_len`, `context_count`.

Each log line includes a `request_id` header for correlation. The response also returns `X-Request-ID`.

## Run directly on host (advanced)

1) Ensure Postgres with pgvector is available locally and `DATABASE_URL` points to it.
2) Create a Python 3.11 venv and install deps:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Start API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Run the same smoke tests listed above (use your local Postgres connection settings).

## Troubleshooting

- Compose warning: `version` is obsolete — safe to ignore; we’ll remove it later.
- DB fails to start: ensure `POSTGRES_DB/USER/PASSWORD` are set in `.env` and `DATABASE_URL` host is `db`.
- 401 from OpenAI: set `DEV_FALLBACKS=true` in `.env` for local development or provide a valid `OPENAI_API_KEY`.
- No context retrieved in RAG: ensure you indexed content first via `/index-memory`.

## Use real OpenAI calls (disable fallbacks)

By default in local Docker runs, we recommend `DEV_FALLBACKS=true` so you can test without external keys. To exercise real models end-to-end:

1) Set your OpenAI API key in `.env` and disable fallbacks:

```
OPENAI_API_KEY=sk-...
DEV_FALLBACKS=false
```

2) Rebuild and restart (if using Docker Compose):

```bash
docker compose up -d --build
```

3) Verify endpoints now use OpenAI:

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi from a real model."}]}' | jq .
```

If the key is missing/invalid while `DEV_FALLBACKS=false`, you should see a clear error. Re-enable fallbacks by setting `DEV_FALLBACKS=true` to return to deterministic local behavior.

## Notes

- Dev fallbacks are meant for local usage only. Consider guarding these behind `DEV_FALLBACKS=false` in production deployments.
- Current similarity metric is L2 distance. Consider adding cosine distance and IVFFlat index for scale.

## Logging and Error Model (dev)

- Structured JSON logs are emitted to stdout. Control verbosity via `LOG_LEVEL` (e.g., `DEBUG`, `INFO`).
- Correlation IDs: set `X-Request-ID` in requests to propagate that value to logs and responses; otherwise a UUID is generated.
- Error responses are consistent:

```json
{
  "error": "InternalServerError",
  "detail": "An unexpected error occurred.",
  "status": 500,
  "request_id": "dev-trace-1"
}
```

Try it:

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: dev-trace-1' \
  -d '{"messages":[]}' | jq .
```
