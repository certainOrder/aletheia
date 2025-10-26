# Getting Started (Docker)

This guide shows how to run Aletheia locally using Docker Compose only. No Python setup required.

- API: FastAPI at http://localhost:8000
- DB: Postgres 16 + pgvector at localhost:5432
- UI: OpenWebUI at http://localhost:3000

## Prerequisites
- Docker Desktop (with Compose)
- Ports available: 5432, 8000, 3000

## 1) Clone the repo

```bash
git clone https://github.com/<your-org>/aletheia.git
cd aletheia
```

## 2) Create a .env
A Compose-driven `.env` file is used by the API and the DB. Create it at the repo root:

```bash
cat > .env <<'EOF'
# --- Postgres ---
POSTGRES_DB=aletheia
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# API connects to the DB via the Compose service name `db`
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/aletheia

# --- Provider config ---
# If you have a real key and want to use OpenAI, set it here and leave DEV_FALLBACKS=false
# OPENAI_API_KEY=sk-...

# To run fully offline (no external calls), set this to true
DEV_FALLBACKS=true

# Index chat turns into memory so future queries can recall them
INDEX_CHAT_HISTORY=true

# Optional: enable the /api/debug/search endpoint for quick checks
# ENABLE_DEBUG_ENDPOINTS=true
EOF
```

Notes:
- If `.env.example` exists in your clone, you can start from that instead.
- The repo contains `docker-compose.override.yml` which may set `DEV_FALLBACKS=false` by default. If you want offline runs, either set `DEV_FALLBACKS=true` in your `.env` and remove the override entry locally, or temporarily flip it to `true` in the override file.

## 3) Build and start the stack

```bash
docker compose build --no-cache
docker compose up -d
```

Check container status:

```bash
docker compose ps
```

## 4) Smoke check the API

- Health: http://localhost:8000/api/health
- Models list (OpenAI-compatible): http://localhost:8000/v1/models

```bash
# Optional curl checks
curl -sS http://localhost:8000/api/health
curl -sS http://localhost:8000/v1/models | jq
```

If `DEV_FALLBACKS=false` and you didn’t set `OPENAI_API_KEY`, chat and embedding requests will return HTTP 500 by design. For local/offline use, set `DEV_FALLBACKS=true`.

## 5) OpenWebUI

Open http://localhost:3000.

To scope retrieval to your user (recommended), add a custom header so the API can link you to a DB user profile automatically:
- Key: `X-User-Id`
- Value: your preferred username or id (e.g., `alice`)

Where to set headers:
- OpenWebUI → Settings → Providers (or Model settings) → Custom headers

With `INDEX_CHAT_HISTORY=true`, the API will automatically index your chat turns. You don’t need to say “Remember: …” — normal statements like “My favorite foods are sushi and tacos.” are saved and will be retrieved on follow-up.

## 6) Quick end-to-end test

In OpenWebUI (with `X-User-Id` set):
- Say: “My favorite foods are sushi and tacos.”
- Then: “What are my favorite foods?”

You should see the correct answer, and the API response includes `aletheia_context` with retrieved snippets.

Optional: enable `ENABLE_DEBUG_ENDPOINTS=true` in your `.env` and restart the API, then:

```bash
curl -sS 'http://localhost:8000/api/debug/search?q=favorite%20foods&top_k=5&user_id=alice' | jq
```

## 7) Stopping and cleaning up

```bash
docker compose down              # stop containers
# If you want to reset the DB completely (drop volumes):
docker compose down -v
```

## 8) Switching between real OpenAI and offline

- Real OpenAI
  - Set `DEV_FALLBACKS=false` (in `.env` and/or override) and provide `OPENAI_API_KEY`.
  - Rebuild or restart the API container.
- Offline/local fallbacks
  - Set `DEV_FALLBACKS=true`. No external calls will be made; responses/embeddings are deterministic.

## Troubleshooting
- 500 errors on chat/embeddings: typically `DEV_FALLBACKS=false` without `OPENAI_API_KEY`.
- Port conflicts: ensure 5432/8000/3000 are free or edit port mappings in `docker-compose.yml`.
- DB connection errors: confirm `DATABASE_URL` in `.env` points to `db` (service name), not `localhost`.
- Data not recalled across new chats: set the `X-User-Id` header so shards are scoped to your user; indexing will still happen unscoped if omitted.
- Need a clean slate: `docker compose down -v` to drop volumes and restart.

## What’s running
- `db` (Postgres 16 + pgvector)
- `aletheia-api` (FastAPI, OpenAI-compatible endpoints)
- `openwebui` (UI bound to the API’s `/v1` endpoints)

For deeper docs, see `docs/` in the repo (architecture, migrations, dev environment). Enjoy!
