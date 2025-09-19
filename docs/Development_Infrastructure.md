# Development Infrastructure Plan — OpenWebUI + Postgres (Docker)

Date: 2025-09-19

References:
- OpenWebUI docs: https://github.com/open-webui/open-webui
- pgvector image: https://hub.docker.com/r/pgvector/pgvector

## Goals
- Run the full stack locally with minimal setup: Postgres (with pgvector), our API, and OpenWebUI.
- Keep data persistent via Docker volumes.
- Be easy to extend for remote/deployed environments (Chromebook compatibility concerns).

## Overview
- `postgres` (with `pgvector`) stores `memory_shards` and other tables.
- `aletheia-api` (this repo) exposes `/v1/*` OpenAI-compatible endpoints and RAG endpoints.
- `openwebui` provides the chat UI, connecting to our API via OpenAI-compatible interface.
- All services share a Docker network and use service names for DNS (e.g., `db`, `aletheia-api`).

## Directory and env layout
- `.env` in repo root: API config and secrets for local dev (not committed)
- `docker-compose.yml` in repo root
- Named volumes for Postgres and OpenWebUI data

Example `.env` (dev)
```env
# API
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/aletheia
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080

# Postgres
POSTGRES_DB=aletheia
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# OpenWebUI (optional defaults; can also configure in UI)
# If set, OpenWebUI will point to our API inside the compose network and can skip auth for local dev.
# OPENAI_API_BASE_URL=http://aletheia-api:8000/v1
# OPENAI_API_KEY=dev-placeholder-key
# WEBUI_AUTH=false
# TZ=UTC
```

## Docker Compose (proposed)
Create `docker-compose.yml`:

```yaml
version: "3.9"

services:
  db:
    image: pgvector/pgvector:pg16
    container_name: aletheia-db
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 10

  aletheia-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: aletheia-api
    env_file: .env
    environment:
      # Ensure the API uses the in-network DB hostname
      DATABASE_URL: ${DATABASE_URL}
    depends_on:
      db:
        condition: service_healthy
    ports:
      - "8000:8000"
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: openwebui
    restart: unless-stopped
    environment:
      # Preconfigure an OpenAI-compatible provider that targets our API inside the Compose network.
      # Use defaults if not provided in .env; remove these to configure manually in the UI.
      OPENAI_API_BASE_URL: ${OPENAI_API_BASE_URL:-http://aletheia-api:8000/v1}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-dev-placeholder-key}
      TZ: ${TZ:-UTC}
      # Optional: disable auth for local dev (use with caution)
      # WEBUI_AUTH: ${WEBUI_AUTH:-false}
    depends_on:
      - aletheia-api
    ports:
      - "3000:8080"  # OpenWebUI UI on localhost:3000
    volumes:
      - openwebui-data:/app/backend/data
    # Optional: enable GPU (requires NVIDIA Container Toolkit on host)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]

volumes:
  pgdata:
  openwebui-data:
```

Notes:
- `pgvector/pgvector:pg16` ships with the extension, removing the need to install it manually.
- `aletheia-api` uses the in-network hostname `db` via `DATABASE_URL` (see `.env` example above).
- OpenWebUI uses the official image. We preconfigure it via env vars to point to our API; remove those lines to set the provider in the UI after first run.

## Dockerfile (proposed)
Create a minimal Dockerfile for the API:

```Dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps (optional): build tools, libpq for psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Bring-up (local)
1) Create `.env` from `.env.example` and set `OPENAI_API_KEY`.
2) Start the stack:
```bash
docker compose up -d --build
```
3) Verify services:
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- OpenWebUI: http://localhost:3000
4) Ensure `pgvector` is available (should be included in the image):
```bash
docker compose exec db psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
5) Index some content:
```bash
curl -X POST http://127.0.0.1:8000/index-memory \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Aletheia is our FastAPI + pgvector service.",
    "user_id": "00000000-0000-0000-0000-000000000001",
    "tags": ["docs","overview"]
  }'
```
6) Configure OpenWebUI provider in the UI (if not pre-configured):
- Base URL: `http://aletheia-api:8000/v1` (inside compose network) or `http://localhost:8000/v1` from your browser
- API Key: any non-empty string (server uses real key from `.env`)
- Model: as listed by `/v1/models`

## Troubleshooting
- OpenWebUI cannot list models:
  - Check API is up: `curl http://localhost:8000/v1/models`
  - Ensure OpenWebUI points to `http://aletheia-api:8000/v1` inside Compose, or `http://localhost:8000/v1` from browser.
- Embeddings fail:
  - Confirm `OPENAI_API_KEY` is set and available to the API container.
  - Inspect logs: `docker compose logs -f aletheia-api`
- Postgres connection issues:
  - Verify `DATABASE_URL` uses host `db` when running in Compose.

## Chromebook considerations (Crostini)
Some ChromeOS devices may not support Docker smoothly. Options:

1) Remote Docker host
- Provision a small VM (e.g., Oracle Always Free, Fly.io Machines, Hetzner, Lightsail) and install Docker & docker-compose.
- Clone repo, copy `.env` (set real secrets), and run `docker compose up -d`.
- Expose OpenWebUI on an HTTPS domain via a reverse proxy (Caddy/Traefik) and firewall rules.
- Secure SSH access; consider Tailscale/Zerotier to avoid public exposure.

2) GitHub Codespaces / Dev Containers
- Use Codespaces or a remote dev container that supports Docker.
- Run the compose stack in that environment and expose forwarded ports.

3) Split architecture
- Host `db` + `aletheia-api` remotely.
- Run OpenWebUI locally in the browser, pointing to the remote API base URL.

Security notes:
- Never commit real secrets; use `.env` locally and secrets in environment on remote.
- Restrict inbound ports; prefer TLS and private networking when possible.

## Future expansion
- Add a `docker-compose.override.yml` for local tweaks.
- Add `Makefile` targets: `make up`, `make down`, `make logs`, `make psql`.
- Use Alembic migrations container/service.
- Add Traefik/Caddy reverse proxy with HTTPS for remote deployments.
- CI/CD pipeline to build/push images (GitHub Actions) and deploy to a target host.

## Acceptance criteria
- `docker compose up -d --build` starts Postgres, the API, and OpenWebUI locally.
- OpenWebUI can list models and chat via the API.
- Data persists across restarts (named volumes).
- Chromebook path has at least one workable remote/deployed option documented.
