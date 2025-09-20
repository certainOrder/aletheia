# Phase 2 Plan: “Real model” testing, retrieval, and UX

Date: 2025-09-20

This phase moves from scaffolding to production-like behavior: use the real OpenAI provider end-to-end, improve retrieval quality/perf, harden migrations/tests/CI, and refine the UX via OpenWebUI.

Key context updates since initial draft:
- Frontend is OpenWebUI pointing at our OpenAI-compatible `/v1` endpoints; the static `chat.html` is not used.
- Alembic migrations exist (0001/0002) and are applied in Docker; we will extend them for retrieval indexes/fields.
- Observability is in place (structured JSON logs, `request_id`), and we’ve verified provider="openai" in `embed_*` and `chat_*` logs with `DEV_FALLBACKS=false`.

## Goals
- Exercise embeddings and chat with a real `OPENAI_API_KEY` (no local fallback in Phase 2 paths)
- Improve retrieval quality and performance (cosine similarity + IVFFlat)
- Extend migrations and keep CI/tests reliable (≥85% coverage target)
- Add optional streaming and clear grounding visibility in OpenWebUI
 - Incorporate recent conversation history (e.g., last 5 turns) per architecture overview

## Milestones & Tasks

### M1: Real OpenAI usage, safely
- Enforce provider readiness when `DEV_FALLBACKS=false`: log a startup warning if `OPENAI_API_KEY` is missing and return a clear 500 at request time with an actionable message and docs URL.
- Update docs: secure key handling, `.env.example` with defaults, Docker Compose env notes for API and OpenWebUI (UI may use any non-empty key; server reads real key from `.env`).
- Add a minimal unit test for provider selection (openai vs fallback) and a smoke test for `/v1/chat/completions`.
- Acceptance:
  - With a valid key and `DEV_FALLBACKS=false`, `/v1/models`, `/v1/chat/completions`, and embeddings succeed; logs show `provider:"openai"`.
  - With missing/invalid key and `DEV_FALLBACKS=false`, endpoints return HTTP 500 with a clear message; logs include `error` and `request_id`.

### M2: Retrieval quality & performance
- Switch to cosine similarity; include score in results and expose in `aletheia_context` payload.
- Add IVFFlat index for `pgvector` with configurable `lists` via env; document build timing and ANALYZE guidance.
- Introduce chunking for long content (configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`).
- Conversation history: include the last N (default 5) user/assistant turns in prompt assembly, aligned with the architecture overview.
- Acceptance:
  - `semantic_search` returns top-k by cosine score (descending), scores included in API response.
  - IVFFlat index exists; query plan uses it for typical `top_k` queries.
  - Ingesting large text creates multiple chunks and improves recall in tests.
  - The final prompt includes the last N conversation turns; N is configurable and enforced with a token budget.

#### M2 Implementation Checklist

- Retrieval metric and scores
  - [x] Add `SIMILARITY_METRIC=cosine` to config with default `cosine`.
  - [x] Update `semantic_search` to use cosine distance for ordering (`pgvector` cosine).
  - [x] Include `score` in each returned context item and in logs; document score semantics.
  - [x] Expose scores in `/v1/chat/completions` response under `aletheia_context`.

- IVFFlat index for performance
  - [x] Add `PGVECTOR_IVFFLAT_LISTS` to config (default 100) and doc trade-offs.
  - [x] Create Alembic migration `0003` to add IVFFlat index on `memory_shards.embedding` (cosine opclass).
  - [x] Optionally gate index creation behind an env toggle to defer building on empty DBs (`PGVECTOR_ENABLE_IVFFLAT`).
  - [x] Add a small script or docs snippet to `ANALYZE` after index creation.

- Chunking & ingestion
  - [x] Add `CHUNK_SIZE` and `CHUNK_OVERLAP` to config with sensible defaults (e.g., 800/100).
  - [x] Implement a chunker utility for long text ingestion (sentence-aware heuristic with fallback slicing).
  - [x] Add `/ingest` endpoint to split content and create multiple shards; env overrides respected at request time.
  - [ ] Ensure chunk tags/metadata are propagated; add `source`/`metadata` fields if needed (covered in M3).

- Conversation history & token budget
  - [x] Add `HISTORY_TURNS` to config (default 5).
  - [x] Assemble prompt with last N turns + retrieved context.
  - [x] Implement a simple token budget and truncate history/context if above ceiling (configurable, e.g., 40k tokens).
  - [x] Log truncation decisions and final prompt token count for observability.

- Observability & logs
  - [x] Log search metric used, `top_k`, and include `score` per item in debug logs.
  - [x] Add structured events for `index_build`, `analyze`, and `retrieval_scores`.

- Tests
  - [x] Unit: verify ordering by cosine score (descending) and score presence in API output.
  - [x] Unit: chunking splits long input into multiple shards as expected.
  - [x] Unit: prompt assembly respects `HISTORY_TURNS` and token budget; truncation occurs when needed.
  - [ ] Migration: ensure Alembic upgrade/downgrade for IVFFlat runs without data loss. (Deferred: covered by container smoke; add focused Alembic test in M3.)
  - [x] Offline-friendly: mock OpenAI where required; avoid network in CI.

- Docs
  - [x] Update `DEV_ENVIRONMENT.md` with new envs and tuning tips (metric, lists, chunk sizes, history, token budget).
  - [x] Update `architecture_overview.md` references to explicitly note cosine + IVFFlat and score exposure.
  - [x] Add a short `DB_MIGRATIONS.md` note about IVFFlat creation timing and ANALYZE guidance.

### M3: Schema migrations (extensions)
- Use existing Alembic setup to add: IVFFlat index operations, `source` and `metadata` (JSONB) on content table(s) as needed.
- Ensure `raw_conversations` persistence on chat calls (request/response metadata, model, provider, request_id) matches the architecture.
- Provide forward/backward migrations with sane defaults and data backfill where applicable.
- Acceptance:
  - `alembic upgrade head` bootstraps a fresh DB including new fields/indexes.
  - `alembic downgrade -1` works at least one step without data loss for core tables.

#### M3 Implementation Checklist

- Schema: content metadata & indexing
  - [ ] Add `source TEXT NULL` to `memory_shards` (optional free-form origin, e.g., url/file)
  - [ ] Add `metadata JSONB` to `memory_shards` (nullable or default `{}`); store arbitrary key/values
  - [ ] Add BTree index on `memory_shards.user_id` (if not already present) for scoped queries
  - [ ] Add GIN index on `memory_shards.metadata` (jsonb_path_ops) for key filtering (optional; doc trade-offs)
  - [ ] Confirm/retain `embedding` vector index (HNSW/IVFFlat as configured) is unchanged by migration

- New table: raw_conversations
  - [ ] Create `raw_conversations` with columns:
    - `id UUID PK`, `created_at TIMESTAMPTZ DEFAULT now()`
    - `request_id TEXT`, `user_id TEXT NULL`, `provider TEXT`, `model TEXT`
    - `messages JSONB` (request payload), `response JSONB` (assistant reply + usage)
    - `status_code INT` (HTTP response code), `latency_ms INT` (optional)
  - [ ] Indexes: BTree on `(created_at)`, `(user_id)`, and optional `(request_id)`

- Alembic migration(s)
  - [ ] Create revision `0004_add_source_metadata_and_raw_conversations`
  - [ ] Upgrade: add columns to `memory_shards`, backfill `metadata` to `{}` where NULL (if default not set)
  - [ ] Upgrade: create `raw_conversations` table + indexes
  - [ ] Downgrade: drop indexes, drop `raw_conversations`, drop columns from `memory_shards`
  - [ ] Ensure idempotence and safety on existing populated DBs (avoid table/column re-creation)

- Ingestion & API propagation
  - [ ] Extend `IndexMemoryRequest` and `IngestRequest` to accept optional `source` and `metadata: dict[str, Any]`
  - [ ] Propagate `source`/`metadata` to `save_embedding_to_db`
  - [ ] Ensure `/ingest` propagates original `tags`, `source`, `metadata` to each chunk
  - [ ] Expose `source`/`metadata` in retrieval results and `aletheia_context`

- Raw conversations persistence (app layer)
  - [ ] In `/v1/chat/completions`, insert a `raw_conversations` record containing: request `messages`, selected `model`, provider, `request_id`, response object, status, and duration
  - [ ] Ensure behavior is offline-friendly with `DEV_FALLBACKS=true` (no network required)
  - [ ] Add log hooks: `raw_conversations_saved` with `id`, `request_id`

- Tests
  - [ ] Migration smoke: `alembic upgrade head` and `downgrade -1` roundtrip on temp DB without data loss
  - [ ] Ingestion propagation: `/index-memory` and `/ingest` persist `source`/`metadata`; retrieval returns them
  - [ ] API response: `aletheia_context` items include `source`/`metadata` when present
  - [ ] Raw conversations: invoking `/v1/chat/completions` creates a `raw_conversations` row with expected fields (mock time for latency determinism)
  - [ ] Offline-friendly: all tests run with `DEV_FALLBACKS=true` and no external calls

- Docs
  - [ ] `architecture_overview.md`: add `source`/`metadata` fields to memory shard description and note `raw_conversations` persistence
  - [ ] `DEV_ENVIRONMENT.md`: document new request fields and example payloads
  - [ ] `DB_MIGRATIONS.md`: add revision notes for `0004`, backfill behavior, and indexes

- Acceptance
  - [ ] Fresh DB: `alembic upgrade head` includes new columns/table; API starts and tests pass
  - [ ] Existing DB: migration applies without data loss; downgrades cleanly one step
  - [ ] `/ingest` and `/index-memory` store & surface `source`/`metadata`; `/v1/chat/completions` logs to `raw_conversations`

### M4: Tests & CI
- Unit tests: deterministic fallback; search ordering; provider selection; basic chat response shape.
- Integration (optional/offline-friendly): mock OpenAI or use deterministic fallbacks; tmp Postgres with `pgvector`; minimal ingest + query.
- CI: GitHub Actions workflow runs lint, typecheck, tests with coverage gate (≥85%).
- Acceptance:
  - PRs run tests automatically and pass the coverage threshold.
  - No network calls in tests unless explicitly allowed; fallbacks/mocks used.

### M5: Streaming & UX improvements (OpenWebUI)
- Add `stream=true` support to `/v1/chat/completions` (SSE) with OpenAI-compatible `delta` payloads.
- Ensure retrieved grounding context is returned in API responses (already present as `aletheia_context`) and surfaced within OpenWebUI.
- Optionally disable/silence Ollama provider probes in OpenWebUI to reduce log noise.
- Acceptance:
  - With `stream=true`, clients receive token deltas via SSE; OpenWebUI renders streaming responses.
  - Grounding snippets visible alongside answers in OpenWebUI.

## Implementation Notes
- Config additions: `SIMILARITY_METRIC=cosine`, `PGVECTOR_IVFFLAT_LISTS=100`, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`.
- Postgres: ensure `pgvector`; consider creating IVFFlat index post-ingest or run `ANALYZE` to avoid poor plans; make `lists` configurable.
- SSE streaming: prefer FastAPI `EventSourceResponse` or chunked responses; keep OpenAI-compatible event shapes.
- Observability: continue structured logs; verify `provider` and search metrics in logs; include scores in logs for search ops.
 - Conversation history: add `HISTORY_TURNS=5` and a simple token budget to cap assembled prompts (e.g., 40k), per architecture overview.

## Risks & Mitigations
- API quota/costs: enable rate limits; keep `DEV_FALLBACKS=true` by default in `.env.example` and document toggling for local dev.
- Index build time: document trade-offs; provide env flag to defer creation; expose `lists` and `top_k` tuning knobs.
- Streaming complexity: ship behind a feature flag until stabilized; keep non-streaming path as default.
- OpenWebUI provider probes: disable non-used providers (e.g., Ollama) or ignore benign probe errors.
 - Token budget risk: ensure conversation history + context stay below token ceiling; make budget configurable and log truncation decisions.

## Rollout
- Work on `feature/phase-2` (already created) or `feature/phase-2-real-models` if preferred; small, focused PRs per milestone.
- Update `DEV_ENVIRONMENT.md` with key validation behavior, logging verification (`provider:"openai"`), and streaming test tips.
- Tag `v0.2.0` after completing M1–M3; M4–M5 can follow in minor increments if needed.
