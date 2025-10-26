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
  - [x] Update `ARCHITECTURE.md` references to explicitly note cosine + IVFFlat and score exposure.
  - [x] Add a short `DB_MIGRATIONS.md` note about IVFFlat creation timing and ANALYZE guidance.

### M3: Schema migrations (extensions)
- Use existing Alembic setup to add: IVFFlat index operations, `source` and `metadata` (JSONB) on content table(s) as needed.
- Ensure `raw_conversations` persistence on chat calls (request/response metadata, model, provider, request_id) matches the architecture.
- Provide forward/backward migrations with sane defaults and data backfill where applicable.
- **Status**: ✅ **COMPLETE** (2025-10-10)
- Acceptance:
  - ✅ `alembic upgrade head` bootstraps a fresh DB including new fields/indexes.
  - ✅ `alembic downgrade -1` works at least one step without data loss for core tables.
  - ✅ M3 fields (source, metadata) persist and surface in `aletheia_context`.
  - ✅ Raw conversations logging operational.
  - ✅ Tests pass (86% coverage); migrations idempotent.

#### M3 Implementation Checklist

- Schema: content metadata & indexing
  - [x] Add `source TEXT NULL` to `memory_shards` (optional free-form origin, e.g., url/file)
  - [x] Add `metadata JSONB` to `memory_shards` (nullable or default `{}`); store arbitrary key/values
  - [x] Add BTree index on `memory_shards.user_id` (if not already present) for scoped queries
  - [x] Add GIN index on `memory_shards.metadata` (jsonb_path_ops) for key filtering (optional; doc trade-offs)
  - [x] Confirm/retain `embedding` vector index (HNSW/IVFFlat as configured) is unchanged by migration

- New table: raw_conversations
  - [x] Create `raw_conversations` with columns:
    - `id UUID PK`, `created_at TIMESTAMPTZ DEFAULT now()`
    - `request_id TEXT`, `user_id TEXT NULL`, `provider TEXT`, `model TEXT`
    - `messages JSONB` (request payload), `response JSONB` (assistant reply + usage)
    - `status_code INT` (HTTP response code), `latency_ms INT` (optional)
  - [x] Indexes: BTree on `(created_at)`, `(user_id)`, and optional `(request_id)`

  Status: Table existed from 0002; augmented in 0005 with `created_at`, `request_id`, `provider`, `model`, `messages`, `response`, `status_code`, and `latency_ms`. Indexes created on `created_at`, `user_id`, and `request_id`.

- Alembic migration(s)
  - [x] Create revision `0004_add_source_metadata_and_raw_conversations` (partial: added source/metadata + indexes; raw_conversations pending)
  - [x] Upgrade: add columns to `memory_shards` (source, metadata), and create indexes (user_id btree, metadata gin)
  - [x] Upgrade: create/augment `raw_conversations` with required columns + indexes (via 0005)
  - [x] Downgrade: drop indexes and columns added in this revision
  - [x] Ensure idempotence and safety on existing populated DBs (IF NOT EXISTS guards for indexes)

- Ingestion & API propagation
  - [x] Extend `IndexMemoryRequest` and `IngestRequest` to accept optional `source` and `metadata: dict[str, Any]`
  - [x] Propagate `source`/`metadata` to `save_embedding_to_db`
  - [x] Ensure `/ingest` propagates original `tags`, `source`, `metadata` to each chunk
  - [x] Expose `source`/`metadata` in retrieval results and `aletheia_context`

- Raw conversations persistence (app layer)
  - [x] In `/v1/chat/completions`, insert a `raw_conversations` record containing: request `messages`, selected `model`, provider, `request_id`, response object, status, and duration
  - [x] Ensure behavior is offline-friendly with `DEV_FALLBACKS=true` (no network required)
  - [x] Add log hooks: `raw_conversations_saved` with `id`, `request_id`

- Tests
  - [x] Migration smoke: `alembic upgrade head` and `downgrade -1` roundtrip on temp DB without data loss
  - [x] Ingestion propagation: `/index-memory` and `/ingest` persist `source`/`metadata`; retrieval returns them
  - [x] API response: `aletheia_context` items include `source`/`metadata` when present
  - [x] Raw conversations: invoking `/v1/chat/completions` creates a `raw_conversations` row with expected fields (mock time for latency determinism)
  - [x] Offline-friendly: all tests run with `DEV_FALLBACKS=true` and no external calls

- Docs
  - [x] `ARCHITECTURE.md`: add `source`/`metadata` fields to memory shard description and note `raw_conversations` persistence
  - [x] `DEV_ENVIRONMENT.md`: document new request fields and example payloads
  - [x] `DB_MIGRATIONS.md`: add revision notes for `0004`, `0005`, backfill behavior, and indexes

- Acceptance
  - [x] Fresh DB: `alembic upgrade head` includes new columns/table; API starts and tests pass
  - [x] Existing DB: migration applies without data loss; downgrades cleanly one step
  - [x] `/ingest` and `/index-memory` store & surface `source`/`metadata`
  - [x] `/v1/chat/completions` logs to `raw_conversations`

### M4: Tests & CI
- Unit tests: deterministic fallback; search ordering; provider selection; basic chat response shape.
- Integration (optional/offline-friendly): mock OpenAI or use deterministic fallbacks; tmp Postgres with `pgvector`; minimal ingest + query.
- CI: GitHub Actions workflow runs lint, typecheck, tests with coverage gate (≥85%).
- Acceptance:
  - PRs run tests automatically and pass the coverage threshold.
  - No network calls in tests unless explicitly allowed; fallbacks/mocks used.

#### M4 Implementation Checklist

- Unit test coverage expansion
  - [ ] Test provider selection logic: verify `DEV_FALLBACKS=true` uses fallback, `=false` with valid key uses OpenAI
  - [ ] Test search ordering: verify results sorted by cosine score (descending); scores present in response
  - [ ] Test chat response shape: validate OpenAI-compatible structure (choices, model, usage, aletheia_context)
  - [ ] Test chunking edge cases: empty input, single chunk, exact boundaries, overlap behavior
  - [ ] Test token budget enforcement: verify truncation when history + context exceeds limit
  - [ ] Test error handling: missing API key returns 500 with actionable message; fallback errors handled gracefully

- Integration test suite (offline-friendly)
  - [ ] Create integration test fixture with temporary Postgres + pgvector (or use existing dummy_db)
  - [ ] Test full ingest → query flow: index content, perform semantic search, verify top-k retrieval
  - [ ] Test migration roundtrip: fresh DB upgrade to head, verify schema; downgrade one step, verify no data loss
  - [ ] Test end-to-end chat: index memory, call `/v1/chat/completions`, verify context injection and raw_conversations logging
  - [ ] Ensure all integration tests use `DEV_FALLBACKS=true` or mocked OpenAI responses (no network calls)

- CI/CD setup (GitHub Actions)
  - [ ] Create `.github/workflows/ci.yml` workflow file
  - [ ] Job: Lint - run `ruff check` on all Python files
  - [ ] Job: Format check - run `black --check` and `ruff format --check`
  - [ ] Job: Type check - run `mypy` with strict config
  - [ ] Job: Tests - run `pytest` with coverage reporting
  - [ ] Coverage gate: fail if coverage < 85% (use `pytest-cov` with `--cov-fail-under=85`)
  - [ ] Matrix strategy: test on Python 3.11 (add 3.12 if desired)
  - [ ] Service container: Postgres 16 with pgvector for integration tests
  - [ ] Environment: set `DEV_FALLBACKS=true` and mock DB credentials for CI runs
  - [ ] Trigger: on push to `main` and `feature/*` branches, and on pull requests

- Test configuration & tooling
  - [ ] Add `pytest.ini` or `pyproject.toml` config: coverage settings, test discovery patterns
  - [ ] Ensure `conftest.py` fixtures are reusable across unit and integration tests
  - [ ] Add `--cov=app --cov-report=term-missing` to pytest invocation for detailed coverage
  - [ ] Document test commands in `README.md` or `DEV_ENVIRONMENT.md` (e.g., `make test`, `pytest -v`)

- Test improvements & missing coverage
  - [ ] Identify gaps: run `pytest --cov=app --cov-report=html` and review uncovered lines
  - [ ] Add tests for `/ingest` endpoint: chunk propagation, source/metadata persistence
  - [ ] Add tests for `/index-memory`: direct embedding save, metadata handling
  - [ ] Add tests for error paths: invalid JSON, missing required fields, DB connection failures
  - [ ] Add tests for migration idempotence: run upgrade twice, verify no errors

- Documentation updates
  - [ ] `README.md`: Add "Running Tests" section with commands and coverage threshold
  - [ ] `DEV_ENVIRONMENT.md`: Document test environment setup, fixtures, and offline testing strategy
  - [ ] `.github/workflows/ci.yml`: Add inline comments explaining each job and step
  - [ ] Add CI badge to `README.md` showing build status (once workflow is active)

- Acceptance verification
  - [ ] Run full test suite locally: `make test` or `pytest` passes with ≥85% coverage
  - [ ] Push to feature branch: GitHub Actions workflow triggers and passes all jobs
  - [ ] Verify no network calls in CI: check workflow logs for absence of external API calls
  - [ ] Coverage report shows all critical paths tested (retrieval, ingestion, chat, migrations)

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
