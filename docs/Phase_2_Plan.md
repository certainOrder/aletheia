# Phase 2 Plan: “Real model” testing, reliability, and UX

Date: 2025-09-19

This phase moves from scaffolding to production-like behavior: enable real OpenAI usage, tighten data model and retrieval, add tests/CI, and improve the chat UX flows.

## Goals
- Use a real OpenAI key to exercise embeddings and chat end-to-end
- Improve retrieval quality and performance (cosine + IVFFlat)
- Establish migrations (Alembic) and basic CI tests for reliability
- Add optional streaming and better grounding visibility in the UI

## Milestones & Tasks

### M1: Enable real OpenAI usage safely
- Add `OPENAI_API_KEY` guidance and validation on startup (when `DEV_FALLBACKS=false`)
- Update docs with secure key handling; add `.env.example` notes for Docker Compose
- Acceptance:
  - With a valid key and `DEV_FALLBACKS=false`, `/v1/chat/completions` and `/embeddings` succeed
  - With missing/invalid key and `DEV_FALLBACKS=false`, clear 500 with actionable error

### M2: Retrieval quality & performance
- Switch to cosine similarity; expose score in responses
- Add IVFFlat index for `pgvector` with configurable `lists` via env
- Add chunking strategy for long content (configurable chunk size & overlap)
- Acceptance:
  - `semantic_search` returns top-k with cosine scores descending
  - IVFFlat index present; query plan uses index
  - Ingesting large text creates multiple chunks and improves recall in tests

### M3: Schema migrations
- Introduce Alembic; generate baseline migration from current models
- Add migration for IVFFlat-related ops and any new fields (e.g., `source`, `metadata` JSONB)
- Acceptance:
  - `alembic upgrade head` bootstraps a fresh DB
  - `alembic downgrade` works for at least one step

### M4: Tests & CI
- Unit tests: fallback determinism; search ordering; basic chat shape
- Integration tests (optional): tmp Postgres with pgvector; minimal ingestion + query
- GitHub Actions workflow to run tests and report
- Acceptance:
  - PRs run tests automatically
  - Minimum coverage gate optional (e.g., 60%)

### M5: Streaming & UX improvements
- Add `stream=true` support to `/v1/chat/completions` (Server-Sent Events)
- Surface retrieved context in responses clearly (already available for RAG); show in `chat.html`
- Acceptance:
  - With `stream=true`, clients receive token deltas
  - Static chat page shows grounding snippets for each answer

## Implementation Notes
- Config additions: `SIMILARITY_METRIC=cosine`, `PGVECTOR_IVFFLAT_LISTS=100`, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`
- Postgres: ensure `pgvector` extension; create `ivfflat` index only after data load or with `ANALYZE` to avoid poor plans
- SSE streaming: prefer FastAPI `EventSourceResponse` or manual chunked responses; maintain OpenAI-compatible `delta` payloads

## Risks & Mitigations
- API quota/costs: enable rate limiting and keep `DEV_FALLBACKS=true` by default in `.env.example`
- Index build time: document trade-offs and provide toggle to defer building
- Streaming complexity: behind a feature flag until stabilized

## Rollout
- Branch `feature/phase-2-real-models`
- Sub-PRs by milestone (M1 → M5)
- Update `DEV_ENVIRONMENT.md` for new envs and streaming testing tips
- Tag v0.2.0 after merge
