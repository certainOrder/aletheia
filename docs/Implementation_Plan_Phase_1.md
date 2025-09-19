# Implementation Plan: Phase 1 (OpenWebUI + Embeddings-backed RAG)

Date: 2025-09-19

This plan breaks Phase 1 into steps with clear milestones, contracts, and acceptance tests.

## 1. Goals
- Provide an OpenWebUI-compatible API that augments chat with private memory via RAG
- Containerize dev environment (Postgres + API + OpenWebUI) and document workflows
- Enable local development without external API keys via deterministic fallbacks

## 2. Milestones & Tasks

### Milestone A: Core RAG API
- A1: Endpoint scaffolding
  - Add `/v1/models`, `/v1/chat/completions`, `/index-memory`, `/rag-chat`
  - Acceptance: curl tests return expected shapes
- A2: Embeddings & search
  - Implement `convert_to_embedding`, `semantic_search`, `save_embedding_to_db`
  - Acceptance: index sample content and retrieve via RAG
- A3: Context augmentation
  - Prepend retrieved context to chat messages
  - Acceptance: `/v1/chat/completions` returns context-informed answers

### Milestone B: Dev fallbacks & config
- B1: Config flag
  - Add `DEV_FALLBACKS` and guard fallbacks for embeddings/chat
  - Acceptance: with flag on, endpoints function without OpenAI key
- B2: Env templates
  - Update `.env.example` and link from `README`
  - Acceptance: new dev follows docs to run stack successfully

### Milestone C: Dev environment (Docker Compose)
- C1: Compose stack
  - Services: `db` (pgvector), `aletheia-api`, `openwebui`
  - Acceptance: `docker compose up -d --build` yields healthy services
- C2: Docs
  - Add `docs/DEV_ENVIRONMENT.md` with smoke tests and troubleshooting
  - Acceptance: contributor can run tests and see expected outputs

### Milestone D: Validation & PR
- D1: Merge main -> feature branch, resolve conflicts, re-test
- D2: Open PR with detailed summary, run Copilot review, request human review

## 3. Contracts & Inputs/Outputs
- `/index-memory`
  - Input: `{ content: string, user_id?: uuid, tags?: string[] }`
  - Output: `{ id: uuid }`
  - Errors: 400 invalid payload; 500 DB/embedding failures
- `/v1/chat/completions`
  - Input: OpenAI-style `{ model?: string, messages: [{role, content}], stream?: bool }`
  - Output: OpenAI-style `chat.completion` object; includes `aletheia_context`
  - Errors: 400 invalid; 500 provider failures (unless `DEV_FALLBACKS=true`)
- `/rag-chat`
  - Input: `{ prompt: string, user_id?: uuid, top_k?: number }`
  - Output: `{ answer: string, context: Array<{id, content, ...}> }`

## 4. Edge Cases
- Empty/short inputs: ensure graceful handling; return generic answer
- No matches in vector search: proceed without context
- Invalid/missing API key: dev fallback produces responses; prod returns 500
- DB unavailable: return 503 with actionable message
- Large inputs: consider truncation or chunking strategy later

## 5. Testing Strategy
- Smoke tests via curl (documented in DEV_ENVIRONMENT)
- Add minimal unit tests in future PR:
  - Embedding fallback determinism under `DEV_FALLBACKS=true`
  - Chat fallback returns consistent structure
  - Semantic search returns ordered results

## 6. Quality Gates
- Build: Docker images build successfully
- Lint/Typecheck: basic checks pass (can add CI later)
- Unit tests: N/A now; plan for next phase
- Smoke tests: pass for health, models, index, rag, chat endpoints

## 7. Risks & Mitigations
- OpenAI dependencies: mitigated by fallbacks
- Schema drift: plan Alembic migrations next
- Performance: plan cosine + IVFFlat next
- Security: plan API auth and stricter CORS

## 8. Deliverables
- Working Compose stack
- Updated docs: `DEV_ENVIRONMENT.md`, `phase_1_embeddings_refactored.md`, this plan
- PR merged into `main`
