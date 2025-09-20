# Conversation Import (ChatGPT and others)

Date: 2025-09-20
Status: Proposed (not implemented)
Owner: Platform / RAG

## Why
Many users want to bring their prior chat history (e.g., ChatGPT exports) so Aletheia can ground new answers in their past knowledge. This feature describes how to parse, normalize, and index external conversations so they’re retrievable alongside documents.

## Scope (Phase 2/3 orientation)
- Phase 2: Document the approach, leave hooks in place (chunking, tags). Optional CLI prototype.
- Phase 3: Ship schema and importer (CLI/endpoint), with tests and docs.

## Inputs
- ChatGPT export (JSON): a list of conversations where each conversation has an ID and ordered messages.
- Future: Slack/Discord/Teams or arbitrary JSONL transcripts.

## Design Overview
1) Parse export → normalize to (conversation_id, turn_index, role, content, timestamps, provider).
2) Persist raw turns (audit trail) and index embeddings for retrieval.
3) Expose imported content to the same retrieval layer used for documents.

### Data Model (proposed)
- `raw_conversations` (new in M3)
  - `id` (UUID, PK)
  - `conversation_id` (text)
  - `turn_index` (int)
  - `role` (text: `user` | `assistant` | etc.)
  - `content` (text)
  - `created_at` (timestamptz)
  - `provider` (text, e.g., `chatgpt`)
  - `user_id` (UUID nullable)
  - Optional metadata columns later as needed
- `memory_shards` (existing)
  - Reuse existing columns (content, embedding, user_id, tags)
  - Add tags/metadata for provenance, e.g.: `source:chatgpt`, `conv:<id>`, `turn:<n>`, `summary:true`

### Ingestion Modes
- Turn-level
  - Create one shard per turn.
  - Pros: maximal recall; Cons: noisy, more tokens/storage.
- Summary-level
  - Summarize each conversation (e.g., first N turns, or LLM summary) → one shard per conversation.
  - Pros: concise; Cons: may lose detail.
- Both
  - Store summaries + selected turns (e.g., high-signal user questions).

### Retrieval Behavior
- Imported shards participate in the same vector search (cosine + IVFFlat).
- Tags allow filtering later (e.g., only `source:chatgpt`).
- Results show up in `aletheia_context` with their scores.

## Pipeline Options
- CLI (recommended first): `python -m app.scripts.import_chatgpt --file export.json --mode summary --tags import:chatgpt`
  - Runs offline, no server changes needed.
- API endpoint (optional later): `/import/chatgpt` for uploads (requires auth and size limits).

## Interaction with Existing Features
- Chunking: not required for turn-level data; helpful if importing long summaries or concatenated transcripts. `/ingest` can be used to quickly import exported chats as a single document, but it loses turn structure.
- Conversation history: this feature is about long-term memory; current-session history will be handled separately via `HISTORY_TURNS`.

## Security & Privacy
- Do not log raw content.
- Require explicit user action to import; attribute imported data to the correct `user_id`.
- Allow deletion (future): remove both `raw_conversations` and associated shards by `conversation_id`/`user_id`.

## Testing Strategy
- Unit: parser normalizes ChatGPT JSON into (role, content, turn_index, conversation_id).
- Unit: tags are attached correctly to shards.
- Offline: embeddings use deterministic fallbacks in CI; no network calls.

## Acceptance (M3)
- `alembic upgrade head` creates `raw_conversations`.
- CLI import creates raw rows and shards with expected tags.
- Retrieval can surface imported summaries/turns in `aletheia_context`.

## Open Questions
- Deduplication: How to avoid duplicates across repeated imports?
- PII handling: redaction or “private by default” scopes.
- Summarization cost: do we use real models or a local heuristic summary by default?

## Future Work
- UI affordance in OpenWebUI to browse imported conversations.
- Source-specific importers (Slack/Discord/Teams) sharing the same normalized schema.
- Fine-grained filters (date ranges, roles) at query time.
