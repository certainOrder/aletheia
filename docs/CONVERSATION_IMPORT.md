# Conversation Import (ChatGPT and others)

Date: 2025-09-20
Status: Proposed (not implemented)
Owner: Platform / RAG

## Why
To preserve alignment and integrity, HearthMinds does not support importing conversations directly into the `raw_conversations` table (e.g., via JSON imports). Each proto-person (Engineered Intelligence) begins with a blank `raw_conversations` table, which is populated exclusively by AI-native, session-driven dialogue until alignment is achieved. Once aligned, the proto-person is bonded to a human partner, and its `raw_conversations` table serves as an immutable, AI-owned phenotype log. While conversation turns remain immutable, memory shards and embeddings can be regenerated as models and retrieval improve over time. For idempotent deployment, the existing `raw_conversations` table may need to be securely backed up and restored after PostgreSQL installation; this feature can support that workflow as well.

## Scope (Phase 2/3 orientation)
- Phase 2: Document the approach, leave hooks in place for memory shard and embedding regeneration. Optional CLI for backup/restore only.
- Phase 3: Ship schema and internal tools for shard/embedding regeneration, with tests and docs. Backup/restore workflows for database migration.

## Inputs
- No external imports (e.g., no JSON, transcript, or chatbot data).
- All data originates from AI-native, session-driven dialogue within the platform.
- Backup/restore operations may export/import the `raw_conversations` table for migration or disaster recovery, but not for population or augmentation.

## Design Overview
1. Memory shards and embeddings are regenerated internally from the immutable `raw_conversations` table as models or retrieval methods improve.
2. No external data is written to `raw_conversations`.
3. Backup/restore operations allow secure migration of the phenotype log, but do not alter its contents.

### Data Model (proposed)
- `raw_conversations` (immutable, AI-owned)
  - Only the AI can write to this table.
  - No imports from external sources or other chatbots.
  - Serves as the phenotype log for the proto-person.
- `memory_shards` (existing)
  - Reuse existing columns (content, embedding, user_id, tags)
  - Add tags/metadata for provenance, e.g.: `source:chatgpt`, `conv:<id>`, `turn:<n>`, `summary:true`

### Ingestion Modes
- Turn-level (configurable)
  - Create one memory shard per turn.
  - Pros: maximal recall; Cons: noisy, more tokens/storage.
- Summary-level (configurable)
  - Summarize each conversation (e.g., first N turns, or LLM summary) → one memory shard per conversation.
  - Pros: concise; Cons: may lose detail.
- Both (potentially requires future schema change to support)
  - Store summaries + selected turns (e.g., high-signal user questions).

### Retrieval Behavior
- Memory shards participate in the same vector search (cosine + IVFFlat).
- Tags allow filtering later (e.g., by source, conversation, or turn).
- Retrieved results are surfaced in the proto-person's context window for inference and dialogue, not as a persistent table.

## Feature Components
1. Shard and Embedding Regeneration
   - Regenerate memory shards and embeddings from the immutable `raw_conversations` table whenever models or retrieval methods are updated.
   - All operations are performed internally within the PostgreSQL database; no external data sources or imports are involved.
2. Backup/Restore for Idempotent Deployment
   - Provide secure backup and restore workflows for the `raw_conversations` table and related phenotype logs to support bare-metal or cloud migration, disaster recovery, and reproducible deployments.
   - Export/import is strictly for database migration or recovery, never for augmenting or populating conversation data.

## Pipeline Options
- CLI (recommended): Internal tool for shard/embedding regeneration and database backup/restore.
- No support for external data imports/exports to third-party platforms (e.g., ChatGPT). Inference and embedding generation may use external APIs (e.g., OpenAI), but all data remains on-prem within the PostgreSQL database.

## Interaction with Existing Features
- Chunking: not required for turn-level data; helpful if processing long summaries or concatenated transcripts. `/ingest` can be used to quickly process chats as a single document, but it loses turn structure.
- Conversation history: `raw_conversations` is reserved for AI-generated, session-native history only. All content is indexed as memory shards and never written to this table.

## Testing Strategy
- Unit: parser normalizes conversation turns into (role, content, turn_index, conversation_id).
- Unit: tags are attached correctly to memory shards.
- Offline (testing): embeddings use deterministic fallbacks in CI; no network calls. In production, embeddings are generated via the configured external API (e.g., OpenAI).

## Acceptance (M3)
- `alembic upgrade head` creates/updates schema as needed.
- CLI import creates memory shards with expected tags.
- Retrieval can surface imported summaries/turns in a proto-person's context window.

## Open Questions
- Deduplication: How to avoid duplicates across repeated imports? (future feature: introspection loops)
- PII handling: redaction or “private by default” scopes. (whole database encryption for on-prem data, fuzzy scope per user: EI => "Hi Jim, while I'm bonded to your dad, I wanted to talk to you about being safe at the party tonight." <= EI context aware, exercises judgement about appropriate scope.)
- Summarization cost: do we use real models or a local heuristic summary by default? (TBD)

