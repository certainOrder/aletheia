# Refactoring TODO

## Embeddings Flow - Key Observations

### Current State Analysis

**Embedding Creation Points:**
- `convert_to_embedding()` in `app/utils/embeddings.py`
  - OpenAI API (production)
  - Deterministic SHA256-based fallback (dev/testing)

**APIs Calling Embeddings:**

1. **`/rag-chat` (POST)** - Line 183
   - Embeds user query for context retrieval
   - Uses `semantic_search` for RAG

2. **`/index-memory` (POST)** - Line 221
   - Single-shot content indexing
   - Direct save to DB

3. **`/ingest` (POST)** - Line 265 (loop)
   - Bulk indexing with chunking
   - Multiple embed + save calls

4. **`/v1/chat/completions` (POST)** - Lines 451, 629, 660
   - Query-time RAG retrieval (Line 451)
   - Optional chat history indexing (Lines 629, 660)
   - Includes retry logic for FK constraints

5. **`/api/debug/search` (GET)** - Line 39
   - Development/debugging only

---

## Refactoring Opportunities

### 1. **Duplication in `/v1/chat/completions`** (HIGH PRIORITY)
**Issue:** The embedding + save pattern is duplicated 4 times:
- User message scoped attempt
- User message unscoped retry
- Assistant response scoped attempt
- Assistant response unscoped retry

**Location:** Lines 626-696 in `app/main.py`

**Proposal:**
```python
# Extract to helper function
def _index_chat_turn_with_retry(
    db: Session,
    content: str,
    user_id: str | None,
    role: str,
    metadata: dict
) -> None:
    """Index a chat turn with FK constraint retry fallback."""
    emb = convert_to_embedding(content)
    try:
        save_embedding_to_db(
            db=db,
            content=content,
            embedding=emb,
            user_id=user_id,
            tags=["chat"],
            source="chat",
            metadata=metadata,
        )
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        # Retry without user scoping
        save_embedding_to_db(
            db=db,
            content=content,
            embedding=emb,
            user_id=None,
            tags=["chat"],
            source="chat",
            metadata={**metadata, "retry_unscoped": True},
        )
```

**Impact:** Reduces ~70 lines to ~20 lines; improves maintainability

---

### 2. **Mixed Concerns in `/v1/chat/completions`** (MEDIUM PRIORITY)
**Issue:** Single endpoint handles too many responsibilities:
- Header parsing & user identity
- User profile management
- Token budget enforcement
- RAG retrieval
- Chat completion
- History persistence
- Error handling/retry logic

**Current Size:** ~325 lines (Lines 385-710)

**Proposal:** Break into focused functions:
```python
# app/services/chat_service.py
class ChatService:
    def extract_user_identity(payload, headers) -> tuple[UUID | None, str | None]
    def ensure_user_profile(db, user_uuid, external_id) -> str | None
    def retrieve_context(db, query, user_id) -> list[dict]
    def enforce_token_budget(messages, context, budget) -> list[dict]
    def index_chat_history(db, user_id, query, response, metadata) -> None
```

**Impact:** Better testability, clearer separation of concerns

---

### 3. **Retry Pattern Abstraction** (MEDIUM PRIORITY)
**Issue:** FK constraint retry logic (scoped → unscoped) appears in multiple places

**Proposal:**
```python
# app/utils/db_helpers.py
def save_with_user_fallback(
    db: Session,
    save_fn: Callable,
    user_id: str | None,
    **kwargs
) -> Any:
    """Attempt save with user_id; retry without on FK constraint failure."""
    try:
        return save_fn(db=db, user_id=user_id, **kwargs)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        # Retry without user scoping
        return save_fn(
            db=db,
            user_id=None,
            metadata={**kwargs.get("metadata", {}), "retry_unscoped": True},
            **{k: v for k, v in kwargs.items() if k != "metadata"}
        )
```

**Impact:** Centralizes retry logic, reduces code duplication

---

### 4. **Repository Pattern for Embeddings** (MEDIUM PRIORITY)
**Issue:** Direct DB access scattered across endpoints

**Proposal:**
```python
# app/repositories/embedding_repository.py
class EmbeddingRepository:
    def __init__(self, db: Session):
        self.db = db

    def save(
        self,
        content: str,
        embedding: list[float],
        user_id: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        metadata: dict | None = None,
        retry_unscoped: bool = True
    ) -> MemoryShard:
        """Save embedding with optional user fallback."""
        # Implementation here

    def search(
        self,
        query_embedding: list[float],
        user_id: str | None = None,
        limit: int = 5
    ) -> list[dict]:
        """Semantic search with user scope fallback."""
        # Implementation here
```

**Impact:** Better testing, clearer data access layer

---

### 5. **Token Budget Logic** (LOW PRIORITY)
**Issue:** Token counting logic embedded in endpoint

**Current:** Lines 480-550 in `app/main.py`

**Proposal:**
```python
# app/utils/token_budget.py
class TokenBudget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def enforce(
        self,
        messages: list[dict],
        context: str,
        system_prompt: str
    ) -> list[dict]:
        """Trim messages to fit within budget."""
        # Implementation here
```

**Impact:** Reusable across endpoints, easier to test

---

## Implementation Priority

### Phase 1: Quick Wins (Low Risk)
- [ ] Extract `_index_chat_turn_with_retry` helper
- [ ] Extract retry pattern abstraction
- [ ] Add unit tests for new helpers

### Phase 2: Service Layer (Medium Risk)
- [ ] Create `ChatService` with focused methods
- [ ] Extract token budget logic
- [ ] Update `/v1/chat/completions` to use services
- [ ] Add integration tests

### Phase 3: Repository Pattern (Medium Risk)
- [ ] Create `EmbeddingRepository`
- [ ] Update all endpoints to use repository
- [ ] Ensure test coverage remains ≥85%

### Phase 4: Validation (Continuous)
- [ ] Run full test suite after each phase
- [ ] Verify `make format lint typecheck test` passes
- [ ] Smoke test Docker stack
- [ ] Update documentation

---

## Non-Goals (Current State is Acceptable)

- `convert_to_embedding` abstraction: Already clean, single responsibility
- `semantic_search` function: Well-scoped, good interface
- `/rag-chat` endpoint: Simple and focused
- `/index-memory` endpoint: Single responsibility
- `/ingest` endpoint: Clear chunking logic

---

## Success Criteria

- [ ] Reduce `/v1/chat/completions` from ~325 lines to ~150 lines
- [ ] Eliminate code duplication in chat history indexing
- [ ] Improve test coverage for retry logic
- [ ] Maintain all existing functionality
- [ ] All quality gates pass (lint, typecheck, test ≥85%)
- [ ] No breaking changes to public APIs

---

## Table Conflation Fix (HIGH PRIORITY)

### Problem Statement

**Current Issue:** `raw_conversations` and `memory_shards` tables are conflated in implementation.

**Intended Design:**
- **`raw_conversations`**: Immutable record of individual conversation turns (user + assistant messages)
  - One row per message
  - Includes: author, content, timestamp, conversation_id, parent_id, embedding
  - Purpose: Complete conversation history, threading, retrieval

- **`memory_shards`**: Compressed/synthesized context from conversations
  - One row per summary/compression
  - Includes: synthesized content, embedding, source_ids (links to raw_conversations)
  - Purpose: Semantic search, RAG retrieval, efficient context

**Current Implementation Problems:**
1. Migration `0005` overwrote `raw_conversations` schema with request/response logging schema
2. No per-turn storage - only full request/response JSONB blobs
3. Individual chat turns being saved to `memory_shards` instead of `raw_conversations`
4. No conversation threading (conversation_id, parent_id)
5. No summarization process from raw turns → memory shards

### Decision Matrix (from User Feedback)

1. **Table Strategy:** Option B - Restore original conversation schema
   - Keep `raw_conversations` for conversation turns (NOT API logging)
   - Remove technical logging fields (request_id, provider, model, status_code, latency_ms)
   - Use file-based logging for API mechanics (for now)

2. **Summarization Trigger:** After each full turn (user message + assistant response)
   - Future service will batch-process raw_conversations → memory_shards
   - Background job for compression/synthesis

3. **Source Linkage:** `memory_shards.source_ids` = array of `raw_conversations.id`
   - Tracks which turns were summarized into each shard

4. **Embedding Strategy:** Embed each turn immediately
   - `raw_conversations` gets embeddings on save
   - `memory_shards` gets embeddings during summarization

5. **Chat History Indexing:** Both tables
   - Immediate: Save turns to `raw_conversations` with embeddings
   - Deferred: Summarize into `memory_shards` (background process)

### Implementation Plan

#### Phase 1: Schema Restoration (CRITICAL)

**Step 1.1: Create Migration to Restore `raw_conversations` Schema**

Create `alembic/versions/0007_restore_raw_conversations_schema.py`:

```python
"""restore raw_conversations to conversation turn schema

Revision ID: 0007
Revises: 0006
Create Date: 2025-11-03
"""

def upgrade() -> None:
    # Option A: Rename current table to preserve API logs
    op.execute("ALTER TABLE raw_conversations RENAME TO api_request_logs")

    # Option B: Drop and recreate (if logs not needed)
    # op.execute("DROP TABLE IF EXISTS raw_conversations")

    # Create raw_conversations with correct schema
    op.execute("""
        CREATE TABLE raw_conversations (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            author text NOT NULL,
            content text NOT NULL,
            timestamp timestamptz DEFAULT now() NOT NULL,
            conversation_id uuid NOT NULL,
            user_id uuid REFERENCES user_profiles(user_id),
            embedding vector(1536) NOT NULL,
            parent_id uuid REFERENCES raw_conversations(id),
            entropy float,
            emotional_index float,
            surprise_index float,
            role text CHECK (role IN ('user', 'assistant', 'system')),
            metadata jsonb
        )
    """)

    # Indexes for performance
    op.execute("CREATE INDEX idx_raw_conv_conversation ON raw_conversations(conversation_id)")
    op.execute("CREATE INDEX idx_raw_conv_user ON raw_conversations(user_id)")
    op.execute("CREATE INDEX idx_raw_conv_timestamp ON raw_conversations(timestamp DESC)")
    op.execute("CREATE INDEX idx_raw_conv_parent ON raw_conversations(parent_id)")

def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS raw_conversations")
    # op.execute("ALTER TABLE api_request_logs RENAME TO raw_conversations")
```

**Step 1.2: Create ORM Model for `RawConversation`**

Add to `app/db/models.py`:

```python
class RawConversation(Base):
    """Immutable record of individual conversation turns.

    Each user message and assistant response is stored as a separate row.
    Embeddings enable semantic search across conversation history.
    Links to conversation_id for threading and parent_id for reply chains.
    """
    __tablename__ = "raw_conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    author: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    entropy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    emotional_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    surprise_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    role: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<RawConversation id={self.id} role={self.role}>"
```

**Step 1.3: Update `memory_shards` to Track Source IDs**

Migration already has `source_ids uuid[]` column ✅

Verify in `app/db/models.py`:
```python
class MemoryShard(Base):
    # ...
    source_ids: Mapped[Optional[list[uuid.UUID]]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )  # ✅ Already present
```

---

#### Phase 2: Update Chat Endpoint to Save Turns

**Step 2.1: Create Helper to Save Conversation Turns**

Add to `app/main.py`:

```python
def _save_conversation_turn(
    db: Session,
    role: str,
    content: str,
    conversation_id: UUID,
    user_id: UUID | None,
    parent_id: UUID | None = None,
    author: str | None = None,
) -> RawConversation:
    """Save a single conversation turn to raw_conversations with embedding.

    Args:
        role: 'user', 'assistant', or 'system'
        content: Message content
        conversation_id: Thread/session ID
        user_id: User profile ID
        parent_id: ID of message this is replying to
        author: Display name (defaults to role)

    Returns:
        Created RawConversation instance
    """
    emb = convert_to_embedding(content)
    turn = RawConversation(
        author=author or role,
        content=content,
        conversation_id=conversation_id,
        user_id=user_id,
        embedding=emb,
        parent_id=parent_id,
        role=role,
        metadata_json={"request_id": get_request_id()},
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)

    logger.info(
        "conversation_turn_saved",
        extra={
            "turn_id": str(turn.id),
            "role": role,
            "conversation_id": str(conversation_id),
            "content_len": len(content),
        },
    )
    return turn
```

**Step 2.2: Update `/v1/chat/completions` to Save Both Tables**

Replace Lines 626-696 (chat history indexing) with:

```python
# Save conversation turns to raw_conversations AND index to memory_shards
try:
    if INDEX_CHAT_HISTORY:
        # Generate or extract conversation_id from request
        conv_id = payload.get("conversation_id")
        if not conv_id:
            conv_id = uuid4()  # New conversation
        else:
            conv_id = UUID(conv_id)

        user_turn_id = None
        assistant_turn_id = None

        # Save user turn to raw_conversations
        if query_text:
            user_turn = _save_conversation_turn(
                db=db,
                role="user",
                content=query_text,
                conversation_id=conv_id,
                user_id=user_id,
                author=payload.get("user") or "user",
            )
            user_turn_id = user_turn.id

            # ALSO index to memory_shards for immediate retrieval
            # (Reuse embedding from turn save to avoid duplicate API call)
            try:
                save_embedding_to_db(
                    db=db,
                    content=query_text,
                    embedding=user_turn.embedding,  # Reuse!
                    user_id=(str(user_id) if user_id else None),
                    tags=["chat", "user"],
                    source="chat",
                    metadata={
                        "role": "user",
                        "conversation_id": str(conv_id),
                        "turn_id": str(user_turn_id),
                    },
                )
            except Exception as e:
                logger.warning("memory_shard_index_failed", extra={"error": str(e)})

        # Save assistant turn to raw_conversations
        if content:
            assistant_turn = _save_conversation_turn(
                db=db,
                role="assistant",
                content=content,
                conversation_id=conv_id,
                user_id=user_id,
                parent_id=user_turn_id,  # Link reply chain
                author=resp_model,
            )
            assistant_turn_id = assistant_turn.id

            # ALSO index to memory_shards
            try:
                save_embedding_to_db(
                    db=db,
                    content=content,
                    embedding=assistant_turn.embedding,  # Reuse!
                    user_id=(str(user_id) if user_id else None),
                    tags=["chat", "assistant"],
                    source="chat",
                    metadata={
                        "role": "assistant",
                        "conversation_id": str(conv_id),
                        "turn_id": str(assistant_turn_id),
                        "model": resp_model,
                    },
                )
            except Exception as e:
                logger.warning("memory_shard_index_failed", extra={"error": str(e)})

        logger.info(
            "chat_turn_indexed",
            extra={
                "conversation_id": str(conv_id),
                "user_turn_id": str(user_turn_id) if user_turn_id else None,
                "assistant_turn_id": str(assistant_turn_id) if assistant_turn_id else None,
            },
        )
except Exception as e:
    logger.warning("chat_turn_save_failed", extra={"error": str(e)})
```

**Step 2.3: Remove Old API Logging Code**

Delete Lines 703-773 (raw_conversations SQL insert for request/response logging).

Use structured logging instead:
```python
logger.info(
    "chat_completion",
    extra={
        "request_id": get_request_id(),
        "model": resp_model,
        "provider": provider,
        "latency_ms": latency_ms,
        "prompt_tokens": post_tokens,
        "completion_tokens": (len(content or "") + 3) // 4,
        "user_id": str(user_id) if user_id else None,
    },
)
```

---

#### Phase 3: Conversation Summarization Service (Future)

**Step 3.1: Create Background Service Skeleton**

Create `app/services/conversation_summarizer.py`:

```python
"""Background service to summarize raw_conversations into memory_shards.

Runs periodically to compress conversation threads into semantic summaries.
"""

from uuid import UUID
from sqlalchemy.orm import Session
from app.db.models import RawConversation, MemoryShard
from app.utils.embeddings import convert_to_embedding, save_embedding_to_db

def summarize_conversation(db: Session, conversation_id: UUID) -> MemoryShard:
    """Compress a full conversation into a memory shard.

    Strategy:
    1. Fetch all turns for conversation_id
    2. Use LLM to generate summary
    3. Embed summary
    4. Save to memory_shards with source_ids linking back to turns

    Args:
        conversation_id: UUID of conversation thread

    Returns:
        Created MemoryShard
    """
    # TODO: Implement in future milestone
    #
    # turns = db.query(RawConversation)\
    #     .filter(RawConversation.conversation_id == conversation_id)\
    #     .order_by(RawConversation.timestamp)\
    #     .all()
    #
    # summary = _generate_summary(turns)  # LLM call
    # emb = convert_to_embedding(summary)
    # shard = save_embedding_to_db(
    #     db=db,
    #     content=summary,
    #     embedding=emb,
    #     source_ids=[t.id for t in turns],
    #     tags=["summary", "conversation"],
    #     source="summarization",
    # )
    # return shard
    raise NotImplementedError("Conversation summarization coming in future milestone")
```

**Step 3.2: Add TODO for Background Job**

```python
# TODO (Future Milestone): Background job to run summarization
# - Cron/schedule to find unsummarized conversations
# - Batch process by conversation_id
# - Update memory_shards with source_ids
# - Mark turns as summarized (add column or metadata flag)
```

---

#### Phase 4: Update Retrieval Logic (Optional Enhancement)

**Step 4.1: Hybrid Retrieval Strategy**

Current: Only queries `memory_shards` ✅

Future: Could query BOTH tables:
```python
def hybrid_search(
    db: Session,
    query_embedding: list[float],
    user_id: str | None = None,
    limit: int = 5
) -> list[dict]:
    """Search both memory_shards (summaries) and raw_conversations (turns)."""

    # Get top-k from memory shards (compressed context)
    shard_results = semantic_search(
        db, query_embedding, user_id=user_id, limit=limit//2
    )

    # Get top-k from raw conversations (individual turns)
    turn_results = search_raw_conversations(
        db, query_embedding, user_id=user_id, limit=limit//2
    )

    # Merge and re-rank by score
    combined = sorted(
        shard_results + turn_results,
        key=lambda x: x.get("score", 0),
        reverse=True
    )[:limit]

    return combined
```

**Decision:** Keep current strategy (memory_shards only) for now. Add hybrid search later.

---

### Testing Requirements

- [ ] Migration 0007 runs cleanly on fresh DB
- [ ] Migration 0007 preserves existing memory_shards data
- [ ] `/v1/chat/completions` creates raw_conversation rows
- [ ] Conversation turns have correct embeddings
- [ ] parent_id links user → assistant correctly
- [ ] conversation_id groups related turns
- [ ] Memory shards still get indexed for immediate retrieval
- [ ] Retrieval still works (queries memory_shards)
- [ ] No duplicate embeddings (reuse from turn save)
- [ ] User scoping works for both tables
- [ ] FK constraints enforced correctly

### Migration Path

1. **Dev/Test:** Run migration 0007, verify schema
2. **Staging:** Backup DB, run migration, test E2E flow
3. **Production:**
   - Option A: Rename to `api_request_logs` (preserve logs)
   - Option B: Drop table (if logs not needed)
   - Create new `raw_conversations` with correct schema
   - Deploy updated code

### Documentation Updates

- [ ] Update `docs/ARCHITECTURE.md` with table purposes
- [ ] Update `docs/DATABASE.md` with new schema
- [ ] Add conversation threading examples
- [ ] Document summarization strategy (future)
- [ ] Update `GETTING_STARTED.md` if needed

---

## Task Checklist

### Immediate (Current Sprint)
- [ ] Create migration 0007 (restore raw_conversations schema)
- [ ] Add RawConversation ORM model
- [ ] Create `_save_conversation_turn` helper
- [ ] Update `/v1/chat/completions` to save both tables
- [ ] Remove old API logging code (Lines 703-773)
- [ ] Add tests for conversation turn storage
- [ ] Verify retrieval still works
- [ ] Run full test suite (≥85% coverage)

### Near-term (Next Sprint)
- [ ] Add conversation_id to request schema
- [ ] Implement conversation threading UI support
- [ ] Add endpoint to fetch conversation history
- [ ] Optimize embedding reuse (avoid duplicate API calls)

### Future (Backlog)
- [ ] Implement conversation summarization service
- [ ] Background job for batch summarization
- [ ] Hybrid retrieval (shards + turns)
- [ ] Conversation pruning/archival
- [ ] Advanced threading (reply chains, branching)
