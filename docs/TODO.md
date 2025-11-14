# Refactoring TODO: Two-Table Architecture Implementation

## 🎯 Executive Summary

**Goal:** Separate conversation turn storage from semantic memory compression.

**Current Problem:** `raw_conversations` table is being used for API request logging instead of conversation turn storage. This conflicts with the intended two-table architecture where:
1. `raw_conversations` = individual conversation turns (user/assistant messages)
2. `memory_shards` = compressed summaries created by background process

**Solution:** Restore `raw_conversations` to its intended purpose and implement dual-context prompt scaffolding.

---

## ⚡ Quick Reference (For Fresh Context)

**What to implement:**
1. **Phase 1**: Migration to restore `raw_conversations` schema (drop old table, create new)
2. **Phase 2**: Save chat turns to `raw_conversations` (ONLY this table, NOT memory_shards)
3. **Phase 2b/3**: Background job to compress turns → `memory_shards` (every few minutes)
4. **Phase 4**: Dual-context prompts (latest 5 raw turns + top 10 memory shards)

**Critical rules:**
- ❌ **NO dual-save**: Chat endpoint saves ONLY to `raw_conversations`
- ✅ **Two embeddings**: raw turns get exact content vectors, shards get fresh compressed vectors
- ✅ **Frequent job**: Background summarization runs every few minutes (not nightly)
- ✅ **Dual prompts**: Recent 5 raw turns (detail) + Top 10 shards (historical context)

**Why this matters:**
- `raw_conversations` = detailed recent context for conversation flow
- `memory_shards` = token-efficient historical context for semantic retrieval
- Different embeddings enable different search strategies
- Aletheia "reviews" her conversations to create compressed memories

---

## 🏗️ Architecture Overview (READ THIS FIRST)

### Two Tables, Two Purposes

#### Table 1: `raw_conversations` (Conversation History)
**Purpose:** Store individual conversation turns for threading, history, and recent context

**Schema (from `pp_schema.sql`):**
```sql
CREATE TABLE raw_conversations (
    id UUID PRIMARY KEY,
    intended_recipient TEXT,        -- "User" or "Aletheia"
    author TEXT NOT NULL,            -- Who said it
    timestamp TIMESTAMPTZ NOT NULL,
    conversation_id UUID,            -- Thread grouping
    user_id UUID REFERENCES user_profiles(id),
    content TEXT NOT NULL,           -- Exact message content
    embedding vector(1536),          -- Embedding of exact content
    parent_id UUID REFERENCES raw_conversations(id),  -- Thread linking
    entropy FLOAT,
    emotional_index FLOAT,
    surprise_index FLOAT
);
```

**When data is saved:** Immediately during chat (synchronous)
**What gets embedded:** Exact message content (user or assistant)
**Used for:** 
- Recent context (latest ~5 turns in prompt)
- Conversation threading
- Complete history/audit trail

#### Table 2: `memory_shards` (Compressed Semantic Memory)
**Purpose:** Token-efficient semantic search for historical context

**Schema (existing):**
```sql
CREATE TABLE memory_shards (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,           -- Compressed summary
    embedding vector(1536),          -- Embedding of compressed summary
    source_ids UUID[],               -- Links to raw_conversations.id
    user_id UUID REFERENCES user_profiles(id),
    tags TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);
```

**When data is saved:** Background process every few minutes (asynchronous)
**What gets embedded:** Compressed summary of multiple turns (FRESH embedding)
**Used for:**
- Historical context via semantic search (top 10 shards in prompt)
- Token-efficient RAG retrieval

### 🔑 Key Architectural Principles

1. **NO DUAL-SAVE:** Chat endpoint saves to `raw_conversations` ONLY, not both tables
2. **Different embeddings:** Raw turn embeddings ≠ compressed summary embeddings
3. **Sequential flow:** Chat → raw_conversations (immediate) → background job → memory_shards (async)
4. **Dual-context prompts:**
   - Recent: Latest ~5 raw_conversations turns (detailed, exact content)
   - Historical: Top 10 memory_shards (compressed, semantic search)
5. **Frequent background job:** Runs every few minutes (not nightly), keeps memory fresh

---

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
- [ ] Prompt scaffolding works: Latest 5 raw turns + Top 10 memory_shards

---

## 📋 Table Conflation Fix (HIGH PRIORITY - START HERE)

### Problem Statement

**Current Issue:** The `raw_conversations` table exists in the database but is being used for API request logging instead of its intended purpose (conversation turn storage).

**Schema Mismatch:**
- Current implementation: Logs API requests with technical fields (request_id, provider, model, status_code, latency_ms)
- Intended schema (pp_schema.sql): Stores conversation turns with threading fields (intended_recipient, author, parent_id, conversation_id)

**Impact:** 
- Cannot store conversation history properly
- Cannot implement conversation threading
- Prompt scaffolding strategy blocked (needs raw turns for recent context)

### Intended Design (What to Build)

This system uses **two tables with different purposes and different embeddings:**

#### `raw_conversations` - Individual Turn Storage
- **Saves when:** During chat (immediate, synchronous)
- **Saves what:** Each individual message (user or assistant)
- **Embedding of:** Exact message content
- **Used for:** Recent context in prompts (latest ~5 turns)

#### `memory_shards` - Compressed Summaries  
- **Saves when:** Background job every few minutes (async)
- **Saves what:** Compressed summary of multiple conversation turns
- **Embedding of:** Compressed summary text (FRESH embedding, not reused)
- **Used for:** Historical context in prompts (top 10 via semantic search)

### Visual Flow

```
User sends message
    ↓
/v1/chat/completions endpoint
    ↓
1. Build prompt with:
   - Latest 5 raw_conversations turns (recent context)
   - Top 10 memory_shards by similarity (historical context)
    ↓
2. Call OpenAI API
    ↓
3. Save user message → raw_conversations (immediate)
4. Save assistant response → raw_conversations (immediate)
    ↓
[END OF SYNCHRONOUS FLOW]

... few minutes later ...

Background Job (runs every ~5 minutes)
    ↓
1. Find unsummarized raw_conversations
2. Group related turns (conversation_id)
3. Compress via LLM: "User asked X, explained Y, understood Z"
4. Generate FRESH embedding of compressed summary
5. Save → memory_shards with source_ids
6. Mark raw turns as summarized
```
  - Prompt usage: Top 10 most relevant shards via semantic search (historical context)

### ⚠️ Critical Architectural Rules

**RULE #1: NO DUAL-SAVE**
- Chat endpoint saves ONLY to `raw_conversations`
- Do NOT save individual turns to `memory_shards` during chat
- `memory_shards` are created ONLY by background job

**RULE #2: DIFFERENT EMBEDDINGS**
- `raw_conversations.embedding` = vector of exact message text
- `memory_shards.embedding` = vector of compressed summary text
- These are different embeddings of different content

**RULE #3: FREQUENT BACKGROUND JOB**
- Not "nightly" batch processing
- Runs every few minutes (or after N turns)
- Keeps `memory_shards` fresh for RAG
- Async is OK since recent turns are in prompt

**RULE #4: DUAL-CONTEXT PROMPTS**
```
System: [Aletheia personality]

Historical Context (compressed, from memory_shards):
- [Top 10 shards by semantic similarity to user query]

Recent Conversation (detailed, from raw_conversations):
- [Latest 5 turns in chronological order]

User: [Current message]
```

### Current Implementation Problems

1. ❌ Migration `0005` overwrote `raw_conversations` schema with request/response logging schema
2. ❌ No per-turn storage - only full request/response JSONB blobs
3. ❌ Code attempts to save individual turns to `memory_shards` (wrong - should only contain summaries)
4. ❌ No conversation threading (conversation_id, parent_id)
5. ❌ No summarization process from raw turns → memory shards
6. ❌ No dual-context prompt scaffolding

### Decision Matrix (Implementation Guide)

### Decision Matrix (Implementation Guide)

1. **Table Strategy:** Restore `raw_conversations` to intended schema
   - Drop or rename current `raw_conversations` table (it's just API logs)
   - Create new `raw_conversations` with correct schema from `pp_schema.sql`
   - Use file-based logging for API request/response debugging (structured logs)
   - Remove technical logging fields (request_id, provider, model, status_code, latency_ms)
   - Use file-based logging for API mechanics (for now)

2. **Summarization Trigger:** Frequent background process
   - Runs every few minutes (or after N new turns)
   - Queries unsummarized `raw_conversations` turns
   - Groups by conversation_id, compresses via LLM
   - Creates fresh embedding of compressed summary
   - Saves to `memory_shards` with source_ids array
   - NOT triggered during chat (async only)

3. **Source Linkage:** `memory_shards.source_ids` tracks original turns
   - Array of UUID references to `raw_conversations.id`
   - Allows tracing summary back to original conversation turns
   - Example: `source_ids = [uuid1, uuid2, uuid3, uuid4]` (4 turns compressed into 1 shard)

4. **Embedding Strategy:** Two different embeddings
   - **raw_conversations.embedding:** Created immediately during chat
     - Input: Exact message text (user or assistant)
     - Purpose: Enable semantic search over individual messages (future feature)
   - **memory_shards.embedding:** Created during background summarization
     - Input: Compressed summary text (NOT original message)
     - Purpose: Enable semantic search for RAG context retrieval
   - These are fundamentally different embeddings of different content

5. **Prompt Scaffolding:** Dual-context architecture (both tables)
   - **Phase 2 (Immediate):** 
     - Save user + assistant turns to `raw_conversations` (synchronous)
     - Include latest 5 raw turns in prompt (detailed recent context)
   - **Phase 2b/3 (Parallel development):** 
     - Background job creates `memory_shards` from raw turns (async, every few minutes)
     - Include top 10 memory_shards in prompt (compressed historical context)
   - **Result:** Prompts have both recent detail AND historical semantic context
   - **NOT dual-save:** Individual turns never go to `memory_shards` during chat

### Implementation Plan

---

## 🚀 Phase 1: Schema Restoration (START HERE)

**Goal:** Restore `raw_conversations` table to correct schema for conversation turn storage.

**Step 1.1: Create Migration 0007**

Create `alembic/versions/0007_restore_raw_conversations_schema.py`:

```python
"""restore raw_conversations to conversation turn schema

Revision ID: 0007
Revises: 0006
Create Date: 2025-11-14

Context: The raw_conversations table was repurposed for API request logging
in migration 0005. This migration restores it to the intended purpose: 
storing individual conversation turns (user/assistant messages) for threading
and recent context in prompts.

The old table can be dropped safely - it only contained API request logs
which are now replaced by structured file logging.
"""

def upgrade() -> None:
    # Drop existing table (was being used for API request logging)
    op.execute("DROP TABLE IF EXISTS raw_conversations")

    # Create raw_conversations with correct schema from pp_schema.sql
    # Purpose: Store individual conversation turns for threading and recent context
    op.execute("""
        CREATE TABLE raw_conversations (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            intended_recipient uuid,           -- "User" or "Aletheia" 
            author text NOT NULL,               -- Who said this message
            content text NOT NULL,              -- Exact message text
            timestamp timestamptz DEFAULT now() NOT NULL,
            conversation_id uuid NOT NULL,      -- Thread grouping
            user_id uuid REFERENCES user_profiles(user_id),
            embedding vector(1536) NOT NULL,    -- Embedding of exact message content
            parent_id uuid REFERENCES raw_conversations(id),  -- Links user msg → assistant reply
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
    intended_recipient: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
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

    def __repr__(self) -> str:
        return f"<RawConversation id={self.id} author={self.author}>"
```

**Step 1.3: Verify memory_shards Has source_ids**

Confirm `memory_shards` table already has `source_ids` column (migration 0004):
```sql
ALTER TABLE memory_shards ADD COLUMN IF NOT EXISTS source_ids uuid[];
```

Verify in `app/db/models.py`:
```python
class MemoryShard(Base):
    # ...
    source_ids: Mapped[Optional[list[uuid.UUID]]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True,
        comment="Array of raw_conversations.id that were compressed into this shard"
    )  # ✅ Already present
```

---

## 🔧 Phase 2: Save Conversation Turns (IMMEDIATE PRIORITY)

**Goal:** Update `/v1/chat/completions` to save individual conversation turns to `raw_conversations`.

**Step 2.1: Create Helper Function**

Add to `app/main.py` (before `/v1/chat/completions` endpoint):

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
    
    This function:
    1. Generates embedding of exact message content
    2. Saves to raw_conversations (NOT memory_shards)
    3. Returns created instance with ID for parent_id linking
    
    NOTE: Do NOT save to memory_shards here. Memory shards are created
    by background job that compresses multiple turns into summaries.

    Args:
        role: 'user' or 'assistant' 
        content: Exact message text
        conversation_id: Thread/session ID for grouping related turns
        user_id: User profile ID (for scoping)
        parent_id: ID of message this is replying to (user msg → assistant reply)
        author: Display name (defaults to role)

    Returns:
        Created RawConversation instance with ID
    """
    # Generate embedding of exact message content
    emb = convert_to_embedding(content)
    
    # Create raw_conversations record
    turn = RawConversation(
        author=author or role,
        content=content,
        conversation_id=conversation_id,
        user_id=user_id,
        embedding=emb,
        parent_id=parent_id,
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

**Step 2.2: Update `/v1/chat/completions` to Save Turns**

Locate Lines 626-696 (chat history indexing with retry logic) and replace with:

```python
# Save conversation turns to raw_conversations ONLY (NOT memory_shards)
# Memory shards will be created later by background summarization job
try:
    if INDEX_CHAT_HISTORY:
        # Get or generate conversation_id
        conv_id = payload.get("conversation_id")
        if not conv_id:
            conv_id = uuid4()  # New conversation
        else:
            conv_id = UUID(conv_id)

        # Save user message to raw_conversations
        user_turn_id = None
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

        # Save assistant response to raw_conversations, linked to user message
        assistant_turn_id = None
        if content:
            assistant_turn = _save_conversation_turn(
                db=db,
                role="assistant",
                content=content,
                conversation_id=conv_id,
                user_id=user_id,
                parent_id=user_turn_id,  # Links assistant reply → user message
                author=resp_model,  # e.g., "gpt-4o"
            )
            assistant_turn_id = assistant_turn.id

        logger.info(
            "conversation_turns_saved",
            extra={
                "conversation_id": str(conv_id),
                "user_turn_id": str(user_turn_id) if user_turn_id else None,
                "assistant_turn_id": str(assistant_turn_id) if assistant_turn_id else None,
            },
        )
except Exception as e:
    logger.warning("chat_turn_save_failed", extra={"error": str(e)})
    # Non-fatal - chat still succeeds even if turn save fails
```

**Key changes from old code:**
- ❌ Removed: Retry logic for FK constraint failures
- ❌ Removed: Duplicate save attempts (scoped/unscoped)
- ❌ Removed: Saves to memory_shards (wrong table)
- ✅ Added: Simple save to raw_conversations with parent_id linking
- ✅ Added: Clear comments about background job responsibility

**Step 2.3: Remove Old API Logging Code**

Delete Lines 703-773 in `app/main.py` (raw SQL INSERT for request/response logging).

Replace with structured file logging:
```python
logger.info(
    "chat_completion_request",
    extra={
        "request_id": get_request_id(),
        "model": resp_model,
        "provider": provider,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "user_id": str(user_id) if user_id else None,
        "conversation_id": str(conv_id) if conv_id else None,
    },
)
```

Technical logs go to files; conversation data goes to `raw_conversations` table.
```

---

#### Phase 3: Conversation Summarization Service (Future)

**Step 3.1: Create Background Service Skeleton**

---

## 🔄 Phase 3: Background Summarization (PARALLEL DEVELOPMENT)

**Goal:** Implement background job to compress `raw_conversations` into `memory_shards`.

**Step 3.1: Create Summarization Service**

Create `app/services/conversation_summarizer.py`:

```python
"""Background service to summarize raw_conversations into memory_shards.

This is "Aletheia's introspection loop" - she reviews her conversation
history and creates compressed semantic summaries for efficient RAG retrieval.

Runs every few minutes (not nightly) to keep memory_shards fresh.
"""

from uuid import UUID
from sqlalchemy.orm import Session
from app.db.models import RawConversation, MemoryShard
from app.utils.embeddings import convert_to_embedding
from app.services.openai_service import OpenAIService

def summarize_conversation_segment(
    db: Session, 
    conversation_id: UUID,
    start_turn_id: UUID | None = None
) -> MemoryShard:
    """Compress recent raw_conversations into a memory shard.

    Aletheia's Introspection Strategy:
    1. Fetch recent unsummarized turns for conversation_id from raw_conversations
    2. Use LLM to generate compressed summary
       Example: "User asked about quantum entanglement; I explained Bell's theorem,
                 non-locality, and measurement correlation. User understood the 
                 'spooky action' concept."
    3. Generate FRESH embedding of the COMPRESSED summary (not reused from turns)
    4. Save to memory_shards with source_ids array linking to original turns
    5. Mark raw turns as summarized (add metadata flag)

    CRITICAL: This creates a NEW embedding of the COMPRESSED content.
    - raw_conversations.embedding = vector of "User: what is quantum entanglement?"
    - memory_shards.embedding = vector of "User asked about quantum entanglement..."
    
    These are different embeddings of different text, enabling:
    - Token-efficient context retrieval (summaries << full messages)
    - Different embedding models for raw vs compressed content
    - Semantic search across summaries rather than individual messages

    Args:
        db: Database session
        conversation_id: UUID of conversation thread
        start_turn_id: Start from this turn (for incremental summarization)

    Returns:
        Created MemoryShard with fresh embedding
    """
    # TODO: Implement in Phase 2b/3
    #
    # # 1. Fetch unsummarized turns for this conversation
    # query = db.query(RawConversation)\
    #     .filter(RawConversation.conversation_id == conversation_id)
    # 
    # if start_turn_id:
    #     query = query.filter(RawConversation.id >= start_turn_id)
    #
    # turns = query.order_by(RawConversation.timestamp).all()
    #
    # if not turns:
    #     return None  # Nothing to summarize
    #
    #
    # # 2. Build conversation transcript for summarization
    # transcript = "\n".join([
    #     f"{t.author}: {t.content}" for t in turns
    # ])
    #
    # # 3. Generate compressed summary via LLM (Aletheia reviews the conversation)
    # openai_service = OpenAIService()
    # summary_prompt = f"""Compress this conversation into a concise summary (2-3 sentences).
    # Focus on key topics, concepts discussed, and user understanding.
    # 
    # Conversation:
    # {transcript}
    # 
    # Summary:"""
    # 
    # summary = openai_service.create_chat_completion(
    #     messages=[{"role": "user", "content": summary_prompt}],
    #     model="gpt-4o-mini",  # Cheaper model for summarization
    #     max_tokens=150
    # )
    #
    # # 4. Generate FRESH embedding of the COMPRESSED summary
    # # This is a different embedding from the raw turn embeddings!
    # summary_embedding = convert_to_embedding(summary)
    #
    # # 5. Save to memory_shards for RAG retrieval
    # shard = MemoryShard(
    #     content=summary,
    #     embedding=summary_embedding,  # Fresh vector of compressed text
    #     source_ids=[t.id for t in turns],  # Links back to raw turns
    #     tags=["conversation", "summary", "introspection"],
    #     user_id=turns[0].user_id if turns else None,
    #     metadata_json={"conversation_id": str(conversation_id), "turn_count": len(turns)}
    # )
    # db.add(shard)
    # 
    # # 6. Mark raw turns as summarized
    # for turn in turns:
    #     if not turn.metadata_json:
    #         turn.metadata_json = {}
    #     turn.metadata_json["summarized"] = True
    #     turn.metadata_json["shard_id"] = str(shard.id)
    # 
    # db.commit()
    # db.refresh(shard)
    # return shard
    
    raise NotImplementedError("Conversation summarization - implement in Phase 2b/3")


def _generate_summary_prompt(turns: list[RawConversation]) -> str:
    """Helper to build summarization prompt from conversation turns."""
    transcript = "\n".join([f"{t.author}: {t.content}" for t in turns])
    return f"""Compress this conversation into a concise summary (2-3 sentences).
Focus on: key topics, concepts discussed, questions answered, user understanding.

Conversation:
{transcript}

Summary:"""
```

**Step 3.2: Add Background Job Scheduler**

Create background job to run every few minutes (choose one approach):

**Option A: APScheduler (simple, in-process)**
```python
# Add to app/main.py
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('interval', minutes=5)
def introspection_loop():
    """Aletheia's introspection: review recent conversations, create summaries."""
    with SessionLocal() as db:
        # Find unsummarized conversations
        unsummarized = db.query(RawConversation)\
            .filter(
                ~RawConversation.metadata_json.contains({"summarized": True})
            )\
            .distinct(RawConversation.conversation_id)\
            .all()
        
        for conv in unsummarized:
            try:
                summarize_conversation_segment(db, conv.conversation_id)
            except Exception as e:
                logger.error(f"Summarization failed: {e}")

scheduler.start()
```

**Option B: Celery (distributed, production-ready)**
```python
# app/tasks.py
from celery import Celery

celery_app = Celery('aletheia', broker='redis://localhost:6379/0')

@celery_app.task
def introspection_loop_task():
    """Background task to summarize conversations."""
    # Same logic as Option A

# Schedule every 5 minutes
celery_app.conf.beat_schedule = {
    'introspection-loop': {
        'task': 'app.tasks.introspection_loop_task',
        'schedule': 300.0,  # 5 minutes
    },
}
```

**Frequency Recommendations:**
- **Development:** Every 5 minutes (time-based)
- **Production:** Hybrid - every 5 min OR after 10 new turns (whichever comes first)
- **Why not nightly?** Users need historical context updated frequently for effective RAG

---

## 🎯 Phase 4: Dual-Context Prompt Scaffolding (HIGH PRIORITY)

**Goal:** Build prompts with both recent raw turns and historical memory shards.

**Step 4.1: Implement Dual-Context Retrieval**

Add helper function to `/v1/chat/completions`:

```python
def _build_context_scaffolding(
    db: Session,
    user_query: str,
    conversation_id: UUID,
    user_id: UUID | None,
    top_k_shards: int = 10,
    recent_turns: int = 5
) -> tuple[list[dict], list[dict]]:
    """Build dual-context prompt scaffolding.
    
    Combines:
    1. Historical context: Top K memory_shards by semantic similarity (compressed)
    2. Recent context: Latest N raw_conversations turns (detailed)
    
    Args:
        db: Database session
        user_query: Current user message
        conversation_id: Current conversation thread ID
        user_id: User profile ID for scoping
        top_k_shards: Number of memory shards to retrieve (default 10)
        recent_turns: Number of recent turns to include (default 5)
    
    Returns:
        (historical_context, recent_context)
        - historical_context: List of memory_shard dicts (compressed summaries)
        - recent_context: List of raw_conversation dicts (exact messages)
    """
    
    # 1. Get historical context via semantic search on memory_shards
    query_embedding = convert_to_embedding(user_query)
    historical_context = semantic_search(
        db=db,
        query_embedding=query_embedding,
        user_id=user_id,
        limit=top_k_shards
    )  # Returns top 10 compressed summaries
    
    # 2. Get recent context from raw_conversations
    recent_turns_query = db.query(RawConversation)\
        .filter(RawConversation.conversation_id == conversation_id)\
        .order_by(RawConversation.timestamp.desc())\
        .limit(recent_turns)
    
    recent_context = [
        {
            "role": turn.author,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat(),
        }
        for turn in reversed(recent_turns_query.all())  # Chronological order
    ]
    
    return historical_context, recent_context


def _format_dual_context_prompt(
    system_prompt: str,
    historical_context: list[dict],
    recent_context: list[dict],
    user_query: str
) -> list[dict]:
    """Format dual-context into messages array for LLM.
    
    Structure:
        System: [Personality + instructions]
        User: [Historical context block]
        User: [Recent conversation block]
        User: [Current query]
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add historical context (compressed summaries)
    if historical_context:
        historical_text = "Historical Context (semantic memories):\n\n" + "\n\n".join([
            f"- {ctx['content']}" for ctx in historical_context
        ])
        messages.append({"role": "user", "content": historical_text})
    
    # Add recent conversation (exact turns)
    if recent_context:
        recent_text = "Recent Conversation:\n\n" + "\n".join([
            f"{turn['role']}: {turn['content']}" for turn in recent_context
        ])
        messages.append({"role": "user", "content": recent_text})
    
    # Add current query
    messages.append({"role": "user", "content": user_query})
    
    return messages
```

**Step 4.2: Update `/v1/chat/completions` to Use Dual-Context**

Replace existing context retrieval (Line ~451) with:

```python
# Build dual-context prompt scaffolding
historical_context, recent_context = _build_context_scaffolding(
    db=db,
    user_query=query_text,
    conversation_id=conv_id,  # From request or generated
    user_id=user_id,
    top_k_shards=10,  # Configurable
    recent_turns=5    # Configurable
)

# Format into messages array
messages = _format_dual_context_prompt(
    system_prompt=system_instructions,
    historical_context=historical_context,
    recent_context=recent_context,
    user_query=query_text
)
```

**Benefits:**
- Recent turns: Specific, detailed context for immediate conversation flow
- Memory shards: Broader semantic context without token bloat (compressed summaries)
- Background job can lag by a few minutes without breaking UX
- Balances detail (recent) vs token efficiency (historical)

---

## 📊 Testing Requirements

**Phase 2 Tests:**
- [ ] Migration 0007 runs cleanly on test DB
- [ ] Migration 0007 drops old raw_conversations table cleanly
- [ ] RawConversation ORM model matches schema
- [ ] `/v1/chat/completions` creates raw_conversation rows (NOT memory_shards)
- [ ] Conversation turns have correct embeddings (one per turn)
- [ ] parent_id correctly links user message → assistant reply
- [ ] conversation_id groups related turns
- [ ] Dual-context prompt scaffolding retrieves latest 5 raw turns
- [ ] Dual-context prompt scaffolding retrieves top 10 memory_shards
- [ ] User scoping works for raw_conversations queries
- [ ] FK constraints enforced correctly
- [ ] No immediate memory_shard saves during chat (synchronous flow only saves raw turns)

**Phase 3 Tests:**
- [ ] Background job finds unsummarized raw_conversations
- [ ] Summarization creates compressed text (not just concatenation)
- [ ] Fresh embedding generated for compressed summary
- [ ] memory_shards.source_ids correctly links to original turns
- [ ] Background job marks turns as summarized (metadata flag)
- [ ] Job runs on schedule (time-based or turn-based trigger)
- [ ] Failed summarization doesn't break chat flow

**Integration Tests:**
- [ ] End-to-end: Chat → save turns → background job → summaries created
- [ ] Prompt includes both recent turns AND memory shards
- [ ] Token budget remains reasonable (summaries << full turns)
- [ ] All quality gates pass (lint, typecheck, test ≥85% coverage)
- [ ] Conversation turns have correct embeddings (one per turn)
- [ ] parent_id links user → assistant correctly
- [ ] conversation_id groups related turns
- [ ] Prompt scaffolding works: Latest 5 raw turns + Top 10 memory_shards
- [ ] Background job creates memory_shards from raw_conversations
- [ ] Background job runs frequently enough (every few minutes or N turns)
- [ ] User scoping works for both tables
- [ ] FK constraints enforced correctly
- [ ] No immediate memory_shard saves during chat (async background only)

### Migration Path

1. **Dev/Test:** 
   - Run migration 0007 on test DB
   - Verify schema with `\d raw_conversations`
   - Test chat endpoint creates turns (not shards)
   - Confirm embeddings work

2. **Staging:** 
   - Backup DB before migration
   - Run migration 0007 (drops old table)
   - Test E2E chat flow
   - Verify no memory_shard pollution

3. **Production:**
   - Drop old `raw_conversations` table (API logs replaced by structured logging)
   - Create new `raw_conversations` with correct turn schema
   - Deploy updated code
   - Monitor that chat turns are saved correctly
   - Phase 3: Add background summarization job later

### Documentation Updates

- [ ] Update `docs/ARCHITECTURE.md` with table purposes
- [ ] Update `docs/DATABASE.md` with new schema
- [ ] Add conversation threading examples
- [ ] Document summarization strategy (future)
- [ ] Update `GETTING_STARTED.md` if needed

---

## Task Checklist

### Immediate (Current Sprint - Phase 2)
- [ ] Create migration 0007 (restore raw_conversations schema)
- [ ] Add RawConversation ORM model
- [ ] Create `_save_conversation_turn` helper
- [ ] Update `/v1/chat/completions` to save conversation turns (raw_conversations ONLY)
- [ ] Remove old API logging code (Lines 703-773 in main.py)
- [ ] Implement dual-context prompt scaffolding (latest 5 raw + top 10 shards)
- [ ] Add tests for conversation turn storage
- [ ] Add tests for prompt scaffolding
- [ ] Run full test suite (≥85% coverage)

### Near-term (Parallel Sprint - Phase 2b/3)
- [ ] Implement Aletheia introspection/summarization service
- [ ] Background job to review raw_conversations and create compressed memory_shards
- [ ] Runs every few minutes (or after N turns) to keep memory fresh
- [ ] Fresh embeddings for compressed summaries (not reused from raw turns)
- [ ] Add 'summarized' flag or metadata to raw_conversations
- [ ] Add conversation_id to request schema
- [ ] Implement conversation threading UI support
- [ ] Add endpoint to fetch conversation history (raw_conversations)

### Future (Backlog - Phase 4+)
- [ ] Tune background job frequency (time-based vs turn-based vs hybrid)
- [ ] Optimize summarization triggers (entropy-based, emotional_index-based)
- [ ] Advanced hybrid retrieval (recent raw + relevant shards from earlier in thread)
- [ ] Multi-user conversation support
- [ ] Advanced RAG tuning (chunk size, overlap, reranking)
- [ ] Aletheia introspection analytics dashboard
- [ ] Conversation pruning/archival
- [ ] Advanced threading (reply chains, branching)
