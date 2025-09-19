# 🧠 Aletheia Phase 1 RAG Architecture Overview

**Last updated:** 2025-09-19  
**Target audience:** Engineers integrating or extending the RAG pipeline

---

## 📌 High-Level Flow (Prompt to Response)

```
User Message
   ↓
OpenWebUI (frontend)
   ↓
FastAPI OpenAI-compatible endpoint (/v1/chat/completions)
   ↓
→ Generate embedding (text-embedding-3-small)
→ Query Postgres/pgvector (memory_shards) for nearest context
→ Inject retrieved context into prompt
→ Include prior 5 user/assistant turns (if available)
   ↓
OpenAI Chat Completion (gpt-4o)
   ↓
Response parsed and returned to OpenWebUI
   ↓
Log raw_conversations (request/response metadata)
→ (Optional) MemoryShards created from relevant response content
```

---

## 🧠 Components and Responsibilities

### 🔹 OpenWebUI
- Provides the user interface.
- Forwards user input to the backend via OpenAI-compatible endpoints.
- Displays streamed or final completions.
- Does not handle retrieval — that's the backend’s job.

### 🔹 FastAPI Backend
- Implements a shim between OpenWebUI and OpenAI API with memory-augmented capabilities.
- **Endpoints:**
  - `GET /v1/models`: Returns supported model (e.g., gpt-4o).
  - `POST /v1/chat/completions`: Core endpoint for chat.
    - Extracts user message.
    - Embeds it via OpenAI embedding model.
    - Searches memory_shards using pgvector.
    - Prepends context + conversation history.
    - Calls OpenAI and returns reply.
- **Internal Utilities:**
  - Embedding generation.
  - Semantic search (cosine or L2).
  - Prompt construction with retrieved context.
  - Optional debug info injection.

### 🔹 Postgres + pgvector
- Stores vectorized memory as context.
- **Table:** `memory_shards`
  - `id`, `user_id`, `content`, `embedding`, `tags`
- Embeddings are 1536-dimensional vectors (OpenAI default).
- Search returns top-k similar context chunks using pgvector.
- Also logs chat history to `raw_conversations`.

### 🔹 OpenAI API (External)
- Handles:
  - Embedding generation (`text-embedding-3-small`)
  - Chat completion (`gpt-4o`)
- API key stored in `.env` and not exposed to the user.

---

## 📊 Database Schema (Simplified)
| Table             | Purpose                                 |
|-------------------|-----------------------------------------|
| memory_shards     | Stores contextual memory chunks + vecs   |
| raw_conversations | Logs input/output and metadata           |

---

## 🔄 Prompt Construction
- **Retrieved Memory:** Top N relevant `memory_shards` (ordered by similarity).
- **Recent Conversation:** Up to 5 latest turns, if available.
- **User Message:** Latest input from OpenWebUI.
- This full message list is sent to OpenAI’s chat/completions endpoint.

---

## 🪛 Developer Notes
- CORS is enabled for local development (e.g. OpenWebUI).
- Config values loaded from `.env`: model names, API keys, DB URL.
- Embedding distance defaults to L2 but cosine is supported.
- `.env.example` included for onboarding new developers.
- All context injection is done server-side; OpenWebUI stays unchanged.
- Future phases will support streaming and finer control.

---

## ✅ Summary
This architecture creates a transparent, OpenAI-compatible RAG pipeline using local semantic memory. It enables:
- Seamless plug-in to existing OpenAI UIs (like OpenWebUI).
- Centralized augmentation logic via FastAPI.
- Extensibility through tagging, context filtering, and user-scoped memory.

---

## 1. 🔁 End-to-End Flow (User Prompt to Response)

```
[User Prompt] → [OpenWebUI] → [FastAPI RAG API] →
  [Embedding via OpenAI] → [Vector Search (pgvector)] →
    [Context Assembly + Chat History] → [GPT-4o Call] →
      [Response Parsing + Logging] → [User Response Returned]
```

---

## 2. 🌐 System Components

### **Frontend: OpenWebUI**
* Sends prompts to local API via OpenAI-compatible endpoints (`/v1/chat/completions`).
* Displays response returned from backend (proxying GPT-4o).
* No change required from user point-of-view.

### **Middle Layer: FastAPI RAG Service**
* Accepts OpenAI-style requests.
* On each user prompt:
  1. Extracts the latest user message.
  2. Generates an OpenAI embedding (via `text-embedding-3-small`).
  3. Queries local `pgvector`-enabled Postgres DB for nearest `memory_shards` using cosine similarity.
  4. Constructs new prompt:
     * Prepends top N matching memory snippets.
     * Appends recent chat history (e.g. last 5 turns).
  5. Sends combined payload to GPT-4o (or configured model).
  6. Logs result to `raw_conversations` and parsed chunks back to `memory_shards`.

### **Backend: PostgreSQL + pgvector**
* Stores:
  * `raw_conversations`: Full user-assistant turns.
  * `memory_shards`: Embeddings + content + tags + user_id.
* Uses cosine similarity (recommended for OpenAI embeddings).
* Tunable distance metric, filtering, and indexing (IVFFlat, etc.).

### **LLM (OpenAI Cloud)**
* GPT-4o receives context-enriched prompt.
* Returns assistant response only (no memory management).

---

## 3. 🧩 Memory Flow

```mermaid
graph TD;
    A[User Prompt] --> B[Embedding Generation];
    B --> C[Semantic Search in pgvector];
    C --> D[Top-N Contexts Retrieved];
    D --> E[System + User Message Assembly];
    E --> F[LLM (GPT-4o) Call];
    F --> G[Response Parsed + Displayed];
    G --> H1[raw_conversations];
    G --> H2[memory_shards];
```

---

## 4. 🛠 Deployment Pipeline Notes
* Codebase is modular:
  * `embeddings.py`, `openai_service.py`, `models.py`, etc.
* `.env` configuration supports model switching, key management, CORS, and DB params.
* Initial setup scripts create required extensions + tables (replace `create_all()` with Alembic in prod).
* Supports OpenWebUI with:
  * Base URL: `http://localhost:8000/v1`
  * Any dummy API key (real one stored server-side).

---

## 5. 🚧 Phase 1 Status Checklist (Recap)
* ✅ FastAPI intercepts OpenAI requests from OpenWebUI
* ✅ Embedding + vector search working
* ✅ Results injected as context
* ✅ GPT-4o generates answer
* ✅ Logs raw + embedded output locally

---

## 6. 📦 Next Steps
* [ ] Add `/ingest` endpoint for document batch uploads
* [ ] SSE streaming + token usage metrics
* [ ] Memory shard editing endpoints
* [ ] IVFFlat index for performance
* [ ] Multi-user support (auth + RLS)
* [ ] README + diagram added to repo (← this doc 😘)

---
