"""FastAPI application entrypoint and top-level routes.

This module wires the API router, CORS, and provides OpenAI-compatible chat endpoints
and a simple RAG flow. In development, deterministic fallbacks are available to run
offline tests without external dependencies.
"""

import logging
import os
from contextlib import asynccontextmanager
from time import time as _time
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.routes import router as api_router
from app.config import (
    ALLOWED_ORIGINS,
    DEV_FALLBACKS,
    LOG_LEVEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)
from app.config import (
    CHUNK_OVERLAP as DEFAULT_CHUNK_OVERLAP,
)
from app.config import (
    CHUNK_SIZE as DEFAULT_CHUNK_SIZE,
)
from app.db import engine, get_db
from app.db.models import Base
from app.error_handlers import (
    handle_http_exception,
    handle_unhandled_exception,
    handle_validation_error,
)
from app.logging_utils import RequestIdMiddleware, configure_logging
from app.schemas import (
    ChatRequest,
    IndexMemoryRequest,
    IndexMemoryResponse,
    IngestRequest,
    IngestResponse,
    RAGChatRequest,
    RAGChatResponse,
)
from app.security import SecurityHeadersMiddleware
from app.services.openai_service import OpenAIService
from app.utils.chunking import chunk_text
from app.utils.embeddings import (
    convert_to_embedding,
    save_embedding_to_db,
    semantic_search,
)

# Load environment variables from .env file at startup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
logger = logging.getLogger("app.flow")


def _ensure_openai_ready() -> None:
    """Raise a clear HTTP 500 if running without key while fallbacks are disabled.

    Keeps startup permissive for local/dev while enforcing Phase 2 contract at request time.
    """
    from fastapi import HTTPException

    # Evaluate from environment at request time to support dynamic config in tests/dev
    from app.config import env as _env

    _dev_fallbacks = (_env("DEV_FALLBACKS", "false") or "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _openai_key = _env("OPENAI_API_KEY")

    if not _dev_fallbacks and not _openai_key:
        msg = (
            "OpenAI API key is required when DEV_FALLBACKS=false. "
            "Set OPENAI_API_KEY in your .env (or set DEV_FALLBACKS=true for local dev). "
            "See docs/DEV_ENVIRONMENT.md#use-real-openai-calls-disable-fallbacks"
        )
        logging.getLogger("app").error(
            "provider_config_error",
            extra={
                "detail": "OPENAI_API_KEY missing while fallbacks disabled",
                "docs_url": "docs/DEV_ENVIRONMENT.md#use-real-openai-calls-disable-fallbacks",
            },
        )
        raise HTTPException(status_code=500, detail=msg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan to initialize database extensions and tables in dev.

    In production deployments, prefer Alembic migrations.
    """
    try:
        # Configure structured logging early
        configure_logging(LOG_LEVEL)
        logging.getLogger("app").info(
            "startup_config",
            extra={
                "log_level": LOG_LEVEL,
                "cors_origins_count": len(ALLOWED_ORIGINS),
            },
        )
        if not DEV_FALLBACKS and not OPENAI_API_KEY:
            logging.getLogger("app").warning(
                "startup_key_missing",
                extra={
                    "detail": "OPENAI_API_KEY not set while DEV_FALLBACKS=false",
                    "docs_url": "docs/DEV_ENVIRONMENT.md#use-real-openai-calls-disable-fallbacks",
                },
            )
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=engine)
    except Exception:
        # Silent failure; prefer Alembic in production
        pass
    yield


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware & basic request logging
app.add_middleware(RequestIdMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Include additional API routes under a stable "/api" prefix
app.include_router(api_router, prefix="/api")


# POST endpoint to test OpenAI integration
@app.post("/openai-chat", tags=["chat"], response_model=dict)
async def openai_chat(request: ChatRequest):
    """Invoke the OpenAI service (or fallback) with a simple prompt payload."""
    _ensure_openai_ready()
    logger.info("route_openai_chat", extra={"prompt_len": len(request.prompt or "")})
    service = OpenAIService()
    response = service.get_response(request.prompt)
    return {"response": response}


@app.get("/", tags=["misc"], summary="Liveness root")
def read_root():
    """Basic liveness message for root path."""
    return {"message": "Hello, World!"}


@app.post(
    "/rag-chat",
    tags=["rag"],
    response_model=RAGChatResponse,
)
async def rag_chat(req: RAGChatRequest, db: Session = Depends(get_db)):
    """RAG flow: embed query, retrieve context, and call chat provider (or fallback)."""
    _ensure_openai_ready()
    logger.info(
        "route_rag_chat",
        extra={"prompt_len": len(req.prompt or ""), "top_k": req.top_k, "user_id": req.user_id},
    )
    query_vec = convert_to_embedding(req.prompt)
    results = semantic_search(db, query_vec, user_id=req.user_id, limit=req.top_k)
    context_chunks = [r["content"] for r in results]
    context_text = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant. Use the provided context if relevant. "
        "Cite facts from context explicitly. If context is empty, answer normally."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {req.prompt}"},
    ]
    service = OpenAIService()
    response = service.chat(messages)
    # Support both dict fallback and OpenAI SDK response
    if isinstance(response, dict):
        answer = response["choices"][0]["message"]["content"]
    else:
        answer = response.choices[0].message.content
    logger.info(
        "route_rag_result",
        extra={"context_count": len(results), "answer_len": len(answer or "")},
    )
    return {"answer": answer, "context": results}


@app.post(
    "/index-memory",
    tags=["indexing"],
    response_model=IndexMemoryResponse,
)
async def index_memory(req: IndexMemoryRequest, db: Session = Depends(get_db)):
    """Compute an embedding for content and persist it as a memory shard."""
    _ensure_openai_ready()
    logger.info(
        "route_index_memory",
        extra={"content_len": len(req.content or ""), "tags_count": len(req.tags or [])},
    )
    emb = convert_to_embedding(req.content)
    shard = save_embedding_to_db(
        db=db,
        content=req.content,
        embedding=emb,
        user_id=req.user_id,
        tags=req.tags,
        source=req.source,
        metadata=req.metadata,
    )
    logger.info("route_index_memory_result", extra={"shard_id": str(shard.id)})
    return {"id": str(shard.id)}


@app.post(
    "/ingest",
    tags=["indexing"],
    response_model=IngestResponse,
)
async def ingest(req: IngestRequest, db: Session = Depends(get_db)):
    """Chunk large content and index each chunk as a separate shard.

    Uses CHUNK_SIZE and CHUNK_OVERLAP from config. Tags and user_id are propagated.
    """
    _ensure_openai_ready()
    # Read chunking params from env at request time to allow tests to override
    from app.config import env as _env

    _chunk_size = int(_env("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)) or str(DEFAULT_CHUNK_SIZE))
    _chunk_overlap = int(
        _env("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)) or str(DEFAULT_CHUNK_OVERLAP)
    )
    logger.info(
        "route_ingest",
        extra={
            "content_len": len(req.content or ""),
            "tags_count": len(req.tags or []),
            "chunk_size": _chunk_size,
            "chunk_overlap": _chunk_overlap,
        },
    )
    chunks = chunk_text(req.content, _chunk_size, _chunk_overlap)
    ids: list[str] = []
    for ch in chunks:
        emb = convert_to_embedding(ch)
        shard = save_embedding_to_db(
            db=db,
            content=ch,
            embedding=emb,
            user_id=req.user_id,
            tags=req.tags,
            source=req.source,
            metadata=req.metadata,
        )
        ids.append(str(shard.id))
    logger.info("route_ingest_result", extra={"chunk_count": len(ids)})
    return {"ids": ids}


# OpenAI-compatible chat completions endpoint for OpenWebUI
@app.post("/v1/chat/completions", tags=["openai-compat"], response_model=dict)
async def v1_chat_completions(payload: dict, db: Session = Depends(get_db)):
    """OpenAI-compatible Chat Completions endpoint with optional RAG context injection."""
    _ensure_openai_ready()
    # Expecting { model?: str, messages: [{role, content}, ...], stream?: bool }
    messages = payload.get("messages", [])
    model = payload.get("model")
    logger.info(
        "route_v1_chat_completions",
        extra={"messages_count": len(messages or []), "model": model or OPENAI_CHAT_MODEL},
    )
    # Read dynamic config for history/budget to support test overrides
    from app.config import HISTORY_TURNS as DEFAULT_HISTORY_TURNS
    from app.config import MAX_PROMPT_TOKENS as DEFAULT_MAX_PROMPT_TOKENS
    from app.config import env as _env

    _history_turns = int(
        _env("HISTORY_TURNS", str(DEFAULT_HISTORY_TURNS)) or str(DEFAULT_HISTORY_TURNS)
    )
    _max_prompt_tokens = int(
        _env("MAX_PROMPT_TOKENS", str(DEFAULT_MAX_PROMPT_TOKENS)) or str(DEFAULT_MAX_PROMPT_TOKENS)
    )

    # Helper: select last N user "turns" (counted by user messages),
    # including any assistant replies between them
    def _select_recent_turns(all_msgs: list[dict], n_users: int) -> list[dict]:
        selected: list[dict] = []
        count_users = 0
        for m in reversed(all_msgs):
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            selected.append(m)
            if role == "user":
                count_users += 1
                if count_users >= n_users:
                    break
        selected.reverse()
        return selected

    # Helper: approximate token count by characters/4 (rough heuristic)
    def _approx_tokens(msgs: list[dict]) -> int:
        total_chars = 0
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                total_chars += len(c)
            else:
                # For non-string content (function/tool calls), ignore for now
                continue
        return (total_chars + 3) // 4

    # Trim incoming history to last N turns prior to retrieval/context assembly
    trimmed_history = _select_recent_turns(messages, max(_history_turns, 0))
    dropped_count = len([m for m in messages if m.get("role") in ("user", "assistant")]) - len(
        trimmed_history
    )
    if dropped_count > 0:
        logger.info(
            "history_trim",
            extra={
                "requested_turns": _history_turns,
                "dropped_messages": dropped_count,
                "retained_messages": len(trimmed_history),
            },
        )
    # Heuristic: use last user message as query for retrieval
    user_messages = [m for m in trimmed_history if m.get("role") == "user" and m.get("content")]
    query_text = user_messages[-1]["content"] if user_messages else ""
    context_results = []
    if query_text:
        # Include retrieval settings for observability
        from app.config import SIMILARITY_METRIC as CONF_METRIC

        logger.info(
            "retrieval_begin",
            extra={
                "query_len": len(query_text or ""),
                "metric": (CONF_METRIC or "cosine"),
                "top_k": 5,
            },
        )
        qvec = convert_to_embedding(query_text)
        context_results = semantic_search(db, qvec, limit=5)
        context_text = "\n\n---\n\n".join(r["content"] for r in context_results)
        # Prepend a system message with retrieved context
        system_prompt = (
            "You are a helpful assistant. Use the provided context if relevant. "
            "Cite facts from context explicitly. If context is empty, answer normally."
        )
        # Build assembled messages: our system + context wrapper + trimmed history
        assembled_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}"},
        ] + trimmed_history
    else:
        # No query text; just carry trimmed history and a default system prompt
        context_text = ""
        context_results = []
        system_prompt = "You are a helpful assistant. Answer concisely and helpfully."
        assembled_messages = [{"role": "system", "content": system_prompt}] + trimmed_history

    # Enforce token budget by dropping oldest history first (preserve system + context + last user)
    pre_tokens = _approx_tokens(assembled_messages)
    history_start_index = (
        2 if context_text != "" else 1
    )  # system + optional context occupy the head

    # Identify indices of removable messages (history only)
    def _count_user_msgs(msgs: list[dict]) -> int:
        return sum(1 for m in msgs if m.get("role") == "user")

    # Ensure we preserve at least one user message (the last one)
    while pre_tokens > _max_prompt_tokens and (history_start_index < len(assembled_messages)):
        # If removing next history message would remove the last remaining user, stop
        remaining_history = assembled_messages[history_start_index:]
        if _count_user_msgs(remaining_history) <= 1:
            break
        # Drop the oldest history message
        dropped = assembled_messages.pop(history_start_index)
        logger.debug(
            "token_budget_drop_history",
            extra={
                "dropped_role": dropped.get("role"),
                "dropped_chars": len(dropped.get("content") or ""),
            },
        )
        pre_tokens = _approx_tokens(assembled_messages)

    # If still over budget, trim the context text content
    if pre_tokens > _max_prompt_tokens and context_text:
        # Compute non-context tokens
        non_context = (
            assembled_messages[:history_start_index] + assembled_messages[history_start_index + 1 :]
        )
        non_ctx_tokens = _approx_tokens(non_context)
        # Available tokens for context message
        available_for_context = max(_max_prompt_tokens - non_ctx_tokens, 0)
        # Convert tokens to approx chars then clamp content
        allowed_chars = max(available_for_context * 4, 0)
        original_ctx = assembled_messages[1]["content"]  # the "Context:\n..." message
        if len(original_ctx) > allowed_chars:
            new_ctx = original_ctx[:allowed_chars].rstrip()
            assembled_messages[1]["content"] = new_ctx
            logger.info(
                "token_budget_trim_context",
                extra={
                    "original_chars": len(original_ctx),
                    "trimmed_chars": len(new_ctx),
                    "max_tokens": _max_prompt_tokens,
                },
            )
        # If still over budget, trim the last user message content to fit
        post_tokens = _approx_tokens(assembled_messages)
        if post_tokens > _max_prompt_tokens:
            # Find last user message index
            last_user_idx = None
            for i in range(len(assembled_messages) - 1, -1, -1):
                if assembled_messages[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx is not None:
                other_msgs = (
                    assembled_messages[:last_user_idx] + assembled_messages[last_user_idx + 1 :]
                )
                other_tokens = _approx_tokens(other_msgs)
                allowed_for_user = max(_max_prompt_tokens - other_tokens, 0)
                allowed_chars = max(allowed_for_user * 4, 0)
                user_content = assembled_messages[last_user_idx].get("content") or ""
                if isinstance(user_content, str) and len(user_content) > allowed_chars:
                    new_user = user_content[:allowed_chars].rstrip()
                    assembled_messages[last_user_idx]["content"] = new_user
                    logger.info(
                        "token_budget_trim_last_user",
                        extra={
                            "original_chars": len(user_content),
                            "trimmed_chars": len(new_user),
                            "max_tokens": _max_prompt_tokens,
                        },
                    )
                    post_tokens = _approx_tokens(assembled_messages)

    post_tokens = _approx_tokens(assembled_messages)
    if post_tokens > _max_prompt_tokens:
        logger.info(
            "token_budget_enforced",
            extra={
                "pre_tokens": pre_tokens,
                "post_tokens": post_tokens,
                "max_tokens": _max_prompt_tokens,
            },
        )
    else:
        logger.info(
            "token_budget_enforced",
            extra={
                "pre_tokens": pre_tokens,
                "post_tokens": post_tokens,
                "max_tokens": _max_prompt_tokens,
            },
        )

    service = OpenAIService()
    resp = service.chat(assembled_messages, model=model)
    # Support dict fallback and SDK response shape
    if isinstance(resp, dict):
        content = resp["choices"][0]["message"]["content"]
        resp_model = resp.get("model", model or OPENAI_CHAT_MODEL)
    else:
        content = resp.choices[0].message.content
        resp_model = resp.model
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(_time())
    logger.info(
        "completion_result",
        extra={
            "model": resp_model,
            "content_len": len(content or ""),
            "context_count": len(context_results),
        },
    )
    out = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": resp_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        # Approximate usage; for observability only
        "usage": {
            "prompt_tokens": post_tokens,
            "completion_tokens": (len(content or "") + 3) // 4,
            "total_tokens": post_tokens + ((len(content or "") + 3) // 4),
        },
        "aletheia_context": context_results,
    }
    return out


@app.get("/v1/models", tags=["openai-compat"], response_model=dict)
async def v1_models():
    """List available models, compatible with OpenAI `/v1/models`."""
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_CHAT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "aletheia",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# Exception handlers (registered after routes/middleware)
app.add_exception_handler(Exception, handle_unhandled_exception)
from collections.abc import Awaitable, Callable  # noqa: E402
from typing import cast  # noqa: E402

from fastapi import HTTPException  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from starlette.responses import Response  # noqa: E402

GenericHandler = Callable[..., Response | Awaitable[Response]]
app.add_exception_handler(HTTPException, cast(GenericHandler, handle_http_exception))
app.add_exception_handler(RequestValidationError, cast(GenericHandler, handle_validation_error))
