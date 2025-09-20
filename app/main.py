"""FastAPI application entrypoint and top-level routes.

This module wires the API router, CORS, and provides OpenAI-compatible chat endpoints
and a simple RAG flow. In development, deterministic fallbacks are available to run
offline tests without external dependencies.
"""

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
from app.config import ALLOWED_ORIGINS, LOG_LEVEL, OPENAI_CHAT_MODEL
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
    RAGChatRequest,
    RAGChatResponse,
)
from app.security import SecurityHeadersMiddleware
from app.services.openai_service import OpenAIService
from app.utils.embeddings import (
    convert_to_embedding,
    save_embedding_to_db,
    semantic_search,
)

# Load environment variables from .env file at startup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan to initialize database extensions and tables in dev.

    In production deployments, prefer Alembic migrations.
    """
    try:
        # Configure structured logging early
        configure_logging(LOG_LEVEL)
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
    return {"answer": answer, "context": results}


@app.post(
    "/index-memory",
    tags=["indexing"],
    response_model=IndexMemoryResponse,
)
async def index_memory(req: IndexMemoryRequest, db: Session = Depends(get_db)):
    """Compute an embedding for content and persist it as a memory shard."""
    emb = convert_to_embedding(req.content)
    shard = save_embedding_to_db(
        db=db, content=req.content, embedding=emb, user_id=req.user_id, tags=req.tags
    )
    return {"id": str(shard.id)}


# OpenAI-compatible chat completions endpoint for OpenWebUI
@app.post("/v1/chat/completions", tags=["openai-compat"], response_model=dict)
async def v1_chat_completions(payload: dict, db: Session = Depends(get_db)):
    """OpenAI-compatible Chat Completions endpoint with optional RAG context injection."""
    # Expecting { model?: str, messages: [{role, content}, ...], stream?: bool }
    messages = payload.get("messages", [])
    model = payload.get("model")
    # Heuristic: use last user message as query for retrieval
    user_messages = [m for m in messages if m.get("role") == "user" and m.get("content")]
    query_text = user_messages[-1]["content"] if user_messages else ""
    context_results = []
    if query_text:
        qvec = convert_to_embedding(query_text)
        context_results = semantic_search(db, qvec, limit=5)
        context_text = "\n\n---\n\n".join(r["content"] for r in context_results)
        # Prepend a system message with retrieved context
        system_prompt = (
            "You are a helpful assistant. Use the provided context if relevant. "
            "Cite facts from context explicitly. If context is empty, answer normally."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}"},
        ] + messages

    service = OpenAIService()
    resp = service.chat(messages, model=model)
    # Support dict fallback and SDK response shape
    if isinstance(resp, dict):
        content = resp["choices"][0]["message"]["content"]
        resp_model = resp.get("model", model or OPENAI_CHAT_MODEL)
    else:
        content = resp.choices[0].message.content
        resp_model = resp.model
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(_time())
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
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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
