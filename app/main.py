
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os


# Load environment variables from .env file at startup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))


from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.openai_service import OpenAIService
from app.config import ALLOWED_ORIGINS, OPENAI_CHAT_MODEL
from fastapi.middleware.cors import CORSMiddleware
from app.db import get_db, engine
from sqlalchemy.orm import Session
from app.utils.embeddings import convert_to_embedding, semantic_search, save_embedding_to_db
from app.db.models import Base
from app.api.routes import router as api_router
from sqlalchemy import text


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include additional API routes
app.include_router(api_router)

# Serve static files (for chat UI)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Route for chat UI
@app.get("/chat")
def chat_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "chat.html"))


# Request model for OpenAI chat
class ChatRequest(BaseModel):
    prompt: str

# POST endpoint to test OpenAI integration
@app.post("/openai-chat")
async def openai_chat(request: ChatRequest):
    service = OpenAIService()
    response = service.get_response(request.prompt)
    return {"response": response}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


# RAG Chat endpoint: generates embedding from user prompt, searches memory_shards, adds context, and calls OpenAI
class RAGChatRequest(BaseModel):
    prompt: str
    user_id: str | None = None
    top_k: int = 5


@app.post("/rag-chat")
async def rag_chat(req: RAGChatRequest, db: Session = Depends(get_db)):
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


@app.on_event("startup")
def on_startup():
    # Create tables if they do not exist (dev convenience)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=engine)
    except Exception:
        # Silent failure; prefer Alembic in production
        pass


# Index content into memory_shards
from typing import Optional, List


class IndexMemoryRequest(BaseModel):
    content: str
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None


@app.post("/index-memory")
async def index_memory(req: IndexMemoryRequest, db: Session = Depends(get_db)):
    emb = convert_to_embedding(req.content)
    shard = save_embedding_to_db(db=db, content=req.content, embedding=emb, user_id=req.user_id, tags=req.tags)
    return {"id": str(shard.id)}


# OpenAI-compatible chat completions endpoint for OpenWebUI
from time import time as _time
from uuid import uuid4


@app.post("/v1/chat/completions")
async def v1_chat_completions(payload: dict, db: Session = Depends(get_db)):
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
        messages = (
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context:\n{context_text}"}]
            + messages
        )

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


@app.get("/v1/models")
async def v1_models():
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