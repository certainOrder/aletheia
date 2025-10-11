import os
from typing import Optional
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env from repo root by default
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_ENV_PATH = os.path.join(REPO_ROOT, ".env")
load_dotenv(DEFAULT_ENV_PATH)


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

# Default database URL for the HearthMinds infrastructure database
DATABASE_URL = env("DATABASE_URL") or "postgresql+psycopg://hearthminds:hearthminds@10.0.0.1:5432/hearthminds"

class Settings(BaseModel):
    DATABASE_URL: str = DATABASE_URL
    POSTGRES_ADMIN_PASSWORD: str = env("POSTGRES_ADMIN_PASSWORD", "postgres") or "postgres"
    LOGOS_PASSWORD: str = env("LOGOS_PASSWORD", "logos") or "logos"
    ALETHEIA_PASSWORD: str = env("ALETHEIA_PASSWORD", "aletheia") or "aletheia"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

OPENAI_API_KEY: Optional[str] = env("OPENAI_API_KEY")

# Models
OPENAI_CHAT_MODEL: str = env("OPENAI_CHAT_MODEL", "gpt-4o") or "gpt-4o"
OPENAI_EMBEDDING_MODEL: str = (
    env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
)

# Embedding dimension should match DB schema
_emb_dim_str: str = env("EMBEDDING_DIM", "1536") or "1536"
EMBEDDING_DIM: int = int(_emb_dim_str)

# Retrieval metric (cosine or l2). Default to cosine for OpenAI embeddings
SIMILARITY_METRIC: str = (env("SIMILARITY_METRIC", "cosine") or "cosine").lower()

# Chunking configuration (character-based with optional sentence awareness)
_chunk_size_str: str = env("CHUNK_SIZE", "800") or "800"
CHUNK_SIZE: int = int(_chunk_size_str)
_chunk_overlap_str: str = env("CHUNK_OVERLAP", "100") or "100"
CHUNK_OVERLAP: int = int(_chunk_overlap_str)

# Conversation history & token budget
_history_turns_str: str = env("HISTORY_TURNS", "5") or "5"
HISTORY_TURNS: int = int(_history_turns_str)
_max_prompt_tokens_str: str = env("MAX_PROMPT_TOKENS", "40000") or "40000"
MAX_PROMPT_TOKENS: int = int(_max_prompt_tokens_str)

# pgvector IVFFlat tuning
_ivf_lists_str: str = env("PGVECTOR_IVFFLAT_LISTS", "100") or "100"
PGVECTOR_IVFFLAT_LISTS: int = int(_ivf_lists_str)
PGVECTOR_ENABLE_IVFFLAT: bool = (env("PGVECTOR_ENABLE_IVFFLAT", "true") or "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# CORS
_origins: str = (
    env("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080")
    or "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080"
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _origins.split(",") if o.strip()]

# Development fallbacks (disable external providers in dev)
DEV_FALLBACKS: bool = (env("DEV_FALLBACKS", "false") or "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Logging
LOG_LEVEL: str = env("LOG_LEVEL", "INFO") or "INFO"
