import os
from dotenv import load_dotenv


# Load .env from repo root by default
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_ENV_PATH = os.path.join(REPO_ROOT, ".env")
load_dotenv(DEFAULT_ENV_PATH)


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


DATABASE_URL = env(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/aletheia",
)

OPENAI_API_KEY = env("OPENAI_API_KEY")

# Models
OPENAI_CHAT_MODEL = env("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Embedding dimension should match DB schema
_emb_dim = env("EMBEDDING_DIM", "1536") or "1536"
EMBEDDING_DIM = int(_emb_dim)

# CORS
_origins = env("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080") or "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080"
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]

# Development fallbacks (disable external providers in dev)
DEV_FALLBACKS = (env("DEV_FALLBACKS", "false") or "false").lower() in {"1", "true", "yes", "on"}
