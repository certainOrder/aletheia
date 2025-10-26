"""Database engine and session utilities.

Provides a SQLAlchemy engine, a session factory, and FastAPI dependency `get_db`
that yields a session and ensures cleanup after request handling.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for request handling and ensure closure.

    This function is intended to be used as a FastAPI dependency.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ["engine", "SessionLocal", "get_db"]
