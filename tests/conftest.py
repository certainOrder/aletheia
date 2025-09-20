import os
import sys
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Ensure repository root is on sys.path for `import app`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEV_FALLBACKS", "true")

from app.db import get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


class DummySession:
    def __init__(self) -> None:
        self.added = []
        self.committed = False
        self.refreshed = []
        # minimal MemoryShard-like store
        self._rows = []

    def add(self, obj):  # pragma: no cover - simple passthrough
        self.added.append(obj)
        # persist a shallow copy to rows with an id
        obj.id = uuid.uuid4()
        self._rows.append(obj)

    def commit(self):  # pragma: no cover
        self.committed = True

    def refresh(self, obj):  # pragma: no cover
        self.refreshed.append(obj)

    # emulate sqlalchemy query for MemoryShard
    def query(self, model):
        class Q:
            def __init__(self, rows):
                self._rows = rows
                self._user_id = None
                self._limit = None

            def filter(self, cond):
                # naive filter by equality of user_id
                self._user_id = (
                    getattr(cond.right, "value", None)
                    or getattr(cond.right, "literal_processor", lambda _=None: None)()
                )
                return self

            def order_by(self, _):
                return self

            def limit(self, n):
                self._limit = n
                return self

            def all(self):
                rows = self._rows
                if self._user_id:
                    rows = [r for r in rows if str(r.user_id) == str(self._user_id)]
                if self._limit is not None:
                    rows = rows[: self._limit]
                return rows

        return Q(self._rows)


@pytest.fixture()
def dummy_db() -> Session:
    # Return a minimal duck-typed Session for utils and API tests
    return DummySession()  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def override_dependencies(dummy_db: Session):
    # Override FastAPI dependency for DB session
    def _get_db():
        yield dummy_db

    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.clear()
