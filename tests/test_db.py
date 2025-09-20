from app.db import get_db


def test_get_db_generator_closes(monkeypatch):
    closed = {"v": False}

    class FakeSession:
        def close(self):
            closed["v"] = True

    def fake_session_local():
        return FakeSession()

    monkeypatch.setattr("app.db.SessionLocal", fake_session_local)

    gen = get_db()
    db = next(gen)
    assert isinstance(db, FakeSession)
    try:
        next(gen)
    except StopIteration:
        pass
    assert closed["v"] is True
