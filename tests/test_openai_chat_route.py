from fastapi.testclient import TestClient

from app.main import app


def test_openai_chat_route(monkeypatch):
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    client = TestClient(app)
    resp = client.post("/openai-chat", json={"prompt": "Hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("response"), str)
