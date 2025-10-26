from fastapi.testclient import TestClient

from app.main import app


def test_v1_chat_completions_requires_key_when_no_fallback(monkeypatch):
    """When DEV_FALLBACKS=false and no OPENAI_API_KEY, endpoint should return 500."""
    monkeypatch.setenv("DEV_FALLBACKS", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert "OpenAI API key is required" in body.get("detail", "")


def test_v1_chat_completions_allows_fallback_without_key(monkeypatch):
    """When DEV_FALLBACKS=true, endpoint should work without a key."""
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
