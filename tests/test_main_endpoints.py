from fastapi.testclient import TestClient

from app.main import app


def test_v1_chat_completions_with_context(monkeypatch):
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    client = TestClient(app)
    # Index something first via API to populate our dummy DB
    client.post("/index-memory", json={"content": "context about apples"})
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Tell me about apples"},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    # our endpoint attaches aletheia_context for debugging
    assert isinstance(body.get("aletheia_context", []), list)
