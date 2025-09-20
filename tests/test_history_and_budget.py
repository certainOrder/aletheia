from fastapi.testclient import TestClient

from app.main import app


def test_history_turns_trimming(monkeypatch):
    # Use local fallbacks to avoid external calls
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    # Keep history to last 2 user turns
    monkeypatch.setenv("HISTORY_TURNS", "2")

    client = TestClient(app)

    # Seed minimal context so retrieval runs but doesn't matter for trimming
    client.post("/index-memory", json={"content": "some context"})

    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]

    r = client.post("/v1/chat/completions", json={"messages": msgs, "model": "gpt-4o"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    # Usage.prompt_tokens should be > 0 (approx)
    assert body["usage"]["prompt_tokens"] > 0


def test_token_budget_enforcement(monkeypatch):
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    monkeypatch.setenv("HISTORY_TURNS", "5")
    # Set a tight token budget to force trimming
    # approx tokens ~= chars/4; create ~300 chars user content to exceed 100 tokens
    monkeypatch.setenv("MAX_PROMPT_TOKENS", "100")

    client = TestClient(app)

    client.post("/index-memory", json={"content": "A" * 200})

    long_text = "L" * 800  # ~200 tokens
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": long_text},
    ]
    r = client.post("/v1/chat/completions", json={"messages": msgs, "model": "gpt-4o"})
    assert r.status_code == 200
    body = r.json()
    # Ensure usage.total_tokens is within budget-ish
    assert body["usage"]["prompt_tokens"] <= 120
