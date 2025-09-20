from app.services.openai_service import OpenAIService


def test_local_response_is_deterministic(monkeypatch):
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    svc = OpenAIService(api_key=None)
    p = "What is the capital of France?"
    r1 = svc.get_response(p)
    r2 = svc.get_response(p)
    assert r1 == r2
    assert isinstance(r1, str)


def test_local_chat_response_shape(monkeypatch):
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    svc = OpenAIService(api_key=None)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hi"},
    ]
    resp = svc.chat(messages)
    assert isinstance(resp, dict)
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(resp["choices"][0]["message"]["content"], str)
