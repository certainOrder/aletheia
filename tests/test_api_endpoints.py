def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"]


def test_health(client):
    r = client.get("/api/health")
    # health route is mounted under /api
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_openai_chat_endpoint(client, monkeypatch):
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    r = client.post("/openai-chat", json={"prompt": "Hello"})
    assert r.status_code == 200
    assert isinstance(r.json()["response"], str)


def test_v1_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1


def test_index_and_rag_chat_flow(client, monkeypatch):
    # With DEV_FALLBACKS true, embedding + chat fallbacks are deterministic
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    # Index a small memory
    r = client.post("/index-memory", json={"content": "Foo bar", "tags": ["note"]})
    assert r.status_code == 200
    shard_id = r.json()["id"]
    assert shard_id
    # Now rag-chat should search and respond
    r2 = client.post("/rag-chat", json={"prompt": "What about Foo?"})
    assert r2.status_code == 200
    data = r2.json()
    assert "answer" in data and "context" in data
    assert isinstance(data["answer"], str)
