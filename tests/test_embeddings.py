from app.utils.embeddings import convert_to_embedding, save_embedding_to_db, semantic_search


def test_convert_to_embedding_deterministic_in_dev(monkeypatch):
    monkeypatch.setenv("DEV_FALLBACKS", "true")
    text = "hello world"
    emb1 = convert_to_embedding(text)
    emb2 = convert_to_embedding(text)
    assert emb1 == emb2
    assert isinstance(emb1, list)
    assert len(emb1) > 0


def test_save_and_search_with_dummy_db(dummy_db, monkeypatch):
    # Save two items and then search returns both in order (limit 2)
    e1 = [0.1, 0.2, 0.3]
    e2 = [0.9, 0.1, 0.5]
    s1 = save_embedding_to_db(db=dummy_db, content="c1", embedding=e1, user_id=None, tags=["t"])  # type: ignore[arg-type]
    s2 = save_embedding_to_db(db=dummy_db, content="c2", embedding=e2, user_id=None, tags=None)  # type: ignore[arg-type]
    out = semantic_search(dummy_db, e1, user_id=None, limit=5)  # type: ignore[arg-type]
    contents = [r["content"] for r in out]
    assert "c1" in contents and "c2" in contents
    assert str(s1.id) != str(s2.id)


def test_semantic_search_returns_scores_and_order(dummy_db, monkeypatch):
    monkeypatch.setenv("SIMILARITY_METRIC", "cosine")
    # Create three shards with known cosine similarity to query
    q = [1.0, 0.0, 0.0]
    s_a = save_embedding_to_db(db=dummy_db, content="A", embedding=[1.0, 0.0, 0.0])  # type: ignore[arg-type]
    s_b = save_embedding_to_db(db=dummy_db, content="B", embedding=[0.8, 0.6, 0.0])  # type: ignore[arg-type]
    s_c = save_embedding_to_db(db=dummy_db, content="C", embedding=[0.0, 1.0, 0.0])  # type: ignore[arg-type]
    res = semantic_search(dummy_db, q, user_id=None, limit=3)  # type: ignore[arg-type]
    # Expect A (score ~1.0), then B (~0.8), then C (~0.0)
    assert [r["content"] for r in res] == ["A", "B", "C"]
    assert res[0]["score"] is None or res[0]["score"] >= (res[1]["score"] or 0)
    # Scores should be present for cosine mode (computed locally in DummySession path)
    assert res[0]["score"] is not None
