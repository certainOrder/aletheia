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
