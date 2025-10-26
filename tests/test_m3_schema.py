"""Tests for M3: Schema migrations and content metadata.

Covers:
- Source and metadata field persistence and retrieval
- Raw conversations logging
- Aletheia context response fields
- Migration idempotence and offline-friendly behavior
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def test_index_memory_with_source_and_metadata(client, dummy_db):
    """Test that /index-memory persists source and metadata fields."""
    # Enable fallbacks for offline testing
    with patch.dict("os.environ", {"DEV_FALLBACKS": "true"}):
        response = client.post(
            "/index-memory",
            json={
                "content": "Test content with metadata",
                "tags": ["test"],
                "source": "https://example.com/test",
                "metadata": {"topic": "testing", "priority": "high"},
            },
        )
        assert response.status_code == 200
        _shard_id = response.json()["id"]

        # Verify source and metadata were stored (via dummy DB)
        added_shard = dummy_db.added[-1]
        assert added_shard.source == "https://example.com/test"
        assert added_shard.metadata_json == {"topic": "testing", "priority": "high"}


def test_ingest_propagates_source_and_metadata(client, dummy_db):
    """Test that /ingest propagates source and metadata to all chunks."""
    with patch.dict("os.environ", {"DEV_FALLBACKS": "true", "CHUNK_SIZE": "50"}):
        long_content = "Alpha beta gamma delta. " * 10  # ~240 chars, should create chunks
        response = client.post(
            "/ingest",
            json={
                "content": long_content,
                "tags": ["chunked"],
                "source": "batch-import",
                "metadata": {"batch_id": "123"},
            },
        )
        assert response.status_code == 200
        shard_ids = response.json()["ids"]
        assert len(shard_ids) > 1  # Should create multiple chunks

        # Verify all chunks have source and metadata
        for shard in dummy_db.added[-len(shard_ids) :]:
            assert shard.source == "batch-import"
            assert shard.metadata_json == {"batch_id": "123"}
            assert "chunked" in shard.tags


def test_aletheia_context_includes_source_and_metadata(client, dummy_db):
    """Test that chat responses include source and metadata in aletheia_context."""
    with patch.dict("os.environ", {"DEV_FALLBACKS": "true"}):
        # First, index content with metadata
        client.post(
            "/index-memory",
            json={
                "content": "Python is a programming language",
                "source": "wikipedia",
                "metadata": {"category": "programming"},
            },
        )

        # Chat about Python
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Tell me about Python"}],
            },
        )
        assert response.status_code == 200
        body = response.json()
        ctx = body.get("aletheia_context", [])
        assert len(ctx) > 0

        # Verify context item has all M3 fields
        item = ctx[0]
        assert "content" in item
        assert "score" in item
        assert "source" in item
        assert item["source"] == "wikipedia"
        assert "metadata" in item
        assert item["metadata"]["category"] == "programming"


def test_raw_conversations_logging(client, dummy_db):
    """Test that /v1/chat/completions logs to raw_conversations table.

    Note: This test verifies the endpoint succeeds; actual DB persistence
    would require a real DB session. The main.py code handles raw_conversations
    via SQL insert, which is tested in Docker smoke tests.
    """
    with patch.dict("os.environ", {"DEV_FALLBACKS": "true"}):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        body = response.json()

        # Verify response structure
        assert "choices" in body
        assert "model" in body
        assert "aletheia_context" in body
        # Raw conversations logging tested via Docker/integration tests


def test_offline_friendly_with_dev_fallbacks(client):
    """Test that all M3 features work offline with DEV_FALLBACKS=true."""
    with patch.dict("os.environ", {"DEV_FALLBACKS": "true"}):
        # Index
        r1 = client.post(
            "/index-memory",
            json={"content": "Offline test", "source": "local", "metadata": {"env": "test"}},
        )
        assert r1.status_code == 200

        # Ingest
        r2 = client.post(
            "/ingest",
            json={
                "content": "Long offline content " * 50,
                "source": "bulk",
                "metadata": {"type": "batch"},
            },
        )
        assert r2.status_code == 200

        # Chat
        r3 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Test"}]},
        )
        assert r3.status_code == 200
        assert "aletheia_context" in r3.json()
