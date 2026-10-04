"""
Tests for Kokoro memory and cache RAG.
"""

import time

import pytest

from core.memory.memory_rag import CacheRecord, MemoryRAG, MemoryRecord


@pytest.fixture
def memory():
    """Create a fresh MemoryRAG instance."""
    return MemoryRAG()


def test_add_and_get_session_memory(memory):
    """Memory should be stored and retrieved by session."""
    memory.add_memory(
        session_id="session-1",
        user_query="Find Python developers",
        assistant_response="Found 5 Python developers.",
        candidate_ids=["candidate-1"],
        document_ids=["doc-1"],
        jd_ids=["jd-1"],
    )

    records = memory.get_session_memory("session-1")

    assert len(records) == 1
    assert records[0].user_query == "Find Python developers"
    assert records[0].assistant_response == "Found 5 Python developers."


def test_memory_is_isolated_by_session(memory):
    """Different sessions must not share session memory."""
    memory.add_memory(
        session_id="session-1",
        user_query="Python developers",
        assistant_response="Result 1",
    )

    memory.add_memory(
        session_id="session-2",
        user_query="Data scientists",
        assistant_response="Result 2",
    )

    session_1 = memory.get_session_memory("session-1")
    session_2 = memory.get_session_memory("session-2")

    assert len(session_1) == 1
    assert len(session_2) == 1
    assert session_1[0].assistant_response == "Result 1"
    assert session_2[0].assistant_response == "Result 2"


def test_memory_limit(memory):
    """Session memory should remain bounded."""
    memory.max_memory_items = 3

    for i in range(5):
        memory.add_memory(
            session_id="session-1",
            user_query=f"query-{i}",
            assistant_response=f"response-{i}",
        )

    records = memory.get_session_memory("session-1")

    assert len(records) == 3
    assert records[-1].user_query == "query-4"


def test_get_relevant_memory(memory):
    """Relevant lexical memory should be returned."""
    memory.add_memory(
        session_id="session-1",
        user_query="Find Python developers with FastAPI experience",
        assistant_response="Found matching candidates.",
    )

    memory.add_memory(
        session_id="session-1",
        user_query="Find Java developers",
        assistant_response="Found Java candidates.",
    )

    relevant = memory.get_relevant_memory(
        session_id="session-1",
        query="Python FastAPI developer",
    )

    assert len(relevant) >= 1
    assert "Python" in relevant[0].user_query


def test_format_memory(memory):
    """Memory should be converted into readable context."""
    memory.add_memory(
        session_id="session-1",
        user_query="Find Python developers",
        assistant_response="Found 5 candidates.",
    )

    records = memory.get_session_memory("session-1")
    formatted = memory.format_memory(records)

    assert isinstance(formatted, str)
    assert "Find Python developers" in formatted
    assert "Found 5 candidates" in formatted


def test_clear_session(memory):
    """Clearing a session should remove only that session."""
    memory.add_memory(
        session_id="session-1",
        user_query="Query 1",
        assistant_response="Response 1",
    )

    memory.add_memory(
        session_id="session-2",
        user_query="Query 2",
        assistant_response="Response 2",
    )

    memory.clear_session("session-1")

    assert memory.get_session_memory("session-1") == []
    assert len(memory.get_session_memory("session-2")) == 1


def test_clear_all_memory(memory):
    """All memory should be removable."""
    memory.add_memory(
        session_id="session-1",
        user_query="Query 1",
        assistant_response="Response 1",
    )

    memory.add_memory(
        session_id="session-2",
        user_query="Query 2",
        assistant_response="Response 2",
    )

    memory.clear_all_memory()

    assert memory.get_session_memory("session-1") == []
    assert memory.get_session_memory("session-2") == []


def test_cache_set_and_get(memory):
    """Cache should return a previously stored response."""
    memory.set_cache(
        key="cache-key-1",
        response={"answer": "Python candidates"},
        session_id="session-1",
    )

    result = memory.get_cache("cache-key-1")

    assert result == {"answer": "Python candidates"}


def test_cache_miss(memory):
    """Unknown cache keys should return None."""
    assert memory.get_cache("does-not-exist") is None


def test_cache_expiration(memory):
    """Expired cache entries should not be returned."""
    memory.set_cache(
        key="cache-key-1",
        response={"answer": "expired"},
        ttl=0,
    )

    time.sleep(0.01)

    assert memory.get_cache("cache-key-1") is None


def test_delete_cache(memory):
    """A cache entry should be removable."""
    memory.set_cache(
        key="cache-key-1",
        response="cached response",
    )

    assert memory.get_cache("cache-key-1") == "cached response"

    memory.delete_cache("cache-key-1")

    assert memory.get_cache("cache-key-1") is None


def test_clear_cache(memory):
    """All cache entries should be removable."""
    memory.set_cache("key-1", "response-1")
    memory.set_cache("key-2", "response-2")

    memory.clear_cache()

    assert memory.get_cache("key-1") is None
    assert memory.get_cache("key-2") is None


def test_cache_key_is_deterministic(memory):
    """Same inputs should generate the same cache key."""
    key_1 = memory.build_cache_key(
        query="Find Python developers",
        session_id="session-1",
    )

    key_2 = memory.build_cache_key(
        query="Find Python developers",
        session_id="session-1",
    )

    assert key_1 == key_2


def test_cache_key_changes_with_query(memory):
    """Different queries should generate different keys."""
    key_1 = memory.build_cache_key(
        query="Find Python developers",
        session_id="session-1",
    )

    key_2 = memory.build_cache_key(
        query="Find Java developers",
        session_id="session-1",
    )

    assert key_1 != key_2


def test_invalidate_cache_by_session(memory):
    """Session-specific cache entries should be invalidated."""
    memory.set_cache(
        "key-1",
        "response-1",
        session_id="session-1",
    )

    memory.set_cache(
        "key-2",
        "response-2",
        session_id="session-2",
    )

    memory.invalidate_cache(session_id="session-1")

    assert memory.get_cache("key-1") is None
    assert memory.get_cache("key-2") == "response-2"


def test_invalidate_cache_by_document(memory):
    """Document-specific cache entries should be invalidated."""
    memory.set_cache(
        "key-1",
        "response-1",
        document_ids=["doc-1"],
    )

    memory.set_cache(
        "key-2",
        "response-2",
        document_ids=["doc-2"],
    )

    memory.invalidate_cache(document_ids=["doc-1"])

    assert memory.get_cache("key-1") is None
    assert memory.get_cache("key-2") == "response-2"


def test_memory_record_dataclass():
    """MemoryRecord should store expected values."""
    record = MemoryRecord(
        session_id="session-1",
        user_query="Python developers",
        assistant_response="Found candidates",
        timestamp=time.time(),
        candidate_ids=["candidate-1"],
        document_ids=["doc-1"],
        jd_ids=["jd-1"],
        metadata={"source": "test"},
    )

    assert record.session_id == "session-1"
    assert record.candidate_ids == ["candidate-1"]
    assert record.metadata["source"] == "test"


def test_cache_record_dataclass():
    """CacheRecord should store expected values."""
    now = time.time()

    record = CacheRecord(
        key="cache-key",
        response="response",
        created_at=now,
        expires_at=now + 3600,
        session_id="session-1",
        document_ids=["doc-1"],
        jd_ids=["jd-1"],
        candidate_ids=["candidate-1"],
        metadata={},
    )

    assert record.key == "cache-key"
    assert record.response == "response"
    assert record.session_id == "session-1"
