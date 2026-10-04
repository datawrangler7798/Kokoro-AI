"""
Tests for Kokoro vector store module.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.retrieval.vector_store import PineconeVectorStore

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_pinecone():
    """
    Create a mocked Pinecone client.
    """

    return MagicMock()


@pytest.fixture
def vector_store(mock_pinecone):
    """
    Create a vector store with mocked Pinecone dependencies.

    The exact constructor can be adjusted if the final
    vector_store implementation exposes different dependency
    injection parameters.
    """

    return PineconeVectorStore(
        client=mock_pinecone,
        index_name="test-index",
        dimension=768,
        metric="cosine",
    )


# ============================================================
# Initialization
# ============================================================


def test_vector_store_initialization(
    vector_store,
):
    assert vector_store is not None

    assert vector_store.index_name == "test-index"

    assert vector_store.dimension == 768


# ============================================================
# Dimension Validation
# ============================================================


def test_valid_vector_dimension(
    vector_store,
):
    vector = [0.1] * 768

    # The store should accept a vector matching
    # the configured dimension.
    vector_store._validate_vector(vector)


def test_invalid_vector_dimension(
    vector_store,
):
    vector = [0.1] * 767

    with pytest.raises(ValueError):
        vector_store._validate_vector(vector)


# ============================================================
# Empty Vector
# ============================================================


def test_empty_vector_rejected(
    vector_store,
):
    with pytest.raises(ValueError):
        vector_store._validate_vector([])


# ============================================================
# Metadata
# ============================================================


def test_metadata_is_preserved(
    vector_store,
):
    metadata = {
        "document_id": "doc-001",
        "candidate_id": "candidate-001",
        "chunk_id": "chunk-001",
        "document_type": "resume",
    }

    assert vector_store._prepare_metadata(metadata) == metadata


# ============================================================
# Upsert
# ============================================================


def test_upsert_calls_pinecone(
    vector_store,
):
    vector_store.index = MagicMock()

    vector = [0.1] * 768

    vector_store.upsert(
        vectors=[
            {
                "id": "chunk-001",
                "values": vector,
                "metadata": {
                    "document_id": "doc-001",
                },
            }
        ]
    )

    vector_store.index.upsert.assert_called_once()


# ============================================================
# Query
# ============================================================


def test_query_calls_pinecone(
    vector_store,
):
    vector_store.index = MagicMock()

    vector_store.index.query.return_value = {"matches": []}

    vector = [0.1] * 768

    result = vector_store.query(
        vector,
        top_k=5,
    )

    vector_store.index.query.assert_called_once()

    assert result is not None


# ============================================================
# Query Dimension Validation
# ============================================================


def test_query_rejects_wrong_dimension(
    vector_store,
):
    vector = [0.1] * 767

    with pytest.raises(ValueError):
        vector_store.query(
            vector,
            top_k=5,
        )


# ============================================================
# Top K Validation
# ============================================================


def test_query_rejects_invalid_top_k(
    vector_store,
):
    vector = [0.1] * 768

    with pytest.raises(ValueError):
        vector_store.query(
            vector,
            top_k=0,
        )


# ============================================================
# Delete
# ============================================================


def test_delete_calls_pinecone(
    vector_store,
):
    vector_store.index = MagicMock()

    vector_store.delete(
        ids=[
            "chunk-001",
            "chunk-002",
        ]
    )

    vector_store.index.delete.assert_called_once()


# ============================================================
# Stats
# ============================================================


def test_stats(
    vector_store,
):
    vector_store.index = MagicMock()

    vector_store.index.describe_index_stats.return_value = {"total_vector_count": 10}

    stats = vector_store.stats()

    assert stats is not None
