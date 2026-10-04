"""
Tests for Kokoro hybrid search.

Tests:
    - BM25 indexing
    - BM25 retrieval
    - Dense + sparse score fusion
    - Score normalization
    - Deduplication
    - Top-K selection
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.retrieval.hybrid_indexer import BM25Backend, BM25Record, HybridIndexer

# ============================================================
# Test Semantic Retriever
# ============================================================


@dataclass
class MockSemanticResult:
    chunk_id: str
    score: float
    content: str
    metadata: dict


class MockSemanticRetriever:
    """
    Small deterministic semantic retriever for unit tests.
    """

    def __init__(self, results):
        self.results = results

    def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        filters=None,
    ):
        return self.results[:top_k]


# ============================================================
# BM25 Fixtures
# ============================================================


@pytest.fixture
def bm25_backend():
    backend = BM25Backend()

    backend.upsert(
        [
            BM25Record(
                chunk_id="chunk-1",
                text=("Python data scientist " "with machine learning experience"),
                metadata={
                    "candidate_id": "candidate-1",
                },
            ),
            BM25Record(
                chunk_id="chunk-2",
                text=("Java backend developer " "with Spring Boot experience"),
                metadata={
                    "candidate_id": "candidate-2",
                },
            ),
            BM25Record(
                chunk_id="chunk-3",
                text=("Python machine learning " "and generative AI engineer"),
                metadata={
                    "candidate_id": "candidate-3",
                },
            ),
        ]
    )

    return backend


# ============================================================
# BM25 Tests
# ============================================================


def test_bm25_indexing(bm25_backend):
    assert bm25_backend.size() == 3


def test_bm25_search_returns_results(
    bm25_backend,
):
    results = bm25_backend.search(
        "Python machine learning",
        top_k=2,
    )

    assert results
    assert len(results) <= 2


def test_bm25_python_query_matches_python_documents(
    bm25_backend,
):
    results = bm25_backend.search(
        "Python",
        top_k=3,
    )

    chunk_ids = [record.chunk_id for record, _ in results]

    assert "chunk-1" in chunk_ids
    assert "chunk-3" in chunk_ids


def test_bm25_filtering(
    bm25_backend,
):
    results = bm25_backend.search(
        "Python",
        top_k=5,
        filters={"candidate_id": "candidate-3"},
    )

    assert all(
        record.metadata.get("candidate_id") == "candidate-3"
        for record, _ in results
    )


# ============================================================
# BM25 Delete
# ============================================================


def test_bm25_delete(
    bm25_backend,
):
    bm25_backend.delete(["chunk-1"])

    results = bm25_backend.search(
        "Python",
        top_k=5,
    )

    chunk_ids = [record.chunk_id for record, _ in results]

    assert "chunk-1" not in chunk_ids


# ============================================================
# Hybrid Fixtures
# ============================================================


@pytest.fixture
def hybrid_indexer(
    bm25_backend,
):
    semantic_results = [
        MockSemanticResult(
            chunk_id="chunk-1",
            score=0.95,
            content="Python data scientist",
            metadata={"candidate_id": "candidate-1"},
        ),
        MockSemanticResult(
            chunk_id="chunk-3",
            score=0.85,
            content="Python GenAI engineer",
            metadata={"candidate_id": "candidate-3"},
        ),
    ]

    semantic_retriever = MockSemanticRetriever(semantic_results)

    return HybridIndexer(
        semantic_retriever=semantic_retriever,
        bm25_backend=bm25_backend,
        semantic_weight=0.6,
        keyword_weight=0.4,
    )


# ============================================================
# Score Normalization
# ============================================================


def test_score_normalization():
    indexer = HybridIndexer(
        semantic_retriever=MockSemanticRetriever([]),
        bm25_backend=BM25Backend(),
        semantic_weight=0.6,
        keyword_weight=0.4,
    )

    scores = {
        "a": 10.0,
        "b": 5.0,
        "c": 0.0,
    }

    normalized = indexer._normalize_scores(scores)

    assert normalized["a"] == pytest.approx(1.0)

    assert normalized["c"] == pytest.approx(0.0)


# ============================================================
# Hybrid Search
# ============================================================


def test_hybrid_search_returns_results(
    hybrid_indexer,
):
    results = hybrid_indexer.search(
        "Python machine learning",
        top_k=5,
    )

    assert results
    assert len(results) <= 5


def test_hybrid_result_contains_scores(
    hybrid_indexer,
):
    results = hybrid_indexer.search(
        "Python",
        top_k=5,
    )

    assert results

    result = results[0]

    assert hasattr(
        result,
        "semantic_score",
    )

    assert hasattr(
        result,
        "keyword_score",
    )

    assert hasattr(
        result,
        "hybrid_score",
    )


def test_hybrid_results_are_ranked(
    hybrid_indexer,
):
    results = hybrid_indexer.search(
        "Python machine learning",
        top_k=5,
    )

    scores = [result.hybrid_score for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )


# ============================================================
# Deduplication
# ============================================================


def test_hybrid_search_deduplicates_chunks(
    hybrid_indexer,
):
    results = hybrid_indexer.search(
        "Python",
        top_k=10,
    )

    chunk_ids = [result.chunk_id for result in results]

    assert len(chunk_ids) == len(set(chunk_ids))


# ============================================================
# Top-K
# ============================================================


def test_hybrid_search_respects_top_k(
    hybrid_indexer,
):
    results = hybrid_indexer.search(
        "Python",
        top_k=1,
    )

    assert len(results) <= 1


# ============================================================
# Empty Query
# ============================================================


def test_empty_query_returns_no_results(
    hybrid_indexer,
):
    assert hybrid_indexer.search("", top_k=5) == []
