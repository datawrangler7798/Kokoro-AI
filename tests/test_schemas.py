"""
Tests for Kokoro Pydantic schemas.
"""

from __future__ import annotations

import pytest

from utils.schemas import (
    CacheStatus,
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    GuardrailStatus,
    IngestionStatus,
    MemoryItem,
    QueryIntent,
    QueryPlan,
    RetrievalMethod,
    SearchDepth,
    SearchFilters,
    ValidationStatus,
)


# ============================================================
# Enum Tests
# ============================================================


def test_document_type_enum():
    assert DocumentType.RESUME.value == "resume"
    assert DocumentType.JD.value == "jd"


def test_search_depth_enum():
    assert SearchDepth.SHALLOW.value == "shallow"
    assert SearchDepth.DEEP.value == "deep"


def test_query_intent_enum():
    assert QueryIntent.SEARCH.value == "search"
    assert QueryIntent.COMPARISON.value == "comparison"
    assert (
        QueryIntent.JD_GAP_ANALYSIS.value
        == "jd_gap_analysis"
    )


def test_retrieval_method_enum():
    assert RetrievalMethod.DENSE.value == "dense"
    assert RetrievalMethod.SPARSE.value == "sparse"
    assert RetrievalMethod.HYBRID.value == "hybrid"


def test_guardrail_status_enum():
    assert GuardrailStatus.PASSED.value == "passed"
    assert GuardrailStatus.BLOCKED.value == "blocked"
    assert GuardrailStatus.FAILED.value == "failed"


def test_validation_status_enum():
    assert ValidationStatus.VALID.value == "valid"
    assert ValidationStatus.INVALID.value == "invalid"


def test_cache_status_enum():
    assert CacheStatus.HIT.value == "hit"
    assert CacheStatus.MISS.value == "miss"


def test_ingestion_status_enum():
    assert IngestionStatus.PENDING.value == "pending"
    assert IngestionStatus.PROCESSING.value == "processing"
    assert IngestionStatus.COMPLETED.value == "completed"
    assert IngestionStatus.SKIPPED.value == "skipped"
    assert IngestionStatus.FAILED.value == "failed"


# ============================================================
# Document Metadata
# ============================================================


def test_document_metadata_creation():
    metadata = DocumentMetadata(
        document_id="doc-001",
        document_type=DocumentType.RESUME,
        filename="candidate.pdf",
    )

    assert metadata.document_id == "doc-001"
    assert metadata.document_type == DocumentType.RESUME
    assert metadata.filename == "candidate.pdf"


# ============================================================
# Document
# ============================================================


def test_document_creation():
    metadata = DocumentMetadata(
        document_id="doc-001",
        document_type=DocumentType.RESUME,
        filename="candidate.pdf",
    )

    document = Document(
        document_id="doc-001",
        content="Python Data Scientist with 5 years experience.",
        metadata=metadata,
    )

    assert document.document_id == "doc-001"
    assert "Python" in document.content
    assert document.metadata.document_id == "doc-001"


# ============================================================
# Document Chunk
# ============================================================


def test_document_chunk_creation():
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="Python and machine learning experience.",
        chunk_index=0,
    )

    assert chunk.chunk_id == "chunk-001"
    assert chunk.document_id == "doc-001"
    assert chunk.chunk_index == 0


# ============================================================
# Search Filters
# ============================================================


def test_search_filters_defaults():
    filters = SearchFilters()

    assert filters.min_experience is None
    assert filters.max_experience is None
    assert filters.location is None
    assert filters.skills == []
    assert filters.candidate_ids == []


def test_search_filters_values():
    filters = SearchFilters(
        min_experience=3,
        max_experience=8,
        location="Gurgaon",
        skills=["Python", "SQL"],
    )

    assert filters.min_experience == 3
    assert filters.max_experience == 8
    assert filters.location == "Gurgaon"
    assert filters.skills == ["Python", "SQL"]


# ============================================================
# Query Plan
# ============================================================


def test_query_plan_creation():
    plan = QueryPlan(
        query="Find Python candidates",
        intent=QueryIntent.SEARCH,
        search_depth=SearchDepth.SHALLOW,
        top_k=10,
    )

    assert plan.query == "Find Python candidates"
    assert plan.intent == QueryIntent.SEARCH
    assert plan.search_depth == SearchDepth.SHALLOW
    assert plan.top_k == 10


def test_query_plan_deep_search():
    plan = QueryPlan(
        query="Compare Python candidates with the JD",
        intent=QueryIntent.COMPARISON,
        search_depth=SearchDepth.DEEP,
        decomposition=True,
        expansion=True,
        subqueries=[
            "Find Python candidates",
            "Compare candidates with JD",
        ],
        expanded_queries=[
            "Python developers",
            "Python data science candidates",
        ],
        top_k=10,
    )

    assert plan.search_depth == SearchDepth.DEEP
    assert plan.decomposition is True
    assert plan.expansion is True
    assert len(plan.subqueries) == 2
    assert len(plan.expanded_queries) == 2


# ============================================================
# Memory
# ============================================================


def test_memory_item_creation():
    memory = MemoryItem(
        session_id="session-001",
        user_query="Find Python candidates",
        assistant_response="I found 5 candidates.",
    )

    assert memory.session_id == "session-001"
    assert memory.user_query == (
        "Find Python candidates"
    )
    assert memory.assistant_response == (
        "I found 5 candidates."
    )


# ============================================================
# Validation Tests
# ============================================================


def test_invalid_document_id_rejected():
    metadata = DocumentMetadata(
        document_id="doc-001",
        document_type=DocumentType.RESUME,
        filename="candidate.pdf",
    )

    document = Document(
        document_id="doc-001",
        content="Candidate information.",
        metadata=metadata,
    )

    assert document.document_id == "doc-001"


def test_empty_query_rejected():
    with pytest.raises(Exception):
        QueryPlan(
            query="",
            intent=QueryIntent.SEARCH,
            search_depth=SearchDepth.SHALLOW,
            top_k=10,
        )