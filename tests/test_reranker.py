"""
Tests for Kokoro reranking.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.retrieval.re_ranker import GeminiReranker
from utils.schemas import RetrievalResult


def make_retrieval_result(
    chunk_id: str = "chunk-1",
    candidate_id: str = "candidate-1",
    content: str = "Python developer with 5 years of experience.",
    score: float = 0.85,
) -> RetrievalResult:
    """Create a sample retrieval result for testing."""
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document-1",
        candidate_id=candidate_id,
        content=content,
        score=score,
        metadata={
            "candidate_name": "John Doe",
            "document_type": "resume",
        },
    )


@pytest.fixture
def reranker():
    """Create a reranker instance."""
    return GeminiReranker()


def test_build_prompt_contains_query_and_results(reranker):
    """Prompt should contain the query and retrieved evidence."""
    results = [
        make_retrieval_result(),
        make_retrieval_result(
            chunk_id="chunk-2",
            content="Experienced in SQL and machine learning.",
        ),
    ]

    prompt = reranker._build_rerank_prompt(
        query="Find Python developers",
        results=results,
    )

    assert "Find Python developers" in prompt
    assert "Python developer" in prompt
    assert "SQL" in prompt


def test_parse_response_valid_json(reranker):
    """Valid JSON response should be parsed correctly."""
    response = json.dumps(
        [
            {
                "chunk_id": "chunk-1",
                "match_score": 0.95,
                "matched_skills": ["Python"],
                "missing_skills": [],
                "experience_match": True,
                "explanation": "Strong Python experience.",
                "evidence": ["Python developer with 5 years of experience."],
            }
        ]
    )

    parsed = reranker._parse_response(response)

    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["chunk_id"] == "chunk-1"
    assert parsed[0]["match_score"] == 0.95


def test_parse_response_percentage_score(reranker):
    """Scores expressed as percentages should be normalized."""
    response = json.dumps(
        [
            {
                "chunk_id": "chunk-1",
                "match_score": 95,
                "matched_skills": ["Python"],
                "missing_skills": [],
                "experience_match": True,
                "explanation": "Strong match.",
                "evidence": [],
            }
        ]
    )

    parsed = reranker._parse_response(response)

    assert parsed[0]["match_score"] == pytest.approx(0.95)


def test_parse_response_invalid_json(reranker):
    """Invalid JSON should raise an appropriate error."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        reranker._parse_response("not valid json")


def test_rerank_returns_results(reranker):
    """Reranking should return grounded rerank results."""
    results = [
        make_retrieval_result(),
        make_retrieval_result(
            chunk_id="chunk-2",
            content="Experienced in SQL and machine learning.",
        ),
    ]

    model_response = json.dumps(
        [
            {
                "chunk_id": "chunk-1",
                "match_score": 0.95,
                "matched_skills": ["Python"],
                "missing_skills": [],
                "experience_match": True,
                "explanation": "Strong Python match.",
                "evidence": ["Python developer with 5 years of experience."],
            },
            {
                "chunk_id": "chunk-2",
                "match_score": 0.75,
                "matched_skills": ["SQL"],
                "missing_skills": ["Python"],
                "experience_match": True,
                "explanation": "Good technical match.",
                "evidence": ["Experienced in SQL and machine learning."],
            },
        ]
    )

    mock_response = MagicMock()
    mock_response.text = model_response

    with patch.object(
        reranker.client.models,
        "generate_content",
        return_value=mock_response,
    ):
        ranked = reranker.rerank(
            query="Find Python developers",
            results=results,
        )

    assert len(ranked) == 2
    assert ranked[0].candidate_id == "candidate-1"
    assert ranked[0].match_score == pytest.approx(0.95)


def test_rerank_ignores_unknown_chunk_ids(reranker):
    """Reranker output must be grounded in retrieved chunks."""
    results = [make_retrieval_result(chunk_id="chunk-1")]

    model_response = json.dumps(
        [
            {
                "chunk_id": "unknown-chunk",
                "match_score": 0.99,
                "matched_skills": ["Python"],
                "missing_skills": [],
                "experience_match": True,
                "explanation": "Unknown evidence.",
                "evidence": [],
            }
        ]
    )

    mock_response = MagicMock()
    mock_response.text = model_response

    with patch.object(
        reranker.client.models,
        "generate_content",
        return_value=mock_response,
    ):
        ranked = reranker.rerank(
            query="Find Python developers",
            results=results,
        )

    assert ranked == []


def test_rerank_empty_results(reranker):
    """Empty retrieval results should return an empty list."""
    ranked = reranker.rerank(
        query="Find Python developers",
        results=[],
    )

    assert ranked == []


def test_rerank_limits_input_results(reranker):
    """Only the configured reranking candidate set should be sent."""
    results = [
        make_retrieval_result(chunk_id=f"chunk-{i}")
        for i in range(20)
    ]

    prompt = reranker._build_rerank_prompt(
        query="Find developers",
        results=results[:5],
    )

    assert "chunk-0" in prompt
    assert "chunk-4" in prompt


def test_factory_returns_reranker():
    """Factory should create a Gemini reranker."""
    from core.retrieval.re_ranker import create_reranker

    reranker = create_reranker()

    assert isinstance(reranker, GeminiReranker)