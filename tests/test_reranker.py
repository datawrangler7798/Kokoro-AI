"""Tests for evidence-based Gemini candidate reranking."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.retrieval.re_ranker import GeminiReranker
from utils.config import settings
from utils.schemas import DocumentType, RetrievalMethod, RetrievalResult


def make_result(
    chunk_id: str = "chunk-1",
    candidate_id: str = "candidate-1",
    text: str = "Python engineer with five years of experience.",
    score: float = 0.85,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc-{candidate_id}",
        document_type=DocumentType.RESUME,
        candidate_id=candidate_id,
        candidate_name="Alex Candidate",
        text=text,
        retrieval_method=RetrievalMethod.HYBRID,
        raw_score=score,
        normalized_score=score,
        rank=1,
        metadata={"candidate_name": "Alex Candidate"},
    )


def make_reranker() -> GeminiReranker:
    reranker = GeminiReranker.__new__(GeminiReranker)
    reranker.model = "test-model"
    reranker.max_retries = 0
    reranker._client = None
    return reranker


def make_candidate(
    candidate_id: str = "candidate-1",
    source_chunk_ids: list[str] | None = None,
    scores: dict[str, float] | None = None,
    **overrides,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_name": "Alex Candidate",
        "profile_summary": "Python engineer with five years of experience.",
        "fit_scores": scores
        or {
            "role_relevance": 90,
            "skills_match": 80,
            "experience_match": 70,
            "domain_relevance": 60,
            "evidence_strength": 100,
        },
        "matched_skills": ["Python"],
        "missing_skills": ["Kubernetes"],
        "advantages": ["Five years of Python engineering experience."],
        "gaps": ["Kubernetes is not shown in the resume."],
        "recommendation": "Consider for interview; validate Kubernetes experience.",
        "experience_match": "Five years stated; meets the requested minimum.",
        "explanation": "Strong role evidence with one skill gap.",
        "evidence": ["Python engineer with five years of experience."],
        "source_chunk_ids": source_chunk_ids or ["chunk-1"],
        "rank": 1,
        **overrides,
    }


def test_prompt_requests_structured_fit_assessment():
    prompt = GeminiReranker._build_prompt(
        "Python engineer with Kubernetes",
        [make_result()],
    )
    assert "Score fit dimensions from 0 to 100" in prompt
    assert "recommendation" in prompt
    assert "Kubernetes" in prompt
    assert "skills match 35%" in prompt


def test_score_is_deterministically_weighted_and_details_are_parsed():
    reranker = make_reranker()
    parsed = reranker._parse_results(
        json.dumps({"results": [make_candidate()]}),
        [make_result()],
    )
    assert len(parsed) == 1
    assert parsed[0].match_score == pytest.approx(0.80)
    assert parsed[0].score_breakdown.skills_match == 80
    assert parsed[0].profile_summary.startswith("Python engineer")
    assert parsed[0].advantages
    assert parsed[0].gaps
    assert parsed[0].recommendation.startswith("Consider")


def test_low_relevance_candidates_are_filtered():
    reranker = make_reranker()
    weak_scores = {key: 35 for key in reranker.SCORE_WEIGHTS}
    parsed = reranker._parse_results(
        json.dumps({"results": [make_candidate(scores=weak_scores)]}),
        [make_result()],
    )
    assert parsed == []


def test_empty_gemini_results_mean_no_matches_not_fallback():
    reranker = make_reranker()
    reranker._call_gemini = MagicMock(return_value='{"results": []}')
    assert reranker.rerank("GenAI engineer", [make_result()]) == []


def test_unknown_evidence_chunk_is_rejected():
    with pytest.raises(ValueError, match="source_chunk_ids"):
        GeminiReranker._parse_results(
            json.dumps({"results": [make_candidate(source_chunk_ids=["unknown"])]}),
            [make_result()],
        )


def test_candidate_cannot_claim_another_candidates_evidence():
    with pytest.raises(ValueError, match="wrong candidate"):
        GeminiReranker._parse_results(
            json.dumps({"results": [make_candidate(candidate_id="someone-else")]}),
            [make_result()],
        )


def test_keyword_only_fallback_candidates_survive_gemini_outage():
    result = make_result(
        text="Accounting Intern with Microsoft Excel and QuickBooks experience.",
        score=0.40,
    )
    fallback = GeminiReranker._fallback_results(
        [result], top_k=5, query="accountant with 4 years, Excel and QuickBooks"
    )
    assert len(fallback) == 1
    assert fallback[0].explanation is None


def test_weak_keyword_fallback_is_still_filtered():
    result = make_result(
        text="Accountant with accounting experience.",
        score=0.03,
    )
    assert (
        GeminiReranker._fallback_results(
            [result], top_k=5, query="accountant with 4 years"
        )
        == []
    )


def test_unrelated_resume_is_not_returned_for_genai_query_during_outage():
    result = make_result(text="Accountant with accounting experience.", score=0.40)
    assert (
        GeminiReranker._fallback_results(
            [result], top_k=5, query="GenAI engineer with 1 year"
        )
        == []
    )


def test_fallback_respects_minimum_experience_when_resume_has_dates():
    intern = make_result(
        candidate_id="intern",
        text="Accounting Intern | Summer 2023. Created Excel spreadsheets.",
        score=0.40,
    )
    experienced = make_result(
        chunk_id="chunk-2",
        candidate_id="experienced-accountant",
        text=f"Cost Accountant | {datetime.now().year - 6} - Present. Excel and QuickBooks.",
        score=0.25,
    )
    fallback = GeminiReranker._fallback_results(
        [intern, experienced],
        top_k=5,
        query="accountant with 4 years, Excel and QuickBooks",
    )
    assert [candidate.candidate_id for candidate in fallback] == [
        "experienced-accountant"
    ]


def test_gemini_outage_fallback_does_not_claim_a_fit_score():
    reranker = make_reranker()
    reranker._call_gemini = MagicMock(side_effect=RuntimeError("503 overloaded"))
    fallback = reranker.rerank("Python engineer", [make_result(score=0.80)])
    assert len(fallback) == 1
    assert fallback[0].explanation is None
    assert "could not complete candidate fit scoring" in fallback.service_notice


def test_gemini_request_uses_constrained_json_output(monkeypatch):
    reranker = make_reranker()
    response = SimpleNamespace(text='{"results": []}')
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=MagicMock(return_value=response))
    )
    reranker._client = client
    acquire = MagicMock()
    monkeypatch.setattr(
        "core.retrieval.re_ranker.get_llm_rate_limiter",
        lambda: SimpleNamespace(acquire=acquire),
    )

    assert reranker._call_gemini("test prompt") == response.text
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"]["required"] == ["results"]
    assert config["max_output_tokens"] == max(settings.LLM_MAX_OUTPUT_TOKENS, 4096)
    acquire.assert_called_once_with()


def test_gemini_auth_failure_has_actionable_fallback_notice():
    reranker = make_reranker()
    reranker._call_gemini = MagicMock(
        side_effect=RuntimeError("API_KEY_INVALID: invalid API key")
    )
    fallback = reranker.rerank("Python engineer", [make_result(score=0.80)])
    assert "Check GOOGLE_API_KEY" in fallback.service_notice


def test_gemini_quota_exhaustion_is_exposed_to_the_application():
    reranker = make_reranker()
    reranker._call_gemini = MagicMock(
        side_effect=RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
    )
    fallback = reranker.rerank("Python engineer", [make_result(score=0.80)])
    assert "quota is exhausted" in fallback.service_notice


def test_empty_retrieval_returns_no_candidates():
    assert make_reranker().rerank("Python engineer", []) == []
