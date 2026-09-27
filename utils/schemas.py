"""
Pydantic schemas used across Kokoro AI.

These models define stable contracts between:

    ingestion
        ↓
    retrieval
        ↓
    reranking
        ↓
    generation
        ↓
    guardrails
        ↓
    Streamlit

The goal is to keep data passed between modules
validated and predictable.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Search Configuration
# ============================================================


class SearchDepth(str, Enum):
    """
    Controls how deeply Kokoro searches the candidate index.
    """

    SHALLOW = "shallow"
    DEEP = "deep"


class QueryIntent(str, Enum):
    """
    Supported recruiter query types.
    """

    CANDIDATE_SEARCH = "candidate_search"
    FILTERING = "filtering"
    COMPARISON = "comparison"
    CANDIDATE_EXPLANATION = "candidate_explanation"
    JD_GAP_ANALYSIS = "jd_gap_analysis"
    MULTI_PART = "multi_part"
    CONVERSATIONAL_FOLLOW_UP = "conversational_follow_up"
    GENERAL = "general"
    UNSUPPORTED = "unsupported"


# ============================================================
# Query Schemas
# ============================================================


class QueryRequest(BaseModel):
    """
    Incoming recruiter query.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=4000,
        description="Recruiter's natural-language query.",
    )

    session_id: str = Field(
        min_length=1,
        max_length=200,
        description="Conversation/session identifier.",
    )

    jd_id: str | None = Field(
        default=None,
        max_length=200,
        description="Optional job-description identifier.",
    )


class QueryPlan(BaseModel):
    """
    Execution plan created by the query router/analyzer.
    """

    model_config = ConfigDict(extra="forbid")

    intent: QueryIntent

    search_depth: SearchDepth

    requires_decomposition: bool = False

    requires_multi_query: bool = False

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )


class SubQuery(BaseModel):
    """
    Individual query generated during deep search
    or multi-query expansion.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=4000,
    )

    intent: QueryIntent = QueryIntent.CANDIDATE_SEARCH


# ============================================================
# Document Metadata
# ============================================================


class DocumentMetadata(BaseModel):
    """
    Metadata associated with an ingested resume or JD chunk.
    """

    model_config = ConfigDict(extra="allow")

    candidate_id: str | None = None

    candidate_name: str | None = None

    document_id: str

    document_type: str

    chunk_id: str

    section: str | None = None

    source_file: str

    document_hash: str


# ============================================================
# Retrieval
# ============================================================


class RetrievedChunk(BaseModel):
    """
    Represents one chunk returned by the retrieval layer.

    Scores are optional because different retrieval stages
    may populate different score fields.
    """

    model_config = ConfigDict(extra="allow")

    chunk_id: str

    candidate_id: str | None = None

    candidate_name: str | None = None

    text: str = Field(
        min_length=1,
    )

    semantic_score: float | None = None

    keyword_score: float | None = None

    hybrid_score: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# Reranking
# ============================================================


class RerankedCandidate(BaseModel):
    """
    Structured output produced by the Gemini reranker.

    The reranker evaluates candidates using retrieved evidence.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str

    candidate_name: str | None = None

    match_score: float = Field(
        ge=0,
        le=100,
    )

    matched_skills: list[str] = Field(
        default_factory=list,
    )

    missing_skills: list[str] = Field(
        default_factory=list,
    )

    experience_match: bool | None = None

    explanation: str = Field(
        min_length=1,
    )

    evidence: list[str] = Field(
        default_factory=list,
    )


# ============================================================
# Final Candidate Match
# ============================================================


class CandidateMatch(BaseModel):
    """
    Candidate information returned to the generation/UI layer.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str

    candidate_name: str | None = None

    match_score: float = Field(
        ge=0,
        le=100,
    )

    strengths: list[str] = Field(
        default_factory=list,
    )

    gaps: list[str] = Field(
        default_factory=list,
    )

    jd_alignment: str | None = None

    evidence: list[str] = Field(
        default_factory=list,
    )

    explanation: str = Field(
        min_length=1,
    )


# ============================================================
# Final Application Response
# ============================================================


class FinalResponse(BaseModel):
    """
    Final validated response returned by Kokoro.

    This is the final contract between the backend
    and Streamlit UI.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
    )

    candidates: list[CandidateMatch] = Field(
        default_factory=list,
    )

    evidence: list[str] = Field(
        default_factory=list,
    )

    request_id: str | None = None