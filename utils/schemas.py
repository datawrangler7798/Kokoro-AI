"""
utils/schemas.py

Shared Pydantic schemas for Kokoro.

This module contains the data contracts shared across:

    Ingestion
        ↓
    Retrieval
        ↓
    Query Planning
        ↓
    Reranking
        ↓
    Generation
        ↓
    Memory
        ↓
    Guardrails
        ↓
    Evaluation

The purpose of this file is to provide stable, validated interfaces
between different parts of the application.

Architecture principle:

    Streamlit / API
          ↓
    Application Layer
          ↓
    Pydantic Schemas
          ↓
    Core Components

Do not place business logic in this file.
Schemas should represent and validate data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# ENUMS
# ============================================================


class DocumentType(str, Enum):
    """
    Supported document types in Kokoro.
    """

    RESUME = "resume"
    JD = "jd"


class SearchDepth(str, Enum):
    """
    Search depth selected by the query router.

    SHALLOW:
        Used for straightforward questions.

    DEEP:
        Used for complex, multi-intent, comparison,
        or gap-analysis questions.
    """

    SHALLOW = "shallow"
    DEEP = "deep"


class QueryIntent(str, Enum):
    """
    High-level query intent.

    These values help the query router and downstream
    components understand what the recruiter is asking.
    """

    SEARCH = "search"
    FILTER = "filter"
    COMPARISON = "comparison"
    JD_GAP_ANALYSIS = "jd_gap_analysis"
    SKILL_MATCH = "skill_match"
    EXPERIENCE_MATCH = "experience_match"
    CANDIDATE_DETAILS = "candidate_details"
    GENERAL = "general"


class RetrievalMethod(str, Enum):
    """
    Retrieval method used to obtain evidence.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class GuardrailStatus(str, Enum):
    """
    Guardrail processing status.
    """

    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"


class ValidationStatus(str, Enum):
    """
    Output validation status.
    """

    VALID = "valid"
    INVALID = "invalid"


class CacheStatus(str, Enum):
    """
    Cache lookup status.
    """

    HIT = "hit"
    MISS = "miss"


class IngestionStatus(str, Enum):
    """
    Document ingestion status.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


# ============================================================
# BASE CONFIGURATION
# ============================================================


class KokoroBaseModel(BaseModel):
    """
    Base Pydantic model used throughout Kokoro.

    extra="forbid":
        Prevents unexpected fields from silently entering
        internal contracts.

    validate_assignment=True:
        Validates values if a model field is changed later.

    use_enum_values=False:
        Keeps Enum values as Enum instances internally.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


# ============================================================
# DOCUMENT METADATA
# ============================================================


class DocumentMetadata(KokoroBaseModel):
    """
    Metadata associated with an ingested resume or JD.

    This metadata is used for:

        - document identification
        - duplicate detection
        - Pinecone metadata
        - BM25 indexing
        - filtering
        - observability
        - candidate aggregation
    """

    document_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the document.",
    )

    document_type: DocumentType = Field(
        ...,
        description="resume or job description.",
    )

    document_hash: str = Field(
        ...,
        min_length=1,
        description="SHA-256 hash of the original document.",
    )

    source_file: str = Field(
        ...,
        min_length=1,
        description="Original local file path/name.",
    )

    candidate_id: str | None = Field(
        default=None,
        description="Unique candidate identifier for resumes.",
    )

    candidate_name: str | None = Field(
        default=None,
        description="Candidate name when available.",
    )

    jd_id: str | None = Field(
        default=None,
        description="Unique job-description identifier.",
    )

    jd_title: str | None = Field(
        default=None,
        description="Job title when available.",
    )

    page_count: int | None = Field(
        default=None,
        ge=1,
        description="Number of pages in the source PDF.",
    )

    file_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Original file size in bytes.",
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Document ingestion timestamp.",
    )

    # --------------------------------------------------------
    # Validators
    # --------------------------------------------------------

    @field_validator(
        "document_hash",
        "source_file",
        "candidate_id",
        "candidate_name",
        "jd_id",
        "jd_title",
    )
    @classmethod
    def validate_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Normalize optional string values.

        Empty strings are converted to None for optional
        metadata fields.
        """

        if value is None:
            return None

        value = value.strip()

        return value if value else None


# ============================================================
# DOCUMENT
# ============================================================


class Document(KokoroBaseModel):
    """
    Represents a parsed source document.

    This is the document-level object produced during ingestion
    before chunking.
    """

    metadata: DocumentMetadata

    text: str = Field(
        ...,
        min_length=1,
        description="Normalized document text.",
    )

    sections: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Section name → section text mapping "
            "created by section-aware parsing."
        ),
    )

    status: IngestionStatus = Field(
        default=IngestionStatus.PENDING,
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """
        Ensure document text is not empty.
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "Document text cannot be empty."
            )

        return value


# ============================================================
# CHUNK
# ============================================================


class DocumentChunk(KokoroBaseModel):
    """
    Represents a chunk generated from a document.

    A chunk is the basic unit indexed in:

        - Pinecone
        - BM25

    Chunk metadata is intentionally explicit because it is
    required for retrieval, filtering, evidence tracing,
    and candidate aggregation.
    """

    chunk_id: str = Field(
        ...,
        min_length=1,
    )

    document_id: str = Field(
        ...,
        min_length=1,
    )

    document_type: DocumentType

    document_hash: str = Field(
        ...,
        min_length=1,
    )

    candidate_id: str | None = None

    candidate_name: str | None = None

    jd_id: str | None = None

    jd_title: str | None = None

    section: str = Field(
        ...,
        min_length=1,
        description="Resume/JD section containing this chunk.",
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Actual chunk text.",
    )

    source_file: str = Field(
        ...,
        min_length=1,
    )

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    chunk_index: int = Field(
        ...,
        ge=0,
    )

    total_chunks: int | None = Field(
        default=None,
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for retrieval.",
    )

    @model_validator(mode="after")
    def validate_chunk_position(self):
        """
        Validate chunk index against total chunks when
        total_chunks is available.
        """

        if (
            self.total_chunks is not None
            and self.chunk_index >= self.total_chunks
        ):
            raise ValueError(
                "chunk_index must be smaller than total_chunks."
            )

        return self


# ============================================================
# SEARCH FILTERS
# ============================================================


class SearchFilters(KokoroBaseModel):
    """
    Optional structured filters supplied by the recruiter.

    These filters are collected by Streamlit but applied
    inside the retrieval/application layer.

    Example:

        minimum_experience = 5
        location = "Gurgaon"
        skills = ["Python", "SQL"]
    """

    minimum_experience: float | None = Field(
        default=None,
        ge=0,
        description="Minimum years of experience.",
    )

    maximum_experience: float | None = Field(
        default=None,
        ge=0,
        description="Maximum years of experience.",
    )

    location: str | None = None

    skills: list[str] = Field(
        default_factory=list,
    )

    education: list[str] = Field(
        default_factory=list,
    )

    certifications: list[str] = Field(
        default_factory=list,
    )

    jd_id: str | None = None

    document_type: DocumentType | None = None

    candidate_ids: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_experience_range(self):
        """
        Ensure maximum experience is not lower than
        minimum experience.
        """

        if (
            self.minimum_experience is not None
            and self.maximum_experience is not None
            and self.maximum_experience
            < self.minimum_experience
        ):
            raise ValueError(
                "maximum_experience must be greater than "
                "or equal to minimum_experience."
            )

        return self

    @field_validator(
        "skills",
        "education",
        "certifications",
        "candidate_ids",
    )
    @classmethod
    def normalize_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Remove empty values and normalize whitespace.
        """

        cleaned = []

        for value in values:
            value = value.strip()

            if value:
                cleaned.append(value)

        return cleaned


# ============================================================
# QUERY PLAN
# ============================================================


class QueryPlan(KokoroBaseModel):
    """
    Query execution plan generated by the query analyzer/router.

    Example:

        User:
            "Find candidates with 5+ years of finance
             experience and SAP, then identify gaps
             against this JD."

        QueryPlan:
            intent = jd_gap_analysis
            search_depth = deep
            decomposition = True
            multi_query = True
            top_k = 10
    """

    original_query: str = Field(
        ...,
        min_length=1,
    )

    intent: QueryIntent = Field(
        default=QueryIntent.GENERAL,
    )

    search_depth: SearchDepth = Field(
        default=SearchDepth.SHALLOW,
    )

    should_decompose: bool = False

    should_expand_query: bool = False

    sub_queries: list[str] = Field(
        default_factory=list,
    )

    expanded_queries: list[str] = Field(
        default_factory=list,
    )

    top_k: int = Field(
        default=10,
        ge=1,
    )

    filters: SearchFilters = Field(
        default_factory=SearchFilters,
    )

    reasoning: str | None = Field(
        default=None,
        description=(
            "Short explanation of why the router selected "
            "this execution plan."
        ),
    )

    @field_validator(
        "sub_queries",
        "expanded_queries",
    )
    @classmethod
    def clean_queries(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Remove blank subqueries/expanded queries.
        """

        return [
            query.strip()
            for query in values
            if query and query.strip()
        ]

    @model_validator(mode="after")
    def validate_query_plan(self):
        """
        Ensure the plan is internally consistent.
        """

        if self.should_decompose and not self.sub_queries:
            raise ValueError(
                "sub_queries are required when "
                "should_decompose=True."
            )

        if (
            self.should_expand_query
            and not self.expanded_queries
        ):
            raise ValueError(
                "expanded_queries are required when "
                "should_expand_query=True."
            )

        return self


# ============================================================
# RETRIEVAL RESULT
# ============================================================


class RetrievalResult(KokoroBaseModel):
    """
    Single result returned by a retrieval method.

    This represents evidence before Gemini reranking.
    """

    chunk_id: str = Field(
        ...,
        min_length=1,
    )

    document_id: str = Field(
        ...,
        min_length=1,
    )

    document_type: DocumentType

    candidate_id: str | None = None

    candidate_name: str | None = None

    section: str | None = None

    text: str = Field(
        ...,
        min_length=1,
    )

    source_file: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    retrieval_method: RetrievalMethod

    raw_score: float

    normalized_score: float | None = None

    rank: int = Field(
        ...,
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# HYBRID RESULT
# ============================================================


class HybridResult(KokoroBaseModel):
    """
    Result produced after combining Pinecone and BM25 results.

    Score calculation is performed by the hybrid retrieval
    component, not by this schema.
    """

    chunk_id: str = Field(
        ...,
        min_length=1,
    )

    document_id: str = Field(
        ...,
        min_length=1,
    )

    document_type: DocumentType

    candidate_id: str | None = None

    candidate_name: str | None = None

    section: str | None = None

    text: str = Field(
        ...,
        min_length=1,
    )

    source_file: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    semantic_score: float | None = None

    lexical_score: float | None = None

    semantic_rank: int | None = Field(
        default=None,
        ge=1,
    )

    lexical_rank: int | None = Field(
        default=None,
        ge=1,
    )

    hybrid_score: float

    rank: int = Field(
        ...,
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# RETRIEVAL RESPONSE
# ============================================================


class RetrievalResponse(KokoroBaseModel):
    """
    Complete retrieval response before reranking.
    """

    query: str = Field(
        ...,
        min_length=1,
    )

    retrieval_method: RetrievalMethod

    results: list[
        RetrievalResult | HybridResult
    ] = Field(
        default_factory=list,
    )

    total_results: int = Field(
        default=0,
        ge=0,
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_result_count(self):
        """
        Keep total_results consistent with the result list
        when total_results is explicitly provided.
        """

        if self.total_results == 0 and self.results:
            self.total_results = len(self.results)

        return self


# ============================================================
# RERANK RESULT
# ============================================================


class CandidateFitScores(KokoroBaseModel):
    """Evidence-based dimensions used to calculate a JD fit score."""

    role_relevance: float = Field(..., ge=0, le=100)
    skills_match: float = Field(..., ge=0, le=100)
    experience_match: float = Field(..., ge=0, le=100)
    domain_relevance: float = Field(..., ge=0, le=100)
    evidence_strength: float = Field(..., ge=0, le=100)


class RerankResult(KokoroBaseModel):
    """
    Structured result returned by the Gemini reranker.

    Gemini evaluates retrieved evidence against the recruiter
    query.

    The reranker does not invent candidate information.

    Evidence must come from retrieved source chunks.
    """

    candidate_id: str = Field(
        ...,
        min_length=1,
    )

    candidate_name: str | None = None

    profile_summary: str | None = None

    match_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Normalized relevance score from 0 to 1.",
    )

    score_breakdown: CandidateFitScores | None = None

    matched_skills: list[str] = Field(
        default_factory=list,
    )

    missing_skills: list[str] = Field(
        default_factory=list,
    )

    advantages: list[str] = Field(default_factory=list)

    gaps: list[str] = Field(default_factory=list)

    recommendation: str | None = None

    experience_match: str | None = None

    explanation: str | None = None

    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence statements grounded in retrieved text.",
    )

    source_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    rank: int = Field(
        ...,
        ge=1,
    )

    @field_validator(
        "matched_skills",
        "missing_skills",
        "advantages",
        "gaps",
        "evidence",
        "source_chunk_ids",
    )
    @classmethod
    def clean_string_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Normalize list values.
        """

        return [
            value.strip()
            for value in values
            if value and value.strip()
        ]


# ============================================================
# CANDIDATE RESULT
# ============================================================


class CandidateResult(KokoroBaseModel):
    """
    Candidate-level result returned to the application layer.

    Multiple retrieved chunks belonging to the same candidate
    can be aggregated into this object.
    """

    candidate_id: str = Field(
        ...,
        min_length=1,
    )

    candidate_name: str | None = None

    profile_summary: str | None = None

    match_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    score_breakdown: CandidateFitScores | None = None

    matched_skills: list[str] = Field(
        default_factory=list,
    )

    missing_skills: list[str] = Field(
        default_factory=list,
    )

    advantages: list[str] = Field(default_factory=list)

    gaps: list[str] = Field(default_factory=list)

    recommendation: str | None = None

    experience_match: str | None = None

    explanation: str | None = None

    evidence: list[str] = Field(
        default_factory=list,
    )

    source_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    source_documents: list[str] = Field(
        default_factory=list,
    )

    rank: int | None = Field(
        default=None,
        ge=1,
    )


# ============================================================
# MEMORY
# ============================================================


class MemoryItem(KokoroBaseModel):
    """
    Single session-memory item.

    Memory helps resolve conversational references such as:

        "What about the second candidate?"

    Memory is NOT the source of truth for candidate facts.

    Candidate facts must be grounded in current retrieved
    resume/JD evidence.
    """

    memory_id: str = Field(
        ...,
        min_length=1,
    )

    session_id: str = Field(
        ...,
        min_length=1,
    )

    role: str = Field(
        ...,
        min_length=1,
        description="user or assistant.",
    )

    content: str = Field(
        ...,
        min_length=1,
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )

    importance: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# CONVERSATION TURN
# ============================================================


class ConversationTurn(KokoroBaseModel):
    """
    Represents one user/assistant interaction.
    """

    session_id: str = Field(
        ...,
        min_length=1,
    )

    user_query: str = Field(
        ...,
        min_length=1,
    )

    assistant_response: str = Field(
        ...,
        min_length=1,
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )

    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    candidate_ids: list[str] = Field(
        default_factory=list,
    )


# ============================================================
# PROMPT CONTEXT
# ============================================================


class PromptContext(KokoroBaseModel):
    """
    Context supplied to the prompt builder.

    The prompt builder combines:

        1. User query
        2. Relevant memory
        3. Retrieved evidence
        4. JD context
        5. Guardrail instructions
    """

    query: str = Field(
        ...,
        min_length=1,
    )

    memory: list[MemoryItem] = Field(
        default_factory=list,
    )

    retrieved_results: list[
        RetrievalResult | HybridResult
    ] = Field(
        default_factory=list,
    )

    reranked_results: list[RerankResult] = Field(
        default_factory=list,
    )

    jd_context: list[DocumentChunk] = Field(
        default_factory=list,
    )

    system_instructions: list[str] = Field(
        default_factory=list,
    )

    guardrail_instructions: list[str] = Field(
        default_factory=list,
    )


# ============================================================
# GUARDRAIL RESULT
# ============================================================


class GuardrailResult(KokoroBaseModel):
    """
    Result produced by input or output guardrails.
    """

    status: GuardrailStatus

    passed: bool

    reason: str | None = None

    violations: list[str] = Field(
        default_factory=list,
    )

    sanitized_text: str | None = None

    @property
    def valid(self) -> bool:
        """Backward-compatible alias for callers using the earlier result shape."""
        return self.passed

    @property
    def is_safe(self) -> bool:
        return self.passed

    @property
    def is_valid(self) -> bool:
        return self.passed

    @model_validator(mode="after")
    def validate_status(self):
        """
        Keep passed/status logically consistent.
        """

        if self.status == GuardrailStatus.PASSED and not self.passed:
            raise ValueError(
                "Guardrail status PASSED requires passed=True."
            )

        if self.status == GuardrailStatus.BLOCKED and self.passed:
            raise ValueError(
                "Guardrail status BLOCKED requires passed=False."
            )

        return self


# ============================================================
# OUTPUT VALIDATION
# ============================================================


class OutputValidationResult(KokoroBaseModel):
    """
    Result of Pydantic/output validation.
    """

    status: ValidationStatus

    valid: bool

    errors: list[str] = Field(
        default_factory=list,
    )

    validated_output: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_status(self):
        """
        Keep status and valid flag consistent.
        """

        if self.status == ValidationStatus.VALID and not self.valid:
            raise ValueError(
                "VALID status requires valid=True."
            )

        if self.status == ValidationStatus.INVALID and self.valid:
            raise ValueError(
                "INVALID status requires valid=False."
            )

        return self


# ============================================================
# FINAL RESPONSE
# ============================================================


class KokoroResponse(KokoroBaseModel):
    """
    Final response returned by the application layer.

    This is the main contract consumed by Streamlit.
    """

    request_id: str = Field(
        ...,
        min_length=1,
    )

    session_id: str = Field(
        ...,
        min_length=1,
    )

    query: str = Field(
        ...,
        min_length=1,
    )

    answer: str = Field(
        ...,
        min_length=1,
    )

    query_plan: QueryPlan | None = None

    candidates: list[CandidateResult] = Field(
        default_factory=list,
    )

    evidence: list[str] = Field(
        default_factory=list,
    )

    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    guardrail_result: GuardrailResult | None = None

    validation_result: OutputValidationResult | None = None

    cache_status: CacheStatus = CacheStatus.MISS

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# INGESTION RESULT
# ============================================================


class IngestionResult(KokoroBaseModel):
    """
    Result returned after processing one document.
    """

    document_id: str = Field(
        ...,
        min_length=1,
    )

    source_file: str = Field(
        ...,
        min_length=1,
    )

    document_type: DocumentType

    status: IngestionStatus

    document_hash: str | None = None

    candidate_id: str | None = None

    candidate_name: str | None = None

    chunk_count: int = Field(
        default=0,
        ge=0,
    )

    error: str | None = None

    skipped_reason: str | None = None

    processing_time_ms: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# BATCH INGESTION RESULT
# ============================================================


class BatchIngestionResult(KokoroBaseModel):
    """
    Result returned after processing a batch of documents.

    A failure in one document must not silently stop the
    processing of successful documents.
    """

    total_files: int = Field(
        ...,
        ge=0,
    )

    successful_files: int = Field(
        default=0,
        ge=0,
    )

    skipped_files: int = Field(
        default=0,
        ge=0,
    )

    failed_files: int = Field(
        default=0,
        ge=0,
    )

    total_chunks: int = Field(
        default=0,
        ge=0,
    )

    results: list[IngestionResult] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_counts(self):
        """
        Ensure batch counters are consistent with results.
        """

        if self.total_files != len(self.results):
            raise ValueError(
                "total_files must match the number of "
                "ingestion results."
            )

        calculated_success = sum(
            result.status == IngestionStatus.COMPLETED
            for result in self.results
        )

        calculated_skipped = sum(
            result.status == IngestionStatus.SKIPPED
            for result in self.results
        )

        calculated_failed = sum(
            result.status == IngestionStatus.FAILED
            for result in self.results
        )

        if self.successful_files != calculated_success:
            raise ValueError(
                "successful_files does not match ingestion results."
            )

        if self.skipped_files != calculated_skipped:
            raise ValueError(
                "skipped_files does not match ingestion results."
            )

        if self.failed_files != calculated_failed:
            raise ValueError(
                "failed_files does not match ingestion results."
            )

        return self


# ============================================================
# DOCUMENT REGISTRY
# ============================================================


class DocumentRegistryEntry(KokoroBaseModel):
    """
    Persistent registry entry used for incremental ingestion.

    SHA-256 is used to determine whether a document has already
    been indexed.
    """

    document_id: str = Field(
        ...,
        min_length=1,
    )

    document_hash: str = Field(
        ...,
        min_length=1,
    )

    source_file: str = Field(
        ...,
        min_length=1,
    )

    document_type: DocumentType

    candidate_id: str | None = None

    candidate_name: str | None = None

    chunk_count: int = Field(
        default=0,
        ge=0,
    )

    indexed: bool = False

    indexed_at: datetime | None = None

    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
    )


# ============================================================
# EVALUATION DATASET
# ============================================================


class EvaluationSample(KokoroBaseModel):
    """
    One ground-truth evaluation example.

    Precision@K and Recall@K require known relevant documents
    or candidates.

    RAGAS evaluation can additionally use reference answers
    depending on the selected RAGAS metrics.
    """

    sample_id: str = Field(
        ...,
        min_length=1,
    )

    query: str = Field(
        ...,
        min_length=1,
    )

    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    relevant_document_ids: list[str] = Field(
        default_factory=list,
    )

    relevant_candidate_ids: list[str] = Field(
        default_factory=list,
    )

    reference_answer: str | None = None

    context: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# RETRIEVAL EVALUATION RESULT
# ============================================================


class RetrievalEvaluationResult(KokoroBaseModel):
    """
    Retrieval-specific evaluation result.

    Includes:

        Precision@K
        Recall@K

    for a particular retrieval method.
    """

    query: str = Field(..., min_length=1)

    retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID

    precision_at_k: dict[str, float] = Field(default_factory=dict)

    recall_at_k: dict[str, float] = Field(default_factory=dict)

    mrr: float = Field(default=0.0, ge=0, le=1)

    ndcg_at_k: dict[str, float] = Field(default_factory=dict)

    retrieved_ids: list[str] = Field(
        default_factory=list,
    )

    relevant_ids: list[str] = Field(
        default_factory=list,
    )


# ============================================================
# RAGAS EVALUATION RESULT
# ============================================================


class RagasEvaluationResult(KokoroBaseModel):
    """
    RAGAS evaluation metrics.

    RAGAS is used for overall RAG quality evaluation.

    Precision@K and Recall@K remain separate retrieval-specific
    metrics.
    """

    sample_id: str = Field(default="interactive", min_length=1)

    faithfulness: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    answer_relevancy: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    context_precision: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    context_recall: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    answer_correctness: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    answer_similarity: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# COMPLETE EVALUATION RESULT
# ============================================================


class EvaluationResult(KokoroBaseModel):
    """
    Complete evaluation result displayed by the Evaluation tab.

    Contains:

        - RAGAS metrics
        - Precision@K
        - Recall@K
        - retrieval method
        - evaluation metadata
    """

    evaluation_id: str = Field(
        ...,
        min_length=1,
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )

    retrieval_method: RetrievalMethod

    k_values: list[int] = Field(
        default_factory=list,
    )

    retrieval_results: list[
        RetrievalEvaluationResult
    ] = Field(
        default_factory=list,
    )

    ragas_results: list[
        RagasEvaluationResult
    ] = Field(
        default_factory=list,
    )

    dataset_path: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("k_values")
    @classmethod
    def validate_k_values(
        cls,
        values: list[int],
    ) -> list[int]:
        """
        Validate evaluation K values.
        """

        if any(value <= 0 for value in values):
            raise ValueError(
                "Evaluation K values must be greater than 0."
            )

        return sorted(set(values))


# ============================================================
# CACHE ENTRY
# ============================================================


class CacheEntry(KokoroBaseModel):
    """
    Cached response entry.

    Response cache is separate from conversational memory.
    """

    cache_key: str = Field(
        ...,
        min_length=1,
    )

    query: str = Field(
        ...,
        min_length=1,
    )

    response: KokoroResponse

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )

    expires_at: datetime

    @model_validator(mode="after")
    def validate_expiration(self):
        """
        Ensure expiration is after creation.
        """

        if self.expires_at <= self.created_at:
            raise ValueError(
                "expires_at must be after created_at."
            )

        return self


# ============================================================
# OBSERVABILITY
# ============================================================


class RetrievalTelemetry(KokoroBaseModel):
    """
    Retrieval telemetry used for observability.

    The architecture requires retrieval scores and latency
    to be observable while avoiding unnecessary resume PII.
    """

    retrieval_method: RetrievalMethod

    pinecone_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    bm25_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    fusion_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    reranker_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    pinecone_result_count: int = Field(
        default=0,
        ge=0,
    )

    bm25_result_count: int = Field(
        default=0,
        ge=0,
    )

    hybrid_result_count: int = Field(
        default=0,
        ge=0,
    )

    reranked_result_count: int = Field(
        default=0,
        ge=0,
    )

    top_chunk_ids: list[str] = Field(
        default_factory=list,
    )

    top_scores: list[float] = Field(
        default_factory=list,
    )


class RequestTelemetry(KokoroBaseModel):
    """
    Complete request-level observability record.
    """

    request_id: str = Field(
        ...,
        min_length=1,
    )

    session_id: str = Field(
        ...,
        min_length=1,
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    llm_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    cache_status: CacheStatus = CacheStatus.MISS

    retrieval: RetrievalTelemetry | None = None

    guardrail_status: GuardrailStatus | None = None

    validation_status: ValidationStatus | None = None

    error: str | None = None


# ============================================================
# SEARCH REQUEST
# ============================================================


class SearchRequest(KokoroBaseModel):
    """
    Request entering the Kokoro application/search layer.
    """

    request_id: str = Field(
        ...,
        min_length=1,
    )

    session_id: str = Field(
        ...,
        min_length=1,
    )

    query: str = Field(
        ...,
        min_length=1,
    )

    filters: SearchFilters = Field(
        default_factory=SearchFilters,
    )

    use_memory: bool = True

    use_cache: bool = True

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """
        Remove unnecessary whitespace from the query.
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "Query cannot be empty."
            )

        return value


# ============================================================
# SEARCH RESPONSE
# ============================================================


class SearchResponse(KokoroBaseModel):
    """
    Internal search response.

    This represents the result before final response
    generation.
    """

    request: SearchRequest

    query_plan: QueryPlan

    retrieval: RetrievalResponse

    reranked_results: list[RerankResult] = Field(
        default_factory=list,
    )

    candidates: list[CandidateResult] = Field(
        default_factory=list,
    )

    memory: list[MemoryItem] = Field(
        default_factory=list,
    )

    telemetry: RetrievalTelemetry | None = None


# ============================================================
# LLM RESPONSE CONTRACT
# ============================================================


class LLMResponse(KokoroBaseModel):
    """
    Structured response returned from an LLM call.

    This separates the provider response from the rest
    of the application.
    """

    text: str = Field(
        ...,
        min_length=1,
    )

    model: str = Field(
        ...,
        min_length=1,
    )

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )
