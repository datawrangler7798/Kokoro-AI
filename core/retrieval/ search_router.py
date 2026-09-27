"""
Search routing and hybrid retrieval for Kokoro AI.

Responsibilities:
- Analyze recruiter queries
- Determine query intent
- Select shallow or deep search
- Retrieve from Pinecone and BM25
- Normalize retrieval scores
- Fuse semantic and keyword results
- Return the final Top-K candidates/chunks

This module does not perform Gemini reranking.
Reranking is handled by re_ranker.py.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from core.retrieval.hybrid_indexer import (
    HybridIndexer,
    normalize_scores,
)
from core.retrieval.vector_store import PineconeVectorStore
from utils.logger import logger
from utils.schemas import (
    QueryIntent,
    QueryPlan,
    QueryRequest,
    RetrievedChunk,
    SearchDepth,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_TOP_K = 5

DEFAULT_SEMANTIC_WEIGHT = 0.6

DEFAULT_KEYWORD_WEIGHT = 0.4


# ============================================================
# Query Router
# ============================================================


class SearchRouter:
    """
    Routes recruiter queries through Kokoro's retrieval pipeline.

    Search modes:

        SHALLOW
            Query
              ↓
            Pinecone
              +
            BM25
              ↓
            Hybrid Top-K

        DEEP
            Complex Query
              ↓
            Decomposition
              ↓
            Multiple retrieval operations
              ↓
            Result fusion
              ↓
            Hybrid Top-K

    Deep-query decomposition itself is kept deliberately
    deterministic here. LLM-based query expansion is handled
    separately by utils.multi_query.py.
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        hybrid_indexer: HybridIndexer,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ) -> None:
        """
        Initialize the search router.

        Parameters
        ----------
        vector_store:
            Pinecone vector-store wrapper.

        hybrid_indexer:
            BM25 indexer.

        semantic_weight:
            Weight assigned to semantic retrieval.

        keyword_weight:
            Weight assigned to keyword retrieval.
        """

        if semantic_weight < 0:
            raise ValueError(
                "semantic_weight cannot be negative."
            )

        if keyword_weight < 0:
            raise ValueError(
                "keyword_weight cannot be negative."
            )

        if semantic_weight + keyword_weight <= 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        self.vector_store = vector_store

        self.hybrid_indexer = hybrid_indexer

        total_weight = (
            semantic_weight + keyword_weight
        )

        self.semantic_weight = (
            semantic_weight / total_weight
        )

        self.keyword_weight = (
            keyword_weight / total_weight
        )

    # ========================================================
    # Query Planning
    # ========================================================

    def create_query_plan(
        self,
        request: QueryRequest,
    ) -> QueryPlan:
        """
        Create a retrieval plan for a recruiter query.

        The router does not rely only on query length.

        It considers:
        - explicit comparison
        - filtering constraints
        - JD-gap language
        - candidate-specific questions
        - multiple independent requirements
        - conversational follow-ups
        - unsupported/general questions
        """

        query = request.query.strip()

        intent = self._detect_intent(
            query
        )

        requires_decomposition = (
            self._requires_decomposition(
                query,
                intent,
            )
        )

        requires_multi_query = (
            self._requires_multi_query(
                query,
                intent,
            )
        )

        # Deep search is used for complex retrieval tasks.
        deep_search = (
            requires_decomposition
            or requires_multi_query
            or intent
            in {
                QueryIntent.COMPARISON,
                QueryIntent.JD_GAP_ANALYSIS,
                QueryIntent.MULTI_PART,
            }
        )

        # General/unsupported queries should not trigger
        # expensive retrieval.
        if intent in {
            QueryIntent.GENERAL,
            QueryIntent.UNSUPPORTED,
        }:
            deep_search = False
            requires_decomposition = False
            requires_multi_query = False

        search_depth = (
            SearchDepth.DEEP
            if deep_search
            else SearchDepth.SHALLOW
        )

        plan = QueryPlan(
            intent=intent,
            search_depth=search_depth,
            requires_decomposition=requires_decomposition,
            requires_multi_query=requires_multi_query,
            top_k=DEFAULT_TOP_K,
        )

        logger.info(
            "Query plan created | "
            "intent=%s | depth=%s | "
            "decomposition=%s | multi_query=%s",
            plan.intent.value,
            plan.search_depth.value,
            plan.requires_decomposition,
            plan.requires_multi_query,
        )

        return plan

    # ========================================================
    # Intent Detection
    # ========================================================

    def _detect_intent(
        self,
        query: str,
    ) -> QueryIntent:
        """
        Detect the primary recruiter query intent.

        This is a deterministic baseline router.

        An LLM-based router can be introduced later without
        changing the QueryPlan contract.
        """

        normalized = query.lower().strip()

        # ----------------------------------------------------
        # Unsupported / general
        # ----------------------------------------------------

        if not normalized:
            return QueryIntent.UNSUPPORTED

        general_patterns = [
            r"^(hi|hello|hey)\b",
            r"what can you do",
            r"how does kokoro work",
        ]

        if any(
            re.search(pattern, normalized)
            for pattern in general_patterns
        ):
            return QueryIntent.GENERAL

        # ----------------------------------------------------
        # JD gap analysis
        # ----------------------------------------------------

        jd_gap_patterns = [
            "gap analysis",
            "skill gap",
            "missing skills",
            "missing skill",
            "what is missing",
            "gaps against",
            "gap against",
            "does the candidate meet the jd",
            "does the candidate match the job description",
            "job description match",
        ]

        if any(
            phrase in normalized
            for phrase in jd_gap_patterns
        ):
            return QueryIntent.JD_GAP_ANALYSIS

        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        comparison_patterns = [
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bversus\b",
            r"\bvs\b",
            r"\bwhich candidate\b",
            r"\bdifference between\b",
        ]

        if any(
            re.search(pattern, normalized)
            for pattern in comparison_patterns
        ):
            return QueryIntent.COMPARISON

        # ----------------------------------------------------
        # Candidate explanation
        # ----------------------------------------------------

        explanation_patterns = [
            "why is",
            "why was",
            "why does",
            "explain this candidate",
            "explain the candidate",
            "why does this candidate",
            "tell me why this candidate",
            "how does this candidate match",
        ]

        if any(
            phrase in normalized
            for phrase in explanation_patterns
        ):
            return QueryIntent.CANDIDATE_EXPLANATION

        # ----------------------------------------------------
        # Conversational follow-up
        # ----------------------------------------------------

        follow_up_patterns = [
            r"^what about\b",
            r"^how about\b",
            r"^and\b",
            r"^also\b",
            r"^what if\b",
            r"^among them\b",
            r"^which one\b",
        ]

        if any(
            re.search(pattern, normalized)
            for pattern in follow_up_patterns
        ):
            return QueryIntent.CONVERSATIONAL_FOLLOW_UP

        # ----------------------------------------------------
        # Filtering
        # ----------------------------------------------------

        filter_patterns = [
            "with at least",
            "more than",
            "less than",
            "minimum",
            "maximum",
            "years of experience",
            "experience in",
            "located in",
            "location",
            "remote",
            "hybrid",
            "full time",
            "full-time",
            "available",
            "notice period",
            "certified",
            "certification",
        ]

        if any(
            phrase in normalized
            for phrase in filter_patterns
        ):
            return QueryIntent.FILTERING

        # ----------------------------------------------------
        # Multi-part query
        # ----------------------------------------------------

        if self._contains_multiple_requirements(
            normalized
        ):
            return QueryIntent.MULTI_PART

        # ----------------------------------------------------
        # Candidate search
        # ----------------------------------------------------

        candidate_search_patterns = [
            "find candidates",
            "find candidate",
            "show candidates",
            "show candidate",
            "search candidates",
            "search candidate",
            "candidates with",
            "candidate with",
            "looking for candidates",
            "people with",
            "profiles with",
        ]

        if any(
            phrase in normalized
            for phrase in candidate_search_patterns
        ):
            return QueryIntent.CANDIDATE_SEARCH

        # Default recruitment-oriented queries to candidate search.
        recruitment_terms = [
            "candidate",
            "resume",
            "resume",
            "profile",
            "developer",
            "engineer",
            "data scientist",
            "machine learning",
            "generative ai",
            "genai",
            "rag",
            "llm",
            "python",
            "sql",
        ]

        if any(
            term in normalized
            for term in recruitment_terms
        ):
            return QueryIntent.CANDIDATE_SEARCH

        return QueryIntent.UNSUPPORTED

    # ========================================================
    # Search Complexity
    # ========================================================

    @staticmethod
    def _contains_multiple_requirements(
        query: str,
    ) -> bool:
        """
        Identify queries containing multiple independent
        requirements.

        Example:

            "Find candidates with Python, RAG, GCP,
             and 5 years of experience"

        is more complex than:

            "Find Python candidates"
        """

        requirement_terms = [
            "and",
            "also",
            "plus",
            "along with",
            "as well as",
            "with experience in",
        ]

        matches = sum(
            query.count(term)
            for term in requirement_terms
        )

        # Multiple technical constraints are also a signal.
        technical_terms = [
            "python",
            "sql",
            "java",
            "aws",
            "gcp",
            "azure",
            "rag",
            "llm",
            "langchain",
            "langgraph",
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "pandas",
        ]

        technical_count = sum(
            1
            for term in technical_terms
            if term in query
        )

        return (
            matches >= 2
            or technical_count >= 4
        )

    @staticmethod
    def _requires_decomposition(
        query: str,
        intent: QueryIntent,
    ) -> bool:
        """
        Determine whether the query should be decomposed
        into multiple retrieval operations.
        """

        if intent in {
            QueryIntent.COMPARISON,
            QueryIntent.JD_GAP_ANALYSIS,
            QueryIntent.MULTI_PART,
        }:
            return True

        normalized = query.lower()

        # Explicit multi-part language.
        decomposition_terms = [
            "and compare",
            "compare and",
            "as well as",
            "along with",
            "both",
            "and also",
            "plus",
        ]

        if any(
            term in normalized
            for term in decomposition_terms
        ):
            return True

        # Several independent constraints.
        return (
            normalized.count(" and ")
            >= 3
        )

    @staticmethod
    def _requires_multi_query(
        query: str,
        intent: QueryIntent,
    ) -> bool:
        """
        Determine whether semantic query expansion may help.
        """

        if intent in {
            QueryIntent.CANDIDATE_SEARCH,
            QueryIntent.FILTERING,
            QueryIntent.CANDIDATE_EXPLANATION,
            QueryIntent.JD_GAP_ANALYSIS,
        }:
            return True

        normalized = query.lower()

        # Longer natural-language recruitment questions
        # can benefit from alternate formulations.
        words = normalized.split()

        return len(words) >= 18

    # ========================================================
    # Retrieval
    # ========================================================

    def search(
        self,
        request: QueryRequest,
        query_vector: list[float],
        candidate_filter: dict[str, Any] | None = None,
    ) -> tuple[QueryPlan, list[RetrievedChunk]]:
        """
        Execute Kokoro retrieval.

        Parameters
        ----------
        request:
            Validated recruiter query.

        query_vector:
            Embedding of the recruiter query.

        candidate_filter:
            Optional metadata filter.

        Returns
        -------
        tuple[QueryPlan, list[RetrievedChunk]]
            Query plan and final hybrid Top-K results.
        """

        plan = self.create_query_plan(
            request
        )

        # ----------------------------------------------------
        # General / unsupported query
        # ----------------------------------------------------

        if plan.intent in {
            QueryIntent.GENERAL,
            QueryIntent.UNSUPPORTED,
        }:
            return plan, []

        # ----------------------------------------------------
        # Current baseline:
        #
        # search_router receives the query vector from the
        # embedding layer.
        #
        # Multi-query/decomposition can be added around this
        # method without changing Pinecone/BM25 contracts.
        # ----------------------------------------------------

        semantic_results = (
            self.vector_store.search(
                query_vector=query_vector,
                top_k=plan.top_k,
                filter=candidate_filter,
            )
        )

        keyword_results = (
            self.hybrid_indexer.search(
                query=request.query,
                top_k=plan.top_k,
                candidate_filter=candidate_filter,
            )
        )

        results = self._fuse_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=plan.top_k,
        )

        return plan, results

    # ========================================================
    # Hybrid Fusion
    # ========================================================

    def _fuse_results(
        self,
        semantic_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Fuse semantic and keyword retrieval results.

        Process:

            Pinecone scores
                 ↓
            normalize

            BM25 scores
                 ↓
            normalize

            weighted fusion
                 ↓
            sort
                 ↓
            Top-K
        """

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

        # ----------------------------------------------------
        # Normalize scores independently
        # ----------------------------------------------------

        semantic_results = normalize_scores(
            semantic_results,
            score_type="semantic",
        )

        keyword_results = normalize_scores(
            keyword_results,
            score_type="keyword",
        )

        # ----------------------------------------------------
        # Merge by chunk ID
        # ----------------------------------------------------

        merged: dict[str, RetrievedChunk] = {}

        for result in semantic_results:

            merged[result.chunk_id] = result

        for result in keyword_results:

            if result.chunk_id not in merged:

                merged[result.chunk_id] = result

            else:

                existing = merged[
                    result.chunk_id
                ]

                if (
                    existing.text
                    != result.text
                ):
                    logger.warning(
                        "Text mismatch for same chunk_id "
                        "during hybrid fusion | chunk_id=%s",
                        result.chunk_id,
                    )

                # Preserve missing metadata.
                existing.metadata.update(
                    result.metadata
                )

                if (
                    existing.candidate_id
                    is None
                ):
                    existing.candidate_id = (
                        result.candidate_id
                    )

                if (
                    existing.candidate_name
                    is None
                ):
                    existing.candidate_name = (
                        result.candidate_name
                    )

                if (
                    existing.keyword_score
                    is None
                ):
                    existing.keyword_score = (
                        result.keyword_score
                    )

        # ----------------------------------------------------
        # Calculate hybrid score
        # ----------------------------------------------------

        for result in merged.values():

            semantic_score = (
                result.semantic_score
                if result.semantic_score is not None
                else 0.0
            )

            keyword_score = (
                result.keyword_score
                if result.keyword_score is not None
                else 0.0
            )

            result.hybrid_score = (
                self.semantic_weight
                * semantic_score
                + self.keyword_weight
                * keyword_score
            )

        # ----------------------------------------------------
        # Sort by hybrid score
        # ----------------------------------------------------

        ranked_results = sorted(
            merged.values(),
            key=lambda result: (
                result.hybrid_score
                if result.hybrid_score is not None
                else 0.0
            ),
            reverse=True,
        )

        final_results = ranked_results[
            :top_k
        ]

        logger.info(
            "Hybrid retrieval completed | "
            "semantic=%d | keyword=%d | "
            "merged=%d | top_k=%d",
            len(semantic_results),
            len(keyword_results),
            len(merged),
            len(final_results),
        )

        return final_results