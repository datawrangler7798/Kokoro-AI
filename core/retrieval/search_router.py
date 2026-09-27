"""
Kokoro AI - Search Router.

Responsibilities
----------------
- Analyze recruiter queries.
- Identify the primary retrieval intent.
- Decide shallow vs deep search.
- Detect constraints and multi-part queries.
- Decompose complex queries into independent sub-queries.
- Trigger multi-query expansion when appropriate.
- Build an explicit QueryPlan for downstream retrieval.

Important
---------
Question decomposition and multi-query expansion are different:

Question decomposition:
    One complex question
            ↓
    Multiple independent questions

Multi-query expansion:
    One search intent
            ↓
    Multiple alternative formulations

This module handles routing/decomposition.
utils.multi_query handles query expansion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from utils.config import get_settings
from utils.logger import get_logger
from utils.utils import get_llm_rate_limiter
from utils.multi_query import (
    get_queries_for_retrieval,
)
from utils.schemas import (
    QueryIntent,
    QueryPlan,
    SearchDepth,
)


logger = get_logger(__name__)
settings = get_settings()


# ============================================================
# Protocols
# ============================================================


class QueryLLM(Protocol):
    """
    Minimal interface required for LLM-based query analysis.

    Keeping this as a protocol allows Gemini to be replaced
    or mocked during testing.
    """

    def generate(
        self,
        prompt: str,
    ) -> str:
        ...


# ============================================================
# Internal Route Decision
# ============================================================


@dataclass(slots=True)
class RouteDecision:
    """
    Internal representation of the router decision.
    """

    intent: QueryIntent

    search_depth: SearchDepth

    requires_decomposition: bool

    requires_multi_query: bool

    constraints: list[str] = field(
        default_factory=list
    )

    sub_queries: list[str] = field(
        default_factory=list
    )

    reasoning: str = ""


# ============================================================
# Gemini Query Analyzer
# ============================================================


class GeminiQueryAnalyzer:
    """
    Gemini-backed query analyzer.

    The LLM is used for semantic intent classification and
    decomposition only.

    It does NOT perform retrieval.
    It does NOT rank candidates.
    It does NOT generate the final recruiter answer.
    """

    SYSTEM_INSTRUCTION = """
You are the query-analysis component of a recruitment RAG system.

Your task is to analyze a recruiter query and return structured
routing information.

Possible intents:
- search
- filter
- comparison
- jd_gap_analysis
- skill_match
- experience_match
- candidate_details
- general

Possible search depths:
- shallow
- deep

Rules:
1. Use shallow search for simple, single-intent questions.
2. Use deep search for complex or multi-part questions.
3. Do not answer the recruiter query.
4. Do not invent candidate information.
5. Do not invent skills, qualifications, companies, or experience.
6. Identify explicit constraints.
7. Comparison questions normally require deep search.
8. JD gap analysis normally requires deep search.
9. Multiple independent retrieval operations require deep search.
10. Decomposition means splitting a complex question into
    independent retrieval questions.
11. Multi-query expansion is different from decomposition.
12. Return valid JSON only.
"""

    def __init__(
        self,
        *,
        generator: Callable[
            [str],
            str,
        ] | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:

        self.model_name = (
            model_name
            or settings.LLM_MODEL
        )

        configured_key = (
            settings.GOOGLE_API_KEY
        )

        if hasattr(
            configured_key,
            "get_secret_value",
        ):
            configured_key = (
                configured_key.get_secret_value()
            )

        self.api_key = (
            api_key
            or configured_key
        )

        self._generator = generator

    # --------------------------------------------------------
    # Gemini Client
    # --------------------------------------------------------

    def _generate_with_gemini(
        self,
        prompt: str,
    ) -> str:

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for query analysis."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for query analysis."
            ) from exc

        client = genai.Client(
            api_key=self.api_key
        )

        get_llm_rate_limiter().acquire()
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        text = getattr(
            response,
            "text",
            None,
        )

        if not text:
            raise RuntimeError(
                "Query analyzer returned an empty response."
            )

        return text

    def _generate(
        self,
        prompt: str,
    ) -> str:

        if self._generator is not None:
            return self._generator(
                prompt
            )

        return self._generate_with_gemini(
            prompt
        )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        *,
        conversation_context: str | None = None,
    ) -> str:

        context = (
            conversation_context
            or "No previous conversation context."
        )

        return f"""
{self.SYSTEM_INSTRUCTION}

Previous conversation context:
{context}

Recruiter query:
{query}

Return exactly this JSON structure:

{{
  "intent": "search",
  "search_depth": "shallow",
  "requires_decomposition": false,
  "requires_multi_query": false,
  "constraints": [],
  "sub_queries": [],
  "reasoning": "brief routing explanation"
}}

If the query is complex, populate sub_queries with the
independent retrieval questions.

Do not include markdown.
"""

    # --------------------------------------------------------
    # JSON Parsing
    # --------------------------------------------------------

    @staticmethod
    def _extract_json(
        text: str,
    ) -> dict[str, Any]:

        cleaned = (
            text
            .strip()
        )

        if cleaned.startswith(
            "```"
        ):
            cleaned = re.sub(
                r"^```(?:json)?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            cleaned = re.sub(
                r"```$",
                "",
                cleaned,
            ).strip()

        try:
            payload = json.loads(
                cleaned
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "Analyzer JSON must be an object."
                )

            return payload

        except json.JSONDecodeError:

            match = re.search(
                r"\{.*\}",
                cleaned,
                flags=re.DOTALL,
            )

            if not match:
                raise ValueError(
                    "Could not parse query analyzer JSON."
                )

            payload = json.loads(
                match.group(0)
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "Analyzer JSON must be an object."
                )

            return payload

    # --------------------------------------------------------
    # Enum Parsing
    # --------------------------------------------------------

    @staticmethod
    def _parse_intent(
        value: Any,
    ) -> QueryIntent:

        if isinstance(
            value,
            QueryIntent,
        ):
            return value

        normalized = str(
            value or ""
        ).strip().lower()

        aliases = {
            "candidate_search": (
                QueryIntent.SEARCH
            ),
            "candidate search": (
                QueryIntent.SEARCH
            ),
            "search": QueryIntent.SEARCH,
            "filter": QueryIntent.FILTER,
            "comparison": (
                QueryIntent.COMPARISON
            ),
            "compare": (
                QueryIntent.COMPARISON
            ),
            "jd_gap_analysis": (
                QueryIntent.JD_GAP_ANALYSIS
            ),
            "jd gap analysis": (
                QueryIntent.JD_GAP_ANALYSIS
            ),
            "skill_match": (
                QueryIntent.SKILL_MATCH
            ),
            "skill match": (
                QueryIntent.SKILL_MATCH
            ),
            "experience_match": (
                QueryIntent.EXPERIENCE_MATCH
            ),
            "experience match": (
                QueryIntent.EXPERIENCE_MATCH
            ),
            "candidate_details": (
                QueryIntent.CANDIDATE_DETAILS
            ),
            "candidate details": (
                QueryIntent.CANDIDATE_DETAILS
            ),
            "general": QueryIntent.GENERAL,
        }

        return aliases.get(
            normalized,
            QueryIntent.GENERAL,
        )

    @staticmethod
    def _parse_depth(
        value: Any,
    ) -> SearchDepth:

        if isinstance(
            value,
            SearchDepth,
        ):
            return value

        normalized = str(
            value or ""
        ).strip().lower()

        if normalized == "deep":
            return SearchDepth.DEEP

        return SearchDepth.SHALLOW

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        query: str,
        *,
        conversation_context: str | None = None,
    ) -> RouteDecision:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        prompt = self._build_prompt(
            query.strip(),
            conversation_context=conversation_context,
        )

        logger.debug(
            "Query analyzer prompt:\n%s",
            prompt,
        )

        raw_response = self._generate(
            prompt
        )

        logger.debug(
            "Query analyzer response:\n%s",
            raw_response,
        )

        try:
            payload = self._extract_json(
                raw_response
            )

        except Exception:

            logger.exception(
                "Failed to parse query analyzer response."
            )

            # Safe fallback:
            # simple search, no decomposition.
            return RouteDecision(
                intent=QueryIntent.SEARCH,
                search_depth=SearchDepth.SHALLOW,
                requires_decomposition=False,
                requires_multi_query=False,
                reasoning=(
                    "Analyzer response could not be parsed; "
                    "using safe shallow-search fallback."
                ),
            )

        intent = self._parse_intent(
            payload.get(
                "intent"
            )
        )

        depth = self._parse_depth(
            payload.get(
                "search_depth"
            )
        )

        sub_queries = (
            payload.get(
                "sub_queries",
                [],
            )
            or []
        )

        sub_queries = [
            str(query).strip()
            for query in sub_queries
            if str(query).strip()
        ]

        constraints = (
            payload.get(
                "constraints",
                [],
            )
            or []
        )

        constraints = [
            str(value).strip()
            for value in constraints
            if str(value).strip()
        ]

        requires_decomposition = bool(
            payload.get(
                "requires_decomposition",
                False,
            )
        )

        requires_multi_query = bool(
            payload.get(
                "requires_multi_query",
                False,
            )
        )

        reasoning = str(
            payload.get(
                "reasoning",
                "",
            )
        )

        # Deterministic safety rules override an
        # under-classified LLM response.

        if intent in (
            QueryIntent.COMPARISON,
            QueryIntent.JD_GAP_ANALYSIS,
        ):
            depth = SearchDepth.DEEP

        if len(sub_queries) > 1:
            depth = SearchDepth.DEEP
            requires_decomposition = True

        return RouteDecision(
            intent=intent,
            search_depth=depth,
            requires_decomposition=(
                requires_decomposition
            ),
            requires_multi_query=(
                requires_multi_query
            ),
            constraints=constraints,
            sub_queries=sub_queries,
            reasoning=reasoning,
        )


# ============================================================
# Rule-Based Query Signals
# ============================================================


class QuerySignalDetector:
    """
    Deterministic signal detector.

    This supplements LLM routing with explicit rules for
    important recruiter query patterns.

    The detector does not replace the LLM analyzer.
    """

    COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference between\b",
    )

    JD_GAP_PATTERNS = (
        r"\bgap\b",
        r"\bmissing\b",
        r"\bmissing skills\b",
        r"\bagainst this jd\b",
        r"\bmatch against\b",
    )

    MULTI_PART_PATTERNS = (
        r"\band then\b",
        r"\bthen\b",
        r"\balso\b",
        r"\bin addition\b",
        r"\bidentify .* and\b",
    )

    FILTER_PATTERNS = (
        r"\bat least\b",
        r"\bat most\b",
        r"\bminimum\b",
        r"\bmaximum\b",
        r"\bmore than\b",
        r"\bless than\b",
        r"\bwith\b",
        r"\bwithout\b",
        r"\bexperience\b",
        r"\blocation\b",
    )

    @staticmethod
    def _matches(
        query: str,
        patterns: Sequence[str],
    ) -> bool:

        normalized = (
            query
            .strip()
            .lower()
        )

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in patterns
        )

    def is_comparison(
        self,
        query: str,
    ) -> bool:

        return self._matches(
            query,
            self.COMPARISON_PATTERNS,
        )

    def is_jd_gap_analysis(
        self,
        query: str,
    ) -> bool:

        return self._matches(
            query,
            self.JD_GAP_PATTERNS,
        )

    def is_multi_part(
        self,
        query: str,
    ) -> bool:

        return self._matches(
            query,
            self.MULTI_PART_PATTERNS,
        )

    def has_filters(
        self,
        query: str,
    ) -> bool:

        return self._matches(
            query,
            self.FILTER_PATTERNS,
        )

    def complexity_score(
        self,
        query: str,
    ) -> int:
        """
        Calculate a lightweight deterministic complexity
        signal.

        This is only a supporting signal. The router does not
        make the decision from query length alone.
        """

        score = 0

        if self.is_comparison(
            query
        ):
            score += 2

        if self.is_jd_gap_analysis(
            query
        ):
            score += 2

        if self.is_multi_part(
            query
        ):
            score += 2

        if self.has_filters(
            query
        ):
            score += 1

        # Multiple conjunctions often indicate multiple
        # requirements.
        conjunction_count = len(
            re.findall(
                r"\b(?:and|then|also)\b",
                query.lower(),
            )
        )

        if conjunction_count >= 2:
            score += 1

        return score


# ============================================================
# Search Router
# ============================================================


class SearchRouter:
    """
    Main Kokoro query router.

    Responsibilities:
        1. Analyze intent.
        2. Determine shallow/deep search.
        3. Decompose complex queries.
        4. Trigger multi-query expansion where appropriate.
        5. Produce an explicit QueryPlan.

    It does not execute retrieval.
    """

    def __init__(
        self,
        *,
        analyzer: QueryLLM | None = None,
        signal_detector: QuerySignalDetector | None = None,
        max_sub_queries: int | None = None,
        max_expanded_queries: int | None = None,
    ) -> None:

        self.analyzer = (
            analyzer
            or GeminiQueryAnalyzer()
        )

        self.signal_detector = (
            signal_detector
            or QuerySignalDetector()
        )

        self.max_sub_queries = (
            max_sub_queries
            if max_sub_queries is not None
            else getattr(
                settings,
                "MAX_DECOMPOSITION_QUERIES",
                4,
            )
        )

        self.max_expanded_queries = (
            max_expanded_queries
            if max_expanded_queries is not None
            else getattr(
                settings,
                "MAX_MULTI_QUERY_VARIANTS",
                3,
            )
        )

    # --------------------------------------------------------
    # Query Cleaning
    # --------------------------------------------------------

    @staticmethod
    def _clean_query(
        query: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            query.strip(),
        )

    # --------------------------------------------------------
    # Deduplication
    # --------------------------------------------------------

    @staticmethod
    def _deduplicate(
        queries: Sequence[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for query in queries:

            cleaned = re.sub(
                r"\s+",
                " ",
                str(query).strip(),
            )

            if not cleaned:
                continue

            key = cleaned.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(
                cleaned
            )

        return result

    # --------------------------------------------------------
    # Deterministic Overrides
    # --------------------------------------------------------

    def _apply_signal_overrides(
        self,
        query: str,
        decision: RouteDecision,
    ) -> RouteDecision:

        if self.signal_detector.is_comparison(
            query
        ):
            decision.intent = (
                QueryIntent.COMPARISON
            )
            decision.search_depth = (
                SearchDepth.DEEP
            )

        elif self.signal_detector.is_jd_gap_analysis(
            query
        ):
            decision.intent = (
                QueryIntent.JD_GAP_ANALYSIS
            )
            decision.search_depth = (
                SearchDepth.DEEP
            )

        complexity = (
            self.signal_detector.complexity_score(
                query
            )
        )

        if complexity >= 3:
            decision.search_depth = (
                SearchDepth.DEEP
            )

        if decision.search_depth == SearchDepth.DEEP:
            decision.requires_multi_query = (
                decision.requires_multi_query
                or getattr(
                    settings,
                    "ENABLE_MULTI_QUERY",
                    True,
                )
            )

        return decision

    # --------------------------------------------------------
    # Fallback Decomposition
    # --------------------------------------------------------

    def _rule_based_decomposition(
        self,
        query: str,
    ) -> list[str]:
        """
        Conservative decomposition fallback.

        This is intentionally simple.

        It only separates obvious independent clauses and
        does not attempt to understand candidate information.
        """

        if not query:
            return []

        # First try explicit semicolon-separated parts.
        parts = re.split(
            r"\s*;\s*",
            query,
        )

        if len(parts) == 1:

            # Split only on strong multi-part patterns.
            parts = re.split(
                r"\s+(?:and then|then identify|then show|also identify)\s+",
                query,
                flags=re.IGNORECASE,
            )

        parts = self._deduplicate(
            parts
        )

        if len(parts) <= 1:
            return []

        return parts[
            : self.max_sub_queries
        ]

    # --------------------------------------------------------
    # Multi Query
    # --------------------------------------------------------

    def _expand_queries(
        self,
        queries: Sequence[str],
    ) -> list[str]:

        all_queries: list[str] = []

        for query in queries:

            expanded = (
                get_queries_for_retrieval(
                    query=query,
                    enable_multi_query=True,
                    max_queries=(
                        self.max_expanded_queries
                    ),
                )
            )

            all_queries.extend(
                expanded
            )

        return self._deduplicate(
            all_queries
        )

    # --------------------------------------------------------
    # QueryPlan Construction
    # --------------------------------------------------------

    def _build_query_plan(
        self,
        query: str,
        decision: RouteDecision,
        expanded_queries: Sequence[str],
        *,
        conversation_context: str | None = None,
        filters: Any | None = None,
        top_k: int | None = None,
    ) -> Any:
        """
        Construct the shared QueryPlan schema.

        Pydantic v2 is preferred. A dictionary fallback is kept
        so routing remains usable if the schema evolves before
        this module is updated.
        """

        payload: dict[str, Any] = {
            "query": query,
            "intent": decision.intent,
            "search_depth": decision.search_depth,
            "requires_decomposition": (
                decision.requires_decomposition
            ),
            "requires_multi_query": (
                decision.requires_multi_query
            ),
            "sub_queries": list(
                decision.sub_queries
            ),
            "expanded_queries": list(
                expanded_queries
            ),
            "top_k": (
                top_k
                if top_k is not None
                else getattr(
                    settings,
                    "HYBRID_TOP_K",
                    settings.RETRIEVAL_TOP_K,
                )
            ),
            "filters": filters,
            "reasoning": decision.reasoning,
        }

        if hasattr(
            QueryPlan,
            "model_validate",
        ):
            try:
                return QueryPlan.model_validate(
                    payload
                )
            except Exception:
                logger.exception(
                    "QueryPlan validation failed."
                )

        # Compatibility fallback.
        return payload

    # --------------------------------------------------------
    # Public Route
    # --------------------------------------------------------

    def route(
        self,
        query: str,
        *,
        conversation_context: str | None = None,
        filters: Any | None = None,
        top_k: int | None = None,
    ) -> Any:
        """
        Analyze and route a recruiter query.

        Shallow:
            query
              ↓
            one retrieval query

        Deep:
            query
              ↓
            decomposition
              ↓
            optional multi-query expansion
              ↓
            retrieval queries
        """

        query = self._clean_query(
            query
        )

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        logger.info(
            "Routing recruiter query."
        )

        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        decision = self.analyzer.analyze(
            query,
            conversation_context=(
                conversation_context
            ),
        )

        decision = (
            self._apply_signal_overrides(
                query,
                decision,
            )
        )

        # ----------------------------------------------------
        # Shallow search
        # ----------------------------------------------------

        if (
            decision.search_depth
            == SearchDepth.SHALLOW
        ):

            decision.requires_decomposition = (
                False
            )

            decision.requires_multi_query = (
                False
            )

            decision.sub_queries = []

            expanded_queries = [
                query
            ]

            return self._build_query_plan(
                query,
                decision,
                expanded_queries,
                conversation_context=(
                    conversation_context
                ),
                filters=filters,
                top_k=top_k,
            )

        # ----------------------------------------------------
        # Deep search
        # ----------------------------------------------------

        sub_queries = (
            self._deduplicate(
                decision.sub_queries
            )
        )

        # If the LLM did not produce usable
        # decomposition, use a conservative fallback.
        if (
            decision.requires_decomposition
            and len(sub_queries) <= 1
        ):

            fallback_queries = (
                self._rule_based_decomposition(
                    query
                )
            )

            if fallback_queries:
                sub_queries = (
                    fallback_queries
                )

        # If decomposition is not required but the
        # query is deep, use the original query as
        # the retrieval intent.
        if not sub_queries:
            sub_queries = [
                query
            ]

        sub_queries = sub_queries[
            : self.max_sub_queries
        ]

        decision.sub_queries = (
            sub_queries
        )

        # ----------------------------------------------------
        # Multi-query expansion
        # ----------------------------------------------------

        if (
            decision.requires_multi_query
            and getattr(
                settings,
                "ENABLE_MULTI_QUERY",
                True,
            )
        ):

            try:
                expanded_queries = (
                    self._expand_queries(
                        sub_queries
                    )
                )

            except Exception:

                logger.exception(
                    "Multi-query expansion failed; "
                    "falling back to sub-queries."
                )

                expanded_queries = list(
                    sub_queries
                )

        else:
            expanded_queries = list(
                sub_queries
            )

        if not expanded_queries:
            expanded_queries = list(
                sub_queries
            )

        # Always retain the original query.
        expanded_queries = (
            self._deduplicate(
                [
                    query,
                    *expanded_queries,
                ]
            )
        )

        logger.info(
            "Query routed: intent=%s depth=%s "
            "decomposition=%s sub_queries=%d "
            "retrieval_queries=%d",
            decision.intent,
            decision.search_depth,
            decision.requires_decomposition,
            len(sub_queries),
            len(expanded_queries),
        )

        return self._build_query_plan(
            query,
            decision,
            expanded_queries,
            conversation_context=(
                conversation_context
            ),
            filters=filters,
            top_k=top_k,
        )


# ============================================================
# Convenience Functions
# ============================================================


def create_search_router(
    *,
    analyzer: QueryLLM | None = None,
) -> SearchRouter:
    """Create a configured search router."""

    return SearchRouter(
        analyzer=analyzer
    )


def route_query(
    query: str,
    *,
    router: SearchRouter | None = None,
    conversation_context: str | None = None,
    filters: Any | None = None,
    top_k: int | None = None,
) -> Any:
    """
    Convenience wrapper for query routing.
    """

    active_router = (
        router
        or create_search_router()
    )

    return active_router.route(
        query,
        conversation_context=(
            conversation_context
        ),
        filters=filters,
        top_k=top_k,
    )


__all__ = [
    "QueryLLM",
    "RouteDecision",
    "GeminiQueryAnalyzer",
    "QuerySignalDetector",
    "SearchRouter",
    "create_search_router",
    "route_query",
]
