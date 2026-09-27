"""
utils/multi_query.py

Multi-query expansion utilities for Kokoro.

Purpose:
    Convert one user query into multiple semantically related
    search queries to improve retrieval coverage.

Example:

    Original query:
        "Find candidates with Python and GenAI experience"

    Expanded queries:
        1. "candidates with Python experience"
        2. "candidates with Generative AI experience"
        3. "candidates with Python and LLM experience"
        4. "candidates with RAG and GenAI experience"

Important:
    Multi-query expansion is different from question decomposition.

    Multi-query expansion:
        One search intent
            ↓
        Multiple alternative search formulations

    Question decomposition:
        Complex question
            ↓
        Multiple independent sub-questions

This module only handles multi-query expansion.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from utils.config import settings
from utils.logger import get_logger


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MAX_QUERIES = 3

MIN_QUERY_LENGTH = 3


# ============================================================
# QUERY CLEANING
# ============================================================


def clean_query(
    query: str,
) -> str:
    """
    Clean a query before expansion.

    Responsibilities:
        - Remove leading/trailing whitespace
        - Normalize repeated whitespace
        - Remove empty lines
    """

    if not query:
        return ""

    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query,
    )

    return query


# ============================================================
# QUERY VALIDATION
# ============================================================


def validate_query(
    query: str,
) -> bool:
    """
    Validate whether a query is suitable for expansion.
    """

    query = clean_query(query)

    if len(query) < MIN_QUERY_LENGTH:
        return False

    return True


# ============================================================
# DUPLICATE REMOVAL
# ============================================================


def remove_duplicate_queries(
    queries: list[str],
) -> list[str]:
    """
    Remove duplicate queries while preserving order.

    Duplicate detection is case-insensitive.
    """

    unique_queries: list[str] = []

    seen: set[str] = set()

    for query in queries:

        cleaned = clean_query(query)

        if not cleaned:
            continue

        normalized = cleaned.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        unique_queries.append(cleaned)

    return unique_queries


# ============================================================
# QUERY LIMIT
# ============================================================


def limit_queries(
    queries: list[str],
    max_queries: int,
) -> list[str]:
    """
    Limit the number of expanded queries.

    The original query should normally be included separately
    by the caller.
    """

    if max_queries <= 0:
        return []

    return queries[:max_queries]


# ============================================================
# PROMPT BUILDER
# ============================================================


def build_multi_query_prompt(
    query: str,
    max_queries: int,
) -> str:
    """
    Build the prompt used for multi-query expansion.

    The LLM is instructed to generate alternative search
    formulations while preserving the original search intent.
    """

    return f"""
You are a search-query expansion component for a recruitment
RAG system.

Your task is to generate alternative search queries for the
user's original query.

Rules:
1. Preserve the original search intent.
2. Do not answer the user's question.
3. Do not invent candidate information.
4. Do not introduce unrelated skills, technologies,
   qualifications, companies, or experience.
5. Use different wording where useful.
6. Keep each query concise and suitable for semantic and
   lexical retrieval.
7. Return only a JSON array of strings.
8. Generate at most {max_queries} alternative queries.
9. Do not include numbering or explanations.

Original query:
{query}

Return JSON only.
""".strip()


# ============================================================
# RESPONSE EXTRACTION
# ============================================================


def _extract_json_array(
    response: str,
) -> list[Any]:
    """
    Extract a JSON array from an LLM response.

    Handles cases where the model accidentally wraps the
    JSON in a markdown code block.
    """

    if not response:
        return []

    response = response.strip()

    # --------------------------------------------------------
    # Remove markdown code fences.
    # --------------------------------------------------------

    response = re.sub(
        r"^```(?:json)?\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )

    response = re.sub(
        r"\s*```$",
        "",
        response,
    )

    response = response.strip()

    # --------------------------------------------------------
    # Direct JSON parsing.
    # --------------------------------------------------------

    try:

        parsed = json.loads(response)

        if isinstance(parsed, list):
            return parsed

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Try extracting the first JSON array.
    # --------------------------------------------------------

    start = response.find("[")
    end = response.rfind("]")

    if start == -1 or end == -1:
        return []

    candidate = response[
        start : end + 1
    ]

    try:

        parsed = json.loads(candidate)

        if isinstance(parsed, list):
            return parsed

    except json.JSONDecodeError:

        logger.debug(
            "Unable to parse multi-query response as JSON."
        )

    return []


# ============================================================
# RESPONSE NORMALIZATION
# ============================================================


def normalize_expanded_queries(
    queries: list[Any],
    original_query: str,
    max_queries: int,
) -> list[str]:
    """
    Normalize and validate LLM-generated queries.

    Invalid values are ignored.

    The original query is excluded because it is normally
    already available to the retrieval pipeline.
    """

    original_normalized = clean_query(
        original_query
    ).lower()

    normalized_queries: list[str] = []

    for query in queries:

        # ----------------------------------------------------
        # Only accept string queries.
        # ----------------------------------------------------

        if not isinstance(query, str):
            continue

        query = clean_query(query)

        if not validate_query(query):
            continue

        # ----------------------------------------------------
        # Do not duplicate the original query.
        # ----------------------------------------------------

        if (
            query.lower()
            == original_normalized
        ):
            continue

        normalized_queries.append(query)

    # --------------------------------------------------------
    # Remove duplicates.
    # --------------------------------------------------------

    normalized_queries = (
        remove_duplicate_queries(
            normalized_queries
        )
    )

    # --------------------------------------------------------
    # Apply maximum query limit.
    # --------------------------------------------------------

    return limit_queries(
        normalized_queries,
        max_queries,
    )


# ============================================================
# LLM INVOCATION
# ============================================================


def _generate_queries_with_llm(
    prompt: str,
) -> str:
    """
    Generate expanded queries using the configured LLM.

    The actual provider integration is intentionally kept
    behind this function.

    This keeps the multi-query module loosely coupled and
    makes it easier to change the LLM implementation later.

    Returns:
        Raw LLM response as a string.

    Raises:
        RuntimeError:
            When the configured LLM client cannot be used.
    """

    try:

        from google import genai

    except ImportError as exc:

        raise RuntimeError(
            "Google GenAI SDK is not installed. "
            "Install the required Gemini dependency."
        ) from exc

    try:

        api_key = (
            settings.GOOGLE_API_KEY
            .get_secret_value()
        )

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
        )

        text = getattr(
            response,
            "text",
            None,
        )

        if not text:
            raise RuntimeError(
                "LLM returned an empty response."
            )

        return text

    except Exception as exc:

        logger.exception(
            "Multi-query LLM generation failed."
        )

        raise RuntimeError(
            "Failed to generate expanded queries."
        ) from exc


# ============================================================
# MAIN EXPANSION FUNCTION
# ============================================================


def expand_query(
    query: str,
    max_queries: Optional[int] = None,
) -> list[str]:
    """
    Expand one user query into multiple search queries.

    Args:
        query:
            Original user query.

        max_queries:
            Maximum number of alternative queries.

            If not supplied, DEFAULT_MAX_QUERIES is used.

    Returns:
        List of expanded queries.

    Example:

        query = "Python GenAI candidates"

        expanded = expand_query(query)

        [
            "candidates with Python experience",
            "candidates with Generative AI experience",
            "candidates with Python and LLM experience"
        ]

    Notes:
        The original query is NOT included in the returned
        list. The retrieval layer should search the original
        query separately.
    """

    query = clean_query(query)

    if not validate_query(query):

        logger.warning(
            "Query is too short or empty for multi-query "
            "expansion."
        )

        return []

    if max_queries is None:
        max_queries = DEFAULT_MAX_QUERIES

    if max_queries <= 0:
        return []

    logger.debug(
        "Starting multi-query expansion."
    )

    prompt = build_multi_query_prompt(
        query=query,
        max_queries=max_queries,
    )

    # --------------------------------------------------------
    # Log prompt only through DEBUG.
    # --------------------------------------------------------

    logger.debug(
        "Multi-query prompt:\n%s",
        prompt,
    )

    raw_response = (
        _generate_queries_with_llm(
            prompt
        )
    )

    # --------------------------------------------------------
    # Log complete LLM response for debugging.
    # --------------------------------------------------------

    logger.debug(
        "Multi-query LLM response:\n%s",
        raw_response,
    )

    parsed_queries = (
        _extract_json_array(
            raw_response
        )
    )

    expanded_queries = (
        normalize_expanded_queries(
            queries=parsed_queries,
            original_query=query,
            max_queries=max_queries,
        )
    )

    logger.info(
        "Multi-query expansion completed | "
        "generated=%d",
        len(expanded_queries),
    )

    logger.debug(
        "Expanded queries:\n%s",
        expanded_queries,
    )

    return expanded_queries


# ============================================================
# BATCH EXPANSION
# ============================================================


def expand_queries(
    queries: list[str],
    max_queries_per_input: Optional[int] = None,
) -> dict[str, list[str]]:
    """
    Expand multiple queries independently.

    Each input query is processed separately.

    Args:
        queries:
            List of original queries.

        max_queries_per_input:
            Maximum expansions per query.

    Returns:
        Dictionary mapping each original query to its
        expanded queries.
    """

    results: dict[str, list[str]] = {}

    for query in queries:

        cleaned_query = clean_query(
            query
        )

        if not validate_query(
            cleaned_query
        ):
            results[query] = []
            continue

        try:

            results[cleaned_query] = (
                expand_query(
                    cleaned_query,
                    max_queries=max_queries_per_input,
                )
            )

        except Exception:

            logger.exception(
                "Multi-query expansion failed "
                "for one query."
            )

            # One query failure should not stop the
            # remaining queries.
            results[cleaned_query] = []

    return results


# ============================================================
# FALLBACK
# ============================================================


def get_queries_for_retrieval(
    query: str,
    enable_multi_query: Optional[bool] = None,
    max_queries: Optional[int] = None,
) -> list[str]:
    """
    Return the queries that should be passed to the retrieval
    layer.

    The original query is ALWAYS returned.

    If multi-query expansion is enabled, expanded queries
    are appended.

    Example:

        Original:
            "Python GenAI candidates"

        Result:
            [
                "Python GenAI candidates",
                "candidates with Python experience",
                "candidates with Generative AI experience"
            ]

    This function is the main integration point for the
    search router/retrieval layer.
    """

    original_query = clean_query(
        query
    )

    if not validate_query(
        original_query
    ):
        return []

    if enable_multi_query is None:
        enable_multi_query = (
            settings.ENABLE_MULTI_QUERY
        )

    # --------------------------------------------------------
    # Multi-query disabled.
    # --------------------------------------------------------

    if not enable_multi_query:

        logger.debug(
            "Multi-query expansion disabled."
        )

        return [original_query]

    # --------------------------------------------------------
    # Generate expanded queries.
    # --------------------------------------------------------

    try:

        expanded_queries = (
            expand_query(
                query=original_query,
                max_queries=max_queries,
            )
        )

    except Exception:

        logger.exception(
            "Multi-query expansion failed. "
            "Falling back to original query."
        )

        return [original_query]

    # --------------------------------------------------------
    # Original query must always remain available.
    # --------------------------------------------------------

    all_queries = [
        original_query,
        *expanded_queries,
    ]

    return remove_duplicate_queries(
        all_queries
    )


# ============================================================
# DEFAULT EXPORTS
# ============================================================


__all__ = [
    "clean_query",
    "validate_query",
    "remove_duplicate_queries",
    "limit_queries",
    "build_multi_query_prompt",
    "normalize_expanded_queries",
    "expand_query",
    "expand_queries",
    "get_queries_for_retrieval",
]