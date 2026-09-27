"""
Multi-query generation for Kokoro AI.

Question decomposition and multi-query expansion are separate
capabilities.

Multi-query expansion creates alternative search formulations
while preserving the original recruiter intent.
"""

import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import config
from utils.logger import logger
from utils.schemas import SubQuery


# ============================================================
# Prompt
# ============================================================

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You generate search-query variations for a recruitment RAG system.

Rules:

1. Preserve the original query intent.
2. Do not invent requirements.
3. Do not add skills, years of experience, companies,
   education, certifications, or technologies that were
   not present in the original query.
4. Produce semantically different search formulations.
5. Keep each query concise and useful for retrieval.
6. Return ONLY a valid JSON array of strings.
7. Do not return markdown.
8. Do not return explanations.
""",
        ),
        (
            "human",
            """
Original recruiter query:

{query}

Generate up to {num_queries} alternative search-query
formulations.
""",
        ),
    ]
)


# ============================================================
# Gemini Client
# ============================================================


def _create_llm() -> ChatGoogleGenerativeAI:
    """
    Create the Gemini chat model used for query expansion.

    The model name comes from centralized application
    configuration.
    """

    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )


# ============================================================
# Response Parsing
# ============================================================


def _parse_query_variations(
    raw_output: str,
) -> list[str]:
    """
    Parse Gemini's JSON response.

    Parameters
    ----------
    raw_output:
        Raw Gemini response.

    Returns
    -------
    list[str]
        Valid, unique query variations.
    """

    try:
        parsed: Any = json.loads(raw_output)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON for multi-query generation."
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            "Multi-query response must be a JSON array."
        )

    queries: list[str] = []

    for item in parsed:

        if not isinstance(item, str):
            continue

        query = item.strip()

        if not query:
            continue

        if query not in queries:
            queries.append(query)

    return queries


# ============================================================
# Public API
# ============================================================


def generate_multi_queries(
    query: str,
    num_queries: int = 3,
) -> list[SubQuery]:
    """
    Generate alternative search queries.

    The original query is always retained.

    Parameters
    ----------
    query:
        Original recruiter query.

    num_queries:
        Maximum number of generated variations.

    Returns
    -------
    list[SubQuery]
        Original query plus generated variations.

    Example
    -------
    Input:
        "Find candidates with Python and RAG experience"

    Possible output:
        [
            SubQuery(
                query="Find candidates with Python and RAG experience"
            ),
            SubQuery(
                query="Candidates experienced in Python and RAG"
            ),
            ...
        ]
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if num_queries < 1:
        raise ValueError(
            "num_queries must be >= 1."
        )

    # --------------------------------------------------------
    # Always preserve the original query
    # --------------------------------------------------------

    results = [
        SubQuery(
            query=query,
        )
    ]

    # --------------------------------------------------------
    # Fallback when Gemini credentials are unavailable
    # --------------------------------------------------------

    if not config.GOOGLE_API_KEY.strip():

        logger.warning(
            "GOOGLE_API_KEY is not configured. "
            "Returning original query only."
        )

        return results

    # --------------------------------------------------------
    # Create Gemini client
    # --------------------------------------------------------

    llm = _create_llm()

    # --------------------------------------------------------
    # Build LCEL chain
    # --------------------------------------------------------

    chain = (
        MULTI_QUERY_PROMPT
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------------
    # Execute Gemini
    # --------------------------------------------------------

    raw_output = chain.invoke(
        {
            "query": query,
            "num_queries": num_queries,
        }
    )

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    variations = _parse_query_variations(
        raw_output
    )

    # --------------------------------------------------------
    # Convert variations into validated SubQuery objects
    # --------------------------------------------------------

    for variation in variations:

        if variation == query:
            continue

        results.append(
            SubQuery(
                query=variation,
            )
        )

    # --------------------------------------------------------
    # Return original + requested number of variations
    # --------------------------------------------------------

    return results[: num_queries + 1]