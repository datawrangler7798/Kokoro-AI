"""
Kokoro - Streamlit Application.

Responsibilities:
    - Provide the recruiter-facing UI.
    - Accept recruiter queries.
    - Display retrieved candidates and answers.
    - Maintain the current chat session.
    - Display basic retrieval and guardrail information.

The UI layer should call backend services through their public
interfaces and should not contain retrieval or LLM business logic.
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from core.guardrails.guardrails import create_guardrails
from core.memory.memory_rag import create_memory_rag
from core.retrieval.search_router import create_search_router
from utils.config import get_settings


# ============================================================
# Configuration
# ============================================================

settings = get_settings()


# ============================================================
# Session State
# ============================================================


def initialize_session() -> None:
    """
    Initialize Streamlit session state.
    """

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(
            uuid.uuid4()
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_response" not in st.session_state:
        st.session_state.last_response = None


# ============================================================
# Services
# ============================================================


@st.cache_resource
def get_memory_service():
    """
    Return shared memory service.
    """

    return create_memory_rag()


@st.cache_resource
def get_guardrail_service():
    """
    Return shared guardrail service.
    """

    return create_guardrails()


@st.cache_resource
def get_search_router():
    """
    Return shared search router.
    """

    return create_search_router()


# ============================================================
# Header
# ============================================================


def render_header() -> None:
    """
    Render application header.
    """

    st.title(
        getattr(
            settings,
            "APP_NAME",
            "Kokoro",
        )
    )

    st.caption(
        "AI-powered recruitment search and candidate analysis"
    )


# ============================================================
# Sidebar
# ============================================================


def render_sidebar() -> None:
    """
    Render sidebar controls.
    """

    st.sidebar.header("Session")

    st.sidebar.text(
        f"Session ID: "
        f"{st.session_state.session_id[:8]}..."
    )

    if st.sidebar.button(
        "Clear Conversation",
        use_container_width=True,
    ):

        memory = get_memory_service()

        memory.clear_session(
            st.session_state.session_id
        )

        st.session_state.messages = []

        st.session_state.last_response = None

        st.rerun()

    st.sidebar.divider()

    st.sidebar.header("Search")

    st.sidebar.caption(
        "Kokoro can route simple questions through "
        "shallow search and complex questions through "
        "deep search."
    )


# ============================================================
# Chat History
# ============================================================


def render_chat_history() -> None:
    """
    Render previous conversation messages.
    """

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(role):
            st.markdown(content)


# ============================================================
# Response Formatting
# ============================================================


def extract_answer(
    response: Any,
) -> str:
    """
    Extract answer text from a backend response.

    Supports the Kokoro response schema as well as a
    plain string fallback.
    """

    if response is None:
        return "No response was generated."

    if isinstance(response, str):
        return response

    answer = getattr(
        response,
        "answer",
        None,
    )

    if answer:
        return str(answer)

    if isinstance(response, dict):

        answer = response.get(
            "answer"
        )

        if answer:
            return str(answer)

        response_text = response.get(
            "response"
        )

        if response_text:
            return str(response_text)

    return str(response)


# ============================================================
# Response Metadata
# ============================================================


def render_response_metadata(
    response: Any,
) -> None:
    """
    Display optional response metadata.
    """

    if response is None:
        return

    retrieval = getattr(
        response,
        "retrieval",
        None,
    )

    if retrieval is None:
        return

    with st.expander(
        "Retrieval details"
    ):

        st.write(
            retrieval
        )


# ============================================================
# Query Processing
# ============================================================


def process_query(
    query: str,
) -> None:
    """
    Process one recruiter query.

    The UI validates the input first and then delegates
    search routing to the backend.

    Actual retrieval/generation orchestration should remain
    outside the Streamlit layer.
    """

    guardrails = get_guardrail_service()

    validation = (
        guardrails.validate_input(
            query
        )
    )

    if validation.status.value != "passed":
        st.error(
            validation.reason
        )
        return

    session_id = (
        st.session_state.session_id
    )

    memory = get_memory_service()

    relevant_memory = (
        memory.get_relevant_memory(
            session_id=session_id,
            query=query,
        )
    )

    memory_text = memory.format_memory(
        relevant_memory
    )

    router = get_search_router()

    try:

        query_plan = router.route(
            query=query
        )

    except Exception as exc:

        st.error(
            "Unable to route the query."
        )

        if settings.ENVIRONMENT == "development":
            st.exception(exc)

        return

    # --------------------------------------------------------
    # Display routing information
    # --------------------------------------------------------

    with st.expander(
        "Search plan",
        expanded=False,
    ):

        st.write(
            {
                "search_depth": str(
                    query_plan.search_depth
                ),
                "intent": str(
                    query_plan.intent
                ),
                "subqueries": (
                    query_plan.subqueries
                ),
                "expanded_queries": (
                    query_plan.expanded_queries
                ),
            }
        )

    # --------------------------------------------------------
    # Backend integration point
    # --------------------------------------------------------

    st.warning(
        "Query routing is ready. "
        "Connect the application orchestration layer "
        "to execute retrieval, reranking, prompt building, "
        "and Gemini generation."
    )

    # Keep memory available for the eventual orchestration
    # layer without making Streamlit responsible for backend
    # business logic.
    _ = memory_text

    _ = session_id


# ============================================================
# Chat Input
# ============================================================


def render_chat_input() -> None:
    """
    Render chat input and process submitted queries.
    """

    query = st.chat_input(
        "Ask about candidates, skills, experience, or a JD..."
    )

    if not query:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your query..."
        ):

            process_query(
                query
            )


# ============================================================
# Main Application
# ============================================================


def main() -> None:
    """
    Streamlit application entry point.
    """

    initialize_session()

    render_header()

    render_sidebar()

    render_chat_history()

    render_chat_input()


if __name__ == "__main__":
    main()