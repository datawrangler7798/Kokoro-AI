"""
Kokoro - Streamlit Application.

Responsibilities:
    - Provide the recruiter-facing UI.
    - Accept recruiter queries.
    - Display retrieved candidates and answers.
    - Maintain the current chat session.
    - Display basic retrieval and guardrail information.
    - Initialize existing resume ingestion.

The UI layer should call backend services through their public
interfaces and should not contain retrieval or LLM business logic.
"""

from __future__ import annotations

# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

import sys
from pathlib import Path

# app.py is inside:
# Kokoro-AI/streamlit/app.py
#
# parents[1] points to:
# Kokoro-AI/
#
# Adding the project root allows imports such as:
# from core...
# from utils...

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import json
import re
import threading
from html import escape
from typing import Any

import streamlit as st

from core.application import create_application
from core.evaluation.evaluator import create_evaluator
from utils.config import get_settings
from utils.logger import logger
from utils.schemas import DocumentType, SearchFilters


# ============================================================
# Configuration
# ============================================================

settings = get_settings()
SESSION_COUNTER_PATH = PROJECT_ROOT / "data" / "index" / "session_counter.json"
SESSION_ID_PATTERN = re.compile(r"^KK(\d+)$")
SESSION_COUNTER_LOCK = threading.Lock()

WELCOME_MESSAGE = (
    "Hi! How are you? I’m your resume search assistant. "
    "To get started, paste the job description in this chat. "
    "I’ll compare it with the indexed resumes and find relevant candidates."
)

RECRUITMENT_SCOPE_PATTERN = re.compile(
    r"\b(candidate|applicant|resume|cv|job|role|position|recruit(?:er|ment)?|"
    r"hir(?:e|ing)|skill|experience|qualification|salary|responsibilit(?:y|ies)|"
    r"interview|accountant|accounting|engineer|developer|designer|analyst|manager|"
    r"sales|nurse|teacher|technician|intern|employee|career)\b",
    re.IGNORECASE,
)


def welcome_messages() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def is_greeting(message: str) -> bool:
    """Recognize short greetings so they are not mistaken for a JD."""

    words = re.sub(r"[^a-z\s]", " ", message.lower()).split()
    common_typos = {
        "hlo": "hello",
        "helo": "hello",
        "helloo": "hello",
        "hii": "hi",
        "hiii": "hi",
        "heyy": "hey",
    }
    normalized = " ".join(common_typos.get(word, word) for word in words)
    return normalized in {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "how are you",
        "hi how are you",
        "hello how are you",
        "good morning",
        "good afternoon",
        "good evening",
    }


def looks_like_job_description(message: str) -> bool:
    """Identify a pasted JD so it replaces the active one in this chat."""

    normalized = message.lower()
    jd_headings = (
        "job description",
        "responsibilities",
        "required skills",
        "preferred qualifications",
        "requirements",
        "what you will do",
        "qualifications",
    )
    return (
        len(message.strip()) >= 500
        and bool(RECRUITMENT_SCOPE_PATTERN.search(message))
    ) or (
        len(message.strip()) >= 180
        and any(heading in normalized for heading in jd_headings)
    )


def is_recruitment_related(message: str) -> bool:
    """Keep unrelated chat text from entering recruiter retrieval."""

    if looks_like_job_description(message):
        return True
    if RECRUITMENT_SCOPE_PATTERN.search(message):
        return True
    return bool(
        re.search(
            r"\b(?:who|which)\s+(?:is|has|are|have|would|should)\b|"
            r"\b(?:compare|rank|shortlist)\b",
            message,
            re.IGNORECASE,
        )
    )


def create_session_id(existing_ids: list[str] | None = None) -> str:
    """Return the next persistent sequential session ID (KK0001, KK0002, ...)."""

    if existing_ids is None:
        existing_ids = []
        if "recent_sessions" in st.session_state:
            existing_ids.extend(st.session_state.recent_sessions)
        if "session_messages" in st.session_state:
            existing_ids.extend(st.session_state.session_messages.keys())
        if "session_id" in st.session_state:
            existing_ids.append(st.session_state.session_id)

    existing_numbers = [
        int(match.group(1))
        for session_id in existing_ids
        if (match := SESSION_ID_PATTERN.fullmatch(str(session_id)))
    ]

    with SESSION_COUNTER_LOCK:
        SESSION_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        last_number = 0
        if SESSION_COUNTER_PATH.exists():
            try:
                stored = json.loads(SESSION_COUNTER_PATH.read_text(encoding="utf-8"))
                last_number = int(stored.get("last_session_number", 0))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.exception("Could not read the session ID counter.")
                raise RuntimeError(
                    f"Could not read session counter at {SESSION_COUNTER_PATH}."
                ) from exc

        next_number = max([last_number, *existing_numbers], default=0) + 1
        temporary_path = SESSION_COUNTER_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps({"last_session_number": next_number}),
            encoding="utf-8",
        )
        temporary_path.replace(SESSION_COUNTER_PATH)

    return f"KK{next_number:04d}"


# ============================================================
# Session State
# ============================================================


def initialize_session() -> None:
    """
    Initialize Streamlit session state.
    """

    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = welcome_messages()

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    if "recent_sessions" not in st.session_state:
        st.session_state.recent_sessions = [st.session_state.session_id]

    if "session_messages" not in st.session_state:
        st.session_state.session_messages = {
            st.session_state.session_id: st.session_state.messages
        }

    if "job_description" not in st.session_state:
        st.session_state.job_description = None

    if "pending_search" not in st.session_state:
        st.session_state.pending_search = None

    if "session_job_descriptions" not in st.session_state:
        st.session_state.session_job_descriptions = {
            st.session_state.session_id: st.session_state.job_description
        }

    # Migrate existing UUID chats so every visible session uses the KK#### format.
    current_id = st.session_state.session_id
    all_ids = list(dict.fromkeys([
        *st.session_state.recent_sessions,
        *st.session_state.session_messages.keys(),
        *st.session_state.session_job_descriptions.keys(),
        current_id,
    ]))
    id_mapping: dict[str, str] = {}
    for old_id in all_ids:
        if SESSION_ID_PATTERN.fullmatch(str(old_id)):
            id_mapping[old_id] = old_id
        else:
            id_mapping[old_id] = create_session_id(
                existing_ids=all_ids + list(id_mapping.values())
            )

    st.session_state.session_messages = {
        id_mapping.get(session_id, session_id): messages
        for session_id, messages in st.session_state.session_messages.items()
    }
    st.session_state.session_job_descriptions = {
        id_mapping.get(session_id, session_id): job_description
        for session_id, job_description in st.session_state.session_job_descriptions.items()
    }
    st.session_state.session_id = id_mapping[current_id]
    st.session_state.recent_sessions = list(dict.fromkeys(
        id_mapping.get(session_id, session_id)
        for session_id in st.session_state.recent_sessions
    ))
    if st.session_state.session_id not in st.session_state.recent_sessions:
        st.session_state.recent_sessions.insert(0, st.session_state.session_id)
    st.session_state.session_messages.setdefault(
        st.session_state.session_id,
        st.session_state.messages,
    )
    st.session_state.session_job_descriptions.setdefault(
        st.session_state.session_id,
        st.session_state.job_description,
    )

    # Clear a greeting that an earlier app version accidentally saved as a JD.
    if is_greeting(str(st.session_state.job_description or "")):
        st.session_state.job_description = None
        st.session_state.pending_search = None
        st.session_state.session_job_descriptions[st.session_state.session_id] = None
        st.session_state.messages = welcome_messages()
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages


# ============================================================
# Services
# ============================================================


@st.cache_resource
def get_application():
    return create_application()


@st.cache_resource
def get_evaluator():
    return create_evaluator()


# ============================================================
# Resume Index Initialization
# ============================================================


@st.cache_resource
def initialize_resume_index() -> dict[str, Any]:
    """
    Discover and index existing PDF resumes from
    data/resumes/.

    The ingestion service handles:

        PDF
          ↓
        validation
          ↓
        parsing
          ↓
        chunking
          ↓
        Gemini embedding
          ↓
        Pinecone upsert

    This runs once per Streamlit process, so local resumes are
    ingested at startup rather than from a UI control.
    """

    try:
        logger.info("Starting startup indexing from data/resumes and data/jds.")
        result = get_application().ingest_existing_resumes()
        logger.info(
            "Startup document indexing completed | total=%d indexed=%d skipped=%d failed=%d chunks=%d",
            result.get("total_files", 0),
            result.get("successful_files", 0),
            result.get("skipped_files", 0),
            result.get("failed_files", 0),
            result.get("total_chunks", 0),
        )
        for item in result.get("results", []):
            if item.get("error"):
                logger.error(
                    "Document indexing failed | file=%s | error_type=%s | error=%s",
                    item.get("source_file", "unknown"),
                    item.get("error_type", "IndexingError"),
                    item["error"],
                )

        return result

    except Exception as exc:
        logger.exception("Startup document indexing failed.")

        result = {
            "total_files": 0,
            "processed_files": 0,
            "successful_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "results": [],
            "error": str(exc),
        }
        return result


# ============================================================
# Header
# ============================================================


def render_header() -> None:
    """
    Render application header.
    """

    app_name = getattr(settings, "APP_NAME", "Kokoro")
    st.markdown(
        f"""
        <div class="kokoro-hero">
          <div class="kokoro-mark">K</div>
          <div>
            <div class="kokoro-eyebrow">TALENT INTELLIGENCE</div>
            <div class="kokoro-title">{escape(str(app_name))} <span>Recruiting</span></div>
            <div class="kokoro-subtitle">Find the right people in your resume library.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        f"Session ID: {st.session_state.session_id}"
    )

    if st.sidebar.button("New chat", width="stretch"):
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
        st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description
        new_id = create_session_id()
        st.session_state.session_id = new_id
        st.session_state.recent_sessions.insert(0, new_id)
        st.session_state.messages = welcome_messages()
        st.session_state.job_description = None
        st.session_state.session_messages[new_id] = st.session_state.messages
        st.session_state.session_job_descriptions[new_id] = None
        st.rerun()

    if len(st.session_state.recent_sessions) > 1:
        selected = st.sidebar.selectbox(
            "Recent chats",
            st.session_state.recent_sessions,
            format_func=lambda value: f"Chat {value[:8]}",
            index=st.session_state.recent_sessions.index(st.session_state.session_id),
        )
        if selected != st.session_state.session_id:
            st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
            st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description
            st.session_state.session_id = selected
            st.session_state.messages = st.session_state.session_messages.get(selected, welcome_messages())
            st.session_state.job_description = st.session_state.session_job_descriptions.get(selected)
            st.rerun()

    if st.sidebar.button(
        "Clear Conversation",
        width="stretch",
    ):

        get_application().memory.clear_session(
            st.session_state.session_id
        )

        st.session_state.messages = welcome_messages()
        st.session_state.job_description = None
        st.session_state.session_job_descriptions[st.session_state.session_id] = None
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages

        st.session_state.last_response = None

        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### Recruiter workspace")
    st.sidebar.caption("Search candidate profiles by sharing a job description in chat.")


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
            if role == "user" and (
                message.get("kind") == "job_description"
                or looks_like_job_description(content)
            ):
                st.markdown("**Job description submitted**")
                with st.expander("View job description"):
                    st.text(content)
            else:
                st.markdown(content)
            if role == "assistant":
                render_candidate_cards(message.get("candidates", []))
                evidence = message.get("evidence", [])
                if evidence:
                    with st.expander("Evidence"):
                        for item in evidence:
                            st.markdown(f"- {item}")


def render_job_description_status() -> None:
    """Show the active job description and allow replacing it from chat."""

    if not st.session_state.job_description:
        st.markdown(
            '<div class="jd-prompt"><span class="jd-prompt-icon">✦</span>'
            '<div><strong>Start with a job description</strong><br>'
            '<span>Paste it in the message box below and I’ll search your resumes.</span></div></div>',
            unsafe_allow_html=True,
        )
        return

    jd = st.session_state.job_description
    with st.container(border=True):
        top, action = st.columns([5, 1])
        with top:
            st.markdown("**Active job description** · Search context")
            st.caption(jd[:240] + ("…" if len(jd) > 240 else ""))
        with action:
            if st.button("Change", key="change_jd", width="stretch"):
                st.session_state.job_description = None
                st.session_state.session_job_descriptions[st.session_state.session_id] = None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sure—paste the new job description in the chat and I’ll use it for the next search.",
                })
                st.rerun()


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

    with st.expander(
        "Retrieval details"
    ):
        plan = getattr(response, "query_plan", None)
        if plan is not None:
            st.write({
                "search_depth": plan.search_depth.value,
                "intent": plan.intent.value,
                "sub_queries": plan.sub_queries,
                "expanded_queries": plan.expanded_queries,
                "retrieved_chunk_ids": response.retrieved_chunk_ids,
                "latency_ms": response.latency_ms,
            })


def render_candidate_cards(candidates: list[Any]) -> None:
    for candidate in candidates:
        with st.container(border=True):
            st.subheader(candidate.get("candidate_name") or candidate.get("candidate_id", "Candidate"))
            score = candidate.get("match_score")
            if score is not None:
                st.metric("Match score", f"{score:.0%}")
            left, right = st.columns(2)
            with left:
                st.markdown("**Matching skills**")
                st.write(", ".join(candidate.get("matched_skills", [])) or "Not specified")
            with right:
                st.markdown("**Potential gaps**")
                st.write(", ".join(candidate.get("missing_skills", [])) or "Not specified")
            if candidate.get("explanation"):
                st.write(candidate["explanation"])
            evidence = candidate.get("evidence", [])
            if evidence:
                with st.expander("View evidence"):
                    for item in evidence:
                        st.markdown(f"- {item}")


# ============================================================
# Query Processing
# ============================================================


def process_query(
    query: str,
) -> Any | None:
    """
    Process one recruiter query.

    The UI validates the input first and then delegates
    search routing to the backend.

    Actual retrieval/generation orchestration should remain
    outside the Streamlit layer.
    """

    try:
        response = get_application().answer(
            query=query,
            session_id=st.session_state.session_id,
        )
    except Exception as exc:
        logger.exception("Recruiter query failed | session_id=%s", st.session_state.session_id)
        st.error(str(exc))
        if settings.APP_ENV == "development":
            st.exception(exc)
        return None
    return response


# ============================================================
# Chat Input
# ============================================================


def process_pending_search() -> None:
    """Run a submitted search after the active JD and chat history rerender."""

    pending = st.session_state.pending_search
    if not pending:
        return

    # Clear first so a Streamlit rerun or recoverable exception cannot submit
    # the same search twice.
    st.session_state.pending_search = None
    with st.chat_message("assistant"):
        if pending["job_description_changed"]:
            st.markdown("**Job description updated.** I’m searching the resume library now.")
        with st.spinner("Searching your resume library..."):
            response = process_query(pending["search_query"])

        if response is not None:
            answer = extract_answer(response)
            st.markdown(answer)
            render_candidate_cards([
                candidate.model_dump()
                for candidate in response.candidates
            ])
            render_response_metadata(response)
            if response.evidence:
                with st.expander("Evidence"):
                    for item in response.evidence:
                        st.markdown(f"- {item}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "candidates": [item.model_dump() for item in response.candidates],
                "evidence": response.evidence,
                "response_metadata": {
                    "query_plan": response.query_plan.model_dump() if response.query_plan else None,
                    "retrieved_chunk_ids": response.retrieved_chunk_ids,
                    "latency_ms": response.latency_ms,
                },
            })
        else:
            error_message = "I couldn’t complete that search. The job description is saved; please try again shortly."
            st.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })

    st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
    st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description
    st.rerun()


def render_chat_input() -> None:
    """Render the input and queue new searches for the next UI pass."""

    query = st.chat_input(
        "Paste the job description to get started..."
        if not st.session_state.job_description
        else "Ask about candidates, skills, or paste a new job description..."
    )
    if not query or not query.strip():
        return

    query = query.strip()
    if is_greeting(query):
        st.session_state.messages.append({"role": "user", "content": query})
        greeting_reply = (
            "Hello! I’m ready to help with candidates for the active job description. What would you like to know?"
            if st.session_state.job_description
            else "Hello! Please paste the job description here, and I’ll look for matching candidates in your resume library."
        )
        st.session_state.messages.append({"role": "assistant", "content": greeting_reply})
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
        st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description
        st.rerun()

    # Reject unsupported requests before they can be mistaken for a job
    # description or combined with the active JD and sent to retrieval.
    input_check = get_application().guardrails.validate_input(query)
    if not input_check.passed:
        st.session_state.messages.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": input_check.reason},
            ]
        )
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
        st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description
        st.rerun()

    if not is_recruitment_related(query):
        invalid_reply = (
            "Invalid input. I can only help with professional job descriptions "
            "and candidate-search questions. Please enter a recruiting-related request."
        )
        st.session_state.messages.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": invalid_reply},
            ]
        )
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
        st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description
        st.rerun()

    is_new_jd = not st.session_state.job_description
    is_replacement_jd = bool(
        st.session_state.job_description and looks_like_job_description(query)
    )
    job_description_changed = is_new_jd or is_replacement_jd

    if is_new_jd and not looks_like_job_description(query):
        st.session_state.messages.append({"role": "user", "content": query})
        prompt = "Hello! Please share the job description, and I’ll search your resume library for matching candidates."
        st.session_state.messages.append({"role": "assistant", "content": prompt})
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
        st.session_state.session_job_descriptions[st.session_state.session_id] = None
        st.rerun()

    if job_description_changed:
        st.session_state.job_description = query
        st.session_state.session_job_descriptions[st.session_state.session_id] = query
        search_query = query
        user_message = {"role": "user", "content": query, "kind": "job_description"}
    else:
        search_query = (
            f"Job description:\n{st.session_state.job_description}\n\n"
            f"Recruiter question:\n{query}"
        )
        user_message = {"role": "user", "content": query}

    st.session_state.messages.append(user_message)
    st.session_state.pending_search = {
        "search_query": search_query,
        "job_description_changed": job_description_changed,
    }
    st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
    st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description
    st.rerun()


def render_evaluation_tab() -> None:
    st.subheader("Evaluate a recruiter response")
    st.caption(
        "Check the hybrid Pinecone + BM25 ranking. To calculate retrieval metrics, "
        "provide candidate IDs known to be relevant for this question."
    )
    with st.form("evaluation_form"):
        question = st.text_input("Evaluation question")
        reference = st.text_area("Reference answer (optional)")
        relevant_ids = st.text_input(
            "Known relevant candidate IDs, comma separated (optional)",
            help="Use candidate IDs from your resume metadata. Without ground-truth IDs, the app can show the ranking but cannot calculate precision or recall.",
        )
        run_ragas = st.checkbox(
            "Also score the generated answer with RAGAS (slower; uses Google Gemini)",
            value=False,
            disabled=not settings.ENABLE_RAGAS,
            help="Enable ENABLE_RAGAS in configuration to use this option.",
        )
        submitted = st.form_submit_button("Run evaluation")
    if not submitted:
        return
    if not question.strip():
        st.error("Enter an evaluation question.")
        return
    try:
        application = get_application()
        retrieved = application.hybrid_indexer.search(
            question.strip(),
            top_k=settings.HYBRID_TOP_K,
            semantic_top_k=settings.VECTOR_TOP_K,
            keyword_top_k=settings.BM25_TOP_K,
            filters=SearchFilters(document_type=DocumentType.RESUME),
        )

        # The index returns chunks. Evaluation compares candidate rankings,
        # so retain each candidate once at its best-ranked chunk position.
        ranked_candidates: list[dict[str, Any]] = []
        ranked_ids: list[str] = []
        seen_ids: set[str] = set()
        for item in retrieved:
            candidate_id = str(item.metadata.get("candidate_id") or "").strip()
            if not candidate_id or candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            ranked_ids.append(candidate_id)
            ranked_candidates.append(
                {
                    "Rank": len(ranked_ids),
                    "Candidate ID": candidate_id,
                    "Name": item.metadata.get("candidate_name") or "",
                    "Hybrid score": round(float(item.hybrid_score), 4),
                }
            )

        st.markdown("**Hybrid retrieval ranking**")
        if ranked_candidates:
            st.dataframe(ranked_candidates, hide_index=True, width="stretch")
        else:
            st.info("The hybrid search found no resume candidates for this question.")

        evaluator = get_evaluator()
        ids = [value.strip() for value in relevant_ids.split(",") if value.strip()]
        if ids:
            retrieval_result = evaluator.evaluate_retrieval(
                query=question.strip(),
                ranked_ids=ranked_ids,
                relevant_ids=ids,
            )
            st.markdown("**Retrieval metrics**")
            st.json(retrieval_result.model_dump())
        else:
            st.info(
                "Ranking is shown above. Add known relevant candidate IDs and run again "
                "to calculate precision@k, recall@k, MRR, and nDCG."
            )

        if run_ragas:
            with st.spinner("Generating an answer and scoring it with RAGAS…"):
                response = application.answer(
                    question.strip(),
                    st.session_state.session_id,
                )
                ragas_result = evaluator.evaluate_generation(
                    question=question.strip(),
                    answer=response.answer,
                    contexts=response.metadata.get("retrieved_contexts", []),
                    reference=reference.strip() or None,
                )
            st.markdown("**Generated answer**")
            st.write(response.answer)
            if ragas_result is not None:
                st.markdown("**RAGAS metrics**")
                st.json(ragas_result.model_dump())
            else:
                st.warning("RAGAS could not complete. Check the application log for the provider error.")
    except Exception as exc:
        logger.exception("Evaluation tab failed | question=%r", question.strip())
        st.error(f"Evaluation failed: {exc}")
        if settings.APP_ENV == "development":
            st.exception(exc)


# ============================================================
# Main Application
# ============================================================


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --primary-color: #1769aa; }
        .stApp { background: #f5f8fc; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e4eaf2; }
        .block-container { max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem; }
        .kokoro-hero { display:flex; align-items:center; gap:16px; padding:4px 0 20px; border-bottom:1px solid #e3eaf3; margin-bottom:22px; }
        .kokoro-mark { width:46px; height:46px; display:grid; place-items:center; border-radius:13px; color:white; font-size:23px; font-weight:800; background:linear-gradient(135deg,#1267a5,#258ed0); box-shadow:0 7px 18px #1769aa2b; }
        .kokoro-eyebrow { color:#1769aa; font-size:10px; font-weight:800; letter-spacing:1.8px; }
        .kokoro-title { color:#18334d; font-size:26px; font-weight:750; line-height:1.25; }
        .kokoro-title span { color:#62758a; font-weight:500; }
        .kokoro-subtitle { color:#65788d; font-size:14px; margin-top:3px; }
        .jd-prompt { display:flex; gap:14px; align-items:center; margin:12px 0 18px; padding:18px 20px; border:1px solid #dce6f0; border-radius:14px; background:#fff; color:#243b53; box-shadow:0 4px 16px #18334d09; }
        .jd-prompt-icon { display:grid; place-items:center; flex:0 0 38px; height:38px; border-radius:11px; color:#1769aa; background:#edf5fb; font-size:20px; }
        .jd-prompt span { color:#64748b; font-size:13px; }
        [data-testid="stChatMessage"] { border:1px solid #e3eaf1; border-radius:15px; padding:14px 18px; background:#fff; box-shadow:0 3px 12px #18334d08; }
        [data-testid="stChatInput"] { border-radius:13px; }
        [data-testid="stTabs"] button { font-weight:650; }
        [data-testid="stTabs"] button[aria-selected="true"] { color:#1769aa; border-bottom-color:#1769aa; }
        [data-testid="stSidebar"] button { border-radius:9px; }
        div[data-testid="stMetric"] { background:#f8fafc; border:1px solid #e4ebf2; padding:12px 14px; border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """
    Streamlit application entry point.
    """

    st.set_page_config(page_title=settings.APP_NAME, page_icon=":material/person_search:")
    initialize_session()

    render_styles()
    render_header()
    render_sidebar()

    # --------------------------------------------------------
    # Initialize existing resumes
    # --------------------------------------------------------

    initialize_resume_index()
    chat_tab, evaluation_tab = st.tabs(["Recruiter chat", "Evaluation"])
    with chat_tab:
        render_chat_history()
        render_job_description_status()
        process_pending_search()
        render_chat_input()
    with evaluation_tab:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
