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
ASSISTANT_AVATAR = PROJECT_ROOT / "streamlit" / "assets" / "assistant-avatar.png"

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
    "Hi, I’m Kira, the recruiting assistant for Kokoro AI. "
    "Tell me what kind of candidate you’re looking for, and I’ll search the resume library."
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
    ) or (
        len(message.strip()) >= 180
        and any(heading in normalized for heading in jd_headings)
    )


def create_session_id(existing_ids: list[str] | None = None) -> str:
    """Return the next persistent sequential session ID (KK00001, KK00002, ...)."""

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

    return f"KK{next_number:05d}"


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

    if "ended_sessions" not in st.session_state:
        st.session_state.ended_sessions = set()

    if "session_messages" not in st.session_state:
        st.session_state.session_messages = {
            st.session_state.session_id: st.session_state.messages
        }

    if "job_description" not in st.session_state:
        st.session_state.job_description = None

    if "pending_search" not in st.session_state:
        st.session_state.pending_search = None

    if "job_prompt_seen_sessions" not in st.session_state:
        st.session_state.job_prompt_seen_sessions = set()

    if "session_job_descriptions" not in st.session_state:
        st.session_state.session_job_descriptions = {
            st.session_state.session_id: st.session_state.job_description
        }

    # Normalize older IDs so visible sessions use the KK##### format.
    current_id = st.session_state.session_id
    all_ids = list(dict.fromkeys([
        *st.session_state.recent_sessions,
        *st.session_state.session_messages.keys(),
        *st.session_state.session_job_descriptions.keys(),
        *st.session_state.ended_sessions,
        current_id,
    ]))
    id_mapping: dict[str, str] = {}
    for old_id in all_ids:
        if SESSION_ID_PATTERN.fullmatch(str(old_id)):
            id_mapping[old_id] = f"KK{int(old_id[2:]):05d}"
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
    st.session_state.ended_sessions = {
        id_mapping.get(session_id, session_id)
        for session_id in st.session_state.ended_sessions
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


def _index_existing_resumes(application: Any) -> None:
    """
    Index existing and newly added resumes in a background worker.
    """

    resume_dir = Path(application.settings.RESUME_DIRECTORY)

    def snapshot() -> dict[Path, tuple[int, int]]:
        if not resume_dir.exists():
            return {}
        files: dict[Path, tuple[int, int]] = {}
        for path in resume_dir.glob("*.pdf"):
            try:
                stat = path.stat()
                files[path] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                logger.debug("Could not inspect resume while scanning: %s", path)
        return files

    try:
        # Keep a pre-ingestion snapshot so files copied in during startup are
        # picked up by the watcher after the initial batch completes.
        known_files = snapshot()
        logger.info("Starting startup indexing from data/resumes and data/jds.")
        result = application.ingest_existing_resumes()
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
    except Exception:
        logger.exception("Startup document indexing failed.")
        known_files = {}

    pending_files: dict[Path, tuple[int, int]] = {}
    threading.Event().wait(2)
    logger.info("Resume folder watcher started | directory=%s", resume_dir)
    while True:
        try:
            current_files = snapshot()
            for path, signature in current_files.items():
                if known_files.get(path) == signature:
                    pending_files.pop(path, None)
                    continue

                # Wait for the file size and modification time to settle so a
                # partially copied PDF is not sent to the parser.
                if pending_files.get(path) != signature:
                    pending_files[path] = signature
                    continue

                logger.info("New or changed resume detected | file=%s", path.name)
                result = application.ingest_files([path], DocumentType.RESUME)
                if result.get("failed_files", 0):
                    logger.error("Resume ingestion failed | file=%s", path.name)
                else:
                    known_files[path] = signature
                    logger.info(
                        "Resume ingestion finished | file=%s indexed=%d skipped=%d chunks=%d",
                        path.name,
                        result.get("successful_files", 0),
                        result.get("skipped_files", 0),
                        result.get("total_chunks", 0),
                    )
                pending_files.pop(path, None)

            for path in known_files.keys() - current_files.keys():
                known_files.pop(path, None)
                pending_files.pop(path, None)
        except Exception:
            logger.exception("Resume folder watcher iteration failed.")

        threading.Event().wait(2)


@st.cache_resource
def initialize_resume_index() -> threading.Thread:
    """Start resume indexing once per process without blocking or updating the UI."""

    worker = threading.Thread(
        target=_index_existing_resumes,
        args=(get_application(),),
        name="kokoro-resume-indexer",
        daemon=True,
    )
    worker.start()
    return worker


# ============================================================
# Header
# ============================================================


def render_header() -> None:
    """
    Render application header.
    """

    st.markdown(
        f"""
        <div class="kokoro-hero">
          <div>
            <div class="kokoro-title">Kokoro AI</div>
            <div class="kokoro-subtitle">Understanding the person behind the resume.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Sidebar
# ============================================================


def save_current_session() -> None:
    st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages
    st.session_state.session_job_descriptions[st.session_state.session_id] = st.session_state.job_description


def start_new_session() -> None:
    save_current_session()
    new_id = create_session_id()
    st.session_state.session_id = new_id
    st.session_state.recent_sessions.insert(0, new_id)
    st.session_state.messages = welcome_messages()
    st.session_state.job_description = None
    st.session_state.pending_search = None
    st.session_state.session_messages[new_id] = st.session_state.messages
    st.session_state.session_job_descriptions[new_id] = None
    st.rerun()


def render_sidebar() -> str:
    """
    Render sidebar controls.
    """

    st.sidebar.markdown("## Kokoro AI")
    st.sidebar.caption("Recruiter workspace")
    page = st.sidebar.radio(
        "Workspace",
        ["Recruiter chat", "Evaluation"],
        label_visibility="collapsed",
        key="workspace_page",
    )
    st.sidebar.divider()

    if page == "Evaluation":
        st.sidebar.caption("Measure retrieval and answer quality.")
        return page

    st.sidebar.markdown("#### Current session")
    chat_started = any(
        message.get("role") == "user"
        for message in st.session_state.messages
    )
    if chat_started:
        st.sidebar.caption(f"Session ID · {st.session_state.session_id}")

    if st.sidebar.button("New chat", type="primary", width="stretch"):
        start_new_session()

    is_ended = st.session_state.session_id in st.session_state.ended_sessions
    if st.sidebar.button(
        "End chat",
        width="stretch",
        disabled=is_ended,
        help="End this chat and keep its history isolated from new chats.",
    ):
        save_current_session()
        st.session_state.ended_sessions.add(st.session_state.session_id)
        st.session_state.pending_search = None
        st.rerun()

    if st.sidebar.button(
        "Clear chat session",
        width="stretch",
        help="Remove the messages and search context from this session.",
    ):

        get_application().memory.clear_session(
            st.session_state.session_id
        )

        st.session_state.messages = welcome_messages()
        st.session_state.job_description = None
        st.session_state.ended_sessions.discard(st.session_state.session_id)
        st.session_state.session_job_descriptions[st.session_state.session_id] = None
        st.session_state.session_messages[st.session_state.session_id] = st.session_state.messages

        st.session_state.last_response = None

        st.rerun()

    if is_ended:
        st.sidebar.info("This chat has ended. Start a new chat to continue.")

    return page

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

        if role == "assistant":
            message_container = st.chat_message(role, avatar=str(ASSISTANT_AVATAR))
        else:
            message_container = st.chat_message(role)

        with message_container:
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
    """Show search context after the user starts chatting."""

    chat_started = any(
        message.get("role") == "user"
        for message in st.session_state.messages
    )
    if not chat_started:
        return

    if not st.session_state.job_description:
        session_id = st.session_state.session_id
        if session_id in st.session_state.job_prompt_seen_sessions:
            return

        st.session_state.job_prompt_seen_sessions.add(session_id)
        st.markdown(
            '<div class="jd-prompt"><span class="jd-prompt-icon">✦</span>'
            '<div><strong>Add a job description for more context</strong><br>'
            '<span>Paste one in the chat when you’re ready. Quick candidate searches work too.</span></div></div>',
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
    has_fit_assessment = any(candidate.get("score_breakdown") for candidate in candidates)
    if candidates and any(candidate.get("score_breakdown") for candidate in candidates):
        with st.expander("How the JD fit score is calculated"):
            st.caption("Evidence-based fit guidance, not a probability of hiring.")
            st.write("Role relevance 30% · Skills match 35% · Experience 20% · Domain relevance 10% · Evidence strength 5%.")
    elif candidates and not has_fit_assessment:
        st.caption("These are keyword-supported resume matches; Gemini fit scoring is temporarily unavailable.")

    for candidate in candidates:
        with st.container(border=True):
            st.subheader(candidate.get("candidate_name") or candidate.get("candidate_id", "Candidate"))
            if candidate.get("profile_summary"):
                st.write(candidate["profile_summary"])
            score = candidate.get("match_score")
            if score is not None:
                st.metric("JD fit score", f"{score:.0%}")
            recommendation = candidate.get("recommendation")
            if recommendation:
                st.markdown(f"**Recruiter suggestion:** {recommendation}")

            score_breakdown = candidate.get("score_breakdown") or {}
            if score_breakdown:
                with st.expander("Fit breakdown"):
                    labels = {
                        "role_relevance": "Role relevance",
                        "skills_match": "Skills match",
                        "experience_match": "Experience match",
                        "domain_relevance": "Domain relevance",
                        "evidence_strength": "Evidence strength",
                    }
                    breakdown_rows = [
                        {"Dimension": label, "Score": f"{score_breakdown[key]:.0f}%"}
                        for key, label in labels.items()
                        if score_breakdown.get(key) is not None
                    ]
                    st.dataframe(breakdown_rows, hide_index=True, width="stretch")

            if has_fit_assessment:
                left, right = st.columns(2)
                with left:
                    st.markdown("**Advantages**")
                    st.write("\n".join(f"- {item}" for item in candidate.get("advantages", [])) or ", ".join(candidate.get("matched_skills", [])) or "No clear strengths found in the retrieved evidence.")
                with right:
                    st.markdown("**Gaps to clarify**")
                    st.write("\n".join(f"- {item}" for item in candidate.get("gaps", [])) or ", ".join(candidate.get("missing_skills", [])) or "No major gaps identified in the retrieved evidence.")
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
    with st.chat_message("assistant", avatar=str(ASSISTANT_AVATAR)):
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


def render_chat_input() -> None:
    """Render the input and queue new searches for the next UI pass."""

    chat_started = any(
        message.get("role") == "user"
        for message in st.session_state.messages
    )
    query = st.chat_input(
        "Type a recruiting request to start..."
        if not chat_started
        else "Ask about candidates, skills, or paste a job description..."
    )
    if not query or not query.strip():
        return

    query = query.strip()
    if is_greeting(query):
        st.session_state.messages.append({"role": "user", "content": query})
        greeting_reply = (
            "Hi, I’m Kira. I’m ready to help with candidates for the active job description. What would you like to know?"
            if st.session_state.job_description
            else "Hi, I’m Kira, Kokoro AI’s recruiting assistant. Ask me for a candidate profile or paste a job description, and I’ll search your resume library."
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

    is_new_jd = not st.session_state.job_description
    is_replacement_jd = bool(
        st.session_state.job_description and looks_like_job_description(query)
    )
    job_description_changed = bool(
        looks_like_job_description(query) and (is_new_jd or is_replacement_jd)
    )

    if job_description_changed:
        st.session_state.job_description = query
        st.session_state.session_job_descriptions[st.session_state.session_id] = query
        search_query = query
        user_message = {"role": "user", "content": query, "kind": "job_description"}
    elif st.session_state.job_description:
        search_query = (
            f"Job description:\n{st.session_state.job_description}\n\n"
            f"Recruiter question:\n{query}"
        )
        user_message = {"role": "user", "content": query}
    else:
        # A short candidate search can run on its own; a full JD is optional.
        search_query = query
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
            "Score the generated answer with RAGAS (uses Google Gemini)",
            value=settings.ENABLE_RAGAS,
            disabled=not settings.ENABLE_RAGAS,
            help="RAGAS scores faithfulness; adding a reference answer also enables context precision and recall.",
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
        :root { --brand:#5556d8; --ink:#1b2433; --muted:#687386; --line:#e4e7ef; --surface:#fff; }
        .stApp { background:#f6f7fb; color:var(--ink); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#eef0f7; border-right:1px solid #e0e4ed; }
        [data-testid="stSidebar"] > div { padding-top:1.35rem; }
        [data-testid="stSidebar"] h2 { letter-spacing:-.035em; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap:.4rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label { padding:.45rem .6rem; border-radius:10px; }
        .block-container { max-width:1120px; padding:1.7rem 2rem 4rem; }
        .kokoro-hero { display:flex; align-items:center; gap:14px; padding:4px 0 19px; border-bottom:1px solid var(--line); margin-bottom:23px; }
        .kokoro-title { color:var(--ink); font-size:28px; font-weight:760; letter-spacing:-.045em; line-height:1.2; }
        .kokoro-subtitle { color:var(--muted); font-size:14px; margin-top:5px; }
        .jd-prompt { display:flex; gap:14px; align-items:center; margin:12px 0 20px; padding:17px 19px; border:1px solid #dfe2f5; border-radius:15px; background:#fff; color:var(--ink); box-shadow:0 8px 24px #20234a08; }
        .jd-prompt-icon { display:grid; place-items:center; flex:0 0 40px; height:40px; border-radius:12px; color:var(--brand); background:#eeefff; font-size:20px; }
        .jd-prompt span { color:var(--muted); font-size:13px; }
        [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:16px; padding:14px 17px; background:var(--surface); box-shadow:0 3px 12px #20234a06; }
        [data-testid="stChatMessageAvatarUser"] { background:var(--brand) !important; color:#fff !important; }
        [data-testid="stChatInput"] { border-radius:14px; box-shadow:0 4px 18px #20234a0a; }
        [data-testid="stSidebar"] button, .stButton button, [data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:600; }
        .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { box-shadow:0 4px 12px #5556d826; }
        div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); padding:12px 14px; border-radius:13px; box-shadow:0 3px 12px #20234a06; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:15px; border-color:var(--line); background:var(--surface); }
        [data-testid="stExpander"] { border-color:var(--line); border-radius:12px; background:var(--surface); }
        input, textarea, [data-baseweb="select"] > div { border-radius:10px !important; }
        input:focus, textarea:focus { border-color:var(--brand) !important; box-shadow:0 0 0 2px #5556d820 !important; }
        hr { border-color:var(--line); }
        @media (max-width: 768px) {
          .block-container { padding:1rem 1rem 2.5rem; }
          .kokoro-hero { gap:12px; margin-bottom:18px; padding-bottom:16px; }
          .kokoro-title { font-size:23px; }
          .kokoro-subtitle { font-size:13px; }
          .jd-prompt { align-items:flex-start; padding:15px; border-radius:15px; }
          [data-testid="stChatMessage"] { padding:12px; border-radius:15px; }
          [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.65rem; }
          [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:min(100%, 240px); }
          div[data-testid="stMetric"] { padding:10px 12px; }
        }
        @media (max-width: 480px) {
          .block-container { padding:.75rem .75rem 2rem; }
          .kokoro-title { font-size:19px; }
          .kokoro-subtitle { font-size:12px; }
          .jd-prompt { gap:10px; padding:13px; }
          [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { overflow-wrap:anywhere; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """
    Streamlit application entry point.
    """

    st.set_page_config(
        page_title=settings.APP_NAME,
        page_icon=":material/person_search:",
        layout="wide",
    )
    initialize_session()

    render_styles()
    render_header()
    page = render_sidebar()

    # --------------------------------------------------------
    # Initialize existing resumes
    # --------------------------------------------------------

    initialize_resume_index()
    if page == "Evaluation":
        with st.container(border=True):
            st.markdown("### Evaluation workspace")
            st.write(
                "Measure hybrid retrieval against known relevant candidates, and "
                "optionally score generated answers with RAGAS."
            )
        render_evaluation_tab()
    else:
        with st.container(border=True):
            st.markdown("### Your recruiting workspace")
            st.write(
                "Search your resume library in plain language and review "
                "evidence-backed candidate matches in chat."
            )
        render_chat_history()
        if st.session_state.session_id in st.session_state.ended_sessions:
            st.info("This chat has ended. Choose **New chat** to start another session.")
        else:
            render_job_description_status()
            process_pending_search()
            render_chat_input()

    st.divider()
    st.caption("Designed and developed by Rupesh Kumar")


if __name__ == "__main__":
    main()
