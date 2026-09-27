"""
utils/logger.py

Centralized logging for Kokoro.

Purpose:
    - Provide one logging mechanism across the application.
    - Show logs in the VS Code/backend terminal.
    - Support detailed debugging during development.
    - Log errors with complete tracebacks.
    - Optionally log resume text, chunks, prompts, and LLM responses.
    - Keep Streamlit UI completely separate from backend logs.

Important:
    This module does NOT display anything in Streamlit.

    Logs are written to:
        1. VS Code / backend terminal
        2. Optional log file, when enabled

Detailed content logging is controlled through configuration.

Recommended development settings:

    LOG_LEVEL=DEBUG
    LOG_RESUME_PII=true
    LOG_FULL_PROMPTS=true
    LOG_FULL_RESPONSES=true

For production, these should normally be disabled.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional


# ============================================================
# CONSTANTS
# ============================================================

LOGGER_NAME = "kokoro"

DEFAULT_LOG_LEVEL = "INFO"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOG_DIRECTORY = "logs"

DEFAULT_LOG_FILE = "kokoro.log"


# ============================================================
# INTERNAL STATE
# ============================================================

_configured = False


# ============================================================
# LOG LEVEL
# ============================================================


def _get_log_level(
    level: Optional[str] = None,
) -> int:
    """
    Convert a string log level to Python logging level.

    Supported levels:

        DEBUG
        INFO
        WARNING
        ERROR
        CRITICAL
    """

    level = (
        level or DEFAULT_LOG_LEVEL
    ).strip().upper()

    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }

    if level not in levels:
        raise ValueError(
            f"Unsupported log level: {level}"
        )

    return levels[level]


# ============================================================
# LOGGER CONFIGURATION
# ============================================================


def configure_logger(
    level: str = DEFAULT_LOG_LEVEL,
    enable_file_logging: bool = False,
    log_directory: str = DEFAULT_LOG_DIRECTORY,
    log_file: str = DEFAULT_LOG_FILE,
) -> logging.Logger:
    """
    Configure the main Kokoro logger.

    Logs are always sent to the terminal.

    File logging is optional.

    Args:
        level:
            Logging level.

        enable_file_logging:
            Whether logs should also be written to a file.

        log_directory:
            Directory used for log files.

        log_file:
            Log filename.

    Returns:
        Configured Kokoro logger.
    """

    global _configured

    logger = logging.getLogger(
        LOGGER_NAME
    )

    log_level = _get_log_level(level)

    logger.setLevel(log_level)

    # Prevent logs from being sent to the root logger.
    # This avoids duplicate messages.
    logger.propagate = False

    # --------------------------------------------------------
    # Configure only once.
    # --------------------------------------------------------

    if not _configured:

        # ----------------------------------------------------
        # Terminal handler
        # ----------------------------------------------------

        console_handler = (
            logging.StreamHandler(
                sys.stdout
            )
        )

        console_handler.setLevel(
            log_level
        )

        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )

        console_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            console_handler
        )

        # ----------------------------------------------------
        # Optional file handler
        # ----------------------------------------------------

        if enable_file_logging:

            Path(log_directory).mkdir(
                parents=True,
                exist_ok=True,
            )

            log_path = (
                Path(log_directory)
                / log_file
            )

            file_handler = (
                logging.FileHandler(
                    log_path,
                    encoding="utf-8",
                )
            )

            file_handler.setLevel(
                log_level
            )

            file_handler.setFormatter(
                formatter
            )

            logger.addHandler(
                file_handler
            )

        _configured = True

    else:

        # Update existing handlers if configure_logger()
        # is called again.
        for handler in logger.handlers:
            handler.setLevel(
                log_level
            )

    return logger


# ============================================================
# LOGGER FACTORY
# ============================================================


def get_logger(
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Return a logger for a specific Kokoro module.

    Example:

        logger = get_logger(__name__)

    If the module is:

        core.ingestion.ingestion

    the logger becomes:

        kokoro.core.ingestion.ingestion
    """

    root_logger = configure_logger()

    if not name:
        return root_logger

    if name.startswith(
        f"{LOGGER_NAME}."
    ):
        logger_name = name
    else:
        logger_name = (
            f"{LOGGER_NAME}.{name}"
        )

    return logging.getLogger(
        logger_name
    )


# ============================================================
# BASIC LOGGING
# ============================================================


def log_debug(
    logger: logging.Logger,
    message: str,
    *args: Any,
) -> None:
    """
    Log a DEBUG message.
    """

    logger.debug(
        message,
        *args,
    )


def log_info(
    logger: logging.Logger,
    message: str,
    *args: Any,
) -> None:
    """
    Log an INFO message.
    """

    logger.info(
        message,
        *args,
    )


def log_warning(
    logger: logging.Logger,
    message: str,
    *args: Any,
) -> None:
    """
    Log a WARNING message.
    """

    logger.warning(
        message,
        *args,
    )


def log_error(
    logger: logging.Logger,
    message: str,
    *args: Any,
) -> None:
    """
    Log an ERROR message.
    """

    logger.error(
        message,
        *args,
    )


def log_exception(
    logger: logging.Logger,
    message: str,
    exception: Optional[Exception] = None,
) -> None:
    """
    Log an exception with its complete traceback.

    Example:

        try:
            ...
        except Exception as exc:
            log_exception(
                logger,
                "Resume parsing failed",
                exc,
            )
    """

    if exception is not None:

        logger.error(
            "%s | %s: %s",
            message,
            type(exception).__name__,
            str(exception),
            exc_info=True,
        )

    else:

        logger.exception(
            message
        )


# ============================================================
# REQUEST LOGGING
# ============================================================


def log_request_start(
    logger: logging.Logger,
    request_id: str,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
) -> None:
    """
    Log the beginning of a request.

    The full query is logged only when DEBUG logging
    is enabled.

    This is useful during local development.
    """

    logger.info(
        "Request started | request_id=%s "
        "| session_id=%s",
        request_id,
        session_id,
    )

    if (
        query is not None
        and logger.isEnabledFor(
            logging.DEBUG
        )
    ):
        logger.debug(
            "Request query | request_id=%s\n%s",
            request_id,
            query,
        )


def log_request_complete(
    logger: logging.Logger,
    request_id: str,
    latency_ms: float,
) -> None:
    """
    Log successful request completion.
    """

    logger.info(
        "Request completed | request_id=%s "
        "| latency_ms=%.2f",
        request_id,
        latency_ms,
    )


def log_request_error(
    logger: logging.Logger,
    request_id: str,
    exception: Exception,
    latency_ms: Optional[float] = None,
) -> None:
    """
    Log request failure with traceback.
    """

    if latency_ms is not None:

        logger.error(
            "Request failed | request_id=%s "
            "| latency_ms=%.2f | error=%s",
            request_id,
            latency_ms,
            str(exception),
            exc_info=True,
        )

    else:

        logger.error(
            "Request failed | request_id=%s "
            "| error=%s",
            request_id,
            str(exception),
            exc_info=True,
        )


# ============================================================
# RESUME / DOCUMENT LOGGING
# ============================================================


def log_resume_text(
    logger: logging.Logger,
    document_id: str,
    resume_text: str,
) -> None:
    """
    Log complete resume text for debugging.

    IMPORTANT:
        This is intended for local development/debugging.

    The function logs to the backend logger only.

    It does NOT send anything to Streamlit.

    Whether this function actually logs is controlled by
    the logger level. DEBUG must be enabled.
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== RESUME TEXT ==========\n"
        "document_id=%s\n"
        "%s\n"
        "========== END RESUME TEXT ==========",
        document_id,
        resume_text,
    )


def log_document_text(
    logger: logging.Logger,
    document_id: str,
    document_text: str,
) -> None:
    """
    Log complete parsed document text.

    Only active at DEBUG level.
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== DOCUMENT TEXT ==========\n"
        "document_id=%s\n"
        "%s\n"
        "========== END DOCUMENT TEXT ==========",
        document_id,
        document_text,
    )


# ============================================================
# CHUNK LOGGING
# ============================================================


def log_chunk(
    logger: logging.Logger,
    chunk_id: str,
    document_id: str,
    chunk_text: str,
    section: Optional[str] = None,
) -> None:
    """
    Log complete chunk content for debugging.

    Useful for debugging:
        - chunk size
        - chunk boundaries
        - section-aware chunking
        - overlap
        - preprocessing
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== CHUNK ==========\n"
        "chunk_id=%s\n"
        "document_id=%s\n"
        "section=%s\n"
        "%s\n"
        "========== END CHUNK ==========",
        chunk_id,
        document_id,
        section,
        chunk_text,
    )


def log_chunk_count(
    logger: logging.Logger,
    document_id: str,
    chunk_count: int,
) -> None:
    """
    Log number of generated chunks.
    """

    logger.info(
        "Chunking completed | document_id=%s "
        "| chunk_count=%d",
        document_id,
        chunk_count,
    )


# ============================================================
# INGESTION LOGGING
# ============================================================


def log_ingestion_start(
    logger: logging.Logger,
    document_id: str,
    source_file: str,
) -> None:
    """
    Log ingestion start.
    """

    logger.info(
        "Ingestion started | document_id=%s "
        "| source_file=%s",
        document_id,
        source_file,
    )


def log_ingestion_complete(
    logger: logging.Logger,
    document_id: str,
    chunk_count: int,
    latency_ms: float,
) -> None:
    """
    Log successful ingestion.
    """

    logger.info(
        "Ingestion completed | document_id=%s "
        "| chunks=%d | latency_ms=%.2f",
        document_id,
        chunk_count,
        latency_ms,
    )


def log_ingestion_skipped(
    logger: logging.Logger,
    document_id: str,
    reason: str,
) -> None:
    """
    Log skipped ingestion.

    Example:
        duplicate document hash
    """

    logger.warning(
        "Ingestion skipped | document_id=%s "
        "| reason=%s",
        document_id,
        reason,
    )


def log_ingestion_error(
    logger: logging.Logger,
    document_id: str,
    exception: Exception,
) -> None:
    """
    Log ingestion failure.
    """

    logger.error(
        "Ingestion failed | document_id=%s "
        "| error=%s",
        document_id,
        str(exception),
        exc_info=True,
    )


# ============================================================
# EMBEDDING LOGGING
# ============================================================


def log_embedding(
    logger: logging.Logger,
    document_id: str,
    chunk_count: int,
    model: str,
    latency_ms: float,
) -> None:
    """
    Log embedding operation.
    """

    logger.info(
        "Embedding completed | document_id=%s "
        "| chunks=%d | model=%s "
        "| latency_ms=%.2f",
        document_id,
        chunk_count,
        model,
        latency_ms,
    )


# ============================================================
# PINECONE LOGGING
# ============================================================


def log_pinecone_upsert(
    logger: logging.Logger,
    document_id: str,
    vector_count: int,
    latency_ms: float,
) -> None:
    """
    Log Pinecone upsert.
    """

    logger.info(
        "Pinecone upsert | document_id=%s "
        "| vectors=%d | latency_ms=%.2f",
        document_id,
        vector_count,
        latency_ms,
    )


def log_pinecone_search(
    logger: logging.Logger,
    query: str,
    result_count: int,
    latency_ms: float,
    results: Optional[Any] = None,
) -> None:
    """
    Log Pinecone search.

    At DEBUG level, the query and optional results are logged.

    This is useful for debugging semantic retrieval.
    """

    logger.info(
        "Pinecone search | results=%d "
        "| latency_ms=%.2f",
        result_count,
        latency_ms,
    )

    if logger.isEnabledFor(
        logging.DEBUG
    ):

        logger.debug(
            "Pinecone query:\n%s",
            query,
        )

        if results is not None:

            logger.debug(
                "Pinecone results:\n%s",
                results,
            )


# ============================================================
# BM25 LOGGING
# ============================================================


def log_bm25_search(
    logger: logging.Logger,
    query: str,
    result_count: int,
    latency_ms: float,
    results: Optional[Any] = None,
) -> None:
    """
    Log BM25 lexical retrieval.

    Full query/results are available in DEBUG mode.
    """

    logger.info(
        "BM25 search | results=%d "
        "| latency_ms=%.2f",
        result_count,
        latency_ms,
    )

    if logger.isEnabledFor(
        logging.DEBUG
    ):

        logger.debug(
            "BM25 query:\n%s",
            query,
        )

        if results is not None:

            logger.debug(
                "BM25 results:\n%s",
                results,
            )


# ============================================================
# HYBRID RETRIEVAL LOGGING
# ============================================================


def log_hybrid_search(
    logger: logging.Logger,
    pinecone_count: int,
    bm25_count: int,
    final_count: int,
    latency_ms: float,
) -> None:
    """
    Log hybrid retrieval statistics.
    """

    logger.info(
        "Hybrid search | pinecone=%d "
        "| bm25=%d | final=%d "
        "| latency_ms=%.2f",
        pinecone_count,
        bm25_count,
        final_count,
        latency_ms,
    )


# ============================================================
# RERANKING LOGGING
# ============================================================


def log_reranker_input(
    logger: logging.Logger,
    query: str,
    candidates: Any,
) -> None:
    """
    Log reranker input at DEBUG level.

    Useful for debugging why Gemini ranked candidates
    in a particular order.
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== RERANKER INPUT ==========\n"
        "QUERY:\n%s\n\n"
        "CANDIDATES:\n%s\n"
        "========== END RERANKER INPUT ==========",
        query,
        candidates,
    )


def log_reranker_output(
    logger: logging.Logger,
    response: Any,
) -> None:
    """
    Log reranker output at DEBUG level.
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== RERANKER OUTPUT ==========\n"
        "%s\n"
        "========== END RERANKER OUTPUT ==========",
        response,
    )


def log_reranking_complete(
    logger: logging.Logger,
    input_count: int,
    output_count: int,
    latency_ms: float,
) -> None:
    """
    Log reranking statistics.
    """

    logger.info(
        "Reranking completed | input=%d "
        "| output=%d | latency_ms=%.2f",
        input_count,
        output_count,
        latency_ms,
    )


# ============================================================
# LLM LOGGING
# ============================================================


def log_llm_prompt(
    logger: logging.Logger,
    operation: str,
    model: str,
    prompt: str,
) -> None:
    """
    Log the complete LLM prompt at DEBUG level.

    This is intentionally backend-only.

    It is NOT displayed in Streamlit.
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== LLM PROMPT ==========\n"
        "operation=%s\n"
        "model=%s\n\n"
        "%s\n"
        "========== END LLM PROMPT ==========",
        operation,
        model,
        prompt,
    )


def log_llm_response(
    logger: logging.Logger,
    operation: str,
    model: str,
    response: Any,
) -> None:
    """
    Log the complete LLM response at DEBUG level.

    This is backend-only.
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== LLM RESPONSE ==========\n"
        "operation=%s\n"
        "model=%s\n\n"
        "%s\n"
        "========== END LLM RESPONSE ==========",
        operation,
        model,
        response,
    )


def log_llm_call(
    logger: logging.Logger,
    operation: str,
    model: str,
    latency_ms: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """
    Log LLM call statistics.
    """

    logger.info(
        "LLM call | operation=%s "
        "| model=%s "
        "| latency_ms=%.2f "
        "| input_tokens=%s "
        "| output_tokens=%s",
        operation,
        model,
        latency_ms,
        input_tokens,
        output_tokens,
    )


def log_llm_error(
    logger: logging.Logger,
    operation: str,
    model: str,
    exception: Exception,
) -> None:
    """
    Log LLM failure with traceback.
    """

    logger.error(
        "LLM call failed | operation=%s "
        "| model=%s | error=%s",
        operation,
        model,
        str(exception),
        exc_info=True,
    )


# ============================================================
# PROMPT BUILDER LOGGING
# ============================================================


def log_prompt_context(
    logger: logging.Logger,
    query: str,
    retrieved_context: Any,
    memory_context: Any = None,
    jd_context: Any = None,
) -> None:
    """
    Log the complete prompt-building context at DEBUG level.

    This is useful when debugging RAG context assembly.
    """

    if not logger.isEnabledFor(
        logging.DEBUG
    ):
        return

    logger.debug(
        "========== PROMPT CONTEXT ==========\n"
        "QUERY:\n%s\n\n"
        "RETRIEVED CONTEXT:\n%s\n\n"
        "MEMORY CONTEXT:\n%s\n\n"
        "JD CONTEXT:\n%s\n"
        "========== END PROMPT CONTEXT ==========",
        query,
        retrieved_context,
        memory_context,
        jd_context,
    )


# ============================================================
# MEMORY LOGGING
# ============================================================


def log_memory_operation(
    logger: logging.Logger,
    session_id: str,
    operation: str,
    item_count: int,
) -> None:
    """
    Log memory operations.

    The actual memory contents are logged separately only
    when DEBUG-level detailed debugging is intentionally added
    by the calling module.
    """

    logger.debug(
        "Memory operation | session_id=%s "
        "| operation=%s | items=%d",
        session_id,
        operation,
        item_count,
    )


# ============================================================
# GUARDRAIL LOGGING
# ============================================================


def log_guardrail_result(
    logger: logging.Logger,
    guardrail_name: str,
    status: str,
    details: Optional[Any] = None,
) -> None:
    """
    Log guardrail result.

    Detailed information is logged only at DEBUG level.
    """

    logger.info(
        "Guardrail | name=%s | status=%s",
        guardrail_name,
        status,
    )

    if (
        details is not None
        and logger.isEnabledFor(
            logging.DEBUG
        )
    ):

        logger.debug(
            "Guardrail details:\n%s",
            details,
        )


# ============================================================
# VALIDATION LOGGING
# ============================================================


def log_validation_result(
    logger: logging.Logger,
    validation_name: str,
    status: str,
    details: Optional[Any] = None,
) -> None:
    """
    Log validation result.
    """

    logger.info(
        "Validation | name=%s | status=%s",
        validation_name,
        status,
    )

    if (
        details is not None
        and logger.isEnabledFor(
            logging.DEBUG
        )
    ):

        logger.debug(
            "Validation details:\n%s",
            details,
        )


# ============================================================
# CACHE LOGGING
# ============================================================


def log_cache_hit(
    logger: logging.Logger,
    cache_key: str,
) -> None:
    """
    Log cache hit.

    Only a cache key is logged, not the cached response.
    """

    logger.debug(
        "Cache hit | key=%s",
        cache_key,
    )


def log_cache_miss(
    logger: logging.Logger,
    cache_key: str,
) -> None:
    """
    Log cache miss.
    """

    logger.debug(
        "Cache miss | key=%s",
        cache_key,
    )


# ============================================================
# EVALUATION LOGGING
# ============================================================


def log_evaluation_start(
    logger: logging.Logger,
    evaluation_id: str,
    dataset_size: int,
) -> None:
    """
    Log evaluation start.
    """

    logger.info(
        "Evaluation started | evaluation_id=%s "
        "| dataset_size=%d",
        evaluation_id,
        dataset_size,
    )


def log_evaluation_complete(
    logger: logging.Logger,
    evaluation_id: str,
    latency_ms: float,
) -> None:
    """
    Log evaluation completion.
    """

    logger.info(
        "Evaluation completed | evaluation_id=%s "
        "| latency_ms=%.2f",
        evaluation_id,
        latency_ms,
    )


# ============================================================
# PERFORMANCE TIMER
# ============================================================


class LogTimer:
    """
    Context manager for measuring operation latency.

    Example:

        with LogTimer(logger, "Pinecone search"):
            results = search(...)
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.DEBUG,
    ) -> None:

        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[float] = None

    def __enter__(self) -> "LogTimer":

        self.start_time = time.perf_counter()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        if self.start_time is None:
            return

        elapsed_ms = (
            time.perf_counter()
            - self.start_time
        ) * 1000

        self.logger.log(
            self.level,
            "Operation completed | "
            "operation=%s | latency_ms=%.2f",
            self.operation,
            elapsed_ms,
        )


# ============================================================
# APPLICATION LOGGING
# ============================================================


def log_application_start(
    logger: logging.Logger,
) -> None:
    """
    Log Kokoro startup.
    """

    logger.info(
        "========== KOKORO STARTED =========="
    )


def log_application_shutdown(
    logger: logging.Logger,
) -> None:
    """
    Log Kokoro shutdown.
    """

    logger.info(
        "========== KOKORO SHUTDOWN =========="
    )


# ============================================================
# DEFAULT LOGGER
# ============================================================


logger = configure_logger()