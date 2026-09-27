"""
Shared utilities for Kokoro AI.

This module contains reusable helpers that are independent
of any specific business module.

Utilities include:
- Request ID generation
- SHA-256 hashing
- Text normalization
- Execution timing
- In-memory rate limiting
"""

import hashlib
import re
import time
import uuid
from collections import deque
from threading import Lock


# ============================================================
# Request ID
# ============================================================


def generate_request_id() -> str:
    """
    Generate a unique request identifier.

    Used for tracing a single recruiter request across
    ingestion, retrieval, reranking, generation, and logging.

    Returns
    -------
    str
        UUID-based request ID.
    """

    return str(uuid.uuid4())


# ============================================================
# SHA-256
# ============================================================


def calculate_sha256(data: bytes) -> str:
    """
    Calculate SHA-256 hash for binary data.

    This is used during document ingestion to identify
    duplicate PDF files.

    Parameters
    ----------
    data:
        Binary document content.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")

    return hashlib.sha256(data).hexdigest()


# ============================================================
# Text Normalization
# ============================================================


def normalize_text(text: str) -> str:
    """
    Perform safe text normalization.

    The goal is to improve retrieval quality while preserving
    useful resume/JD information such as:
    - headings
    - dates
    - acronyms
    - punctuation
    - readable sentence structure

    This function intentionally does NOT:
    - lowercase all text
    - remove stop words
    - perform stemming
    - remove punctuation aggressively
    - alter semantic content

    Parameters
    ----------
    text:
        Raw extracted document text.

    Returns
    -------
    str
        Normalized text.
    """

    if not text:
        return ""

    # Remove non-printable control characters while preserving
    # useful whitespace and newline characters.
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        " ",
        text,
    )

    # Normalize different newline representations.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse repeated spaces/tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Keep meaningful paragraph breaks but remove excessive ones.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# Execution Timing
# ============================================================


def elapsed_seconds(start_time: float) -> float:
    """
    Calculate elapsed execution time.

    Parameters
    ----------
    start_time:
        Value returned by time.perf_counter().

    Returns
    -------
    float
        Elapsed time rounded to four decimal places.
    """

    return round(
        time.perf_counter() - start_time,
        4,
    )


# ============================================================
# Rate Limiter
# ============================================================


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Example:
        limiter = RateLimiter(max_requests=5)

        if limiter.allow():
            # Make LLM request
            pass
        else:
            # Reject/throttle request
            pass

    This provides an application-level protection layer
    against excessive LLM requests.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: int = 60,
    ) -> None:
        """
        Initialize the rate limiter.

        Parameters
        ----------
        max_requests:
            Maximum number of requests allowed within
            the configured time window.

        window_seconds:
            Length of the sliding window.
        """

        if max_requests < 1:
            raise ValueError(
                "max_requests must be >= 1."
            )

        if window_seconds < 1:
            raise ValueError(
                "window_seconds must be >= 1."
            )

        self.max_requests = max_requests

        self.window_seconds = window_seconds

        self._timestamps: deque[float] = deque()

        self._lock = Lock()

    def allow(self) -> bool:
        """
        Check whether another request is allowed.

        Returns
        -------
        bool
            True if the request is allowed.
            False if the rate limit has been reached.
        """

        now = time.monotonic()

        with self._lock:

            cutoff = now - self.window_seconds

            # Remove timestamps outside the current
            # sliding window.
            while (
                self._timestamps
                and self._timestamps[0] <= cutoff
            ):
                self._timestamps.popleft()

            # Reject if the limit has been reached.
            if len(self._timestamps) >= self.max_requests:
                return False

            # Register the new request.
            self._timestamps.append(now)

            return True