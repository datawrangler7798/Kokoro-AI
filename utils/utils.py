"""
utils/utils.py

Common reusable utility functions for Kokoro.

This module contains generic helper functions that are shared
across ingestion, retrieval, evaluation, caching, and application
layers.

Responsibilities:
    - File hashing
    - File validation
    - Text normalization
    - ID generation
    - Safe JSON operations
    - Directory handling
    - Timing helpers

Important:
    This module must remain independent of Streamlit, Pinecone,
    Gemini, BM25, and other business-specific components.

Business logic belongs in the appropriate core module.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


class RateLimiter:
    """Thread-safe sliding-window limiter for provider requests."""

    def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
        if max_calls <= 0 or window_seconds <= 0:
            raise ValueError("Rate limit and window must be positive.")
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window_seconds:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait_for = self.window_seconds - (now - self._calls[0])
            time.sleep(max(0.01, wait_for))


_llm_rate_limiter: RateLimiter | None = None
_llm_rate_limiter_lock = threading.Lock()


def get_llm_rate_limiter() -> RateLimiter:
    """Return the shared Gemini request limiter."""
    global _llm_rate_limiter
    if _llm_rate_limiter is None:
        with _llm_rate_limiter_lock:
            if _llm_rate_limiter is None:
                from utils.config import get_settings

                _llm_rate_limiter = RateLimiter(
                    max_calls=get_settings().LLM_REQUESTS_PER_MINUTE,
                    window_seconds=60.0,
                )
    return _llm_rate_limiter


# ============================================================
# FILE UTILITIES
# ============================================================


def calculate_file_hash(
    file_path: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate a cryptographic hash for a file.

    Kokoro uses SHA-256 for document duplicate detection.

    Args:
        file_path:
            Path to the file.

        algorithm:
            Hash algorithm supported by hashlib.

        chunk_size:
            Number of bytes read at a time.

    Returns:
        Hexadecimal hash string.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        ValueError:
            If the requested hashing algorithm is unavailable.

    Example:
        document_hash = calculate_file_hash(
            "data/resumes/resume.pdf"
        )
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        hash_function = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    with path.open("rb") as file:
        while True:
            data = file.read(chunk_size)

            if not data:
                break

            hash_function.update(data)

    return hash_function.hexdigest()


def get_file_size_bytes(
    file_path: str | Path,
) -> int:
    """
    Return the file size in bytes.

    Args:
        file_path:
            Path to the file.

    Returns:
        File size in bytes.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return path.stat().st_size


def get_file_extension(
    file_path: str | Path,
) -> str:
    """
    Return a normalized lowercase file extension.

    Example:
        resume.PDF → ".pdf"
    """

    return Path(file_path).suffix.lower()


def is_allowed_file_extension(
    file_path: str | Path,
    allowed_extensions: list[str],
) -> bool:
    """
    Check whether a file has an allowed extension.

    Args:
        file_path:
            File path.

        allowed_extensions:
            Example:
                [".pdf"]

    Returns:
        True if the extension is allowed.
    """

    extension = get_file_extension(file_path)

    normalized_extensions = {
        value.strip().lower() for value in allowed_extensions if value and value.strip()
    }

    return extension in normalized_extensions


def validate_file_size(
    file_path: str | Path,
    max_size_mb: int,
) -> None:
    """
    Validate that a file does not exceed the configured
    maximum size.

    Raises:
        ValueError:
            If the file is larger than the allowed limit.
    """

    if max_size_mb <= 0:
        raise ValueError("max_size_mb must be greater than 0.")

    size_bytes = get_file_size_bytes(file_path)

    max_size_bytes = max_size_mb * 1024 * 1024

    if size_bytes > max_size_bytes:
        raise ValueError(f"File exceeds maximum allowed size of " f"{max_size_mb} MB.")


def validate_input_file(
    file_path: str | Path,
    allowed_extensions: list[str],
    max_size_mb: int,
) -> None:
    """
    Perform common file validation.

    Validation includes:

        1. File existence
        2. Extension
        3. File size
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    if not is_allowed_file_extension(
        path,
        allowed_extensions,
    ):
        raise ValueError(f"Unsupported file extension: " f"{get_file_extension(path)}")

    validate_file_size(
        path,
        max_size_mb,
    )


# ============================================================
# DIRECTORY UTILITIES
# ============================================================


def ensure_directory(
    directory: str | Path,
) -> Path:
    """
    Create a directory if it does not exist.

    Args:
        directory:
            Directory path.

    Returns:
        Path object representing the directory.
    """

    path = Path(directory)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def ensure_parent_directory(
    file_path: str | Path,
) -> Path:
    """
    Create the parent directory for a file if required.

    Returns:
        Parent directory Path.
    """

    path = Path(file_path)

    parent = path.parent

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return parent


# ============================================================
# TEXT UTILITIES
# ============================================================


def normalize_whitespace(
    text: str,
) -> str:
    """
    Normalize whitespace while preserving readable text.

    Multiple spaces, tabs, and repeated line breaks are
    reduced appropriately.

    Args:
        text:
            Input text.

    Returns:
        Normalized text.
    """

    if not text:
        return ""

    # Normalize common whitespace characters.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # Remove trailing spaces from each line.
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]

    # Collapse excessive blank lines while preserving
    # section readability.
    normalized_lines: list[str] = []

    previous_blank = False

    for line in lines:
        line = line.strip()

        if not line:
            if not previous_blank:
                normalized_lines.append("")

            previous_blank = True
            continue

        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


def normalize_text(
    text: str,
) -> str:
    """
    Perform lightweight text normalization.

    This intentionally does NOT perform aggressive cleaning.

    Resume information such as:

        C++
        C#
        .NET
        AWS
        GCP
        Python 3.x
        SQL

    must remain readable.

    Heavy preprocessing belongs in the ingestion/parser layer.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    return normalize_whitespace(text)


def clean_string(
    value: str | None,
) -> str | None:
    """
    Strip whitespace from a string.

    Empty strings are converted to None.
    """

    if value is None:
        return None

    value = value.strip()

    return value if value else None


def clean_string_list(
    values: list[str] | None,
) -> list[str]:
    """
    Normalize a list of strings.

    Empty values are removed.
    """

    if not values:
        return []

    cleaned: list[str] = []

    for value in values:
        value = clean_string(value)

        if value is not None:
            cleaned.append(value)

    return cleaned


# ============================================================
# ID UTILITIES
# ============================================================


def generate_uuid() -> str:
    """
    Generate a UUID4 string.

    Used for request IDs, session IDs, document IDs,
    and other runtime identifiers where appropriate.
    """

    return str(uuid.uuid4())


def generate_request_id() -> str:
    """
    Generate a request identifier.
    """

    return f"req_{generate_uuid()}"


def generate_session_id() -> str:
    """
    Generate a session identifier.
    """

    return f"session_{generate_uuid()}"


def generate_document_id(
    document_hash: str,
) -> str:
    """
    Generate a deterministic document ID from a document hash.

    Using the document hash makes the identifier stable for
    the same document.

    Args:
        document_hash:
            SHA-256 document hash.

    Returns:
        Deterministic document identifier.
    """

    if not document_hash:
        raise ValueError("document_hash cannot be empty.")

    return f"doc_{document_hash[:24]}"


def generate_chunk_id(
    document_id: str,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic chunk ID.

    Example:

        doc_abc123_chunk_000001
    """

    if not document_id:
        raise ValueError("document_id cannot be empty.")

    if chunk_index < 0:
        raise ValueError("chunk_index cannot be negative.")

    return f"{document_id}" f"_chunk_{chunk_index:06d}"


def generate_candidate_id(
    document_id: str,
) -> str:
    """
    Generate a candidate ID from a resume document ID.

    Candidate identity can later be handled separately if
    Kokoro supports multiple resume versions per candidate.
    """

    if not document_id:
        raise ValueError("document_id cannot be empty.")

    return f"candidate_{document_id.replace('doc_', '')}"


# ============================================================
# JSON UTILITIES
# ============================================================


def load_json(
    file_path: str | Path,
    default: Any = None,
) -> Any:
    """
    Load JSON from a file.

    Args:
        file_path:
            JSON file path.

        default:
            Value returned when the file does not exist.

    Returns:
        Parsed JSON object or default value.
    """

    path = Path(file_path)

    if not path.exists():
        return default

    if not path.is_file():
        raise ValueError(f"Expected a file but found: {path}")

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def save_json(
    file_path: str | Path,
    data: Any,
    indent: int = 2,
) -> None:
    """
    Save an object as JSON.

    Parent directories are automatically created.

    Args:
        file_path:
            Destination JSON file.

        data:
            JSON-serializable object.

        indent:
            JSON indentation.
    """

    path = Path(file_path)

    ensure_parent_directory(path)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=indent,
                ensure_ascii=False,
            )

        # Replace destination only after successful write.
        temporary_path.replace(path)

    except TypeError as exc:
        if temporary_path.exists():
            temporary_path.unlink()

        raise ValueError("Data is not JSON serializable.") from exc

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


# ============================================================
# HASH UTILITIES
# ============================================================


def calculate_text_hash(
    text: str,
    algorithm: str = "sha256",
) -> str:
    """
    Calculate a hash for text.

    Useful for:

        - cache keys
        - query fingerprints
        - deduplication
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    try:
        hash_function = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    hash_function.update(text.encode("utf-8"))

    return hash_function.hexdigest()


def create_cache_key(
    query: str,
    session_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    Create a deterministic cache key.

    The key includes:

        - query
        - optional session ID
        - optional additional parameters

    This prevents logically different requests from sharing
    the same cached response.

    Note:
        Cache behavior itself belongs to the cache/application
        layer. This function only creates the key.
    """

    payload = {
        "query": query.strip(),
        "session_id": session_id,
        "extra": extra or {},
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        default=str,
    )

    return calculate_text_hash(serialized)


# ============================================================
# SAFE PATH UTILITIES
# ============================================================


def sanitize_filename(
    filename: str,
) -> str:
    """
    Sanitize a filename before saving an uploaded file.

    This prevents path traversal characters from being used
    directly as part of a filesystem path.

    Example:

        ../../resume.pdf

    becomes:

        resume.pdf
    """

    if not filename:
        raise ValueError("Filename cannot be empty.")

    # Remove directory components.
    filename = Path(filename).name

    # Replace unsafe characters.
    filename = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        filename,
    )

    # Avoid hidden/relative names.
    filename = filename.lstrip(".")

    if not filename:
        raise ValueError("Filename became empty after sanitization.")

    return filename


def build_safe_file_path(
    directory: str | Path,
    filename: str,
) -> Path:
    """
    Build a safe path for an uploaded file.

    The filename is sanitized before joining it to the
    destination directory.
    """

    directory_path = ensure_directory(directory)

    safe_filename = sanitize_filename(filename)

    return directory_path / safe_filename


# ============================================================
# ITERATION UTILITIES
# ============================================================


def batch_items(
    items: list[Any],
    batch_size: int,
) -> Iterator[list[Any]]:
    """
    Yield a list in smaller batches.

    Useful for:

        - embedding batches
        - Pinecone upserts
        - large ingestion jobs

    Args:
        items:
            Input list.

        batch_size:
            Maximum number of items per batch.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[start : start + batch_size]


# ============================================================
# TIMING UTILITIES
# ============================================================


@contextmanager
def measure_time() -> Iterator[dict[str, float]]:
    """
    Context manager for measuring elapsed time.

    Example:

        with measure_time() as timer:
            run_retrieval()

        print(timer["elapsed_ms"])

    Returns:
        Dictionary containing elapsed milliseconds.
    """

    start = perf_counter()

    result: dict[str, float] = {
        "elapsed_ms": 0.0,
    }

    try:
        yield result

    finally:
        elapsed_seconds = perf_counter() - start

        result["elapsed_ms"] = elapsed_seconds * 1000


def measure_function_time(
    function,
    *args,
    **kwargs,
) -> tuple[Any, float]:
    """
    Execute a function and return:

        (function_result, elapsed_ms)

    This is useful when a caller needs both the result and
    execution latency.
    """

    start = perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    elapsed_ms = (perf_counter() - start) * 1000

    return result, elapsed_ms


# ============================================================
# NUMERIC UTILITIES
# ============================================================


def normalize_score(
    score: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Min-max normalize a score into the range [0, 1].

    Formula:

        (score - minimum)
        ------------------
        (maximum - minimum)

    If minimum == maximum, return 1.0.

    This helper can be used by hybrid retrieval score
    normalization.
    """

    if maximum < minimum:
        raise ValueError("maximum cannot be smaller than minimum.")

    if minimum == maximum:
        return 1.0

    normalized = (score - minimum) / (maximum - minimum)

    return max(
        0.0,
        min(1.0, normalized),
    )


def min_max_normalize_scores(
    scores: list[float],
) -> list[float]:
    """
    Normalize a list of scores using min-max normalization.

    Example:

        [10, 20, 30]

    becomes:

        [0.0, 0.5, 1.0]
    """

    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    return [
        normalize_score(
            score,
            minimum,
            maximum,
        )
        for score in scores
    ]


# ============================================================
# TEXT TOKEN/WORD HELPERS
# ============================================================


def estimate_word_count(
    text: str,
) -> int:
    """
    Estimate the number of words in text.

    This is intentionally a simple word-count utility.

    It is NOT a tokenizer and should not be used for exact
    LLM token counting.
    """

    if not text:
        return 0

    return len(
        re.findall(
            r"\S+",
            text,
        )
    )


def truncate_text(
    text: str,
    max_characters: int,
) -> str:
    """
    Truncate text to a maximum number of characters.

    This helper is useful for logs and diagnostics.

    It should not be used to truncate evidence before
    generation unless explicitly intended.
    """

    if max_characters <= 0:
        raise ValueError("max_characters must be greater than 0.")

    if len(text) <= max_characters:
        return text

    return text[:max_characters] + "..."


# ============================================================
# BOOLEAN / CONFIGURATION HELPERS
# ============================================================


def parse_bool(
    value: bool | str,
) -> bool:
    """
    Convert common string representations into a boolean.

    Supported true values:

        true
        1
        yes
        y
        on

    Supported false values:

        false
        0
        no
        n
        off
    """

    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "y",
        "on",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "n",
        "off",
    }:
        return False

    raise ValueError(f"Invalid boolean value: {value}")


# ============================================================
# DICTIONARY UTILITIES
# ============================================================


def remove_none_values(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove dictionary entries whose values are None.

    Nested dictionaries are not recursively modified.
    """

    return {key: value for key, value in data.items() if value is not None}


def merge_dicts(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a new dictionary containing values from both
    dictionaries.

    Values in `updates` take precedence.

    The original dictionaries are not modified.
    """

    result = dict(base)

    result.update(updates)

    return result
