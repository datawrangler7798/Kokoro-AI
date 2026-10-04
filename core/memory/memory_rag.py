"""
Kokoro - Memory and Cache RAG Support.

Responsibilities:
    - Maintain bounded conversation memory per session.
    - Retrieve relevant previous conversation turns.
    - Format memory for prompt construction.
    - Provide deterministic response caching.
    - Invalidate cache entries when source data changes.
    - Manage cache expiration and cleanup.

Important:
    - Memory is NOT the source of truth for candidate information.
    - Current retrieved evidence must be used for factual candidate claims.
    - This module does NOT perform vector retrieval.
    - This module does NOT call the LLM.
    - This module does NOT interact with Streamlit.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from utils.config import get_settings
from utils.logger import logger

# ============================================================
# Internal Records
# ============================================================


@dataclass
class MemoryRecord:
    """
    Internal representation of one conversation memory item.
    """

    session_id: str
    user_query: str
    assistant_response: str
    timestamp: float

    candidate_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    jd_ids: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheRecord:
    """
    Internal response-cache record.
    """

    key: str
    response: Any
    created_at: float
    expires_at: float

    session_id: str | None = None

    document_ids: list[str] = field(default_factory=list)
    jd_ids: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Memory + Cache
# ============================================================


class MemoryRAG:
    """
    Session memory and response-cache manager.

    Memory:
        Stores recent conversation turns for a session.

    Relevance:
        Uses lightweight lexical matching plus recency.

    Cache:
        Stores completed responses using deterministic cache keys.

    This class is intentionally independent from:
        - Pinecone
        - BM25
        - Gemini
        - Streamlit
        - retrieval routing
    """

    def __init__(
        self,
        *,
        max_memory_items: int | None = None,
        cache_ttl_seconds: int | None = None,
        cache_enabled: bool | None = None,
    ) -> None:
        self.settings = get_settings()

        self.max_memory_items = (
            max_memory_items
            if max_memory_items is not None
            else self.settings.MAX_MEMORY_ITEMS
        )

        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else self.settings.CACHE_TTL_SECONDS
        )

        self.cache_enabled = (
            cache_enabled if cache_enabled is not None else self.settings.ENABLE_CACHE
        )

        if self.max_memory_items <= 0:
            raise ValueError("max_memory_items must be greater than zero.")

        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be greater than zero.")

        self._memory: dict[str, list[MemoryRecord]] = {}

        self._cache: dict[str, CacheRecord] = {}

        self._lock = threading.RLock()

    # ========================================================
    # Text Helpers
    # ========================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for lightweight lexical matching.
        """

        if not text:
            return ""

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9+#.\s-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """
        Convert text into a set of normalized tokens.
        """

        normalized = cls._normalize_text(text)

        if not normalized:
            return set()

        return {token for token in normalized.split() if len(token) > 1}

    # ========================================================
    # Memory
    # ========================================================

    def add_memory(
        self,
        *,
        session_id: str,
        user_query: str,
        assistant_response: str,
        candidate_ids: Iterable[str] | None = None,
        document_ids: Iterable[str] | None = None,
        jd_ids: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """
        Add one conversation turn to session memory.

        Memory is bounded by max_memory_items.
        """

        if not session_id:
            raise ValueError("session_id is required.")

        if not user_query or not user_query.strip():
            raise ValueError("user_query cannot be empty.")

        if not assistant_response or not assistant_response.strip():
            raise ValueError("assistant_response cannot be empty.")

        record = MemoryRecord(
            session_id=session_id,
            user_query=user_query.strip(),
            assistant_response=assistant_response.strip(),
            timestamp=time.time(),
            candidate_ids=list(candidate_ids or []),
            document_ids=list(document_ids or []),
            jd_ids=list(jd_ids or []),
            metadata=dict(metadata or {}),
        )

        with self._lock:
            session_memory = self._memory.setdefault(
                session_id,
                [],
            )

            session_memory.append(record)

            if len(session_memory) > self.max_memory_items:
                self._memory[session_id] = session_memory[-self.max_memory_items :]

        logger.debug(
            "Added memory item for session=%s",
            session_id,
        )

        return record

    # ========================================================
    # Get Session Memory
    # ========================================================

    def get_session_memory(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """
        Return recent memory items for a session.

        Newest items are returned first.
        """

        if not session_id:
            return []

        with self._lock:
            records = list(
                self._memory.get(
                    session_id,
                    [],
                )
            )

        records.reverse()

        if limit is not None:
            if limit <= 0:
                return []

            records = records[:limit]

        return records

    # ========================================================
    # Memory Relevance
    # ========================================================

    @classmethod
    def _memory_score(
        cls,
        query: str,
        record: MemoryRecord,
    ) -> float:
        """
        Calculate lightweight relevance score.

        The score combines:
            - lexical overlap
            - recency

        This is intentionally deterministic and lightweight.
        """

        query_tokens = cls._tokenize(query)

        if not query_tokens:
            return 0.0

        record_text = f"{record.user_query} " f"{record.assistant_response}"

        record_tokens = cls._tokenize(record_text)

        if not record_tokens:
            return 0.0

        overlap = len(query_tokens & record_tokens) / len(query_tokens)

        age_seconds = max(
            0.0,
            time.time() - record.timestamp,
        )

        # Recency decays gradually over approximately one hour.
        recency = 1.0 / (1.0 + age_seconds / 3600.0)

        return 0.8 * overlap + 0.2 * recency

    def get_relevant_memory(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[MemoryRecord]:
        """
        Retrieve the most relevant previous conversation turns.

        If no query overlap exists, recent memory can still be
        returned as contextual history.
        """

        if not session_id or not query:
            return []

        records = self.get_session_memory(session_id)

        if not records:
            return []

        scored_records = [
            (
                self._memory_score(
                    query,
                    record,
                ),
                record,
            )
            for record in records
        ]

        scored_records.sort(
            key=lambda item: (
                item[0],
                item[1].timestamp,
            ),
            reverse=True,
        )

        if top_k is None:
            top_k = self.max_memory_items

        if top_k <= 0:
            return []

        return [record for _, record in scored_records[:top_k]]

    # ========================================================
    # Format Memory
    # ========================================================

    def format_memory(
        self,
        records: Iterable[MemoryRecord],
    ) -> str:
        """
        Convert memory records into prompt-ready text.

        This output is contextual information only.
        """

        records = list(records)

        if not records:
            return ""

        sections: list[str] = []

        for index, record in enumerate(
            records,
            start=1,
        ):
            sections.append(
                "\n".join(
                    [
                        f"Previous Turn {index}:",
                        f"User: {record.user_query}",
                        ("Assistant: " f"{record.assistant_response}"),
                    ]
                )
            )

        return "\n\n".join(sections)

    # ========================================================
    # Clear Session Memory
    # ========================================================

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Delete all memory for a session.
        """

        if not session_id:
            return

        with self._lock:
            self._memory.pop(
                session_id,
                None,
            )

        logger.debug(
            "Cleared memory for session=%s",
            session_id,
        )

    # ========================================================
    # Clear All Memory
    # ========================================================

    def clear_all_memory(self) -> None:
        """
        Clear all conversation memory.
        """

        with self._lock:
            self._memory.clear()

        logger.debug("Cleared all conversation memory.")

    # ========================================================
    # Cache Key
    # ========================================================

    @staticmethod
    def _stable_json(
        value: Any,
    ) -> str:
        """
        Serialize values deterministically.
        """

        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def build_context_hash(
        cls,
        context: Any = None,
    ) -> str:
        """
        Create deterministic hash for retrieval/context data.
        """

        payload = cls._stable_json(context)

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def create_cache_key(
        self,
        *,
        query: str,
        context: Any = None,
        session_id: str | None = None,
        model_version: str | None = None,
        prompt_version: str = "v1",
    ) -> str:
        """
        Create a deterministic cache key.

        The key changes when relevant query/context/model/prompt
        information changes.
        """

        if not query or not query.strip():
            raise ValueError("query is required for cache key.")

        model = model_version or getattr(
            self.settings,
            "LLM_MODEL",
            "",
        )

        payload = {
            "query": query.strip(),
            "context_hash": self.build_context_hash(context),
            "session_id": session_id or "",
            "model_version": model,
            "prompt_version": prompt_version,
        }

        serialized = self._stable_json(payload)

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ========================================================
    # Cache Set
    # ========================================================

    def set_cache(
        self,
        key: str,
        response: Any,
        *,
        session_id: str | None = None,
        document_ids: Iterable[str] | None = None,
        jd_ids: Iterable[str] | None = None,
        candidate_ids: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> CacheRecord | None:
        """
        Store a response in cache.
        """

        if not self.cache_enabled:
            return None

        if not key:
            raise ValueError("Cache key cannot be empty.")

        ttl = ttl_seconds if ttl_seconds is not None else self.cache_ttl_seconds

        if ttl <= 0:
            raise ValueError("ttl_seconds must be greater than zero.")

        now = time.time()

        record = CacheRecord(
            key=key,
            response=response,
            created_at=now,
            expires_at=now + ttl,
            session_id=session_id,
            document_ids=list(document_ids or []),
            jd_ids=list(jd_ids or []),
            candidate_ids=list(candidate_ids or []),
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._cache[key] = record

        logger.debug(
            "Cache entry created: %s",
            key,
        )

        return record

    # ========================================================
    # Cache Get
    # ========================================================

    def get_cache(
        self,
        key: str,
    ) -> Any | None:
        """
        Return cached response if valid.

        Returns None for:
            - cache disabled
            - missing key
            - expired entry
        """

        record = self.get_cache_record(key)

        if record is None:
            return None

        return record.response

    # ========================================================
    # Cache Record
    # ========================================================

    def get_cache_record(
        self,
        key: str,
    ) -> CacheRecord | None:
        """
        Return full cache record if it has not expired.
        """

        if not self.cache_enabled:
            return None

        if not key:
            return None

        with self._lock:
            record = self._cache.get(key)

            if record is None:
                return None

            if record.expires_at <= time.time():
                self._cache.pop(
                    key,
                    None,
                )

                logger.debug(
                    "Expired cache entry removed: %s",
                    key,
                )

                return None

            return record

    # ========================================================
    # Cache Delete
    # ========================================================

    def delete_cache(
        self,
        key: str,
    ) -> bool:
        """
        Delete one cache entry.

        Returns:
            True if an entry was removed.
        """

        if not key:
            return False

        with self._lock:
            existed = key in self._cache

            self._cache.pop(
                key,
                None,
            )

        return existed

    # ========================================================
    # Cache Invalidation
    # ========================================================

    def invalidate_cache(
        self,
        *,
        document_ids: Iterable[str] | None = None,
        jd_ids: Iterable[str] | None = None,
        candidate_ids: Iterable[str] | None = None,
        session_id: str | None = None,
    ) -> int:
        """
        Invalidate cache entries associated with changed data.

        This is useful after:
            - resume updates
            - JD updates
            - candidate changes
            - session reset
        """

        document_set = set(document_ids or [])

        jd_set = set(jd_ids or [])

        candidate_set = set(candidate_ids or [])

        removed = 0

        with self._lock:
            keys_to_remove: list[str] = []

            for key, record in self._cache.items():
                should_remove = False

                if session_id and record.session_id == session_id:
                    should_remove = True

                if document_set and document_set.intersection(record.document_ids):
                    should_remove = True

                if jd_set and jd_set.intersection(record.jd_ids):
                    should_remove = True

                if candidate_set and candidate_set.intersection(record.candidate_ids):
                    should_remove = True

                if should_remove:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self._cache.pop(
                    key,
                    None,
                )
                removed += 1

        logger.debug(
            "Invalidated %d cache entries.",
            removed,
        )

        return removed

    # ========================================================
    # Clear Cache
    # ========================================================

    def clear_cache(self) -> None:
        """
        Clear all cached responses.
        """

        with self._lock:
            self._cache.clear()

        logger.debug("Cleared all cache entries.")

    # ========================================================
    # Cleanup
    # ========================================================

    def cleanup_expired_cache(self) -> int:
        """
        Remove all expired cache entries.

        Returns:
            Number of removed entries.
        """

        now = time.time()

        removed = 0

        with self._lock:
            expired_keys = [
                key for key, record in self._cache.items() if record.expires_at <= now
            ]

            for key in expired_keys:
                self._cache.pop(
                    key,
                    None,
                )

                removed += 1

        if removed:
            logger.debug(
                "Removed %d expired cache entries.",
                removed,
            )

        return removed

    # ========================================================
    # Statistics
    # ========================================================

    def stats(self) -> dict[str, int | bool]:
        """
        Return basic memory/cache statistics.
        """

        with self._lock:
            memory_sessions = len(self._memory)

            memory_items = sum(len(records) for records in self._memory.values())

            cache_items = len(self._cache)

        return {
            "memory_enabled": True,
            "cache_enabled": self.cache_enabled,
            "memory_sessions": memory_sessions,
            "memory_items": memory_items,
            "cache_items": cache_items,
        }


# ============================================================
# Factory
# ============================================================


_memory_rag: MemoryRAG | None = None


def create_memory_rag() -> MemoryRAG:
    """
    Return the shared MemoryRAG instance.
    """

    global _memory_rag

    if _memory_rag is None:
        _memory_rag = MemoryRAG()

    return _memory_rag
