"""
Hybrid indexing support for Kokoro AI.

Responsibilities:
- Maintain an in-memory BM25 keyword index.
- Store searchable document chunks.
- Tokenize text consistently for BM25.
- Add/update/remove documents from the BM25 index.
- Perform keyword retrieval.
- Combine semantic and keyword candidates later.

Important:
    rank-bm25 is an in-memory library. Therefore the BM25 index
    is rebuilt from the locally available indexed documents when
    the application starts or when the index is explicitly rebuilt.

Pinecone remains the persistent semantic/vector store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from threading import RLock
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from utils.logger import logger
from utils.schemas import RetrievedChunk


# ============================================================
# BM25 Configuration
# ============================================================

DEFAULT_BM25_K1 = 1.5

DEFAULT_BM25_B = 0.75


# ============================================================
# Internal BM25 Record
# ============================================================


@dataclass
class BM25Record:
    """
    Internal representation of one BM25 indexed document chunk.
    """

    chunk_id: str

    text: str

    metadata: dict[str, Any]


# ============================================================
# Tokenization
# ============================================================


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25 retrieval.

    The tokenizer intentionally keeps useful technical
    information such as:

        Python
        Python 3
        SQL
        C++
        .NET
        RAG
        LLM
        GCP
        AWS

    We do not perform aggressive stemming or stop-word removal.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    list[str]
        Tokens suitable for BM25.
    """

    if not text:
        return []

    text = text.lower()

    # Keep:
    # - letters
    # - numbers
    # - +, #, ., -, _
    #
    # This is useful for technical recruitment queries.
    tokens = re.findall(
        r"[a-z0-9][a-z0-9+#._-]*",
        text,
    )

    return tokens


# ============================================================
# Hybrid Indexer
# ============================================================


class HybridIndexer:
    """
    Manages the BM25 keyword index used by Kokoro.

    This class does not directly communicate with Pinecone.
    Pinecone semantic indexing is handled by PineconeVectorStore.

    The two indexes can therefore be updated independently.
    """

    def __init__(
        self,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        """
        Initialize the BM25 indexer.

        Parameters
        ----------
        k1:
            BM25 term-frequency saturation parameter.

        b:
            BM25 document-length normalization parameter.
        """

        if k1 <= 0:
            raise ValueError(
                "BM25 k1 must be greater than 0."
            )

        if not 0 <= b <= 1:
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

        self.k1 = k1

        self.b = b

        self._records: dict[str, BM25Record] = {}

        self._bm25: BM25Okapi | None = None

        self._tokenized_corpus: list[list[str]] = []

        self._record_order: list[str] = []

        self._lock = RLock()

    # ========================================================
    # Properties
    # ========================================================

    @property
    def size(self) -> int:
        """
        Return the number of indexed chunks.
        """

        with self._lock:
            return len(self._records)

    # ========================================================
    # Add Documents
    # ========================================================

    def add_documents(
        self,
        documents: list[Document],
    ) -> None:
        """
        Add or update LangChain Documents in the BM25 index.

        Documents must contain:

            metadata["chunk_id"]

        Parameters
        ----------
        documents:
            Chunked LangChain Documents.
        """

        if not documents:
            return

        with self._lock:

            for document in documents:

                chunk_id = document.metadata.get(
                    "chunk_id"
                )

                if not chunk_id:
                    raise ValueError(
                        "Every document must contain "
                        "metadata['chunk_id']."
                    )

                text = document.page_content.strip()

                if not text:
                    continue

                self._records[str(chunk_id)] = BM25Record(
                    chunk_id=str(chunk_id),
                    text=text,
                    metadata=dict(
                        document.metadata
                    ),
                )

            self._rebuild_locked()

        logger.info(
            "BM25 documents indexed | added=%d | total=%d",
            len(documents),
            self.size,
        )

    # ========================================================
    # Add Raw Records
    # ========================================================

    def add_records(
        self,
        records: list[BM25Record],
    ) -> None:
        """
        Add internal BM25 records.

        Useful when restoring the index from persisted
        application data.
        """

        if not records:
            return

        with self._lock:

            for record in records:

                if not record.chunk_id:
                    raise ValueError(
                        "BM25 record chunk_id cannot be empty."
                    )

                if not record.text.strip():
                    continue

                self._records[
                    record.chunk_id
                ] = record

            self._rebuild_locked()

        logger.info(
            "BM25 records restored | records=%d | total=%d",
            len(records),
            self.size,
        )

    # ========================================================
    # Remove Document
    # ========================================================

    def remove_document(
        self,
        document_id: str,
    ) -> int:
        """
        Remove all chunks belonging to a document.

        Parameters
        ----------
        document_id:
            Document identifier stored in metadata.

        Returns
        -------
        int
            Number of removed chunks.
        """

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty."
            )

        with self._lock:

            to_remove = [
                chunk_id
                for chunk_id, record in self._records.items()
                if str(
                    record.metadata.get(
                        "document_id",
                        "",
                    )
                )
                == document_id
            ]

            for chunk_id in to_remove:
                del self._records[chunk_id]

            self._rebuild_locked()

        logger.info(
            "BM25 document removed | "
            "document_id=%s | chunks_removed=%d",
            document_id,
            len(to_remove),
        )

        return len(to_remove)

    # ========================================================
    # Rebuild
    # ========================================================

    def rebuild(self) -> None:
        """
        Explicitly rebuild the BM25 index.
        """

        with self._lock:
            self._rebuild_locked()

        logger.info(
            "BM25 index rebuilt | documents=%d",
            self.size,
        )

    def _rebuild_locked(self) -> None:
        """
        Rebuild BM25.

        Caller must hold self._lock.
        """

        self._record_order = list(
            self._records.keys()
        )

        self._tokenized_corpus = [
            tokenize_for_bm25(
                self._records[chunk_id].text
            )
            for chunk_id in self._record_order
        ]

        # rank-bm25 requires a corpus.
        if not self._tokenized_corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

    # ========================================================
    # Keyword Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform BM25 keyword retrieval.

        Parameters
        ----------
        query:
            Recruiter's search query.

        top_k:
            Maximum number of results.

        candidate_filter:
            Optional metadata filter.

        Returns
        -------
        list[RetrievedChunk]
            BM25-ranked chunks.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

        query_tokens = tokenize_for_bm25(
            query
        )

        if not query_tokens:
            return []

        with self._lock:

            if self._bm25 is None:
                return []

            scores = self._bm25.get_scores(
                query_tokens
            )

            ranked_indexes = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )

            results: list[RetrievedChunk] = []

            for index in ranked_indexes:

                if len(results) >= top_k:
                    break

                chunk_id = self._record_order[index]

                record = self._records[
                    chunk_id
                ]

                if not self._matches_filter(
                    record.metadata,
                    candidate_filter,
                ):
                    continue

                score = float(
                    scores[index]
                )

                results.append(
                    RetrievedChunk(
                        chunk_id=record.chunk_id,
                        candidate_id=self._optional_string(
                            record.metadata.get(
                                "candidate_id"
                            )
                        ),
                        candidate_name=self._optional_string(
                            record.metadata.get(
                                "candidate_name"
                            )
                        ),
                        text=record.text,
                        keyword_score=score,
                        metadata=dict(
                            record.metadata
                        ),
                    )
                )

        logger.info(
            "BM25 search completed | "
            "top_k=%d | returned=%d",
            top_k,
            len(results),
        )

        return results

    # ========================================================
    # Retrieve Record
    # ========================================================

    def get_record(
        self,
        chunk_id: str,
    ) -> BM25Record | None:
        """
        Retrieve an indexed BM25 record by chunk ID.
        """

        with self._lock:
            return self._records.get(
                chunk_id
            )

    # ========================================================
    # Export Records
    # ========================================================

    def export_records(self) -> list[BM25Record]:
        """
        Export indexed records.

        This can later be used to persist BM25 source data
        and rebuild the in-memory index after application restart.
        """

        with self._lock:
            return list(
                self._records.values()
            )

    # ========================================================
    # Metadata Filtering
    # ========================================================

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        candidate_filter: dict[str, Any] | None,
    ) -> bool:
        """
        Apply a simple equality-based metadata filter.

        Example:

            {
                "document_type": "resume"
            }

        or:

            {
                "candidate_id": "candidate_001"
            }

        This is intentionally simple because more complex
        filtering can be implemented at the retrieval-router
        layer.
        """

        if not candidate_filter:
            return True

        for key, expected_value in candidate_filter.items():

            actual_value = metadata.get(
                key
            )

            if actual_value != expected_value:
                return False

        return True

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        """
        Convert optional metadata to a clean string.
        """

        if value is None:
            return None

        value = str(value).strip()

        return value or None


# ============================================================
# Hybrid Score Normalization
# ============================================================


def normalize_scores(
    results: list[RetrievedChunk],
    score_type: str,
) -> list[RetrievedChunk]:
    """
    Normalize retrieval scores to the range [0, 1].

    This is useful before combining Pinecone semantic scores
    with BM25 keyword scores because their raw score scales
    are different.

    Parameters
    ----------
    results:
        Retrieved chunks.

    score_type:
        Either:
            "semantic"
            "keyword"

    Returns
    -------
    list[RetrievedChunk]
        Results with normalized scores.

    Notes
    -----
    Min-max normalization is used here:

        normalized =
            (score - min_score) /
            (max_score - min_score)

    If all scores are identical, each score becomes 1.0.
    """

    if score_type not in {
        "semantic",
        "keyword",
    }:
        raise ValueError(
            "score_type must be 'semantic' or 'keyword'."
        )

    if not results:
        return []

    scores: list[float] = []

    for result in results:

        if score_type == "semantic":
            score = result.semantic_score
        else:
            score = result.keyword_score

        if score is not None:
            scores.append(
                float(score)
            )

    if not scores:
        return results

    min_score = min(scores)

    max_score = max(scores)

    score_range = max_score - min_score

    for result in results:

        if score_type == "semantic":

            if result.semantic_score is None:
                continue

            if score_range == 0:
                result.semantic_score = 1.0

            else:
                result.semantic_score = (
                    result.semantic_score - min_score
                ) / score_range

        else:

            if result.keyword_score is None:
                continue

            if score_range == 0:
                result.keyword_score = 1.0

            else:
                result.keyword_score = (
                    result.keyword_score - min_score
                ) / score_range

    return results