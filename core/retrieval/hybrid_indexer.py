"""
Kokoro AI - Hybrid Retrieval Indexer.

Responsibilities
----------------
- Maintain the BM25 lexical index.
- Execute BM25 keyword retrieval.
- Execute Pinecone semantic retrieval.
- Normalize semantic and lexical scores.
- Fuse scores using configurable weights.
- Deduplicate results coming from both retrieval systems.
- Return the initial hybrid retrieval pool.

Architecture
------------
                    Query
                      |
              +--------+--------+
              |                 |
              v                 v
          Pinecone             BM25
          Semantic            Keyword
          Search              Search
              |                 |
              +--------+--------+
                      |
                 Score Normalize
                      |
                 Weighted Fusion
                      |
                  Hybrid Top-K
                      |
                   Reranker

Important
---------
This module does NOT:
- classify the query
- perform query decomposition
- perform multi-query generation
- call Gemini for reranking
- generate the final answer
- depend on Streamlit
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from utils.config import get_settings
from utils.logger import get_logger
from utils.schemas import RetrievalMethod, RetrievalResult


logger = get_logger(__name__)
settings = get_settings()


# ============================================================
# Protocols
# ============================================================


class SemanticRetriever(Protocol):
    """
    Interface required from the semantic vector store.

    PineconeVectorStore satisfies this interface.
    """

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Any | None = None,
    ) -> list[RetrievalResult]:
        ...


# ============================================================
# Internal BM25 Record
# ============================================================


@dataclass(slots=True)
class BM25Record:
    """
    Internal representation of one indexed chunk.

    The complete chunk text is retained because BM25 requires
    the indexed corpus to build/query its lexical representation.
    """

    chunk_id: str
    text: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Hybrid Result
# ============================================================


@dataclass(slots=True)
class HybridSearchResult:
    """
    Internal hybrid retrieval result.

    Scores:
        semantic_score
            Raw Pinecone similarity score.

        keyword_score
            Raw BM25 score.

        normalized_semantic_score
            Semantic score normalized to 0..1.

        normalized_keyword_score
            BM25 score normalized to 0..1.

        hybrid_score
            Weighted combination used for final hybrid ranking.
    """

    chunk_id: str

    content: str

    metadata: dict[str, Any]

    semantic_score: float = 0.0

    keyword_score: float = 0.0

    normalized_semantic_score: float = 0.0

    normalized_keyword_score: float = 0.0

    hybrid_score: float = 0.0

    retrieval_method: RetrievalMethod = (
        RetrievalMethod.HYBRID
    )


# ============================================================
# BM25 Backend
# ============================================================


class BM25Backend:
    """
    Lightweight BM25 backend using rank-bm25.

    The backend owns:
        - corpus records
        - tokenized corpus
        - BM25 model

    It can be rebuilt whenever the indexed corpus changes.

    This keeps BM25 implementation isolated from the hybrid
    fusion logic.
    """

    def __init__(
        self,
        *,
        tokenizer: Any | None = None,
    ) -> None:

        self._tokenizer = (
            tokenizer
            or self.default_tokenizer
        )

        self._records: dict[
            str,
            BM25Record,
        ] = {}

        self._tokenized_corpus: list[
            list[str]
        ] = []

        self._record_ids: list[str] = []

        self._bm25: Any | None = None

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    @staticmethod
    def default_tokenizer(
        text: str,
    ) -> list[str]:
        """
        Simple deterministic tokenizer.

        BM25 does not require an LLM tokenizer.

        The tokenizer:
        - lowercases text
        - keeps alphanumeric terms
        - preserves useful technical tokens
        """

        if not text:
            return []

        normalized = text.lower()

        tokens: list[str] = []

        current: list[str] = []

        for character in normalized:

            if (
                character.isalnum()
                or character in (
                    "_",
                    "-",
                    ".",
                    "#",
                    "+",
                )
            ):
                current.append(
                    character
                )

            else:

                if current:
                    tokens.append(
                        "".join(current)
                    )
                    current = []

        if current:
            tokens.append(
                "".join(current)
            )

        return [
            token
            for token in tokens
            if token
        ]

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    def _rebuild(self) -> None:
        """Rebuild the BM25 model from the current records."""

        with self._lock:

            self._record_ids = list(
                self._records.keys()
            )

            self._tokenized_corpus = [
                self._tokenizer(
                    self._records[
                        record_id
                    ].text
                )
                for record_id in self._record_ids
            ]

            if not self._tokenized_corpus:
                self._bm25 = None
                return

            try:
                from rank_bm25 import (
                    BM25Okapi,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "rank-bm25 is required for BM25 retrieval."
                ) from exc

            self._bm25 = BM25Okapi(
                self._tokenized_corpus
            )

    # --------------------------------------------------------
    # Upsert
    # --------------------------------------------------------

    def upsert(
        self,
        records: Sequence[BM25Record],
    ) -> int:
        """
        Add or replace BM25 records.

        chunk_id is the stable key, so repeated ingestion of
        the same chunk does not create duplicate BM25 entries.
        """

        if not records:
            return 0

        with self._lock:

            for record in records:

                if not record.chunk_id:
                    raise ValueError(
                        "BM25 record requires chunk_id."
                    )

                if not record.text.strip():
                    continue

                self._records[
                    record.chunk_id
                ] = record

            self._rebuild()

        logger.info(
            "BM25 index updated: %d records.",
            len(self._records),
        )

        return len(records)

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------

    def delete(
        self,
        chunk_ids: Sequence[str],
    ) -> int:
        """Delete BM25 records by chunk ID."""

        if not chunk_ids:
            return 0

        deleted = 0

        with self._lock:

            for chunk_id in chunk_ids:

                if (
                    chunk_id
                    in self._records
                ):
                    del self._records[
                        chunk_id
                    ]
                    deleted += 1

            if deleted:
                self._rebuild()

        return deleted

    def delete_by_metadata(
        self,
        *,
        document_id: str | None = None,
        candidate_id: str | None = None,
        jd_id: str | None = None,
    ) -> int:
        """
        Delete BM25 records matching document metadata.
        """

        if not any(
            (
                document_id,
                candidate_id,
                jd_id,
            )
        ):
            return 0

        matching_ids: list[str] = []

        with self._lock:

            for chunk_id, record in (
                self._records.items()
            ):

                metadata = record.metadata

                if (
                    document_id is not None
                    and metadata.get(
                        "document_id"
                    )
                    != document_id
                ):
                    continue

                if (
                    candidate_id is not None
                    and metadata.get(
                        "candidate_id"
                    )
                    != candidate_id
                ):
                    continue

                if (
                    jd_id is not None
                    and metadata.get(
                        "jd_id"
                    )
                    != jd_id
                ):
                    continue

                matching_ids.append(
                    chunk_id
                )

            return self.delete(
                matching_ids
            )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[BM25Record, float]]:
        """
        Search BM25 and return:

            [(record, raw_bm25_score), ...]

        Filtering is applied after scoring.
        """

        if (
            not query
            or not query.strip()
            or top_k <= 0
        ):
            return []

        with self._lock:

            if self._bm25 is None:
                return []

            query_tokens = self._tokenizer(
                query
            )

            if not query_tokens:
                return []

            scores = self._bm25.get_scores(
                query_tokens
            )

            ranked_indexes = sorted(
                range(len(scores)),
                key=lambda index: scores[
                    index
                ],
                reverse=True,
            )

            results: list[
                tuple[
                    BM25Record,
                    float,
                ]
            ] = []

            for index in ranked_indexes:

                score = float(
                    scores[index]
                )

                # BM25 ranks every corpus row, including rows with
                # no overlapping query terms. Those zero-score rows
                # are not lexical matches and must not enter fusion.
                if score <= 0.0:
                    break

                record_id = (
                    self._record_ids[
                        index
                    ]
                )

                record = self._records[
                    record_id
                ]

                if not self._matches_filters(
                    record,
                    filters,
                ):
                    continue

                results.append(
                    (
                        record,
                        score,
                    )
                )

                if len(results) >= top_k:
                    break

            return results

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    @staticmethod
    def _matches_filters(
        record: BM25Record,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        """
        Apply application-level metadata filtering to BM25
        results.

        Pinecone handles metadata filtering natively. BM25 does
        not, so the same logical filters are applied here.
        """

        if not filters:
            return True

        metadata = record.metadata

        # --------------------------------------------
        # Document type
        # --------------------------------------------

        document_type = filters.get(
            "document_type"
        )

        if document_type is not None:

            if hasattr(
                document_type,
                "value",
            ):
                document_type = (
                    document_type.value
                )

            if isinstance(
                document_type,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                allowed = {
                    str(value)
                    for value in document_type
                }

                actual = str(
                    metadata.get(
                        "document_type",
                        "",
                    )
                )

                if actual not in allowed:
                    return False

            elif str(
                metadata.get(
                    "document_type",
                    "",
                )
            ) != str(document_type):
                return False

        # --------------------------------------------
        # Candidate ID
        # --------------------------------------------

        candidate_id = filters.get(
            "candidate_id"
        )

        if (
            candidate_id is not None
            and metadata.get(
                "candidate_id"
            )
            != candidate_id
        ):
            return False

        candidate_ids = filters.get(
            "candidate_ids"
        )

        if (
            candidate_ids
            and metadata.get(
                "candidate_id"
            )
            not in candidate_ids
        ):
            return False

        # --------------------------------------------
        # JD ID
        # --------------------------------------------

        jd_id = filters.get(
            "jd_id"
        )

        if (
            jd_id is not None
            and metadata.get(
                "jd_id"
            )
            != jd_id
        ):
            return False

        # --------------------------------------------
        # Location
        # --------------------------------------------

        location = filters.get(
            "location"
        )

        if (
            location is not None
            and str(
                metadata.get(
                    "location",
                    "",
                )
            ).lower()
            != str(location).lower()
        ):
            return False

        # --------------------------------------------
        # Experience
        # --------------------------------------------

        experience = metadata.get(
            "experience_years"
        )

        min_experience = filters.get(
            "min_experience"
        )

        if min_experience is not None:

            if experience is None:
                return False

            try:
                if float(experience) < float(
                    min_experience
                ):
                    return False
            except (
                TypeError,
                ValueError,
            ):
                return False

        max_experience = filters.get(
            "max_experience"
        )

        if max_experience is not None:

            if experience is None:
                return False

            try:
                if float(experience) > float(
                    max_experience
                ):
                    return False
            except (
                TypeError,
                ValueError,
            ):
                return False

        # --------------------------------------------
        # Skills
        # --------------------------------------------

        required_skills = filters.get(
            "skills"
        )

        if required_skills:

            indexed_skills = metadata.get(
                "skills",
                [],
            )

            if isinstance(
                indexed_skills,
                str,
            ):
                indexed_skills = [
                    indexed_skills
                ]

            indexed_normalized = {
                str(skill).lower()
                for skill in indexed_skills
            }

            requested_normalized = {
                str(skill).lower()
                for skill in required_skills
            }

            if not requested_normalized.issubset(
                indexed_normalized
            ):
                return False

        return True

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    def size(self) -> int:
        """Return number of BM25 records."""

        with self._lock:
            return len(
                self._records
            )

    def clear(self) -> None:
        """Remove all BM25 records."""

        with self._lock:

            self._records.clear()

            self._record_ids.clear()

            self._tokenized_corpus.clear()

            self._bm25 = None

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Persist the BM25 corpus as JSON.

        The BM25 model itself is rebuilt when loading rather
        than serializing the rank-bm25 object.
        """

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:

            payload = {
                "records": [
                    {
                        "chunk_id": record.chunk_id,
                        "text": record.text,
                        "metadata": record.metadata,
                    }
                    for record in self._records.values()
                ]
            }

        destination.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Saved BM25 corpus: %s",
            destination,
        )

    def load(
        self,
        path: str | Path,
    ) -> int:
        """Load a previously persisted BM25 corpus."""

        source = Path(path)

        if not source.exists():
            return 0

        payload = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )

        records = payload.get(
            "records",
            [],
        )

        loaded_records: list[
            BM25Record
        ] = []

        for item in records:

            if not isinstance(
                item,
                Mapping,
            ):
                continue

            chunk_id = item.get(
                "chunk_id"
            )

            text = item.get(
                "text",
                "",
            )

            if not chunk_id or not text:
                continue

            loaded_records.append(
                BM25Record(
                    chunk_id=str(
                        chunk_id
                    ),
                    text=str(text),
                    metadata=dict(
                        item.get(
                            "metadata",
                            {},
                        )
                        or {}
                    ),
                )
            )

        self.upsert(
            loaded_records
        )

        logger.info(
            "Loaded BM25 corpus: %d records.",
            len(loaded_records),
        )

        return len(loaded_records)


# ============================================================
# Hybrid Indexer
# ============================================================


class HybridIndexer:
    """
    Combines Pinecone semantic retrieval and BM25 lexical
    retrieval.

    Default fusion:

        hybrid_score =
            0.60 * normalized_semantic_score
          + 0.40 * normalized_keyword_score

    The final list is sorted by hybrid_score.
    """

    def __init__(
        self,
        *,
        semantic_retriever: SemanticRetriever,
        bm25_backend: BM25Backend | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> None:

        self.semantic_retriever = (
            semantic_retriever
        )

        self.bm25 = (
            bm25_backend
            or BM25Backend()
        )

        self.semantic_weight = (
            settings.SEMANTIC_WEIGHT
            if semantic_weight is None
            else semantic_weight
        )

        # Current configuration uses LEXICAL_WEIGHT.
        self.keyword_weight = (
            settings.LEXICAL_WEIGHT
            if keyword_weight is None
            else keyword_weight
        )

        self._validate_weights()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_weights(self) -> None:
        """Validate hybrid fusion weights."""

        if not (
            0.0
            <= self.semantic_weight
            <= 1.0
        ):
            raise ValueError(
                "semantic_weight must be between 0 and 1."
            )

        if not (
            0.0
            <= self.keyword_weight
            <= 1.0
        ):
            raise ValueError(
                "keyword_weight must be between 0 and 1."
            )

        total = (
            self.semantic_weight
            + self.keyword_weight
        )

        if not math.isclose(
            total,
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "semantic_weight + keyword_weight "
                "must equal 1.0."
            )

    # --------------------------------------------------------
    # BM25 Indexing
    # --------------------------------------------------------

    @staticmethod
    def _extract_value(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            obj,
            Mapping,
        ):
            value = obj.get(key, default)
            if value is default and key == "score":
                value = obj.get("raw_score", default)
            if value is default and key == "content":
                value = obj.get("text", default)
            return value

        value = getattr(obj, key, default)
        if value is default and key == "score":
            value = getattr(obj, "raw_score", default)
        if value is default and key == "content":
            value = getattr(obj, "text", default)
        return value

    @classmethod
    def _chunk_to_bm25_record(
        cls,
        chunk: Any,
    ) -> BM25Record:
        """Convert an ingestion chunk into a BM25 record."""

        metadata = (
            cls._extract_value(
                chunk,
                "metadata",
                {},
            )
            or {}
        )
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()
        metadata = dict(metadata)

        chunk_id = (
            cls._extract_value(
                chunk,
                "chunk_id",
            )
            or metadata.get("chunk_id")
            or cls._extract_value(
                chunk,
                "id",
            )
        )

        if not chunk_id:
            raise ValueError(
                "Chunk requires chunk_id for BM25 indexing."
            )

        text = (
            cls._extract_value(
                chunk,
                "content",
            )
            or cls._extract_value(
                chunk,
                "text",
            )
            or cls._extract_value(
                chunk,
                "page_content",
            )
            or ""
        )

        if not text.strip():
            raise ValueError(
                f"Chunk {chunk_id} has empty text."
            )

        # Keep important retrieval metadata available
        # even when it was supplied as top-level fields.
        for key in (
            "document_id",
            "candidate_id",
            "jd_id",
            "document_type",
            "candidate_name",
            "location",
            "experience_years",
            "skills",
            "section",
            "page",
        ):

            value = cls._extract_value(
                chunk,
                key,
            )

            if value is not None:
                metadata.setdefault(
                    key,
                    value,
                )

        return BM25Record(
            chunk_id=str(
                chunk_id
            ),
            text=str(text),
            metadata=metadata,
        )

    def index_chunks(
        self,
        chunks: Sequence[Any],
    ) -> int:
        """
        Add chunks to BM25.

        Pinecone indexing is handled by vector_store.py.
        """

        records = [
            self._chunk_to_bm25_record(
                chunk
            )
            for chunk in chunks
        ]

        return self.bm25.upsert(
            records
        )

    def add_documents(
        self,
        documents: Sequence[Any],
    ) -> int:
        """Index ingestion documents in the BM25 corpus."""

        return self.index_chunks(documents)

    # --------------------------------------------------------
    # Score Normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        scores: Mapping[str, float],
    ) -> dict[str, float]:
        """
        Normalize scores to 0..1.

        The normalization is performed independently for each
        retrieval source because Pinecone similarity and BM25
        scores are not directly comparable.
        """

        if not scores:
            return {}

        values = [
            float(value)
            for value in scores.values()
        ]

        minimum = min(values)
        maximum = max(values)

        if math.isclose(
            minimum,
            maximum,
            abs_tol=1e-12,
        ):
            # If every candidate has the same score, preserve
            # the fact that the candidates were retrieved while
            # avoiding arbitrary differentiation.
            return {
                key: (
                    1.0
                    if value > 0
                    else 0.0
                )
                for key, value in scores.items()
            }

        return {
            key: (
                float(value) - minimum
            )
            / (
                maximum - minimum
            )
            for key, value in scores.items()
        }

    # --------------------------------------------------------
    # Metadata Filtering
    # --------------------------------------------------------

    @staticmethod
    def _filters_to_mapping(
        filters: Any | None,
    ) -> dict[str, Any] | None:
        """Convert SearchFilters/Pydantic model to a mapping."""

        if filters is None:
            return None

        if hasattr(
            filters,
            "model_dump",
        ):
            return filters.model_dump(
                exclude_none=True
            )

        if isinstance(
            filters,
            Mapping,
        ):
            return dict(filters)

        return {
            key: value
            for key, value in vars(
                filters
            ).items()
            if value is not None
        }

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    def _fuse_results(
        self,
        semantic_results: Sequence[
            RetrievalResult
        ],
        keyword_results: Sequence[
            tuple[BM25Record, float]
        ],
        *,
        top_k: int,
    ) -> list[HybridSearchResult]:
        """
        Merge semantic and keyword results by chunk_id.
        """

        semantic_scores: dict[
            str,
            float,
        ] = {}

        semantic_objects: dict[
            str,
            RetrievalResult,
        ] = {}

        for result in semantic_results:

            chunk_id = (
                self._extract_value(
                    result,
                    "chunk_id",
                )
            )

            if not chunk_id:
                continue

            chunk_id = str(
                chunk_id
            )

            score = float(
                self._extract_value(
                    result,
                    "score",
                    0.0,
                )
                or 0.0
            )

            semantic_scores[
                chunk_id
            ] = score

            semantic_objects[
                chunk_id
            ] = result

        keyword_scores: dict[
            str,
            float,
        ] = {}

        keyword_objects: dict[
            str,
            BM25Record,
        ] = {}

        for record, score in keyword_results:

            chunk_id = str(
                record.chunk_id
            )

            keyword_scores[
                chunk_id
            ] = float(score)

            keyword_objects[
                chunk_id
            ] = record

        normalized_semantic = (
            self._normalize_scores(
                semantic_scores
            )
        )

        normalized_keyword = (
            self._normalize_scores(
                keyword_scores
            )
        )

        all_chunk_ids = set(
            semantic_scores
        ) | set(
            keyword_scores
        )

        fused: list[
            HybridSearchResult
        ] = []

        for chunk_id in all_chunk_ids:

            semantic_score = (
                semantic_scores.get(
                    chunk_id,
                    0.0,
                )
            )

            keyword_score = (
                keyword_scores.get(
                    chunk_id,
                    0.0,
                )
            )

            normalized_semantic_score = (
                normalized_semantic.get(
                    chunk_id,
                    0.0,
                )
            )

            normalized_keyword_score = (
                normalized_keyword.get(
                    chunk_id,
                    0.0,
                )
            )

            hybrid_score = (
                self.semantic_weight
                * normalized_semantic_score
                + self.keyword_weight
                * normalized_keyword_score
            )

            semantic_result = (
                semantic_objects.get(
                    chunk_id
                )
            )

            keyword_record = (
                keyword_objects.get(
                    chunk_id
                )
            )

            content = ""

            metadata: dict[
                str,
                Any,
            ] = {}

            if semantic_result is not None:

                content = (
                    self._extract_value(
                        semantic_result,
                        "content",
                        "",
                    )
                    or ""
                )

                result_metadata = (
                    self._extract_value(
                        semantic_result,
                        "metadata",
                        {},
                    )
                    or {}
                )

                if hasattr(
                    result_metadata,
                    "model_dump",
                ):
                    result_metadata = (
                        result_metadata.model_dump()
                    )

                metadata.update(
                    dict(
                        result_metadata
                    )
                )

            if keyword_record is not None:

                if not content:
                    content = (
                        keyword_record.text
                    )

                # Semantic metadata remains primary.
                # BM25 fills in anything missing.
                for key, value in (
                    keyword_record.metadata.items()
                ):
                    metadata.setdefault(
                        key,
                        value,
                    )

            fused.append(
                HybridSearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    normalized_semantic_score=(
                        normalized_semantic_score
                    ),
                    normalized_keyword_score=(
                        normalized_keyword_score
                    ),
                    hybrid_score=hybrid_score,
                )
            )

        fused.sort(
            key=lambda result: (
                result.hybrid_score,
                result.semantic_score,
                result.keyword_score,
            ),
            reverse=True,
        )

        return fused[
            :top_k
        ]

    # --------------------------------------------------------
    # Hybrid Search
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        semantic_top_k: int | None = None,
        keyword_top_k: int | None = None,
        filters: Any | None = None,
    ) -> list[HybridSearchResult]:
        """
        Execute hybrid retrieval.

        Flow:

            Query
              |
              +----> Pinecone semantic retrieval
              |
              +----> BM25 lexical retrieval
              |
              v
          Normalize scores
              |
              v
          Weighted fusion
              |
              v
           Hybrid Top-K
        """

        if not query or not query.strip():
            return []

        # Current configuration:
        # VECTOR_TOP_K = 15
        # BM25_TOP_K = 15
        # HYBRID_TOP_K = 10

        final_top_k = (
            top_k
            if top_k is not None
            else settings.HYBRID_TOP_K
        )

        semantic_k = (
            semantic_top_k
            if semantic_top_k is not None
            else settings.VECTOR_TOP_K
        )

        keyword_k = (
            keyword_top_k
            if keyword_top_k is not None
            else settings.BM25_TOP_K
        )

        if final_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if semantic_k <= 0:
            raise ValueError(
                "semantic_top_k must be greater than zero."
            )

        if keyword_k <= 0:
            raise ValueError(
                "keyword_top_k must be greater than zero."
            )

        filter_mapping = (
            self._filters_to_mapping(
                filters
            )
        )

        logger.info(
            "Starting hybrid search: "
            "top_k=%d semantic_k=%d keyword_k=%d",
            final_top_k,
            semantic_k,
            keyword_k,
        )

        # ----------------------------------------------------
        # Semantic retrieval
        # ----------------------------------------------------

        semantic_results = (
            self.semantic_retriever.similarity_search(
                query,
                top_k=semantic_k,
                filters=filters,
            )
        )

        # ----------------------------------------------------
        # Keyword retrieval
        # ----------------------------------------------------

        keyword_results = (
            self.bm25.search(
                query,
                top_k=keyword_k,
                filters=filter_mapping,
            )
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        results = self._fuse_results(
            semantic_results,
            keyword_results,
            top_k=final_top_k,
        )

        logger.info(
            "Hybrid search completed: "
            "semantic=%d keyword=%d final=%d",
            len(semantic_results),
            len(keyword_results),
            len(results),
        )

        return results

    # --------------------------------------------------------
    # Deletion
    # --------------------------------------------------------

    def delete_by_document_id(
        self,
        document_id: str,
    ) -> int:
        """Remove a document from the BM25 index."""

        return self.bm25.delete_by_metadata(
            document_id=document_id
        )

    def delete_by_candidate_id(
        self,
        candidate_id: str,
    ) -> int:
        """Remove candidate chunks from BM25."""

        return self.bm25.delete_by_metadata(
            candidate_id=candidate_id
        )

    def delete_by_jd_id(
        self,
        jd_id: str,
    ) -> int:
        """Remove JD chunks from BM25."""

        return self.bm25.delete_by_metadata(
            jd_id=jd_id
        )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def save_bm25(
        self,
        path: str | Path | None = None,
    ) -> None:
        """
        Persist the BM25 corpus.

        If no path is provided, use the configured BM25
        directory.
        """

        if path is None:

            directory = Path(
                getattr(
                    settings,
                    "BM25_DIRECTORY",
                    "data/bm25",
                )
            )

            path = (
                directory
                / "bm25_index.json"
            )

        self.bm25.save(
            path
        )

    def load_bm25(
        self,
        path: str | Path | None = None,
    ) -> int:
        """Load the persisted BM25 corpus."""

        if path is None:

            directory = Path(
                getattr(
                    settings,
                    "BM25_DIRECTORY",
                    "data/bm25",
                )
            )

            path = (
                directory
                / "bm25_index.json"
            )

        return self.bm25.load(
            path
        )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    def bm25_size(self) -> int:
        """Return number of chunks currently indexed by BM25."""

        return self.bm25.size()


# ============================================================
# Factory
# ============================================================


def create_hybrid_indexer(
    *,
    semantic_retriever: SemanticRetriever,
    bm25_backend: BM25Backend | None = None,
) -> HybridIndexer:
    """
    Create the application hybrid indexer.

    Dependencies are injectable so unit tests can use mocked
    Pinecone/embedding components without external services.
    """

    return HybridIndexer(
        semantic_retriever=semantic_retriever,
        bm25_backend=bm25_backend,
    )


__all__ = [
    "SemanticRetriever",
    "BM25Record",
    "BM25Backend",
    "HybridSearchResult",
    "HybridIndexer",
    "create_hybrid_indexer",
]
