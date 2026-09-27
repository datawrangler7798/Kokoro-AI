"""
Kokoro - Gemini Embedding Service.

Responsibilities:
    - Generate embeddings using Gemini Embedding API.
    - Use the same embedding model for documents and queries.
    - Use task-specific configuration:
        * RETRIEVAL_DOCUMENT for resume/JD chunks
        * RETRIEVAL_QUERY for recruiter queries
    - Batch document embedding requests.
    - Validate embedding dimensions.
    - Apply rate limiting.
    - Retry transient API failures.

This module does NOT:
    - perform Pinecone operations
    - perform BM25 retrieval
    - perform reranking
    - generate final answers
"""

from __future__ import annotations

import time
from typing import Sequence

from google import genai
from google.genai import types
from langchain_core.documents import Document

from utils.config import get_settings
from utils.logger import logger
from utils.utils import RateLimiter


# ============================================================
# Constants
# ============================================================

DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 8.0


# ============================================================
# Embedding Service
# ============================================================


class GeminiEmbeddingService:
    """
    Production wrapper around the Gemini Embedding API.

    The service creates one Google GenAI client and reuses it.

    Document embeddings and query embeddings use the same model
    but different task types.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
    ) -> None:

        self.config = get_settings()

        self.model_name = (
            model_name
            or self.config.EMBEDDING_MODEL
        )

        self.dimension = (
            dimension
            or self.config.EMBEDDING_DIMENSION
        )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        self.batch_size = batch_size
        self.max_retries = max_retries

        self._client: genai.Client | None = None

        self._rate_limiter = RateLimiter(
            max_calls=self.config.LLM_REQUESTS_PER_MINUTE,
            window_seconds=60.0,
        )

    # ========================================================
    # Client
    # ========================================================

    def _get_client(self) -> genai.Client:
        """
        Lazily initialize the Google GenAI client.
        """

        if self._client is None:

            if not self.config.GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY is required "
                    "for Gemini embeddings."
                )

            self._client = genai.Client(
                api_key=self.config.GOOGLE_API_KEY.get_secret_value()
                if hasattr(self.config.GOOGLE_API_KEY, "get_secret_value")
                else str(self.config.GOOGLE_API_KEY)
            )

        return self._client

    # ========================================================
    # Validation
    # ========================================================

    def _validate_embedding(
        self,
        embedding: Sequence[float],
    ) -> list[float]:
        """
        Validate a single embedding.

        Ensures:
            - embedding exists
            - dimension matches configuration
            - values are numeric
        """

        if not embedding:
            raise ValueError(
                "Gemini returned an empty embedding."
            )

        vector = [
            float(value)
            for value in embedding
        ]

        if len(vector) != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {len(vector)}."
            )

        return vector

    # ========================================================
    # API Request
    # ========================================================

    def _embed_batch(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[list[float]]:
        """
        Send one embedding request to Gemini.

        Args:
            texts:
                Texts to embed.

            task_type:
                Gemini embedding task type.

        Returns:
            List of embedding vectors.
        """

        if not texts:
            return []

        client = self._get_client()

        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):

            try:

                self._rate_limiter.acquire()

                response = client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimension,
                    ),
                )

                embeddings = response.embeddings

                if not embeddings:
                    raise RuntimeError(
                        "Gemini returned no embeddings."
                    )

                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        "Gemini embedding response count "
                        "does not match input count. "
                        f"inputs={len(texts)} "
                        f"outputs={len(embeddings)}"
                    )

                result: list[list[float]] = []

                for embedding in embeddings:

                    values = getattr(
                        embedding,
                        "values",
                        None,
                    )

                    if values is None:
                        raise RuntimeError(
                            "Gemini returned an embedding "
                            "without vector values."
                        )

                    result.append(
                        self._validate_embedding(values)
                    )

                return result

            except Exception as exc:

                last_exception = exc

                if attempt >= self.max_retries:
                    break

                delay = min(
                    INITIAL_RETRY_DELAY_SECONDS
                    * (2 ** attempt),
                    MAX_RETRY_DELAY_SECONDS,
                )

                logger.warning(
                    "Gemini embedding request failed "
                    "(attempt %d/%d). Retrying in %.1fs.",
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )

                time.sleep(delay)

        raise RuntimeError(
            "Gemini embedding request failed "
            f"after {self.max_retries + 1} attempts."
        ) from last_exception

    # ========================================================
    # Document Embeddings
    # ========================================================

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for arbitrary document text.

        This method uses RETRIEVAL_DOCUMENT and is intended for
        resume/JD chunks.

        Args:
            texts:
                Texts to embed.

        Returns:
            List of configured-dimensional vectors.
        """

        if not texts:
            return []

        normalized_texts: list[str] = []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "All embedding inputs must be strings."
                )

            cleaned = text.strip()

            if not cleaned:
                raise ValueError(
                    "Cannot embed empty text."
                )

            normalized_texts.append(cleaned)

        all_embeddings: list[list[float]] = []

        for start in range(
            0,
            len(normalized_texts),
            self.batch_size,
        ):

            batch = normalized_texts[
                start : start + self.batch_size
            ]

            embeddings = self._embed_batch(
                batch,
                task_type="RETRIEVAL_DOCUMENT",
            )

            all_embeddings.extend(embeddings)

        if len(all_embeddings) != len(normalized_texts):
            raise RuntimeError(
                "Final embedding count does not match "
                "input count."
            )

        return all_embeddings

    # ========================================================
    # LangChain Documents
    # ========================================================

    def embed_documents(
        self,
        documents: Sequence[Document],
    ) -> list[list[float]]:
        """
        Generate embeddings for LangChain Documents.

        The page/chunk metadata is not embedded; only
        page_content is sent to Gemini.
        """

        if not documents:
            return []

        texts = [
            document.page_content
            for document in documents
        ]

        return self.embed_texts(texts)

    # ========================================================
    # Query Embedding
    # ========================================================

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a recruiter query.

        Uses RETRIEVAL_QUERY.

        The query uses the same embedding model and
        dimensionality as the indexed documents.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        embeddings = self._embed_batch(
            [query],
            task_type="RETRIEVAL_QUERY",
        )

        if len(embeddings) != 1:
            raise RuntimeError(
                "Expected exactly one query embedding."
            )

        return embeddings[0]

    # ========================================================
    # Dimension
    # ========================================================

    @property
    def embedding_dimension(self) -> int:
        """
        Return configured embedding dimension.
        """

        return self.dimension

    # ========================================================
    # Model
    # ========================================================

    @property
    def embedding_model(self) -> str:
        """
        Return configured embedding model.
        """

        return self.model_name

    # ========================================================
    # Health Check
    # ========================================================

    def health_check(self) -> bool:
        """
        Verify that the embedding service can successfully
        generate an embedding.

        Returns:
            True if successful.

        Raises:
            Exception if the embedding request fails.
        """

        vector = self.embed_query(
            "health check"
        )

        if len(vector) != self.dimension:
            raise RuntimeError(
                "Embedding health check returned "
                "an unexpected dimension."
            )

        return True


# ============================================================
# Singleton
# ============================================================

_embedding_service: GeminiEmbeddingService | None = None


def get_embedding_service() -> GeminiEmbeddingService:
    """
    Return the shared Gemini embedding service.
    """

    global _embedding_service

    if _embedding_service is None:
        _embedding_service = GeminiEmbeddingService()

    return _embedding_service