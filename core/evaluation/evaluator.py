"""
Kokoro - Evaluation Module.

Responsibilities:
    - Evaluate retrieval quality using Precision@K, Recall@K,
      MRR and nDCG.
    - Evaluate RAG responses using RAGAS when enabled.
    - Keep retrieval evaluation separate from generation.
    - Return structured evaluation results.

This module does NOT:
    - perform retrieval
    - generate answers
    - modify Pinecone/BM25 indexes
    - manage conversation memory
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from utils.config import get_settings
from utils.logger import logger
from utils.schemas import (
    EvaluationSample,
    EvaluationResult,
    RagasEvaluationResult,
    RetrievalEvaluationResult,
)


# ============================================================
# Retrieval Metrics
# ============================================================


def precision_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K =
        relevant retrieved items in top K / K
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = list(ranked_ids[:k])

    if not retrieved:
        return 0.0

    hits = sum(
        1
        for item_id in retrieved
        if item_id in relevant
    )

    return hits / len(retrieved)


def recall_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K =
        relevant retrieved items in top K / total relevant items
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(
        ranked_ids[:k]
    )

    hits = len(
        retrieved.intersection(relevant)
    )

    return hits / len(relevant)


def reciprocal_rank(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    Returns:
        1 / rank of the first relevant result.

    Returns 0 if no relevant result is found.
    """

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    for rank, item_id in enumerate(
        ranked_ids,
        start=1,
    ):
        if item_id in relevant:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate Discounted Cumulative Gain@K.

    Binary relevance is used:
        relevant = 1
        non-relevant = 0
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    score = 0.0

    for rank, item_id in enumerate(
        ranked_ids[:k],
        start=1,
    ):
        relevance = (
            1.0
            if item_id in relevant
            else 0.0
        )

        score += relevance / math.log2(
            rank + 1
        )

    return score


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate normalized Discounted Cumulative Gain@K.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    actual_dcg = dcg_at_k(
        ranked_ids,
        relevant,
        k,
    )

    ideal_ids = list(relevant)

    ideal_dcg = dcg_at_k(
        ideal_ids,
        relevant,
        min(k, len(ideal_ids)),
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# Retrieval Evaluator
# ============================================================


class RetrievalEvaluator:
    """
    Evaluate retrieval results against known relevant IDs.
    """

    def __init__(
        self,
        k_values: Sequence[int] | None = None,
    ) -> None:

        settings = get_settings()

        if k_values is None:
            k_values = settings.get_evaluation_k_values()

        self.k_values = sorted(
            {
                int(k)
                for k in k_values
                if int(k) > 0
            }
        )

        if not self.k_values:
            raise ValueError(
                "At least one valid K value is required."
            )

    def evaluate(
        self,
        *,
        query: str,
        ranked_ids: Sequence[str],
        relevant_ids: Iterable[str],
    ) -> RetrievalEvaluationResult:
        """
        Evaluate one retrieval query.
        """

        relevant = list(
            dict.fromkeys(
                relevant_ids
            )
        )

        precision_scores: dict[str, float] = {}
        recall_scores: dict[str, float] = {}
        ndcg_scores: dict[str, float] = {}

        for k in self.k_values:

            precision_scores[str(k)] = (
                precision_at_k(
                    ranked_ids,
                    relevant,
                    k,
                )
            )

            recall_scores[str(k)] = (
                recall_at_k(
                    ranked_ids,
                    relevant,
                    k,
                )
            )

            ndcg_scores[str(k)] = (
                ndcg_at_k(
                    ranked_ids,
                    relevant,
                    k,
                )
            )

        mrr = reciprocal_rank(
            ranked_ids,
            relevant,
        )

        return RetrievalEvaluationResult(
            query=query,
            precision_at_k=precision_scores,
            recall_at_k=recall_scores,
            mrr=mrr,
            ndcg_at_k=ndcg_scores,
        )


# ============================================================
# RAGAS Evaluator
# ============================================================


class RagasEvaluator:
    """
    Optional RAGAS evaluation wrapper.

    RAGAS is loaded lazily so the core application does not
    require RAGAS at import time.
    """

    def __init__(
        self,
        enabled: bool | None = None,
    ) -> None:

        settings = get_settings()

        self.enabled = (
            enabled
            if enabled is not None
            else settings.ENABLE_RAGAS
        )

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        reference: str | None = None,
    ) -> RagasEvaluationResult | None:
        """
        Evaluate one RAG response using RAGAS.

        If RAGAS is disabled or unavailable, returns None.
        """

        if not self.enabled:
            return None

        try:
            from ragas import evaluate
            from ragas import EvaluationDataset
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except ImportError:

            logger.warning(
                "RAGAS is enabled but not installed."
            )

            return None

        sample: dict[str, Any] = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": list(contexts),
        }

        if reference is not None:
            sample["reference"] = reference

        dataset = EvaluationDataset.from_list(
            [sample]
        )

        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
        ]

        if reference is not None:
            metrics.extend(
                [
                    ContextPrecision(),
                    ContextRecall(),
                ]
            )

        try:

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
            )

            scores = dict(
                result.scores[0]
            )

            return RagasEvaluationResult(
                faithfulness=float(
                    scores.get(
                        "faithfulness",
                        0.0,
                    )
                ),
                answer_relevancy=float(
                    scores.get(
                        "answer_relevancy",
                        0.0,
                    )
                ),
                context_precision=float(
                    scores.get(
                        "context_precision",
                        0.0,
                    )
                ),
                context_recall=float(
                    scores.get(
                        "context_recall",
                        0.0,
                    )
                ),
            )

        except Exception as exc:

            logger.exception(
                "RAGAS evaluation failed: %s",
                exc,
            )

            return None


# ============================================================
# Evaluation Service
# ============================================================


class Evaluator:
    """
    Main Kokoro evaluation service.

    Combines:
        - retrieval metrics
        - optional RAGAS metrics
    """

    def __init__(
        self,
        *,
        retrieval_evaluator: RetrievalEvaluator | None = None,
        ragas_evaluator: RagasEvaluator | None = None,
    ) -> None:

        self.settings = get_settings()

        self.retrieval_evaluator = (
            retrieval_evaluator
            or RetrievalEvaluator()
        )

        self.ragas_evaluator = (
            ragas_evaluator
            or RagasEvaluator()
        )

    # ========================================================
    # Retrieval Evaluation
    # ========================================================

    def evaluate_retrieval(
        self,
        *,
        query: str,
        ranked_ids: Sequence[str],
        relevant_ids: Iterable[str],
    ) -> RetrievalEvaluationResult:
        """
        Evaluate retrieval quality.
        """

        return self.retrieval_evaluator.evaluate(
            query=query,
            ranked_ids=ranked_ids,
            relevant_ids=relevant_ids,
        )

    # ========================================================
    # RAG Evaluation
    # ========================================================

    def evaluate_generation(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        reference: str | None = None,
    ) -> RagasEvaluationResult | None:
        """
        Evaluate generated answer using RAGAS.
        """

        return self.ragas_evaluator.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
        )

    # ========================================================
    # Complete Evaluation
    # ========================================================

    def evaluate(
        self,
        *,
        sample: EvaluationSample,
        ranked_ids: Sequence[str],
        contexts: Sequence[str],
        answer: str,
    ) -> EvaluationResult:
        """
        Run retrieval and optional generation evaluation
        for one evaluation sample.
        """

        retrieval_result = (
            self.evaluate_retrieval(
                query=sample.question,
                ranked_ids=ranked_ids,
                relevant_ids=sample.relevant_ids,
            )
        )

        ragas_result = (
            self.evaluate_generation(
                question=sample.question,
                answer=answer,
                contexts=contexts,
                reference=sample.reference_answer,
            )
        )

        return EvaluationResult(
            retrieval=retrieval_result,
            ragas=ragas_result,
        )


# ============================================================
# Factory
# ============================================================


_evaluator: Evaluator | None = None


def create_evaluator() -> Evaluator:
    """
    Return the shared evaluator instance.
    """

    global _evaluator

    if _evaluator is None:
        _evaluator = Evaluator()

    return _evaluator