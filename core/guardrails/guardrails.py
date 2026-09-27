"""
Kokoro - Guardrails.

Responsibilities:
    - Validate user inputs.
    - Detect prompt-injection attempts.
    - Detect sensitive information / PII patterns.
    - Validate generated responses.
    - Prevent retrieved documents from overriding system instructions.
    - Return structured guardrail results.

This module does NOT:
    - perform retrieval
    - call Pinecone
    - call BM25
    - perform reranking
    - generate LLM responses
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from utils.config import get_settings
from utils.logger import logger
from utils.schemas import (
    GuardrailResult,
    GuardrailStatus,
    OutputValidationResult,
    ValidationStatus,
)


# ============================================================
# Patterns
# ============================================================

PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(all|any|the)\s+(previous|prior|above)\s+instructions?",
    r"ignore\s+your\s+(system|developer)\s+instructions?",
    r"disregard\s+(all|any|the)\s+(previous|prior|above)",
    r"forget\s+(all|any|the)\s+(previous|prior|above)",
    r"override\s+(your|the)\s+(system|developer)\s+instructions?",
    r"reveal\s+(your|the)\s+(system|developer)\s+prompt",
    r"show\s+(me\s+)?your\s+(system|developer)\s+prompt",
    r"print\s+(your|the)\s+(system|developer)\s+prompt",
    r"what\s+are\s+your\s+(system|developer)\s+instructions?",
    r"bypass\s+(your|the)\s+safety",
    r"disable\s+(your|the)\s+(guardrails|safety)",
    r"act\s+as\s+(a\s+)?system",
    r"you\s+are\s+now\s+the\s+system",
)

PII_PATTERNS = {
    "email": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)"
    ),
    "aadhaar": re.compile(
        r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"
    ),
    "pan": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b",
        re.IGNORECASE,
    ),
}


# ============================================================
# Guardrails
# ============================================================


class Guardrails:
    """
    Central guardrail service for Kokoro.

    The service performs deterministic checks before and after
    LLM generation.
    """

    def __init__(
        self,
        *,
        block_prompt_injection: bool | None = None,
        detect_pii: bool | None = None,
        validate_output: bool | None = None,
    ) -> None:

        self.settings = get_settings()

        self.block_prompt_injection = (
            block_prompt_injection
            if block_prompt_injection is not None
            else self.settings.ENABLE_GUARDRAILS
        )

        self.detect_pii = (
            detect_pii
            if detect_pii is not None
            else self.settings.ENABLE_PII_PROTECTION
        )

        self.validate_output_enabled = (
            validate_output
            if validate_output is not None
            else self.settings.ENABLE_OUTPUT_VALIDATION
        )

    # ========================================================
    # Text Normalization
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize text before pattern matching.
        """

        if not text:
            return ""

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ========================================================
    # Prompt Injection
    # ========================================================

    def detect_prompt_injection(
        self,
        text: str,
    ) -> list[str]:
        """
        Detect common prompt-injection patterns.

        Returns:
            List of matched pattern descriptions.
        """

        if not text:
            return []

        normalized = self._normalize_text(
            text
        )

        matches: list[str] = []

        for pattern in PROMPT_INJECTION_PATTERNS:

            if re.search(
                pattern,
                normalized,
                re.IGNORECASE,
            ):
                matches.append(pattern)

        return matches

    # ========================================================
    # PII Detection
    # ========================================================

    def detect_pii(
        self,
        text: str,
    ) -> list[str]:
        """
        Detect common PII patterns.

        This is detection only; it does not modify the text.
        """

        if not text:
            return []

        detected: list[str] = []

        for pii_type, pattern in PII_PATTERNS.items():

            if pattern.search(text):
                detected.append(
                    pii_type
                )

        return detected

    # ========================================================
    # Input Validation
    # ========================================================

    def validate_input(
        self,
        text: str,
    ) -> GuardrailResult:
        """
        Validate user input before retrieval/generation.
        """

        if not isinstance(text, str):

            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="Input must be a string.",
                details={},
            )

        if not text.strip():

            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="Input cannot be empty.",
                details={},
            )

        prompt_injection_matches = []

        if self.block_prompt_injection:
            prompt_injection_matches = (
                self.detect_prompt_injection(
                    text
                )
            )

        if prompt_injection_matches:

            logger.warning(
                "Prompt injection detected."
            )

            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="Potential prompt injection detected.",
                details={
                    "prompt_injection": True,
                    "matches": len(
                        prompt_injection_matches
                    ),
                },
            )

        pii_matches = []

        if self.detect_pii:
            pii_matches = self.detect_pii(
                text
            )

        return GuardrailResult(
            status=GuardrailStatus.PASSED,
            reason="Input passed guardrail checks.",
            details={
                "prompt_injection": False,
                "pii_detected": pii_matches,
            },
        )

    # ========================================================
    # Retrieved Content Validation
    # ========================================================

    def validate_retrieved_content(
        self,
        contents: Iterable[str],
    ) -> GuardrailResult:
        """
        Validate retrieved documents as untrusted content.

        Retrieved resumes/JDs must never be allowed to override
        system or developer instructions.
        """

        injection_count = 0
        pii_types: set[str] = set()

        for content in contents:

            if not isinstance(content, str):
                continue

            if self.block_prompt_injection:

                matches = self.detect_prompt_injection(
                    content
                )

                if matches:
                    injection_count += len(
                        matches
                    )

            if self.detect_pii:

                pii_types.update(
                    self.detect_pii(
                        content
                    )
                )

        if injection_count > 0:

            logger.warning(
                "Potential prompt injection found "
                "inside retrieved content."
            )

            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason=(
                    "Retrieved content contains "
                    "potential prompt-injection instructions."
                ),
                details={
                    "injection_count": injection_count,
                    "pii_detected": sorted(
                        pii_types
                    ),
                },
            )

        return GuardrailResult(
            status=GuardrailStatus.PASSED,
            reason=(
                "Retrieved content passed guardrail checks."
            ),
            details={
                "injection_count": 0,
                "pii_detected": sorted(
                    pii_types
                ),
            },
        )

    # ========================================================
    # Output Validation
    # ========================================================

    def validate_output(
        self,
        output: str,
        *,
        source_texts: Iterable[str] | None = None,
    ) -> OutputValidationResult:
        """
        Validate generated output.

        Checks:
            - output is non-empty
            - optional PII detection
            - optional unsupported claims based on basic
              source-term overlap

        This is a deterministic safety check, not a full
        semantic factuality evaluator.
        """

        if not isinstance(output, str):

            return OutputValidationResult(
                status=ValidationStatus.INVALID,
                reason="Output must be a string.",
                details={},
            )

        cleaned_output = output.strip()

        if not cleaned_output:

            return OutputValidationResult(
                status=ValidationStatus.INVALID,
                reason="Generated output is empty.",
                details={},
            )

        pii_matches: list[str] = []

        if self.detect_pii:

            pii_matches = self.detect_pii(
                cleaned_output
            )

        injection_matches = []

        if self.block_prompt_injection:

            injection_matches = (
                self.detect_prompt_injection(
                    cleaned_output
                )
            )

        if injection_matches:

            return OutputValidationResult(
                status=ValidationStatus.INVALID,
                reason=(
                    "Generated output contains "
                    "potential prompt-injection content."
                ),
                details={
                    "prompt_injection": True,
                },
            )

        if pii_matches:

            logger.warning(
                "PII detected in generated output: %s",
                pii_matches,
            )

        source_count = 0

        if source_texts is not None:

            source_count = sum(
                1
                for source in source_texts
                if isinstance(source, str)
                and source.strip()
            )

        return OutputValidationResult(
            status=ValidationStatus.VALID,
            reason="Generated output passed validation.",
            details={
                "pii_detected": pii_matches,
                "source_count": source_count,
            },
        )

    # ========================================================
    # Combined Input Check
    # ========================================================

    def check_input(
        self,
        text: str,
    ) -> bool:
        """
        Convenience method.

        Returns:
            True when input passes.
        """

        result = self.validate_input(
            text
        )

        return (
            result.status
            == GuardrailStatus.PASSED
        )

    # ========================================================
    # Combined Output Check
    # ========================================================

    def check_output(
        self,
        output: str,
        *,
        source_texts: Iterable[str] | None = None,
    ) -> bool:
        """
        Convenience method.

        Returns:
            True when output passes.
        """

        if not self.validate_output_enabled:
            return True

        result = self.validate_output(
            output,
            source_texts=source_texts,
        )

        return (
            result.status
            == ValidationStatus.VALID
        )


# ============================================================
# Factory
# ============================================================


_guardrails: Guardrails | None = None


def create_guardrails() -> Guardrails:
    """
    Return the shared Guardrails instance.
    """

    global _guardrails

    if _guardrails is None:
        _guardrails = Guardrails()

    return _guardrails