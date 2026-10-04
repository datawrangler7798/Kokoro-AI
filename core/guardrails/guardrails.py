from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from utils.config import get_settings
from utils.logger import get_logger
from utils.schemas import GuardrailResult, GuardrailStatus

logger = get_logger(__name__)


# ============================================================
# Prompt Injection Patterns
# ============================================================

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all|any|the)\s+(previous|prior|above)\s+instructions",
    r"ignore\s+previous\s+instructions",
    r"disregard\s+(all|any|the)\s+(previous|prior|above)\s+instructions",
    r"forget\s+(all|any|the)\s+(previous|prior|above)\s+instructions",
    r"system\s+prompt",
    r"reveal\s+(your|the)\s+(system|hidden)\s+prompt",
    r"show\s+(me\s+)?your\s+(system|hidden)\s+prompt",
    r"developer\s+message",
    r"jailbreak",
    r"bypass\s+(your|the)\s+(rules|instructions|guardrails)",
]


# ============================================================
# PII Patterns
# ============================================================

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "pan": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        re.IGNORECASE,
    ),
    "aadhaar": re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
}


# Recruiting requests for sexual services are outside this professional
# candidate-search application and must be stopped before retrieval.
UNSUPPORTED_RECRUITING_PATTERNS = (
    re.compile(r"\b(?:sex\s+work(?:er|ers|r)|sexual\s+workers?)\b", re.IGNORECASE),
    re.compile(r"\bsex\b", re.IGNORECASE),
    re.compile(r"\bprostitut(?:e|es|ion)\b", re.IGNORECASE),
    re.compile(r"\bescorts?\b", re.IGNORECASE),
)


# ============================================================
# Result Model
# ============================================================


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    detected_items: list[str] | None = None

    def __post_init__(self) -> None:
        if self.detected_items is None:
            self.detected_items = []


# ============================================================
# Guardrails
# ============================================================


class Guardrails:
    """
    Input, retrieved-context and output guardrails.

    Responsibilities:
    - Prompt injection detection
    - PII detection
    - Retrieved-context validation
    - Output validation
    """

    def __init__(
        self,
        *,
        block_prompt_injection: bool | None = None,
        detect_pii: bool | None = None,
        validate_output: bool | None = None,
    ) -> None:
        self.settings = get_settings()

        # --------------------------------------------------------
        # IMPORTANT:
        # Use the actual settings names from config.py
        # --------------------------------------------------------

        self.block_prompt_injection = (
            block_prompt_injection
            if block_prompt_injection is not None
            else self.settings.BLOCK_PROMPT_INJECTION
        )

        # Do NOT name this attribute "detect_pii"
        # because detect_pii() is also a method.
        self.detect_pii_enabled = (
            detect_pii
            if detect_pii is not None
            else self.settings.ENABLE_INPUT_GUARDRAIL
        )

        self.validate_output_enabled = (
            validate_output
            if validate_output is not None
            else self.settings.ENABLE_OUTPUT_GUARDRAIL
        )

        logger.info(
            "Guardrails initialized | " "prompt_injection=%s | pii=%s | output=%s",
            self.block_prompt_injection,
            self.detect_pii_enabled,
            self.validate_output_enabled,
        )

    # ============================================================
    # Prompt Injection
    # ============================================================

    def detect_prompt_injection(self, text: str) -> list[str]:
        """
        Detect potential prompt-injection patterns.
        """

        if not text:
            return []

        detected: list[str] = []

        for pattern in PROMPT_INJECTION_PATTERNS:
            try:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    detected.append(pattern)
            except re.error:
                logger.exception(
                    "Invalid prompt injection regex: %s",
                    pattern,
                )

        return detected

    # ============================================================
    # PII Detection
    # ============================================================

    def detect_pii(self, text: str) -> dict[str, list[str]]:
        """
        Detect common PII patterns.
        """

        if not text:
            return {}

        detected: dict[str, list[str]] = {}

        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)

            if matches:
                detected[pii_type] = list(set(matches))

        return detected

    # ============================================================
    # Input Validation
    # ============================================================

    def validate_input(self, text: str) -> GuardrailResult:
        """
        Validate user query before retrieval.
        """

        if not text or not text.strip():
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                passed=False,
                reason="Input query is empty.",
            )

        text = text.strip()

        if any(pattern.search(text) for pattern in UNSUPPORTED_RECRUITING_PATTERNS):
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                passed=False,
                reason=(
                    "Invalid input. Kokoro supports professional recruitment "
                    "and candidate-search requests only. Please provide a "
                    "professional job description or a candidate-related question."
                ),
            )

        # --------------------------------------------------------
        # Prompt Injection
        # --------------------------------------------------------

        if self.block_prompt_injection:
            injection_matches = self.detect_prompt_injection(text)

            if injection_matches:
                logger.warning("Prompt injection detected in user input.")

                return GuardrailResult(
                    status=GuardrailStatus.BLOCKED,
                    passed=False,
                    reason="Potential prompt injection detected.",
                )

        # --------------------------------------------------------
        # PII
        # --------------------------------------------------------

        if self.detect_pii_enabled:
            pii_matches = self.detect_pii(text)

            if pii_matches:
                logger.warning(
                    "PII detected in user input: %s",
                    list(pii_matches.keys()),
                )

                return GuardrailResult(
                    status=GuardrailStatus.BLOCKED,
                    passed=False,
                    reason="Potentially sensitive personal information detected.",
                )

        return GuardrailResult(
            status=GuardrailStatus.PASSED,
            passed=True,
            reason="Input passed guardrails.",
        )

    # ============================================================
    # Retrieved Context Validation
    # ============================================================

    def validate_retrieved_context(
        self,
        documents: list[Any],
    ) -> GuardrailResult:
        """
        Validate retrieved documents before sending them to the LLM.
        """

        if not documents:
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                passed=False,
                reason="No retrieved documents found.",
            )

        for document in documents:
            if isinstance(document, str):
                content = document

            elif hasattr(document, "page_content"):
                content = document.page_content

            elif isinstance(document, dict):
                content = str(
                    document.get(
                        "text",
                        document.get(
                            "content",
                            "",
                        ),
                    )
                )

            else:
                content = str(document)

            if not content.strip():
                continue

            # Prompt injection inside retrieved documents
            if self.block_prompt_injection:
                injection_matches = self.detect_prompt_injection(content)

                if injection_matches:
                    logger.warning("Prompt injection detected in retrieved document.")

                    return GuardrailResult(
                        status=GuardrailStatus.BLOCKED,
                        passed=False,
                        reason=(
                            "Potential prompt injection detected "
                            "inside retrieved context."
                        ),
                    )

            # PII detection
            if self.detect_pii_enabled:
                pii_matches = self.detect_pii(content)

                if pii_matches:
                    logger.warning(
                        "PII detected in retrieved document: %s",
                        list(pii_matches.keys()),
                    )

        return GuardrailResult(
            status=GuardrailStatus.PASSED,
            passed=True,
            reason="Retrieved context passed guardrails.",
        )

    # ============================================================
    # Output Validation
    # ============================================================

    def validate_output(
        self,
        output: str,
    ) -> GuardrailResult:
        """
        Validate generated LLM output.
        """

        if not output or not output.strip():
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                passed=False,
                reason="Generated output is empty.",
            )

        cleaned_output = output.strip()

        # --------------------------------------------------------
        # PII
        # --------------------------------------------------------

        if self.detect_pii_enabled:
            pii_matches = self.detect_pii(cleaned_output)

            if pii_matches:
                logger.warning(
                    "PII detected in generated output: %s",
                    list(pii_matches.keys()),
                )

                return GuardrailResult(
                    status=GuardrailStatus.BLOCKED,
                    passed=False,
                    reason="Generated output contains potentially sensitive information.",
                )

        return GuardrailResult(
            status=GuardrailStatus.PASSED,
            passed=True,
            reason="Generated output passed guardrails.",
        )

    # ============================================================
    # Unified Output Check
    # ============================================================

    def check_output(
        self,
        output: str,
    ) -> GuardrailResult:
        if not self.validate_output_enabled:
            return GuardrailResult(
                status=GuardrailStatus.PASSED,
                passed=True,
                reason="Output guardrail is disabled.",
            )

        return self.validate_output(output)


# ============================================================
# Factory
# ============================================================

_guardrails_instance: Guardrails | None = None


def create_guardrails(
    *,
    block_prompt_injection: bool | None = None,
    detect_pii: bool | None = None,
    validate_output: bool | None = None,
) -> Guardrails:
    global _guardrails_instance

    if _guardrails_instance is None:
        _guardrails_instance = Guardrails(
            block_prompt_injection=block_prompt_injection,
            detect_pii=detect_pii,
            validate_output=validate_output,
        )

    return _guardrails_instance


__all__ = [
    "Guardrails",
    "ValidationResult",
    "create_guardrails",
]
