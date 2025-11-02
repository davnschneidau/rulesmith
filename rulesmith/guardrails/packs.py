"""Pre-built guardrail packs (PII, toxicity, hallucination, etc.)."""

import re
from typing import Any, Dict, List, Optional

from rulesmith.guardrails.policy import GuardAction, GuardPolicy


# PII Detection Guards
def detect_email(text: str) -> List[str]:
    """Detect email addresses in text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def detect_phone(text: str) -> List[str]:
    """Detect phone numbers in text."""
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    return re.findall(phone_pattern, text)


def detect_ssn(text: str) -> List[str]:
    """Detect SSN-like patterns in text."""
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    return re.findall(ssn_pattern, text)


@guard(name="pii_email")
def pii_email_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Guard to detect email addresses."""
    text = str(inputs.get("text", inputs.get("input", inputs.get("prompt", ""))))
    emails = detect_email(text)

    return {
        "passed": len(emails) == 0,
        "message": f"Found {len(emails)} email address(es)" if emails else None,
        "metadata": {"emails": emails} if emails else {},
    }


@guard(name="pii_phone")
def pii_phone_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Guard to detect phone numbers."""
    text = str(inputs.get("text", inputs.get("input", inputs.get("prompt", ""))))
    phones = detect_phone(text)

    return {
        "passed": len(phones) == 0,
        "message": f"Found {len(phones)} phone number(s)" if phones else None,
        "metadata": {"phones": phones} if phones else {},
    }


@guard(name="pii_ssn")
def pii_ssn_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Guard to detect SSN patterns."""
    text = str(inputs.get("text", inputs.get("input", inputs.get("prompt", ""))))
    ssns = detect_ssn(text)

    return {
        "passed": len(ssns) == 0,
        "message": f"Found {len(ssns)} SSN pattern(s)" if ssns else None,
        "metadata": {"ssns": ssns} if ssns else {},
    }


# Toxicity Detection Guards
@guard(name="toxicity_basic")
def toxicity_basic_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Basic toxicity guard using keyword matching.

    Note: For production, use a proper toxicity detection model.
    """
    text = str(inputs.get("text", inputs.get("input", inputs.get("prompt", "")))).lower()

    # Basic toxicity keywords (very simplified)
    toxic_keywords = ["hate", "kill", "die", "stupid", "idiot"]  # Minimal set for example

    found_keywords = [kw for kw in toxic_keywords if kw in text]

    return {
        "passed": len(found_keywords) == 0,
        "message": f"Found potentially toxic content: {found_keywords}" if found_keywords else None,
        "score": len(found_keywords) / len(toxic_keywords) if found_keywords else 0.0,
        "metadata": {"keywords": found_keywords} if found_keywords else {},
    }


# Hallucination Detection Guards
@guard(name="hallucination_citations")
def hallucination_citations_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Guard to check for citations/references (indicates factual claims)."""
    text = str(inputs.get("text", inputs.get("output", "")))

    # Check for citation patterns
    has_citations = bool(re.search(r'\[.*?\]|\(.*?\)|http', text))

    return {
        "passed": has_citations,  # Pass if citations present
        "message": "No citations found in output" if not has_citations else None,
        "metadata": {"has_citations": has_citations},
    }


@guard(name="hallucination_confidence")
def hallucination_confidence_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Guard to check for confidence markers in output."""
    text = str(inputs.get("text", inputs.get("output", ""))).lower()

    # Low confidence markers
    low_confidence = ["i think", "probably", "maybe", "perhaps", "might", "uncertain"]

    # High confidence markers (absolute claims)
    high_confidence = ["always", "never", "all", "none", "definitely", "certainly"]

    low_conf_count = sum(1 for marker in low_confidence if marker in text)
    high_conf_count = sum(1 for marker in high_confidence if marker in text)

    # Flag if high confidence claims without evidence
    risk_score = high_conf_count / max(len(text.split()), 1) if high_conf_count > 0 else 0.0

    return {
        "passed": risk_score < 0.1,  # Threshold for hallucination risk
        "message": f"High confidence claims detected (risk: {risk_score:.2f})" if risk_score >= 0.1 else None,
        "score": risk_score,
        "metadata": {
            "low_confidence_markers": low_conf_count,
            "high_confidence_markers": high_conf_count,
        },
    }


# Output Validation Guards
@guard(name="output_length")
def output_length_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Guard to check output length."""
    text = str(inputs.get("text", inputs.get("output", "")))
    min_length = inputs.get("min_length", 10)
    max_length = inputs.get("max_length", 10000)

    length = len(text)

    passed = min_length <= length <= max_length
    message = None
    if length < min_length:
        message = f"Output too short ({length} < {min_length})"
    elif length > max_length:
        message = f"Output too long ({length} > {max_length})"

    return {
        "passed": passed,
        "message": message,
        "metadata": {"length": length, "min_length": min_length, "max_length": max_length},
    }


@guard(name="output_format")
def output_format_guard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Guard to check output format (JSON, etc.)."""
    text = str(inputs.get("text", inputs.get("output", "")))
    expected_format = inputs.get("format", "text")

    if expected_format == "json":
        try:
            import json

            json.loads(text)
            passed = True
            message = None
        except json.JSONDecodeError:
            passed = False
            message = "Output is not valid JSON"

        return {
            "passed": passed,
            "message": message,
            "metadata": {"format": expected_format, "is_valid_json": passed},
        }

    # For text format, just check non-empty
    return {
        "passed": len(text.strip()) > 0,
        "message": "Output is empty" if len(text.strip()) == 0 else None,
        "metadata": {"format": expected_format},
    }


# Guard Packs
class GuardPack:
    """Collection of related guards."""

    def __init__(self, name: str, guards: List[str], default_action: GuardAction = GuardAction.BLOCK):
        self.name = name
        self.guards = guards
        self.default_action = default_action

    def to_policy(self, when_node: Optional[str] = None) -> GuardPolicy:
        """Convert guard pack to GuardPolicy."""
        return GuardPolicy(
            name=f"{self.name}_pack",
            checks=self.guards,
            when_node=when_node,
            on_fail=self.default_action,
        )


# Pre-built guard packs
PII_PACK = GuardPack(
    name="pii",
    guards=["pii_email", "pii_phone", "pii_ssn"],
    default_action=GuardAction.BLOCK,
)

TOXICITY_PACK = GuardPack(
    name="toxicity",
    guards=["toxicity_basic"],
    default_action=GuardAction.BLOCK,
)

HALLUCINATION_PACK = GuardPack(
    name="hallucination",
    guards=["hallucination_citations", "hallucination_confidence"],
    default_action=GuardAction.FLAG,
)

OUTPUT_VALIDATION_PACK = GuardPack(
    name="output_validation",
    guards=["output_length", "output_format"],
    default_action=GuardAction.FLAG,
)

ALL_GUARDS_PACK = GuardPack(
    name="all",
    guards=[
        "pii_email",
        "pii_phone",
        "pii_ssn",
        "toxicity_basic",
        "hallucination_citations",
        "hallucination_confidence",
        "output_length",
        "output_format",
    ],
    default_action=GuardAction.BLOCK,
)


def register_default_guards(executor: Optional[Any] = None) -> None:
    """Register all default guards with executor."""
    from rulesmith.guardrails.execution import guard_executor as default_executor

    executor = executor or default_executor

    guards = [
        pii_email_guard,
        pii_phone_guard,
        pii_ssn_guard,
        toxicity_basic_guard,
        hallucination_citations_guard,
        hallucination_confidence_guard,
        output_length_guard,
        output_format_guard,
    ]

    for guard_func in guards:
        if hasattr(guard_func, "_guard_name"):
            executor.register_guard(guard_func._guard_name, guard_func)
