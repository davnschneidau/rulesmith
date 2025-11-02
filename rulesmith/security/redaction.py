"""PII redaction and field suppression."""

import re
from typing import Any, Dict, List, Optional


class RedactionConfig:
    """Configuration for PII redaction."""

    def __init__(
        self,
        redact_emails: bool = True,
        redact_phones: bool = True,
        redact_ssns: bool = True,
        redact_custom_patterns: Optional[List[tuple[str, str]]] = None,
        suppress_fields: Optional[List[str]] = None,
        replacement: str = "[REDACTED]",
    ):
        """
        Initialize redaction config.

        Args:
            redact_emails: Whether to redact email addresses
            redact_phones: Whether to redact phone numbers
            redact_ssns: Whether to redact SSN patterns
            redact_custom_patterns: Optional list of (pattern, replacement) tuples
            suppress_fields: Optional list of field names to completely suppress
            replacement: Replacement string for redacted content
        """
        self.redact_emails = redact_emails
        self.redact_phones = redact_phones
        self.redact_ssns = redact_ssns
        self.redact_custom_patterns = redact_custom_patterns or []
        self.suppress_fields = suppress_fields or []
        self.replacement = replacement


def redact_text(text: str, config: RedactionConfig) -> str:
    """
    Redact PII from text.

    Args:
        text: Text to redact
        config: Redaction configuration

    Returns:
        Redacted text
    """
    result = text

    # Redact emails
    if config.redact_emails:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        result = re.sub(email_pattern, config.replacement, result)

    # Redact phone numbers
    if config.redact_phones:
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        result = re.sub(phone_pattern, config.replacement, result)

    # Redact SSNs
    if config.redact_ssns:
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        result = re.sub(ssn_pattern, config.replacement, result)

    # Redact custom patterns
    for pattern, replacement in config.redact_custom_patterns:
        result = re.sub(pattern, replacement, result)

    return result


def redact_dict(data: Dict[str, Any], config: RedactionConfig) -> Dict[str, Any]:
    """
    Redact PII from dictionary.

    Args:
        data: Dictionary to redact
        config: Redaction configuration

    Returns:
        Redacted dictionary
    """
    result = {}

    for key, value in data.items():
        # Suppress field if in suppress list
        if key in config.suppress_fields:
            continue

        if isinstance(value, str):
            result[key] = redact_text(value, config)
        elif isinstance(value, dict):
            result[key] = redact_dict(value, config)
        elif isinstance(value, list):
            result[key] = [
                redact_dict(item, config) if isinstance(item, dict) else redact_text(item, config) if isinstance(item, str) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def suppress_fields(data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Suppress specified fields from dictionary.

    Args:
        data: Dictionary
        fields: List of field names to suppress

    Returns:
        Dictionary with fields suppressed
    """
    result = {}
    for key, value in data.items():
        if key not in fields:
            if isinstance(value, dict):
                result[key] = suppress_fields(value, fields)
            else:
                result[key] = value
    return result
