"""Tests for security features."""

import pytest

from rulesmith.security.redaction import RedactionConfig, redact_dict, redact_text, suppress_fields
from rulesmith.security.secrets import detect_secrets, validate_no_secrets


class TestRedaction:
    """Test PII redaction."""

    def test_redact_email(self):
        """Test email redaction."""
        config = RedactionConfig(redact_emails=True)
        text = "Contact me at user@example.com for details"
        result = redact_text(text, config)
        assert "user@example.com" not in result
        assert "[REDACTED]" in result

    def test_redact_dict(self):
        """Test dictionary redaction."""
        config = RedactionConfig(redact_emails=True, redact_phones=True)
        data = {
            "name": "John",
            "email": "john@example.com",
            "phone": "123-456-7890",
        }
        result = redact_dict(data, config)
        assert result["name"] == "John"  # Not redacted
        assert "[REDACTED]" in result["email"]
        assert "[REDACTED]" in result["phone"]

    def test_suppress_fields(self):
        """Test field suppression."""
        data = {"public": "data", "secret": "hidden"}
        result = suppress_fields(data, ["secret"])
        assert "secret" not in result
        assert "public" in result


class TestSecrets:
    """Test secret detection."""

    def test_detect_secrets(self):
        """Test secret detection."""
        data = {
            "api_key": "sk-1234567890abcdef",
            "password": "secret123",
        }
        secrets = detect_secrets(data)
        assert len(secrets) > 0

    def test_validate_no_secrets(self):
        """Test secret validation."""
        data = {"name": "test", "value": "normal"}
        is_valid, issues = validate_no_secrets(data)
        assert is_valid is True or len(issues) == 0

