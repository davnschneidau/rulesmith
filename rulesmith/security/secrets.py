"""Secret management validation."""

import os
import re
from typing import Dict, List, Optional, Set


def detect_secrets(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Detect potential secrets in data.

    Args:
        data: Data dictionary

    Returns:
        List of detected secrets with location information
    """
    secrets = []

    # Common secret patterns
    patterns = {
        "api_key": [
            r'api[_-]?key["\s:=]+([a-zA-Z0-9_-]{20,})',
            r'apikey["\s:=]+([a-zA-Z0-9_-]{20,})',
        ],
        "access_token": [
            r'access[_-]?token["\s:=]+([a-zA-Z0-9_-]{20,})',
            r'bearer\s+([a-zA-Z0-9_.-]{20,})',
        ],
        "password": [
            r'password["\s:=]+([^\s"]{8,})',
            r'pwd["\s:=]+([^\s"]{8,})',
        ],
        "secret": [
            r'secret["\s:=]+([a-zA-Z0-9_-]{16,})',
            r'secret[_-]?key["\s:=]+([a-zA-Z0-9_-]{16,})',
        ],
    }

    def scan_value(value: Any, path: str = ""):
        """Recursively scan value for secrets."""
        if isinstance(value, str):
            for secret_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.finditer(pattern, value, re.IGNORECASE)
                    for match in matches:
                        secrets.append({
                            "type": secret_type,
                            "location": path,
                            "pattern": pattern,
                            "snippet": match.group(0)[:50],  # Truncate
                        })
        elif isinstance(value, dict):
            for k, v in value.items():
                new_path = f"{path}.{k}" if path else k
                scan_value(v, new_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                new_path = f"{path}[{i}]"
                scan_value(item, new_path)

    scan_value(data)
    return secrets


def validate_no_secrets(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate that data contains no secrets.

    Args:
        data: Data dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    detected = detect_secrets(data)
    if detected:
        issues = [
            f"Potential {s['type']} found at {s['location']}: {s['snippet']}"
            for s in detected
        ]
        return False, issues

    return True, []


def get_secret_from_env(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """
    Get secret from environment variable with validation.

    Args:
        key: Environment variable key
        default: Optional default value
        required: Whether secret is required

    Returns:
        Secret value

    Raises:
        ValueError: If required secret is missing
    """
    value = os.getenv(key, default)

    if value is None and required:
        raise ValueError(f"Required environment variable '{key}' is not set")

    return value
