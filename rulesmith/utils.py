"""Utility functions for code hashing, environment detection, and serialization."""

import hashlib
import inspect
import json
import sys
from typing import Any, Callable, Dict


def hash_code(func: Callable) -> str:
    """
    Compute SHA256 hash of a function's source code.

    Args:
        func: The function to hash

    Returns:
        SHA256 hex digest of the function's source
    """
    try:
        source = inspect.getsource(func)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        # If source unavailable, hash the function's qualified name
        qualname = getattr(func, "__qualname__", str(func))
        return hashlib.sha256(qualname.encode("utf-8")).hexdigest()


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_pip_freeze() -> Dict[str, str]:
    """
    Get installed packages and versions.

    Returns:
        Dictionary mapping package names to versions
    """
    try:
        import subprocess

        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        packages = {}
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                name, version = line.split("==", 1)
                packages[name] = version
        return packages
    except Exception:
        return {}


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string
    """
    try:
        import orjson

        return orjson.dumps(obj, **kwargs).decode("utf-8")
    except ImportError:
        return json.dumps(obj, default=str, **kwargs)


def safe_json_loads(s: str, **kwargs) -> Any:
    """
    Safely deserialize JSON string to object.

    Args:
        s: JSON string
        **kwargs: Additional arguments for json.loads

    Returns:
        Deserialized object
    """
    try:
        import orjson

        return orjson.loads(s, **kwargs)
    except ImportError:
        return json.loads(s, **kwargs)

