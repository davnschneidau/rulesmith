"""Utility modules for Rulesmith."""

# Import hash_code from parent utils.py (one level up)
import sys
from pathlib import Path

# Import from parent utils.py
_utils_py = Path(__file__).parent.parent / "utils.py"
if _utils_py.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_utils", _utils_py)
    if spec and spec.loader:
        _utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_utils)
        hash_code = _utils.hash_code
        del _utils, spec

from rulesmith.utils.logging import configure_logging, get_logger, log_error, log_warning

__all__ = [
    "get_logger",
    "log_error",
    "log_warning",
    "configure_logging",
]

# Add hash_code if imported
if "hash_code" in globals():
    __all__.append("hash_code")

