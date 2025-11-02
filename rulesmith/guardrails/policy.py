"""Guard policy definitions and decorator."""

from enum import Enum
from typing import Any, Callable, Dict, Optional

import functools


class GuardAction(str, Enum):
    """Guard action types."""

    ALLOW = "allow"
    BLOCK = "block"
    OVERRIDE = "override"
    FLAG = "flag"


class GuardPolicy:
    """Guard policy definition."""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        when_node: Optional[str] = None,
        checks: Optional[list[str]] = None,
        on_fail: GuardAction = GuardAction.BLOCK,
        override_template: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.version = version
        self.when_node = when_node
        self.checks = checks or []
        self.on_fail = on_fail
        self.override_template = override_template or {}


def guard(name: str):
    """
    Decorator for guard functions.

    Args:
        name: Guard name

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._guard_name = name
        wrapper._is_guard = True

        return wrapper

    return decorator

