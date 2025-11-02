"""Shadow mode execution."""

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class ShadowExecutor:
    """Execute function in shadow mode (no side effects)."""

    def __init__(self, shadow_function: Callable[..., T]):
        """
        Initialize shadow executor.

        Args:
            shadow_function: Function to execute in shadow mode
        """
        self.shadow_function = shadow_function

    def execute(self, *args, **kwargs) -> T:
        """
        Execute shadow function.

        Args:
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        return self.shadow_function(*args, **kwargs)


def shadow_mode(
    primary_func: Callable[..., T],
    shadow_func: Optional[Callable[..., T]] = None,
    shadow_probability: float = 1.0,
) -> Callable[..., T]:
    """
    Decorator to execute function in shadow mode.

    Args:
        primary_func: Primary function to execute
        shadow_func: Optional shadow function (defaults to primary_func)
        shadow_probability: Probability of executing shadow (0.0-1.0)

    Returns:
        Wrapped function
    """
    import random

    if shadow_func is None:
        shadow_func = primary_func

    def wrapper(*args, **kwargs) -> T:
        # Execute primary
        primary_result = primary_func(*args, **kwargs)

        # Execute shadow with probability
        if random.random() < shadow_probability:
            try:
                shadow_result = shadow_func(*args, **kwargs)
                # Compare results if needed
                # Could log differences, metrics, etc.
            except Exception:
                # Ignore shadow execution errors
                pass

        return primary_result

    return wrapper
