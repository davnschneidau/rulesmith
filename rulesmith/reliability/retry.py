"""Retry logic with backoff and jitter."""

import random
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

T = TypeVar("T")


class RetryConfig:
    """Retry configuration."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            exponential_base: Exponential backoff base
            jitter: Whether to add random jitter
            retryable_exceptions: Tuple of exception types to retry on
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def get_backoff(self, attempt: int) -> float:
        """Calculate backoff time for attempt."""
        backoff = self.initial_backoff * (self.exponential_base ** attempt)
        backoff = min(backoff, self.max_backoff)

        if self.jitter:
            # Add random jitter (up to 25% of backoff)
            jitter_amount = backoff * 0.25 * random.random()
            backoff += jitter_amount

        return backoff


def retry(
    func: Optional[Callable[..., T]] = None,
    config: Optional[RetryConfig] = None,
) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        func: Function to wrap (if used as decorator)
        config: Optional RetryConfig (defaults to RetryConfig())

    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return f(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        backoff = config.get_backoff(attempt)
                        time.sleep(backoff)
                    else:
                        # Final attempt failed
                        raise last_exception

            raise last_exception

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def retry_with_config(config: RetryConfig):
    """Create retry decorator with specific config."""
    return lambda func: retry(func, config=config)
