"""Rate limiting with token buckets."""

import time
from threading import Lock
from typing import Optional


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        initial_tokens: Optional[float] = None,
    ):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum bucket capacity
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = Lock()

    def consume(self, tokens: float = 1.0, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume
            blocking: If True, wait until tokens available
            timeout: Optional timeout in seconds

        Returns:
            True if tokens consumed, False if timeout
        """
        start_time = time.time()

        with self._lock:
            while True:
                # Refill tokens based on elapsed time
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now

                # Check if enough tokens available
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not blocking:
                    return False

                if timeout and (time.time() - start_time) >= timeout:
                    return False

                # Wait a bit before checking again
                time.sleep(0.01)


class RateLimiter:
    """Rate limiter wrapper."""

    def __init__(
        self,
        requests_per_second: float,
        burst_size: Optional[float] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst_size: Optional burst capacity (defaults to requests_per_second)
        """
        self.bucket = TokenBucket(
            capacity=burst_size or requests_per_second,
            refill_rate=requests_per_second,
        )

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire rate limit permit.

        Args:
            blocking: If True, wait until permit available
            timeout: Optional timeout in seconds

        Returns:
            True if permit acquired, False if timeout
        """
        return self.bucket.consume(1.0, blocking=blocking, timeout=timeout)

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
