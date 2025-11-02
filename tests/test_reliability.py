"""Tests for reliability features."""

import pytest
import time

from rulesmith.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState
from rulesmith.reliability.rate_limit import RateLimiter, TokenBucket
from rulesmith.reliability.retry import RetryConfig, retry


class TestRetry:
    """Test retry logic."""

    def test_retry_config(self):
        """Test retry configuration."""
        config = RetryConfig(max_retries=3, initial_backoff=1.0)
        assert config.max_retries == 3
        assert config.initial_backoff == 1.0

    def test_retry_decorator(self):
        """Test retry decorator."""
        call_count = [0]

        @retry(config=RetryConfig(max_retries=2))
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count[0] == 2


class TestRateLimiter:
    """Test rate limiting."""

    def test_token_bucket(self):
        """Test token bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Should be able to consume immediately
        assert bucket.consume(1.0, blocking=False) is True

        # Consume all tokens
        for _ in range(9):
            bucket.consume(1.0, blocking=False)

        # Should fail without blocking
        assert bucket.consume(1.0, blocking=False) is False

    def test_rate_limiter(self):
        """Test rate limiter."""
        limiter = RateLimiter(requests_per_second=2.0)

        # Should be able to acquire
        assert limiter.acquire(blocking=False) is True

        # Context manager
        with limiter:
            pass  # Rate limit acquired


class TestCircuitBreaker:
    """Test circuit breaker."""

    def test_circuit_breaker_closed(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)

        def successful_func():
            return "success"

        result = breaker.call(successful_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_opening(self):
        """Test circuit breaker opening on failures."""
        breaker = CircuitBreaker(failure_threshold=2)

        def failing_func():
            raise ValueError("Error")

        # Fail twice
        for _ in range(2):
            try:
                breaker.call(failing_func)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(failing_func)

