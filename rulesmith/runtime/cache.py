"""Caching interface and implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Cache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MemoryCache(Cache):
    """In-memory cache implementation."""

    def __init__(self):
        self._cache: Dict[str, tuple[Any, Optional[float]]] = {}
        import time

        self._time = time.time

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]
        if expiry is not None and self._time() > expiry:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        expiry = None
        if ttl is not None:
            expiry = self._time() + ttl

        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

