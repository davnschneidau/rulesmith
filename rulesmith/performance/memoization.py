"""Memoization and result caching for rulebook executions."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from rulesmith.io.decision_result import DecisionResult
from rulesmith.runtime.cache import Cache, MemoryCache


@dataclass
class CacheKey:
    """Cache key for memoization."""
    
    rulebook_version: str
    inputs_hash: str
    node_id: Optional[str] = None  # For node-level caching
    
    def to_string(self) -> str:
        """Convert to string key."""
        if self.node_id:
            return f"{self.rulebook_version}:{self.node_id}:{self.inputs_hash}"
        return f"{self.rulebook_version}:{self.inputs_hash}"
    
    @classmethod
    def from_string(cls, key_str: str) -> "CacheKey":
        """Create from string key."""
        parts = key_str.split(":")
        if len(parts) == 3:
            return cls(rulebook_version=parts[0], node_id=parts[1], inputs_hash=parts[2])
        elif len(parts) == 2:
            return cls(rulebook_version=parts[0], inputs_hash=parts[1])
        else:
            raise ValueError(f"Invalid cache key format: {key_str}")


def compute_inputs_hash(inputs: Dict[str, Any], subset: Optional[list] = None) -> str:
    """
    Compute hash of inputs for caching.
    
    Args:
        inputs: Input dictionary
        subset: Optional list of keys to include in hash (if None, uses all)
    
    Returns:
        SHA256 hash
    """
    if subset:
        # Only hash specified keys
        filtered = {k: v for k, v in inputs.items() if k in subset}
    else:
        filtered = inputs
    
    # Sort keys for deterministic hashing
    sorted_items = sorted(filtered.items())
    inputs_json = json.dumps(sorted_items, sort_keys=True)
    return hashlib.sha256(inputs_json.encode("utf-8")).hexdigest()


class MemoizationCache:
    """Memoization cache for rulebook executions."""
    
    def __init__(self, cache: Optional[Cache] = None, default_ttl: float = 3600.0):
        """
        Initialize memoization cache.
        
        Args:
            cache: Optional cache implementation (defaults to MemoryCache)
            default_ttl: Default TTL in seconds
        """
        self.cache = cache or MemoryCache()
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
        }
    
    def get(
        self,
        rulebook_version: str,
        inputs: Dict[str, Any],
        inputs_subset: Optional[list] = None,
        node_id: Optional[str] = None,
    ) -> Optional[DecisionResult]:
        """
        Get cached result.
        
        Args:
            rulebook_version: Rulebook version
            inputs: Input dictionary
            inputs_subset: Optional subset of input keys to use for hashing
            node_id: Optional node ID for node-level caching
        
        Returns:
            Cached DecisionResult or None
        """
        inputs_hash = compute_inputs_hash(inputs, inputs_subset)
        cache_key = CacheKey(
            rulebook_version=rulebook_version,
            inputs_hash=inputs_hash,
            node_id=node_id,
        )
        
        key_str = cache_key.to_string()
        cached = self.cache.get(key_str)
        
        if cached is not None:
            self.stats["hits"] += 1
            # Deserialize from dict
            if isinstance(cached, dict):
                return DecisionResult.from_dict(cached)
            return cached
        else:
            self.stats["misses"] += 1
            return None
    
    def set(
        self,
        rulebook_version: str,
        inputs: Dict[str, Any],
        result: DecisionResult,
        ttl: Optional[float] = None,
        inputs_subset: Optional[list] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """
        Cache a result.
        
        Args:
            rulebook_version: Rulebook version
            inputs: Input dictionary
            result: DecisionResult to cache
            ttl: Optional TTL in seconds (uses default if None)
            inputs_subset: Optional subset of input keys to use for hashing
            node_id: Optional node ID for node-level caching
        """
        inputs_hash = compute_inputs_hash(inputs, inputs_subset)
        cache_key = CacheKey(
            rulebook_version=rulebook_version,
            inputs_hash=inputs_hash,
            node_id=node_id,
        )
        
        key_str = cache_key.to_string()
        
        # Serialize to dict
        result_dict = result.to_dict()
        
        self.cache.set(key_str, result_dict, ttl=ttl or self.default_ttl)
        self.stats["sets"] += 1
    
    def invalidate(
        self,
        rulebook_version: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """
        Invalidate cache entries.
        
        Args:
            rulebook_version: Optional rulebook version to invalidate (all if None)
            node_id: Optional node ID to invalidate (all if None)
        """
        # This is a simplified implementation
        # In production, you'd want more sophisticated invalidation
        if rulebook_version is None and node_id is None:
            self.cache.clear()
        else:
            # Would need to iterate and delete matching keys
            # For now, clear all if specific invalidation needed
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0.0
        
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate": hit_rate,
        }


def memoize(
    rulebook_version: str,
    inputs_subset: Optional[list] = None,
    ttl: Optional[float] = None,
    cache: Optional[MemoizationCache] = None,
):
    """
    Decorator for memoizing rule functions.
    
    Args:
        rulebook_version: Rulebook version
        inputs_subset: Optional subset of input keys to use for hashing
        ttl: Optional TTL in seconds
        cache: Optional cache instance (uses global if None)
    
    Returns:
        Decorator function
    """
    if cache is None:
        from rulesmith.performance.memoization import memoization_cache
        cache = memoization_cache
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Build inputs dict
            inputs = {}
            if args:
                # Try to get input names from function signature
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        inputs[param_names[i]] = arg
            
            inputs.update(kwargs)
            
            # Check cache
            cached_result = cache.get(
                rulebook_version=rulebook_version,
                inputs=inputs,
                inputs_subset=inputs_subset,
                node_id=func.__name__,
            )
            
            if cached_result is not None:
                return cached_result.value  # Return value from cached DecisionResult
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if isinstance(result, dict):
                decision_result = DecisionResult(
                    value=result,
                    version=rulebook_version,
                )
                cache.set(
                    rulebook_version=rulebook_version,
                    inputs=inputs,
                    result=decision_result,
                    ttl=ttl,
                    inputs_subset=inputs_subset,
                    node_id=func.__name__,
                )
            
            return result
        
        return wrapper
    
    return decorator


# Global memoization cache
memoization_cache = MemoizationCache()

