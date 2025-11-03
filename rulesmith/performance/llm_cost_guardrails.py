"""LLM cost guardrails and token budget management."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from rulesmith.runtime.cache import Cache, MemoryCache


@dataclass
class TokenBudget:
    """Token budget for a tenant or user."""
    
    tenant_id: str
    budget_per_day: int = 1_000_000  # 1M tokens per day
    budget_per_month: int = 30_000_000  # 30M tokens per month
    used_today: int = 0
    used_this_month: int = 0
    last_reset_date: str = field(default_factory=lambda: datetime.utcnow().date().isoformat())
    cost_per_1k_tokens: float = 0.01  # Default cost per 1k tokens (USD)
    
    def reset_if_needed(self) -> None:
        """Reset budget if date has changed."""
        today = datetime.utcnow().date().isoformat()
        last_reset = datetime.fromisoformat(self.last_reset_date).date()
        
        if today != last_reset:
            # Check if it's a new month
            if last_reset.month != datetime.utcnow().month or last_reset.year != datetime.utcnow().year:
                self.used_this_month = 0
            
            self.used_today = 0
            self.last_reset_date = today
    
    def can_consume(self, tokens: int) -> bool:
        """
        Check if tokens can be consumed.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if budget available
        """
        self.reset_if_needed()
        
        return (
            self.used_today + tokens <= self.budget_per_day and
            self.used_this_month + tokens <= self.budget_per_month
        )
    
    def consume(self, tokens: int) -> bool:
        """
        Consume tokens.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if consumed successfully, False if budget exceeded
        """
        self.reset_if_needed()
        
        if not self.can_consume(tokens):
            return False
        
        self.used_today += tokens
        self.used_this_month += tokens
        return True
    
    def remaining_today(self) -> int:
        """Get remaining tokens for today."""
        self.reset_if_needed()
        return max(0, self.budget_per_day - self.used_today)
    
    def remaining_this_month(self) -> int:
        """Get remaining tokens for this month."""
        self.reset_if_needed()
        return max(0, self.budget_per_month - self.used_this_month)
    
    def get_cost(self, tokens: int) -> float:
        """Calculate cost for tokens."""
        return (tokens / 1000.0) * self.cost_per_1k_tokens


@dataclass
class LLMCacheEntry:
    """Cache entry for LLM responses."""
    
    prompt_hash: str
    model: str
    response: str
    tokens: Dict[str, int]
    cost: float
    cached_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class LLMCostGuardrails:
    """Manages LLM cost guardrails and caching."""
    
    def __init__(self, cache: Optional[Cache] = None):
        """
        Initialize LLM cost guardrails.
        
        Args:
            cache: Optional cache for LLM responses (defaults to MemoryCache)
        """
        self.budgets: Dict[str, TokenBudget] = {}  # tenant_id -> TokenBudget
        self.llm_cache: Cache = cache or MemoryCache()
        self.cache_ttl: float = 3600.0  # 1 hour default
        self.route_to_hitl_on_exhaustion: bool = True
        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "cached_responses": 0,
            "budget_exceeded": 0,
            "routed_to_hitl": 0,
        }
    
    def get_budget(self, tenant_id: str) -> TokenBudget:
        """Get or create token budget for a tenant."""
        if tenant_id not in self.budgets:
            self.budgets[tenant_id] = TokenBudget(tenant_id=tenant_id)
        
        return self.budgets[tenant_id]
    
    def check_budget(
        self,
        tenant_id: str,
        estimated_tokens: int,
    ) -> Dict[str, Any]:
        """
        Check if request is within budget.
        
        Args:
            tenant_id: Tenant identifier
            estimated_tokens: Estimated tokens for request
        
        Returns:
            Dictionary with:
            - allowed: bool
            - reason: Optional error message
            - remaining_today: int
            - remaining_this_month: int
            - should_route_hitl: bool
        """
        budget = self.get_budget(tenant_id)
        
        can_consume = budget.can_consume(estimated_tokens)
        
        result = {
            "allowed": can_consume,
            "remaining_today": budget.remaining_today(),
            "remaining_this_month": budget.remaining_this_month(),
            "should_route_hitl": False,
        }
        
        if not can_consume:
            result["reason"] = (
                f"Token budget exhausted for tenant {tenant_id}. "
                f"Remaining today: {budget.remaining_today()}, "
                f"this month: {budget.remaining_this_month()}"
            )
            
            if self.route_to_hitl_on_exhaustion:
                result["should_route_hitl"] = True
                self.stats["routed_to_hitl"] += 1
            
            self.stats["budget_exceeded"] += 1
        
        return result
    
    def record_usage(
        self,
        tenant_id: str,
        tokens: Dict[str, int],
        cost: Optional[float] = None,
    ) -> None:
        """
        Record token usage.
        
        Args:
            tenant_id: Tenant identifier
            tokens: Token counts (input_tokens, output_tokens, total_tokens)
            cost: Optional cost in USD
        """
        budget = self.get_budget(tenant_id)
        total_tokens = tokens.get("total_tokens", 0)
        
        if total_tokens > 0:
            budget.consume(total_tokens)
    
    def get_cached_response(
        self,
        prompt: str,
        model: str,
        guard_version: Optional[str] = None,
    ) -> Optional[LLMCacheEntry]:
        """
        Get cached LLM response.
        
        Args:
            prompt: Prompt text
            model: Model name
            guard_version: Optional guardrail version (for cache invalidation)
        
        Returns:
            Cached entry or None
        """
        # Compute cache key
        prompt_hash = hashlib.sha256(f"{prompt}:{model}:{guard_version or ''}".encode()).hexdigest()
        cache_key = f"llm:{prompt_hash}"
        
        cached = self.llm_cache.get(cache_key)
        if cached:
            self.stats["cached_responses"] += 1
            if isinstance(cached, dict):
                return LLMCacheEntry(**cached)
            return cached
        
        return None
    
    def cache_response(
        self,
        prompt: str,
        model: str,
        response: str,
        tokens: Dict[str, int],
        cost: float,
        guard_version: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Cache LLM response.
        
        Args:
            prompt: Prompt text
            model: Model name
            response: Response text
            tokens: Token counts
            cost: Cost in USD
            guard_version: Optional guardrail version
            ttl: Optional TTL in seconds
        """
        prompt_hash = hashlib.sha256(f"{prompt}:{model}:{guard_version or ''}".encode()).hexdigest()
        cache_key = f"llm:{prompt_hash}"
        
        entry = LLMCacheEntry(
            prompt_hash=prompt_hash,
            model=model,
            response=response,
            tokens=tokens,
            cost=cost,
        )
        
        # Store as dict
        self.llm_cache.set(
            cache_key,
            {
                "prompt_hash": entry.prompt_hash,
                "model": entry.model,
                "response": entry.response,
                "tokens": entry.tokens,
                "cost": entry.cost,
                "cached_at": entry.cached_at,
            },
            ttl=ttl or self.cache_ttl,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cached_responses"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0
                else 0.0
            ),
        }


# Global LLM cost guardrails
llm_cost_guardrails = LLMCostGuardrails()

