"""Performance optimization modules (compilation, memoization, LLM cost guardrails)."""

from rulesmith.performance.compilation import (
    PredicateCompiler,
    predicate_compiler,
)
from rulesmith.performance.llm_cost_guardrails import (
    LLMCostGuardrails,
    LLMCacheEntry,
    TokenBudget,
    llm_cost_guardrails,
)
from rulesmith.performance.memoization import (
    CacheKey,
    MemoizationCache,
    compute_inputs_hash,
    memoize,
    memoization_cache,
)

__all__ = [
    # Compilation
    "PredicateCompiler",
    "predicate_compiler",
    # Memoization
    "MemoizationCache",
    "CacheKey",
    "compute_inputs_hash",
    "memoize",
    "memoization_cache",
    # LLM Cost Guardrails
    "LLMCostGuardrails",
    "TokenBudget",
    "LLMCacheEntry",
    "llm_cost_guardrails",
]

