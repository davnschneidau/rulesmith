"""Tests for Phase 8: Performance features."""

import pytest

from rulesmith.performance.compilation import PredicateCompiler, predicate_compiler
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
    memoization_cache,
)
from rulesmith.io.decision_result import DecisionResult


class TestPredicateCompilation:
    """Test predicate compilation."""
    
    def test_compile_predicate(self):
        """Test predicate compilation."""
        compiler = PredicateCompiler()
        
        # Compile a simple predicate
        compiled = compiler.compile_predicate("age >= 18 and income > 30000")
        
        # Test evaluation
        result = compiled({"age": 25, "income": 50000})
        assert result is True
        
        result = compiled({"age": 15, "income": 50000})
        assert result is False
    
    def test_compile_rule_function(self):
        """Test rule function compilation."""
        compiler = PredicateCompiler()
        
        def simple_rule(age: int, income: float) -> dict:
            return {"eligible": age >= 18 and income > 30000}
        
        compiled = compiler.compile_rule_function(simple_rule)
        
        # Should return same result
        result1 = simple_rule(25, 50000)
        result2 = compiled(25, 50000)
        
        assert result1 == result2


class TestMemoization:
    """Test memoization and caching."""
    
    def test_compute_inputs_hash(self):
        """Test input hash computation."""
        inputs1 = {"x": 5, "y": 10}
        inputs2 = {"y": 10, "x": 5}  # Same, different order
        
        hash1 = compute_inputs_hash(inputs1)
        hash2 = compute_inputs_hash(inputs2)
        
        assert hash1 == hash2  # Should be deterministic
    
    def test_memoization_cache(self):
        """Test memoization cache."""
        cache = MemoizationCache()
        
        result = DecisionResult(
            value={"result": 10},
            version="1.0.0",
        )
        
        # Cache result
        cache.set(
            rulebook_version="1.0.0",
            inputs={"x": 5},
            result=result,
        )
        
        # Retrieve cached result
        cached = cache.get(
            rulebook_version="1.0.0",
            inputs={"x": 5},
        )
        
        assert cached is not None
        assert cached.value == {"result": 10}
        
        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
    
    def test_memoization_cache_subset(self):
        """Test memoization with input subset."""
        cache = MemoizationCache()
        
        result = DecisionResult(
            value={"result": 10},
            version="1.0.0",
        )
        
        # Cache with subset of inputs
        cache.set(
            rulebook_version="1.0.0",
            inputs={"x": 5, "y": 10, "timestamp": 1234567890},
            result=result,
            inputs_subset=["x", "y"],  # Only hash x and y
        )
        
        # Retrieve with different timestamp (should still match)
        cached = cache.get(
            rulebook_version="1.0.0",
            inputs={"x": 5, "y": 10, "timestamp": 9876543210},  # Different timestamp
            inputs_subset=["x", "y"],
        )
        
        assert cached is not None
    
    def test_cache_key(self):
        """Test cache key."""
        key = CacheKey(
            rulebook_version="1.0.0",
            inputs_hash="abc123",
            node_id="rule1",
        )
        
        key_str = key.to_string()
        assert "1.0.0" in key_str
        assert "abc123" in key_str
        assert "rule1" in key_str
        
        # Round trip
        key2 = CacheKey.from_string(key_str)
        assert key2.rulebook_version == key.rulebook_version
        assert key2.inputs_hash == key.inputs_hash
        assert key2.node_id == key.node_id


class TestLLMCostGuardrails:
    """Test LLM cost guardrails."""
    
    def test_token_budget(self):
        """Test token budget."""
        budget = TokenBudget(
            tenant_id="tenant1",
            budget_per_day=1000,
        )
        
        # Consume tokens
        success = budget.consume(500)
        assert success is True
        assert budget.used_today == 500
        
        # Try to exceed budget
        success = budget.consume(600)  # Would exceed 1000
        assert success is False
        
        # Check remaining
        assert budget.remaining_today() == 500
    
    def test_llm_cost_guardrails_budget_check(self):
        """Test LLM cost guardrails budget check."""
        guardrails = LLMCostGuardrails()
        
        budget = guardrails.get_budget("tenant1")
        budget.budget_per_day = 1000
        
        # Check budget
        result = guardrails.check_budget("tenant1", 500)
        assert result["allowed"] is True
        
        # Exceed budget
        result = guardrails.check_budget("tenant1", 600)  # Would exceed
        assert result["allowed"] is False
    
    def test_llm_caching(self):
        """Test LLM response caching."""
        guardrails = LLMCostGuardrails()
        
        # Cache a response
        guardrails.cache_response(
            prompt="What is 2+2?",
            model="gpt-4",
            response="4",
            tokens={"total_tokens": 100},
            cost=0.01,
        )
        
        # Retrieve cached
        cached = guardrails.get_cached_response(
            prompt="What is 2+2?",
            model="gpt-4",
        )
        
        assert cached is not None
        assert cached.response == "4"
        assert cached.tokens["total_tokens"] == 100
    
    def test_llm_cost_guardrails_stats(self):
        """Test LLM cost guardrails statistics."""
        guardrails = LLMCostGuardrails()
        
        # Record some usage
        guardrails.record_usage(
            tenant_id="tenant1",
            tokens={"total_tokens": 1000},
            cost=0.1,
        )
        
        stats = guardrails.get_stats()
        assert "total_requests" in stats
        assert "budget_exceeded" in stats


class TestPerformanceIntegration:
    """Integration tests for performance features."""
    
    def test_memoization_with_execution(self):
        """Test memoization with execution engine."""
        from rulesmith.dag.graph import Rulebook
        from rulesmith.dag.decorators import rule
        
        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}
        
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="test_rule")
        
        # First execution (cache miss)
        result1 = rb.run({"x": 5}, return_decision_result=True)
        assert isinstance(result1, DecisionResult)
        
        # Second execution with same inputs (should hit cache)
        # Note: Memoization is enabled by default in run()
        # The cache should be checked automatically
        
        # Verify cache stats
        stats = memoization_cache.get_stats()
        assert stats["sets"] >= 1  # At least one set
    
    def test_gate_with_compilation(self):
        """Test gate function with compiled predicates."""
        from rulesmith.dag.functions import gate
        
        state = {"age": 25, "income": 50000}
        
        # Use compiled predicate
        result = gate("age >= 18 and income > 30000", state, use_compiled=True)
        assert result["passed"] is True
        
        result = gate("age < 18", state, use_compiled=True)
        assert result["passed"] is False

