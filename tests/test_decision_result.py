"""Tests for DecisionResult standardization."""

import pytest

from rulesmith.io.decision_result import (
    DecisionResult,
    FiredRule,
    ensure_decision_result,
)


class TestDecisionResult:
    """Test DecisionResult type."""
    
    def test_basic_decision_result(self):
        """Test basic DecisionResult creation."""
        result = DecisionResult(
            value={"approved": True, "score": 0.85},
            version="1.0.0",
        )
        
        assert result.value == {"approved": True, "score": 0.85}
        assert result.version == "1.0.0"
        assert result.fired == []
        assert result.skipped == []
        assert result.metrics == {}
        assert result.costs == {}
        assert result.warnings == []
    
    def test_decision_result_with_fired_rules(self):
        """Test DecisionResult with fired rules."""
        fired_rule = FiredRule(
            id="rule1",
            name="check_eligibility",
            salience=10,
            inputs={"age": 25, "income": 50000},
            outputs={"eligible": True},
            reason="Age >= 18 and income > 30000",
            duration_ms=5.2,
        )
        
        result = DecisionResult(
            value={"approved": True},
            version="1.0.0",
            fired=[fired_rule],
        )
        
        assert len(result.fired) == 1
        assert result.fired[0].id == "rule1"
        assert result.fired[0].reason == "Age >= 18 and income > 30000"
    
    def test_decision_result_to_dict(self):
        """Test DecisionResult serialization."""
        fired_rule = FiredRule(
            id="rule1",
            name="test_rule",
            salience=0,
            inputs={},
            outputs={},
            reason="Test",
            duration_ms=1.0,
        )
        
        result = DecisionResult(
            value={"test": True},
            version="1.0.0",
            fired=[fired_rule],
            metrics={"latency_ms": 100.0},
            costs={"token_cost_usd": 0.01},
            warnings=["Test warning"],
        )
        
        data = result.to_dict()
        assert data["value"] == {"test": True}
        assert data["version"] == "1.0.0"
        assert len(data["fired"]) == 1
        assert data["fired"][0]["id"] == "rule1"
        assert data["metrics"]["latency_ms"] == 100.0
        assert data["costs"]["token_cost_usd"] == 0.01
        assert data["warnings"] == ["Test warning"]
    
    def test_decision_result_from_dict(self):
        """Test DecisionResult deserialization."""
        data = {
            "value": {"approved": True},
            "version": "1.0.0",
            "fired": [
                {
                    "id": "rule1",
                    "name": "test_rule",
                    "salience": 0,
                    "inputs": {},
                    "outputs": {},
                    "reason": "Test",
                    "duration_ms": 1.0,
                }
            ],
            "skipped": ["rule2"],
            "metrics": {"latency_ms": 100.0},
            "costs": {},
            "warnings": [],
        }
        
        result = DecisionResult.from_dict(data)
        assert result.value == {"approved": True}
        assert result.version == "1.0.0"
        assert len(result.fired) == 1
        assert result.fired[0].id == "rule1"
        assert "rule2" in result.skipped
    
    def test_decision_result_helpers(self):
        """Test DecisionResult helper methods."""
        fired_rule = FiredRule(
            id="rule1",
            name="test",
            salience=0,
            inputs={},
            outputs={},
            reason="Reason 1",
            duration_ms=1.0,
        )
        
        result = DecisionResult(
            value={},
            version="1.0.0",
            fired=[fired_rule],
            metrics={"total_duration_ms": 150.0},
            costs={"token_cost_usd": 0.01, "model_cost_usd": 0.02},
        )
        
        reasons = result.get_reasons()
        assert len(reasons) == 1
        assert reasons[0] == "Reason 1"
        
        assert result.get_total_cost() == 0.03
        assert result.get_total_duration_ms() == 150.0
    
    def test_ensure_decision_result_from_result(self):
        """Test ensure_decision_result with DecisionResult."""
        result = DecisionResult(
            value={"test": True},
            version="1.0.0",
        )
        
        ensured = ensure_decision_result(result, version="1.0.0")
        assert ensured is result  # Should return same object
    
    def test_ensure_decision_result_from_dict(self):
        """Test ensure_decision_result with Dict."""
        data = {"approved": True, "score": 0.85}
        
        result = ensure_decision_result(data, version="1.0.0")
        assert isinstance(result, DecisionResult)
        assert result.value == {"approved": True, "score": 0.85}
        assert result.version == "1.0.0"
    
    def test_ensure_decision_result_from_other(self):
        """Test ensure_decision_result with other types."""
        result = ensure_decision_result("test", version="1.0.0")
        assert isinstance(result, DecisionResult)
        assert result.value == "test"


class TestDecisionResultIntegration:
    """Integration tests for DecisionResult with execution."""
    
    def test_execution_returns_decision_result(self):
        """Test that execution engine returns DecisionResult."""
        from rulesmith.dag.graph import Rulebook
        from rulesmith.dag.decorators import rule
        
        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}
        
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="test_rule")
        
        result = rb.run({"x": 5}, return_decision_result=True)
        
        assert isinstance(result, DecisionResult)
        assert result.value["y"] == 10
        assert result.version == "1.0.0"
        assert len(result.fired) == 1
        assert result.fired[0].name == "test_rule"
    
    def test_execution_legacy_mode(self):
        """Test execution engine in legacy mode returns Dict."""
        from rulesmith.dag.graph import Rulebook
        from rulesmith.dag.decorators import rule
        
        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}
        
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="test_rule")
        
        result = rb.run({"x": 5}, return_decision_result=False)
        
        assert isinstance(result, dict)
        assert result["y"] == 10

