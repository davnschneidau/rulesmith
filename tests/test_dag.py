"""Tests for DAG execution engine."""

import pytest

from rulesmith import rule, rulebook, Rulebook
from rulesmith.dag.registry import rule_registry, rulebook_registry
from rulesmith.runtime.context import RunContext


class TestRuleDecorator:
    """Test @rule decorator."""

    def test_register_rule(self):
        """Test rule registration."""
        rule_registry.clear()

        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}

        assert "test_rule" in rule_registry.list()
        assert rule_registry.get("test_rule") == test_rule

    def test_rule_execution(self):
        """Test rule execution."""
        rule_registry.clear()

        @rule(name="add", inputs=["a", "b"], outputs=["sum"])
        def add(a: int, b: int) -> dict:
            return {"sum": a + b}

        func = rule_registry.get("add")
        result = func(a=1, b=2)
        assert result == {"sum": 3}


class TestRulebookBuilder:
    """Test Rulebook builder."""

    def test_basic_rulebook(self):
        """Test basic rulebook construction."""
        rule_registry.clear()

        @rule(name="double", inputs=["x"], outputs=["y"])
        def double(x: int) -> dict:
            return {"y": x * 2}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(double, as_name="double_node")
        spec = rb.to_spec()

        assert spec.name == "test"
        assert spec.version == "1.0.0"
        assert len(spec.nodes) == 1
        assert spec.nodes[0].name == "double_node"

    def test_rulebook_execution(self):
        """Test rulebook execution."""
        rule_registry.clear()

        @rule(name="increment", inputs=["x"], outputs=["x"])
        def increment(x: int) -> dict:
            return {"x": x + 1}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(increment, as_name="inc")
        result = rb.run({"x": 5})

        assert result["x"] == 6

    def test_connected_nodes(self):
        """Test rulebook with connected nodes."""
        rule_registry.clear()

        @rule(name="add_one", inputs=["x"], outputs=["x"])
        def add_one(x: int) -> dict:
            return {"x": x + 1}

        @rule(name="multiply_two", inputs=["x"], outputs=["x"])
        def multiply_two(x: int) -> dict:
            return {"x": x * 2}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(add_one, as_name="add")
        rb.add_rule(multiply_two, as_name="mult")
        rb.connect("add", "mult")

        result = rb.run({"x": 5})
        # add_one(5) = 6, then multiply_two(6) = 12
        assert result["x"] == 12


class TestGateNode:
    """Test GateNode conditional routing."""

    def test_gate_passes(self):
        """Test gate with passing condition."""
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_gate("age_check", "age >= 18")

        result = rb.run({"age": 20})
        assert result["passed"] is True

    def test_gate_fails(self):
        """Test gate with failing condition."""
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_gate("age_check", "age >= 18")

        result = rb.run({"age": 15})
        assert result["passed"] is False

