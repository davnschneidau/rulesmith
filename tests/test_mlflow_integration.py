"""Tests for MLflow integration."""

import pytest

from rulesmith import rule, Rulebook
from rulesmith.dag.registry import rule_registry
from rulesmith.runtime.mlflow_context import MLflowRunContext
from rulesmith.runtime.context import RunContext


class TestMLflowContext:
    """Test MLflow-aware context."""

    def test_mlflow_context_creation(self):
        """Test creating MLflow context."""
        rule_registry.clear()

        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="double")
        spec = rb.to_spec()

        # Create context without MLflow (should work even if MLflow not available)
        context = MLflowRunContext(
            rulebook_spec=spec,
            enable_mlflow=False,  # Disable MLflow for testing
        )

        assert context.rulebook_spec == spec
        assert context.enable_mlflow is False

    def test_context_manager(self):
        """Test context manager functionality."""
        rule_registry.clear()

        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="double")
        spec = rb.to_spec()

        context = MLflowRunContext(
            rulebook_spec=spec,
            enable_mlflow=False,
        )

        # Should work as context manager
        with context:
            assert context.enable_mlflow is False

    def test_node_execution_context(self):
        """Test node execution context."""
        rule_registry.clear()

        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="double")
        spec = rb.to_spec()

        context = MLflowRunContext(
            rulebook_spec=spec,
            enable_mlflow=False,
        )

        node_ctx = context.start_node_execution("double", "rule", inputs={"x": 5})
        assert node_ctx.node_name == "double"
        assert node_ctx.node_kind == "rule"
        assert node_ctx.inputs == {"x": 5}

        node_ctx.finish({"y": 10})
        assert node_ctx.outputs == {"y": 10}
        assert node_ctx.execution_time is not None


class TestRulebookMLflowIntegration:
    """Test rulebook execution with MLflow integration."""

    def test_rulebook_with_mlflow_disabled(self):
        """Test rulebook execution with MLflow disabled."""
        rule_registry.clear()

        @rule(name="add_one", inputs=["x"], outputs=["x"])
        def add_one(x: int) -> dict:
            return {"x": x + 1}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(add_one, as_name="add")

        # Execute with MLflow disabled
        result = rb.run({"x": 5}, enable_mlflow=False)
        assert result["x"] == 6

    def test_rulebook_with_custom_context(self):
        """Test rulebook execution with custom context."""
        rule_registry.clear()

        @rule(name="multiply", inputs=["x"], outputs=["x"])
        def multiply(x: int) -> dict:
            return {"x": x * 2}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(multiply, as_name="mult")

        # Use basic context
        context = RunContext(identity="test_user", seed=42)
        result = rb.run({"x": 5}, context=context, enable_mlflow=False)
        assert result["x"] == 10
        assert context.identity == "test_user"

