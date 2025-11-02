"""Tests for hooks system."""

import pytest

from rulesmith import rule, Rulebook
from rulesmith.dag.registry import rule_registry
from rulesmith.runtime.hooks import HookRegistry, Hooks


class TestHooks(Hooks):
    """Test hooks implementation."""

    def __init__(self):
        self.before_calls = []
        self.after_calls = []
        self.error_calls = []

    def before_node(self, node_name: str, state, context) -> None:
        """Called before node execution."""
        self.before_calls.append(node_name)

    def after_node(self, node_name: str, state, context, outputs) -> None:
        """Called after node execution."""
        self.after_calls.append((node_name, outputs))

    def on_error(self, node_name: str, state, context, error) -> None:
        """Called on error."""
        self.error_calls.append((node_name, str(error)))

    def on_guard(self, node_name: str, state, context, guard_result) -> None:
        """Called on guard."""
        pass

    def on_hitl(self, node_name: str, state, context, request_id: str) -> None:
        """Called on HITL."""
        pass


class TestHooksSystem:
    """Test hooks system."""

    def test_hooks_registry(self):
        """Test hook registry."""
        registry = HookRegistry()
        hooks = TestHooks()

        registry.register(hooks)
        assert len(registry._hooks) == 1

    def test_hooks_execution(self):
        """Test hooks during rulebook execution."""
        rule_registry.clear()

        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}

        hooks = TestHooks()
        from rulesmith.runtime.hooks import hook_registry

        hook_registry.register(hooks)

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="rule1")

        result = rb.run({"x": 5}, enable_mlflow=False)

        # Check hooks were called
        assert len(hooks.before_calls) >= 1
        assert len(hooks.after_calls) >= 1

