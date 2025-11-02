"""Tests for guardrails system."""

import pytest

from rulesmith import Rulebook
from rulesmith.guardrails.execution import GuardExecutor, GuardResult, guard_executor
from rulesmith.guardrails.packs import (
    ALL_GUARDS_PACK,
    HALLUCINATION_PACK,
    OUTPUT_VALIDATION_PACK,
    PII_PACK,
    TOXICITY_PACK,
    register_default_guards,
)
from rulesmith.guardrails.policy import GuardAction, GuardPolicy, guard


class TestGuardDecorator:
    """Test @guard decorator."""

    def test_guard_decorator(self):
        """Test guard decorator."""
        @guard(name="test_guard")
        def test_guard(inputs):
            return {"passed": True}

        assert hasattr(test_guard, "_guard_name")
        assert test_guard._guard_name == "test_guard"
        assert hasattr(test_guard, "_is_guard")


class TestGuardExecutor:
    """Test guard execution."""

    def test_guard_executor_registration(self):
        """Test guard registration."""
        executor = GuardExecutor()

        @guard(name="test_guard")
        def test_guard(inputs):
            return {"passed": True}

        executor.register_guard("test_guard", test_guard)
        assert "test_guard" in executor._guards

    def test_guard_evaluation(self):
        """Test guard evaluation."""
        executor = GuardExecutor()

        @guard(name="always_pass")
        def always_pass(inputs):
            return {"passed": True}

        executor.register_guard("always_pass", always_pass)

        result = executor.evaluate("always_pass", {"text": "test"})
        assert isinstance(result, GuardResult)
        assert result.passed is True

    def test_guard_evaluation_failure(self):
        """Test guard evaluation with failure."""
        executor = GuardExecutor()

        @guard(name="always_fail")
        def always_fail(inputs):
            return {"passed": False, "message": "Test failure"}

        executor.register_guard("always_fail", always_fail)

        result = executor.evaluate("always_fail", {"text": "test"})
        assert result.passed is False
        assert "Test failure" in result.message

    def test_guard_policy_evaluation(self):
        """Test guard policy evaluation."""
        executor = GuardExecutor()

        @guard(name="guard1")
        def guard1(inputs):
            return {"passed": True}

        @guard(name="guard2")
        def guard2(inputs):
            return {"passed": False, "message": "Failed"}

        executor.register_guard("guard1", guard1)
        executor.register_guard("guard2", guard2)

        policy = GuardPolicy(name="test_policy", checks=["guard1", "guard2"])
        results = executor.evaluate_policy(policy, {"text": "test"})

        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    def test_guard_policy_block(self):
        """Test guard policy blocking."""
        executor = GuardExecutor()

        @guard(name="blocking_guard")
        def blocking_guard(inputs):
            return {"passed": False}

        executor.register_guard("blocking_guard", blocking_guard)

        policy = GuardPolicy(
            name="block_policy",
            checks=["blocking_guard"],
            on_fail=GuardAction.BLOCK,
        )

        result = executor.apply_policy(policy, {"text": "test"}, {"output": "result"})
        assert result.get("_guard_blocked") is True

    def test_guard_policy_override(self):
        """Test guard policy override."""
        executor = GuardExecutor()

        @guard(name="failing_guard")
        def failing_guard(inputs):
            return {"passed": False}

        executor.register_guard("failing_guard", failing_guard)

        policy = GuardPolicy(
            name="override_policy",
            checks=["failing_guard"],
            on_fail=GuardAction.OVERRIDE,
            override_template={"output": "safe_output"},
        )

        result = executor.apply_policy(policy, {"text": "test"}, {"output": "unsafe"})
        assert result["output"] == "safe_output"
        assert result.get("_guard_blocked") is None

    def test_guard_policy_flag(self):
        """Test guard policy flagging."""
        executor = GuardExecutor()

        @guard(name="flagging_guard")
        def flagging_guard(inputs):
            return {"passed": False, "message": "Flagged"}

        executor.register_guard("flagging_guard", flagging_guard)

        policy = GuardPolicy(
            name="flag_policy",
            checks=["flagging_guard"],
            on_fail=GuardAction.FLAG,
        )

        result = executor.apply_policy(policy, {"text": "test"}, {"output": "result"})
        assert result.get("_guard_flags") is not None
        assert len(result["_guard_flags"]) > 0


class TestGuardPacks:
    """Test guard packs."""

    def test_pii_pack(self):
        """Test PII guard pack."""
        assert PII_PACK.name == "pii"
        assert "pii_email" in PII_PACK.guards

    def test_guard_pack_to_policy(self):
        """Test converting guard pack to policy."""
        policy = PII_PACK.to_policy(when_node="test_node")
        assert isinstance(policy, GuardPolicy)
        assert policy.when_node == "test_node"
        assert len(policy.checks) == 3

    def test_register_default_guards(self):
        """Test registering default guards."""
        executor = GuardExecutor()
        register_default_guards(executor)

        # Check that PII guards are registered
        assert "pii_email" in executor._guards
        assert "pii_phone" in executor._guards


class TestRulebookGuards:
    """Test guardrails with rulebooks."""

    def test_attach_guard_to_node(self):
        """Test attaching guard to rulebook node."""
        from rulesmith import rule

        @rule(name="test_rule", inputs=["text"], outputs=["output"])
        def test_rule(text: str) -> dict:
            return {"output": text}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="rule1")

        # Attach PII pack
        rb.attach_guard("rule1", PII_PACK)

        spec = rb.to_spec()
        # Guard policies are stored on node, not in spec yet
        # This is fine for now

    def test_guard_execution_in_rulebook(self):
        """Test guard execution during rulebook run."""
        from rulesmith import rule

        @rule(name="test_rule", inputs=["text"], outputs=["output"])
        def test_rule(text: str) -> dict:
            return {"output": text}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="rule1")

        # Attach guard that will pass
        policy = GuardPolicy(name="test", checks=[])  # No checks = always pass
        rb.attach_guard("rule1", policy)

        # Should execute normally
        result = rb.run({"text": "safe text"}, enable_mlflow=False)
        assert "output" in result

