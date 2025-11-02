"""Integration tests for full workflow."""

import pytest

from rulesmith import rule, Rulebook
from rulesmith.dag.registry import rule_registry
from rulesmith.guardrails.packs import PII_PACK
from rulesmith.io.ser import ABArm
from rulesmith.runtime.hooks import hook_registry


@rule(name="input_validation", inputs=["age", "income"], outputs=["valid", "age", "income"])
def input_validation(age: int, income: float) -> dict:
    """Validate input."""
    return {"valid": age >= 18 and income > 0, "age": age, "income": income}


@rule(name="scoring", inputs=["age", "income"], outputs=["score"])
def scoring(age: int, income: float) -> dict:
    """Calculate score."""
    return {"score": (age * 10) + (income / 1000)}


@rule(name="decision", inputs=["valid", "score"], outputs=["approved", "reason"])
def decision(valid: bool, score: float) -> dict:
    """Make decision."""
    if not valid:
        return {"approved": False, "reason": "Invalid input"}
    elif score >= 700:
        return {"approved": True, "reason": "High score"}
    else:
        return {"approved": False, "reason": "Low score"}


class TestFullWorkflow:
    """Test complete workflow."""

    def test_complete_rulebook_workflow(self):
        """Test complete rulebook execution."""
        rule_registry.clear()

        rb = Rulebook(name="credit_workflow", version="1.0.0")

        # Add rules
        rb.add_rule(input_validation, as_name="validate")
        rb.add_rule(scoring, as_name="score")
        rb.add_rule(decision, as_name="decide")

        # Connect
        rb.connect("validate", "score")
        rb.connect("score", "decide")

        # Attach guard
        rb.attach_guard("decide", PII_PACK)

        # Execute
        result = rb.run({
            "age": 30,
            "income": 75000,
        }, enable_mlflow=False)

        assert "approved" in result
        assert result["approved"] is True

    def test_workflow_with_fork(self):
        """Test workflow with A/B testing."""
        rule_registry.clear()

        rb = Rulebook(name="ab_workflow", version="1.0.0")

        rb.add_rule(input_validation, as_name="validate")

        # Add A/B fork
        arms = [
            ABArm(node="scoring_v1", weight=0.5),
            ABArm(node="scoring_v2", weight=0.5),
        ]
        rb.add_fork("ab_test", arms, policy="hash")

        rb.add_rule(scoring, as_name="scoring_v1")
        rb.add_rule(scoring, as_name="scoring_v2")

        rb.connect("validate", "ab_test")
        rb.connect("ab_test", "scoring_v1")
        rb.connect("ab_test", "scoring_v2")

        result = rb.run({
            "age": 30,
            "income": 75000,
            "identity": "user123",
        }, enable_mlflow=False)

        assert "_ab_selection" in result

