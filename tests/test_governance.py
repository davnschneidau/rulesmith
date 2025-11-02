"""Tests for governance system."""

import pytest

from rulesmith import Rulebook
from rulesmith.governance.audit import AuditLogger, hash_rulebook_spec, log_deployment, log_promotion
from rulesmith.governance.diff import RulebookDiff, diff_rulebooks
from rulesmith.governance.lineage import LineageGraph, build_lineage
from rulesmith.governance.promotion import PromotionPolicy, SLO, check_slos


class TestPromotion:
    """Test promotion system."""

    def test_slo_evaluation(self):
        """Test SLO evaluation."""
        slo = SLO(metric_name="accuracy", threshold=0.95, operator=">=")

        assert slo.evaluate(0.96) is True
        assert slo.evaluate(0.94) is False
        assert slo.evaluate(0.95) is True

    def test_promotion_policy(self):
        """Test promotion policy."""
        slos = [
            SLO("accuracy", 0.95),
            SLO("latency_ms", 100, operator="<="),
        ]

        policy = PromotionPolicy(name="prod_policy", slos=slos, require_all=True)
        assert policy.name == "prod_policy"
        assert len(policy.slos) == 2
        assert policy.require_all is True


class TestDiff:
    """Test diff system."""

    def test_diff_empty_rulebooks(self):
        """Test diff of empty rulebooks."""
        from rulesmith.io.ser import RulebookSpec

        spec1 = RulebookSpec(name="test", version="1.0.0")
        spec2 = RulebookSpec(name="test", version="1.0.0")

        diff = diff_rulebooks(spec1, spec2)
        assert diff.has_changes() is False

    def test_diff_added_node(self):
        """Test diff with added node."""
        from rulesmith import rule
        from rulesmith.dag.registry import rule_registry

        rule_registry.clear()

        @rule(name="rule1", inputs=["x"], outputs=["y"])
        def rule1(x: int) -> dict:
            return {"y": x * 2}

        rb1 = Rulebook(name="test", version="1.0.0")
        spec1 = rb1.to_spec()

        rb2 = Rulebook(name="test", version="1.0.1")
        rb2.add_rule(rule1, as_name="rule1")
        spec2 = rb2.to_spec()

        diff = diff_rulebooks(spec1, spec2)
        assert diff.has_changes() is True
        assert len(diff.added_nodes) == 1

    def test_diff_removed_node(self):
        """Test diff with removed node."""
        from rulesmith import rule
        from rulesmith.dag.registry import rule_registry

        rule_registry.clear()

        @rule(name="rule1", inputs=["x"], outputs=["y"])
        def rule1(x: int) -> dict:
            return {"y": x * 2}

        rb1 = Rulebook(name="test", version="1.0.0")
        rb1.add_rule(rule1, as_name="rule1")
        spec1 = rb1.to_spec()

        rb2 = Rulebook(name="test", version="1.0.1")
        spec2 = rb2.to_spec()

        diff = diff_rulebooks(spec1, spec2)
        assert diff.has_changes() is True
        assert len(diff.removed_nodes) == 1


class TestLineage:
    """Test lineage system."""

    def test_build_lineage(self):
        """Test building lineage graph."""
        from rulesmith import rule
        from rulesmith.dag.registry import rule_registry

        rule_registry.clear()

        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_rule(test_rule, as_name="rule1")
        spec = rb.to_spec()

        lineage = build_lineage(spec)
        assert isinstance(lineage, LineageGraph)
        assert "rule1" in lineage.nodes


class TestAudit:
    """Test audit system."""

    def test_audit_logger(self):
        """Test audit logger."""
        logger = AuditLogger()

        entry = logger.log("promote", "model", "test_model", actor="user123")
        assert entry.action == "promote"
        assert entry.entity_type == "model"
        assert entry.entity_id == "test_model"

        entries = logger.get_entries(entity_type="model")
        assert len(entries) == 1

    def test_hash_rulebook_spec(self):
        """Test rulebook spec hashing."""
        rb = Rulebook(name="test", version="1.0.0")
        spec = rb.to_spec()

        hash1 = hash_rulebook_spec(spec)
        hash2 = hash_rulebook_spec(spec)

        assert hash1 == hash2  # Same spec, same hash
        assert len(hash1) == 64  # SHA256 hex length

    def test_log_promotion(self):
        """Test logging promotion."""
        logger = AuditLogger()

        entry = log_promotion(logger, "test_model", "@staging", "@prod", actor="user123")
        assert entry.action == "promote"
        assert entry.metadata["from_stage"] == "@staging"
        assert entry.metadata["to_stage"] == "@prod"

    def test_log_deployment(self):
        """Test logging deployment."""
        logger = AuditLogger()

        rb = Rulebook(name="test", version="1.0.0")
        spec = rb.to_spec()

        entry = log_deployment(logger, spec, actor="user123", environment="prod")
        assert entry.action == "deploy"
        assert entry.metadata["rulebook_name"] == "test"
        assert "spec_hash" in entry.metadata

