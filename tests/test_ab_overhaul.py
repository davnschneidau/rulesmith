"""Tests for Phase 4: A/B testing overhaul."""

import pytest

from rulesmith.ab.lineage import ABLineageTracker, ForkLineage, lineage_tracker
from rulesmith.ab.metrics import (
    ABMetricsCollector,
    ArmMetrics,
    ComparativeMetrics,
    metrics_collector,
)
from rulesmith.ab.reasoning import (
    ForkReason,
    explain_fork_selection,
    explain_why_not,
)
from rulesmith.ab.traffic import pick_arm
from rulesmith.dag.functions import fork
from rulesmith.io.ser import ABArm


class TestABMetrics:
    """Test A/B metrics collection."""
    
    def test_arm_metrics_basic(self):
        """Test basic arm metrics collection."""
        collector = ABMetricsCollector()
        
        # Record some executions
        collector.record_arm_execution(
            arm_name="control",
            outcome=True,
            latency_ms=100.0,
            cost=0.01,
        )
        collector.record_arm_execution(
            arm_name="control",
            outcome=True,
            latency_ms=120.0,
            cost=0.01,
        )
        collector.record_arm_execution(
            arm_name="control",
            outcome=False,
            latency_ms=150.0,
            cost=0.01,
        )
        
        collector.record_arm_execution(
            arm_name="treatment",
            outcome=True,
            latency_ms=90.0,
            cost=0.01,
        )
        collector.record_arm_execution(
            arm_name="treatment",
            outcome=True,
            latency_ms=95.0,
            cost=0.01,
        )
        
        # Compute metrics
        control_metrics = collector.compute_arm_metrics("control", conversion_key=None)
        treatment_metrics = collector.compute_arm_metrics("treatment", conversion_key=None)
        
        assert control_metrics.sample_size == 3
        assert control_metrics.conversion_rate == pytest.approx(2/3, abs=0.01)
        assert control_metrics.latency_p50 > 0
        assert control_metrics.latency_p95 > control_metrics.latency_p50
        
        assert treatment_metrics.sample_size == 2
        assert treatment_metrics.conversion_rate == 1.0
        
        # Compare arms
        comparison = collector.compare_arms("control", "treatment", conversion_key=None)
        assert comparison.arm_a == "control"
        assert comparison.arm_b == "treatment"
        assert comparison.conversion_rate_delta == pytest.approx(1.0 - 2/3, abs=0.01)


class TestABReasoning:
    """Test A/B reasoning and explainability."""
    
    def test_explain_hash_policy(self):
        """Test explanation for hash policy."""
        arms = [
            ABArm(node="control", weight=0.5),
            ABArm(node="treatment", weight=0.5),
        ]
        
        reason = explain_fork_selection(
            fork_name="test_fork",
            selected_arm="control",
            policy="hash",
            arms=arms,
            context={"identity": "user123"},
        )
        
        assert reason.fork_name == "test_fork"
        assert reason.selected_arm == "control"
        assert reason.policy == "hash"
        assert "deterministic" in reason.policy_explanation.lower()
        assert "user123" in reason.policy_explanation
    
    def test_explain_thompson_sampling(self):
        """Test explanation for Thompson Sampling."""
        arms = [
            ABArm(node="control", weight=0.5),
            ABArm(node="treatment", weight=0.5),
        ]
        
        arms_history = {
            "control": {"successes": 10, "failures": 2},
            "treatment": {"successes": 8, "failures": 1},
        }
        
        reason = explain_fork_selection(
            fork_name="bandit_test",
            selected_arm="control",
            policy="thompson_sampling",
            arms=arms,
            context={},
            arms_history=arms_history,
        )
        
        assert reason.policy == "thompson_sampling"
        assert "thompson" in reason.policy_explanation.lower()
        assert reason.historical_performance is not None
        assert "control" in reason.historical_performance
    
    def test_explain_why_not(self):
        """Test why-not explanation."""
        explanation = explain_why_not(
            fork_name="test",
            selected_arm="control",
            other_arm="treatment",
            policy="hash",
        )
        
        assert "treatment" in explanation.lower()
        assert "not selected" in explanation.lower()


class TestABLineage:
    """Test A/B lineage tracking."""
    
    def test_lineage_tracking(self):
        """Test basic lineage tracking."""
        tracker = ABLineageTracker()
        
        # Record a fork
        lineage = tracker.record_fork(
            fork_name="experiment",
            policy="hash",
            selected_arm="control",
            all_arms=["control", "treatment"],
            arm_weights={"control": 0.5, "treatment": 0.5},
            identity="user123",
        )
        
        assert lineage.fork_name == "experiment"
        assert lineage.selected_arm == "control"
        
        # Record arm executions
        tracker.record_arm_execution(
            fork_name="experiment",
            arm_name="control",
            node_name="rule1",
            outcome=True,
            latency_ms=100.0,
        )
        
        tracker.record_arm_execution(
            fork_name="experiment",
            arm_name="control",
            node_name="rule2",
            outcome=True,
            latency_ms=50.0,
        )
        
        # Get complete lineage
        complete = tracker.get_complete_lineage("experiment")
        assert complete is not None
        assert complete["fork_name"] == "experiment"
        assert "control" in complete["executions"]
        assert len(complete["executions"]["control"]) == 2


class TestEnhancedFork:
    """Test enhanced fork function."""
    
    def test_fork_with_reasoning(self):
        """Test fork function with reasoning enabled."""
        arms = [
            ABArm(node="control", weight=0.5),
            ABArm(node="treatment", weight=0.5),
        ]
        
        result = fork(
            arms=arms,
            policy="hash",
            identity="user123",
            fork_name="test_experiment",
            enable_reasoning=True,
        )
        
        assert result["selected_variant"] in ["control", "treatment"]
        assert result["ab_test"] == "test_experiment"
        assert result["ab_policy"] == "hash"
        assert "_fork_reasoning" in result
        assert "explanation" in result["_fork_reasoning"]
        assert "_all_arms" in result
        assert len(result["_all_arms"]) == 2
    
    def test_fork_without_reasoning(self):
        """Test fork function with reasoning disabled."""
        arms = [
            ABArm(node="control", weight=0.5),
            ABArm(node="treatment", weight=0.5),
        ]
        
        result = fork(
            arms=arms,
            policy="hash",
            enable_reasoning=False,
        )
        
        assert result["selected_variant"] in ["control", "treatment"]
        assert "_fork_reasoning" not in result
    
    def test_ghost_fork(self):
        """Test ghost fork mode."""
        arms = [
            ABArm(node="control", weight=0.5),
            ABArm(node="treatment", weight=0.5),
        ]
        
        result = fork(
            arms=arms,
            policy="hash",
            ghost_fork=True,
            fork_name="shadow_test",
        )
        
        assert result["_ghost_fork"] is True
        assert "_ghost_arms" in result
        assert len(result["_ghost_arms"]) == 2
        assert "_ghost_primary" in result
        assert result["_ghost_primary"] in ["control", "treatment"]
    
    def test_fork_with_metrics(self):
        """Test fork function with metrics tracking."""
        arms = [
            ABArm(node="control", weight=0.5),
            ABArm(node="treatment", weight=0.5),
        ]
        
        # Reset metrics collector
        global metrics_collector
        metrics_collector = ABMetricsCollector()
        
        result = fork(
            arms=arms,
            policy="hash",
            identity="user123",
            track_metrics=True,
        )
        
        assert "_metrics_ref" in result
        assert result["_metrics_ref"]["arm"] in ["control", "treatment"]
        
        # Check that lineage was recorded
        lineage = lineage_tracker.get_fork_lineage(result["ab_test"])
        assert lineage is not None
        assert lineage.selected_arm == result["selected_variant"]


class TestABIntegration:
    """Integration tests for A/B testing."""
    
    def test_complete_ab_flow(self):
        """Test complete A/B testing flow."""
        # Reset trackers
        global metrics_collector, lineage_tracker
        metrics_collector = ABMetricsCollector()
        lineage_tracker = ABLineageTracker()
        
        arms = [
            ABArm(node="control", weight=0.5),
            ABArm(node="treatment", weight=0.5),
        ]
        
        # Select arm
        result = fork(
            arms=arms,
            policy="hash",
            identity="user123",
            fork_name="experiment",
            track_metrics=True,
        )
        
        selected_arm = result["selected_variant"]
        
        # Simulate execution and record outcomes
        metrics_collector.record_arm_execution(
            arm_name=selected_arm,
            outcome={"approved": True},
            latency_ms=150.0,
            cost=0.02,
        )
        
        lineage_tracker.record_arm_execution(
            fork_name="experiment",
            arm_name=selected_arm,
            node_name="approval_node",
            outcome={"approved": True},
            latency_ms=150.0,
        )
        
        # Get metrics
        arm_metrics = metrics_collector.compute_arm_metrics(selected_arm)
        assert arm_metrics.sample_size > 0
        
        # Get lineage
        complete_lineage = lineage_tracker.get_complete_lineage("experiment")
        assert complete_lineage is not None
        assert selected_arm in complete_lineage["executions"]

