"""Tests for Phase 10: Analytics features."""

import pytest
from datetime import datetime, timedelta

from rulesmith.analytics.dead_rule_detection import DeadRuleDetector, DeadRuleReport
from rulesmith.analytics.impact_analysis import ImpactAnalyzer, ImpactReport, RuleImpact
from rulesmith.analytics.ab_outcomes import ABOutcomesLibrary, ABOutcome, ABTestReport
from rulesmith.io.decision_result import DecisionResult, FiredRule
from rulesmith.io.ser import RulebookSpec, NodeSpec, Edge


class TestDeadRuleDetection:
    """Test dead rule detection."""
    
    def test_never_fired_detection(self):
        """Test detection of never-fired rules."""
        detector = DeadRuleDetector()
        
        # Create rulebook spec
        spec = RulebookSpec(
            name="test",
            version="1.0.0",
            nodes=[
                NodeSpec(name="rule1", kind="rule", rule_ref="rule1"),
                NodeSpec(name="rule2", kind="rule", rule_ref="rule2"),
            ],
            edges=[
                Edge(source="rule1", target="rule2"),
            ],
        )
        
        # Create execution history with only rule1 firing
        history = [
            DecisionResult(
                value={"result": "approved"},
                version="1.0.0",
                fired=[
                    FiredRule(
                        id="rule1",
                        name="rule1",
                        salience=0,
                        inputs={},
                        outputs={},
                        reason="",
                        duration_ms=10.0,
                    ),
                ],
            ),
        ]
        
        reports = detector.analyze_rulebook(spec, execution_history=history)
        
        # rule2 should be detected as never fired
        rule2_reports = [r for r in reports if r.rule_id == "rule2"]
        assert len(rule2_reports) > 0
        assert rule2_reports[0].status in ["never_fired", "unreachable"]
    
    def test_reachability_check(self):
        """Test reachability checking."""
        detector = DeadRuleDetector()
        
        spec = RulebookSpec(
            name="test",
            version="1.0.0",
            nodes=[
                NodeSpec(name="entry", kind="rule", rule_ref="entry"),
                NodeSpec(name="unreachable", kind="rule", rule_ref="unreachable"),
            ],
            edges=[
                # No edges to unreachable node
            ],
        )
        
        reports = detector.analyze_rulebook(spec)
        
        unreachable_reports = [r for r in reports if r.rule_name == "unreachable"]
        assert len(unreachable_reports) > 0
        assert unreachable_reports[0].status == "unreachable"
    
    def test_health_summary(self):
        """Test health summary generation."""
        detector = DeadRuleDetector()
        
        spec = RulebookSpec(
            name="test",
            version="1.0.0",
            nodes=[
                NodeSpec(name="rule1", kind="rule", rule_ref="rule1"),
                NodeSpec(name="rule2", kind="rule", rule_ref="rule2"),
            ],
            edges=[],
        )
        
        summary = detector.get_rule_health_summary(spec)
        
        assert summary["total_rules"] == 2
        assert "dead_rules" in summary
        assert "active_rules" in summary


class TestImpactAnalysis:
    """Test impact analysis."""
    
    def test_impact_analysis(self):
        """Test impact analysis."""
        analyzer = ImpactAnalyzer(outcome_field="result")
        
        # Create execution history
        history = [
            DecisionResult(
                value={"result": "approved"},
                version="1.0.0",
                fired=[
                    FiredRule(
                        id="rule1",
                        name="rule1",
                        salience=0,
                        inputs={},
                        outputs={"result": "approved"},
                        reason="",
                        duration_ms=10.0,
                    ),
                ],
            ),
            DecisionResult(
                value={"result": "rejected"},
                version="1.0.0",
                fired=[
                    FiredRule(
                        id="rule2",
                        name="rule2",
                        salience=0,
                        inputs={},
                        outputs={"result": "rejected"},
                        reason="",
                        duration_ms=10.0,
                    ),
                ],
            ),
        ]
        
        report = analyzer.analyze_impact(execution_history=history)
        
        assert report.total_executions == 2
        assert len(report.rule_impacts) > 0
        assert "summary" in report.summary
    
    def test_rule_dependency(self):
        """Test rule dependency analysis."""
        analyzer = ImpactAnalyzer()
        
        history = [
            DecisionResult(
                value={},
                version="1.0.0",
                fired=[
                    FiredRule(id="rule1", name="rule1", salience=0, inputs={}, outputs={}, reason="", duration_ms=10.0),
                    FiredRule(id="rule2", name="rule2", salience=0, inputs={}, outputs={}, reason="", duration_ms=10.0),
                ],
            ),
        ]
        
        dependency = analyzer.analyze_rule_dependency("rule1", execution_history=history)
        
        assert dependency["rule_id"] == "rule1"
        assert "co_occurring_rules" in dependency


class TestABOutcomes:
    """Test A/B outcomes library."""
    
    def test_record_outcome(self):
        """Test recording outcomes."""
        library = ABOutcomesLibrary()
        
        outcome = library.record_outcome(
            fork_name="test_fork",
            arm_name="arm_a",
            outcome_value="approved",
            metrics={"latency_ms": 100.0},
            costs={"token_cost": 0.01},
            latency_ms=100.0,
        )
        
        assert outcome.fork_name == "test_fork"
        assert outcome.arm_name == "arm_a"
        assert outcome.outcome_value == "approved"
    
    def test_record_from_decision_result(self):
        """Test recording from DecisionResult."""
        library = ABOutcomesLibrary()
        
        result = DecisionResult(
            value={"result": "approved"},
            version="1.0.0",
            metrics={"latency_ms": 100.0},
            costs={"token_cost": 0.01},
        )
        
        outcome = library.record_from_decision_result(
            result,
            fork_name="test_fork",
            arm_name="arm_a",
        )
        
        assert outcome.fork_name == "test_fork"
        assert outcome.arm_name == "arm_a"
    
    def test_get_outcomes(self):
        """Test getting outcomes with filters."""
        library = ABOutcomesLibrary()
        
        library.record_outcome("fork1", "arm_a", outcome_value="approved")
        library.record_outcome("fork1", "arm_b", outcome_value="rejected")
        library.record_outcome("fork2", "arm_a", outcome_value="approved")
        
        # Filter by fork
        fork1_outcomes = library.get_outcomes(fork_name="fork1")
        assert len(fork1_outcomes) == 2
        
        # Filter by arm
        arm_a_outcomes = library.get_outcomes(arm_name="arm_a")
        assert len(arm_a_outcomes) == 2
    
    def test_generate_report(self):
        """Test report generation."""
        library = ABOutcomesLibrary()
        
        # Record some outcomes
        for i in range(50):
            library.record_outcome(
                "test_fork",
                "arm_a",
                outcome_value="approved" if i % 2 == 0 else "rejected",
                latency_ms=100.0 + i,
            )
        
        for i in range(50):
            library.record_outcome(
                "test_fork",
                "arm_b",
                outcome_value="approved" if i % 3 == 0 else "rejected",
                latency_ms=150.0 + i,
            )
        
        report = library.generate_report("test_fork")
        
        assert report.fork_name == "test_fork"
        assert len(report.arm_metrics) == 2
        assert report.recommendation
    
    def test_export_outcomes(self):
        """Test exporting outcomes."""
        library = ABOutcomesLibrary()
        
        library.record_outcome("fork1", "arm_a", outcome_value="approved")
        library.record_outcome("fork1", "arm_b", outcome_value="rejected")
        
        # Export as JSON
        json_data = library.export_outcomes(fork_name="fork1", format="json")
        assert "fork1" in json_data
        assert "arm_a" in json_data
        
        # Export as CSV
        csv_data = library.export_outcomes(fork_name="fork1", format="csv")
        assert "fork1" in csv_data
        assert "arm_a" in csv_data

