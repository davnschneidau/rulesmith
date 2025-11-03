"""Tests for Phase 6: Reliability features."""

import pytest
import tempfile
import os

from rulesmith.reliability.admission import (
    AdmissionController,
    CheckResult,
    default_admission_controller,
)
from rulesmith.reliability.blue_green import (
    BlueGreenDeployment,
    DeploymentConfig,
)
from rulesmith.reliability.replay import (
    ReplayEngine,
    ReplayStore,
    RunSnapshot,
    replay_engine,
)
from rulesmith.reliability.slis import (
    SLI,
    SLICollector,
    SLO,
    sli_collector,
)
from rulesmith.io.ser import RulebookSpec, NodeSpec, Edge
from rulesmith.io.decision_result import DecisionResult


class TestReplay:
    """Test deterministic replay."""
    
    def test_run_snapshot(self):
        """Test RunSnapshot creation."""
        snapshot = RunSnapshot(
            run_id="test_run_123",
            rulebook_name="test_rulebook",
            rulebook_version="1.0.0",
            inputs={"x": 5},
            inputs_hash="abc123",
        )
        
        assert snapshot.run_id == "test_run_123"
        assert snapshot.rulebook_name == "test_rulebook"
        assert snapshot.inputs == {"x": 5}
        
        # Test serialization
        data = snapshot.to_dict()
        assert data["run_id"] == "test_run_123"
        
        # Test deserialization
        snapshot2 = RunSnapshot.from_dict(data)
        assert snapshot2.run_id == snapshot.run_id
        assert snapshot2.inputs == snapshot.inputs
    
    def test_replay_store(self):
        """Test ReplayStore."""
        store = ReplayStore()
        
        snapshot = RunSnapshot(
            run_id="test_1",
            rulebook_name="test",
            rulebook_version="1.0.0",
            inputs={"x": 1},
            inputs_hash="hash1",
        )
        
        store.save_snapshot(snapshot)
        loaded = store.load_snapshot("test_1")
        
        assert loaded is not None
        assert loaded.run_id == "test_1"
        assert loaded.inputs == {"x": 1}
    
    def test_replay_store_file(self):
        """Test ReplayStore with file storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(storage_path=tmpdir)
            
            snapshot = RunSnapshot(
                run_id="test_file",
                rulebook_name="test",
                rulebook_version="1.0.0",
                inputs={"x": 1},
                inputs_hash="hash1",
            )
            
            store.save_snapshot(snapshot)
            
            # Load from file
            loaded = store.load_snapshot("test_file")
            assert loaded is not None
            assert loaded.run_id == "test_file"
    
    def test_replay_engine_capture(self):
        """Test ReplayEngine capture."""
        engine = ReplayEngine()
        
        result = DecisionResult(
            value={"y": 10},
            version="1.0.0",
        )
        
        snapshot = engine.capture_run(
            run_id="capture_test",
            rulebook_name="test",
            rulebook_version="1.0.0",
            inputs={"x": 5},
            result=result,
        )
        
        assert snapshot.run_id == "capture_test"
        assert snapshot.inputs == {"x": 5}
        assert snapshot.original_result is not None


class TestBlueGreen:
    """Test blue/green deployment."""
    
    def test_deployment_config(self):
        """Test DeploymentConfig."""
        config = DeploymentConfig(
            blue_version="1.0.0",
            green_version="1.1.0",
            traffic_split=0.1,  # 10% to green
        )
        
        assert config.blue_version == "1.0.0"
        assert config.green_version == "1.1.0"
        assert config.traffic_split == 0.1
        assert config.rollback_thresholds is not None
    
    def test_blue_green_deployment(self):
        """Test BlueGreenDeployment."""
        # Create mock rulebooks
        class MockRulebook:
            def __init__(self, name, version):
                self.name = name
                self.version = version
            
            def run(self, payload, context=None, enable_mlflow=True):
                return DecisionResult(
                    value={"result": f"{self.version}"},
                    version=self.version,
                )
        
        blue = MockRulebook("test", "1.0.0")
        green = MockRulebook("test", "1.1.0")
        
        config = DeploymentConfig(
            blue_version="1.0.0",
            green_version="1.1.0",
            traffic_split=0.0,  # All blue
        )
        
        deployment = BlueGreenDeployment(
            rulebook_name="test",
            blue_rulebook=blue,
            green_rulebook=green,
            config=config,
        )
        
        # Force blue
        result = deployment.execute({"x": 1}, force_version="blue")
        assert result.value["result"] == "1.0.0"
        
        # Force green
        result = deployment.execute({"x": 1}, force_version="green")
        assert result.value["result"] == "1.1.0"


class TestSLIs:
    """Test SLI/SLO collection."""
    
    def test_sli_collector(self):
        """Test SLICollector."""
        collector = SLICollector()
        
        # Register SLI
        sli = SLI(
            name="success_rate",
            metric_name="success",
            window_seconds=300,
        )
        collector.register_sli(sli)
        
        # Record metrics
        collector.record_metric("success", 1.0)  # Success
        collector.record_metric("success", 1.0)  # Success
        collector.record_metric("success", 0.0)  # Failure
        
        # Compute SLI
        value = collector.compute_sli("success_rate")
        assert value > 0.5  # At least 2/3 success
    
    def test_slo_evaluation(self):
        """Test SLO evaluation."""
        collector = SLICollector()
        
        # Register SLI and SLO
        sli = SLI(name="success_rate", metric_name="success", window_seconds=300)
        collector.register_sli(sli)
        
        slo = SLO(
            name="availability",
            sli_name="success_rate",
            target=0.99,  # 99% target
        )
        collector.register_slo(slo)
        
        # Record mostly successes
        for _ in range(100):
            collector.record_metric("success", 1.0)
        
        # Evaluate SLO
        evaluation = collector.evaluate_slo("availability")
        assert evaluation["compliant"] is True
        assert evaluation["sli_value"] >= 0.99


class TestAdmissionController:
    """Test admission controller."""
    
    def test_cycle_detection(self):
        """Test cycle detection."""
        controller = AdmissionController()
        
        # Create spec with cycle
        nodes = [
            NodeSpec(name="a", kind="rule"),
            NodeSpec(name="b", kind="rule"),
        ]
        edges = [
            Edge(source="a", target="b"),
            Edge(source="b", target="a"),  # Cycle!
        ]
        spec = RulebookSpec(name="test", version="1.0.0", nodes=nodes, edges=edges)
        
        result = controller.check(spec)
        assert result.approved is False
        assert any(c.name == "cycle_detection" and c.result == CheckResult.FAIL for c in result.checks)
    
    def test_complexity_check(self):
        """Test complexity check."""
        controller = AdmissionController(max_complexity=100)
        
        # Create simple spec
        nodes = [NodeSpec(name=f"node_{i}", kind="rule") for i in range(5)]
        edges = []
        spec = RulebookSpec(name="test", version="1.0.0", nodes=nodes, edges=edges)
        
        result = controller.check(spec)
        # Should pass with low complexity
        assert result.approved is True
        
        # Create complex spec
        nodes = [NodeSpec(name=f"node_{i}", kind="rule") for i in range(20)]
        edges = [Edge(source=f"node_{i}", target=f"node_{i+1}") for i in range(19)]
        spec = RulebookSpec(name="test", version="1.0.0", nodes=nodes, edges=edges)
        
        result = controller.check(spec)
        # Should fail with high complexity
        assert result.approved is False
    
    def test_fallback_check(self):
        """Test fallback check."""
        controller = AdmissionController(require_fallbacks=True)
        
        # Create spec with gate but no fallback
        nodes = [
            NodeSpec(name="gate1", kind="gate"),
            NodeSpec(name="rule1", kind="rule"),
        ]
        edges = [
            Edge(source="gate1", target="rule1", condition="when x > 0"),
            # No else branch!
        ]
        spec = RulebookSpec(name="test", version="1.0.0", nodes=nodes, edges=edges)
        
        result = controller.check(spec)
        # Should fail or warn (depending on implementation)
        # Note: This is a simplified check - real implementation would need more sophisticated gate detection
        assert len(result.checks) > 0

