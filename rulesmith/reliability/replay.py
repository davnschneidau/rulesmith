"""Deterministic replay for execution runs."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from rulesmith.io.decision_result import DecisionResult


@dataclass
class RunSnapshot:
    """Snapshot of a run for deterministic replay."""
    
    run_id: str
    rulebook_name: str
    rulebook_version: str
    inputs: Dict[str, Any]
    inputs_hash: str
    env_vars: Dict[str, str] = field(default_factory=dict)
    seed: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_uri: Optional[str] = None
    original_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "rulebook_name": self.rulebook_name,
            "rulebook_version": self.rulebook_version,
            "inputs": self.inputs,
            "inputs_hash": self.inputs_hash,
            "env_vars": self.env_vars,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "trace_uri": self.trace_uri,
            "original_result": self.original_result,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSnapshot":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            rulebook_name=data["rulebook_name"],
            rulebook_version=data["rulebook_version"],
            inputs=data["inputs"],
            inputs_hash=data["inputs_hash"],
            env_vars=data.get("env_vars", {}),
            seed=data.get("seed"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            trace_uri=data.get("trace_uri"),
            original_result=data.get("original_result"),
        )


class ReplayStore:
    """Store and retrieve run snapshots for replay."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize replay store.
        
        Args:
            storage_path: Optional path for storing snapshots (defaults to in-memory)
        """
        self.storage_path = storage_path
        self.snapshots: Dict[str, RunSnapshot] = {}
    
    def save_snapshot(self, snapshot: RunSnapshot) -> None:
        """Save a run snapshot."""
        self.snapshots[snapshot.run_id] = snapshot
        
        if self.storage_path:
            os.makedirs(self.storage_path, exist_ok=True)
            file_path = os.path.join(self.storage_path, f"{snapshot.run_id}.json")
            with open(file_path, "w") as f:
                json.dump(snapshot.to_dict(), f, indent=2)
    
    def load_snapshot(self, run_id: str) -> Optional[RunSnapshot]:
        """Load a run snapshot."""
        if run_id in self.snapshots:
            return self.snapshots[run_id]
        
        if self.storage_path:
            file_path = os.path.join(self.storage_path, f"{run_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                return RunSnapshot.from_dict(data)
        
        return None
    
    def list_snapshots(
        self,
        rulebook_name: Optional[str] = None,
        rulebook_version: Optional[str] = None,
    ) -> List[RunSnapshot]:
        """List snapshots with optional filters."""
        snapshots = list(self.snapshots.values())
        
        if rulebook_name:
            snapshots = [s for s in snapshots if s.rulebook_name == rulebook_name]
        
        if rulebook_version:
            snapshots = [s for s in snapshots if s.rulebook_version == rulebook_version]
        
        return sorted(snapshots, key=lambda s: s.timestamp, reverse=True)


class ReplayEngine:
    """Engine for deterministic replay of runs."""
    
    def __init__(self, store: Optional[ReplayStore] = None):
        """
        Initialize replay engine.
        
        Args:
            store: Optional replay store (defaults to in-memory)
        """
        self.store = store or ReplayStore()
    
    def capture_run(
        self,
        run_id: str,
        rulebook_name: str,
        rulebook_version: str,
        inputs: Dict[str, Any],
        result: DecisionResult,
        trace_uri: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> RunSnapshot:
        """
        Capture a run for later replay.
        
        Args:
            run_id: Unique run identifier
            rulebook_name: Name of rulebook
            rulebook_version: Version of rulebook
            inputs: Input payload
            result: Execution result
            trace_uri: Optional trace URI
            seed: Optional random seed for deterministic replay
        
        Returns:
            RunSnapshot object
        """
        # Hash inputs for verification
        inputs_json = json.dumps(inputs, sort_keys=True)
        inputs_hash = hashlib.sha256(inputs_json.encode("utf-8")).hexdigest()
        
        # Capture environment variables (filter sensitive ones)
        env_vars = {}
        for key, value in os.environ.items():
            if not any(sensitive in key.upper() for sensitive in ["PASSWORD", "SECRET", "KEY", "TOKEN"]):
                env_vars[key] = value
        
        snapshot = RunSnapshot(
            run_id=run_id,
            rulebook_name=rulebook_name,
            rulebook_version=rulebook_version,
            inputs=inputs,
            inputs_hash=inputs_hash,
            env_vars=env_vars,
            seed=seed,
            trace_uri=trace_uri,
            original_result=result.to_dict(),
        )
        
        self.store.save_snapshot(snapshot)
        return snapshot
    
    def replay(
        self,
        run_id: str,
        rulebook: Any,  # Rulebook instance
        verify: bool = True,
    ) -> DecisionResult:
        """
        Replay a captured run.
        
        Args:
            run_id: Run ID to replay
            rulebook: Rulebook instance to execute
            verify: If True, verify inputs hash matches
        
        Returns:
            DecisionResult from replay
        
        Raises:
            ValueError: If snapshot not found or verification fails
        """
        snapshot = self.store.load_snapshot(run_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found for run_id: {run_id}")
        
        # Verify inputs hash
        if verify:
            inputs_json = json.dumps(snapshot.inputs, sort_keys=True)
            inputs_hash = hashlib.sha256(inputs_json.encode("utf-8")).hexdigest()
            if inputs_hash != snapshot.inputs_hash:
                raise ValueError("Inputs hash mismatch - inputs may have been modified")
        
        # Set environment variables if needed
        original_env = {}
        try:
            for key, value in snapshot.env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
        except Exception:
            # If setting env vars fails, restore and continue
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        
        # Set random seed if provided
        if snapshot.seed is not None:
            import random
            random.seed(snapshot.seed)
        
        try:
            # Execute rulebook
            result = rulebook.run(snapshot.inputs, enable_mlflow=False)
            
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            
            return result
        except Exception as e:
            # Restore environment on error
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            raise
    
    def compare_replay(
        self,
        run_id: str,
        rulebook: Any,
    ) -> Dict[str, Any]:
        """
        Replay a run and compare with original result.
        
        Args:
            run_id: Run ID to replay
            rulebook: Rulebook instance to execute
        
        Returns:
            Comparison dictionary with differences
        """
        snapshot = self.store.load_snapshot(run_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found for run_id: {run_id}")
        
        # Replay
        replayed_result = self.replay(run_id, rulebook, verify=True)
        
        # Compare with original
        original_result = DecisionResult.from_dict(snapshot.original_result) if snapshot.original_result else None
        
        comparison = {
            "run_id": run_id,
            "rulebook_version": snapshot.rulebook_version,
            "inputs_match": True,  # Verified by hash
            "outputs_match": False,
            "differences": [],
        }
        
        if original_result:
            # Compare values
            if replayed_result.value != original_result.value:
                comparison["outputs_match"] = False
                comparison["differences"].append({
                    "type": "value_mismatch",
                    "original": original_result.value,
                    "replayed": replayed_result.value,
                })
            else:
                comparison["outputs_match"] = True
            
            # Compare metrics
            if replayed_result.metrics != original_result.metrics:
                comparison["differences"].append({
                    "type": "metrics_mismatch",
                    "original_metrics": original_result.metrics,
                    "replayed_metrics": replayed_result.metrics,
                })
            
            # Compare fired rules
            if len(replayed_result.fired) != len(original_result.fired):
                comparison["differences"].append({
                    "type": "fired_rules_count_mismatch",
                    "original_count": len(original_result.fired),
                    "replayed_count": len(replayed_result.fired),
                })
        
        return comparison


# Global replay engine instance
replay_engine = ReplayEngine()

