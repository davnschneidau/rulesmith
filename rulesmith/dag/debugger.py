"""Debug mode for rulebook execution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rulesmith.io.decision_result import DecisionResult


@dataclass
class NodeExecutionStep:
    """Represents a single node execution step."""
    
    node_name: str
    node_kind: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time_ms: float
    error: Optional[str] = None


@dataclass
class DebugInfo:
    """Debug information for rulebook execution."""
    
    step_by_step: List[NodeExecutionStep] = field(default_factory=list)
    state_snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_path: List[str] = field(default_factory=list)
    metrics_per_node: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_by_step": [
                {
                    "node_name": step.node_name,
                    "node_kind": step.node_kind,
                    "execution_time_ms": step.execution_time_ms,
                    "error": step.error,
                    "inputs_keys": list(step.inputs.keys()),
                    "outputs_keys": list(step.outputs.keys()),
                }
                for step in self.step_by_step
            ],
            "execution_path": self.execution_path,
            "metrics_per_node": self.metrics_per_node,
        }


class DebugContext:
    """Context manager for debug mode execution."""
    
    def __init__(self, rulebook: "Rulebook", payload: Dict[str, Any]):
        """
        Initialize debug context.
        
        Args:
            rulebook: Rulebook to debug
            payload: Input payload
        """
        self.rulebook = rulebook
        self.payload = payload
        self.debug_info = DebugInfo()
        self._state_snapshots: Dict[str, Dict[str, Any]] = {}
    
    def execute(self) -> DecisionResult:
        """
        Execute rulebook in debug mode.
        
        Returns:
            DecisionResult with debug information attached
        """
        # Store original execute method
        from rulesmith.dag.execution import ExecutionEngine
        
        spec = self.rulebook.to_spec()
        engine = ExecutionEngine(spec)
        
        # Register all nodes
        for name, node in self.rulebook._nodes.items():
            engine.register_node(name, node)
        
        # Create context
        from rulesmith.runtime.context import RunContext
        context = RunContext()
        
        # Monkey-patch execution to capture debug info
        # This is a simplified approach - in production, we'd want a cleaner way
        original_execute = engine.execute
        
        def debug_execute(*args, **kwargs):
            # Call original execute but capture steps
            # For now, we'll add debug info after execution
            result = original_execute(*args, **kwargs)
            
            # Add debug info to result
            if isinstance(result, DecisionResult):
                # Attach debug info
                result.debug_info = self.debug_info
            
            return result
        
        engine.execute = debug_execute
        
        # Execute
        result = engine.execute(
            payload=self.payload,
            context=context,
            nodes=self.rulebook._nodes,
            return_decision_result=True,
            enable_memoization=False,  # Disable memoization for debugging
        )
        
        return result
    
    def get_state_at(self, node_name: str) -> Optional[Dict[str, Any]]:
        """
        Get state snapshot at a specific node.
        
        Args:
            node_name: Node name
        
        Returns:
            State dictionary or None if not found
        """
        return self._state_snapshots.get(node_name)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

