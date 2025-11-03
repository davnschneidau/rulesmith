"""Standard DecisionResult type for all execution outputs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class FiredRule:
    """Information about a fired rule."""
    
    id: str
    name: str
    salience: int
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reason: str  # Human-readable explanation
    duration_ms: float


@dataclass(frozen=True)
class DecisionResult:
    """
    Standardized result type for all rulebook executions.
    
    This provides a consistent interface across all entry points (Python API, HTTP, CLI)
    and makes it easy to integrate, test, and audit decisions.
    """
    
    value: Any  # Primary decision payload
    version: str  # Rulebook version
    trace_uri: Optional[str] = None  # MLflow or internal trace URI
    fired: List[FiredRule] = field(default_factory=list)  # Fired rules (ordered)
    skipped: List[str] = field(default_factory=list)  # Rule IDs skipped + optional reasons
    metrics: Dict[str, float] = field(default_factory=dict)  # Execution metrics
    costs: Dict[str, float] = field(default_factory=dict)  # Costs (e.g., token_cost_usd)
    warnings: List[str] = field(default_factory=list)  # Warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "version": self.version,
            "trace_uri": self.trace_uri,
            "fired": [
                {
                    "id": fr.id,
                    "name": fr.name,
                    "salience": fr.salience,
                    "inputs": fr.inputs,
                    "outputs": fr.outputs,
                    "reason": fr.reason,
                    "duration_ms": fr.duration_ms,
                }
                for fr in self.fired
            ],
            "skipped": self.skipped,
            "metrics": self.metrics,
            "costs": self.costs,
            "warnings": self.warnings,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionResult":
        """Create from dictionary."""
        fired = [
            FiredRule(
                id=fr["id"],
                name=fr["name"],
                salience=fr["salience"],
                inputs=fr["inputs"],
                outputs=fr["outputs"],
                reason=fr["reason"],
                duration_ms=fr["duration_ms"],
            )
            for fr in data.get("fired", [])
        ]
        
        return cls(
            value=data["value"],
            version=data["version"],
            trace_uri=data.get("trace_uri"),
            fired=fired,
            skipped=data.get("skipped", []),
            metrics=data.get("metrics", {}),
            costs=data.get("costs", {}),
            warnings=data.get("warnings", []),
        )
    
    def get_reasons(self) -> List[str]:
        """Get human-readable reasons for all fired rules."""
        return [fr.reason for fr in self.fired]
    
    def get_total_cost(self) -> float:
        """Get total cost across all costs."""
        return sum(self.costs.values())
    
    def get_total_duration_ms(self) -> float:
        """Get total execution duration."""
        return self.metrics.get("total_duration_ms", 0.0)


def ensure_decision_result(result: Union[DecisionResult, Dict[str, Any]], version: str = "unknown") -> DecisionResult:
    """
    Ensure result is a DecisionResult, converting from Dict if needed.
    
    This is a compatibility helper for code that might still return Dict.
    
    Args:
        result: Either DecisionResult or Dict
        version: Rulebook version (used if converting from Dict)
    
    Returns:
        DecisionResult
    """
    if isinstance(result, DecisionResult):
        return result
    
    # Convert Dict to DecisionResult
    if isinstance(result, dict):
        # Extract value (remove internal tracking fields)
        value = {
            k: v for k, v in result.items()
            if not k.startswith("_") or k in ["_executed_", "_ab_selection", "_fork_selection"]
        }
        
        return DecisionResult(
            value=value,
            version=version,
            fired=[],  # Can't reconstruct from dict
            skipped=[],
            metrics={},
            costs={},
            warnings=[],
        )
    
    # If it's not a dict or DecisionResult, wrap it
    return DecisionResult(
        value=result,
        version=version,
        fired=[],
        skipped=[],
        metrics={},
        costs={},
        warnings=[],
    )
