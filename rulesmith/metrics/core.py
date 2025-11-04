"""Unified metrics system with MLflow integration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Metric:
    """
    Unified metric representation.
    
    All metrics in Rulesmith are represented as Metric objects that can be
    logged to MLflow and tracked over time.
    """
    
    name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[float] = None
    threshold_operator: str = "<="  # <=, >=, ==, <, >
    timestamp: Optional[float] = None
    
    def breached(self) -> bool:
        """Check if metric has breached its threshold."""
        if self.threshold is None:
            return False
        
        if self.threshold_operator == "<=":
            return self.value > self.threshold
        elif self.threshold_operator == ">=":
            return self.value < self.threshold
        elif self.threshold_operator == "==":
            return abs(self.value - self.threshold) > 1e-6
        elif self.threshold_operator == "<":
            return self.value >= self.threshold
        elif self.threshold_operator == ">":
            return self.value <= self.threshold
        else:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "threshold": self.threshold,
            "threshold_operator": self.threshold_operator,
            "breached": self.breached(),
        }


class MetricRegistry:
    """
    Registry for tracking all metrics across a rulebook execution.
    
    Integrates with MLflow for persistence and querying.
    """
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._thresholds: Dict[str, Dict[str, Any]] = {}
    
    def register(self, metric: Metric) -> None:
        """Register a metric."""
        self._metrics[metric.name] = metric
        
        # Check threshold if set
        if metric.threshold is not None:
            self._thresholds[metric.name] = {
                "threshold": metric.threshold,
                "operator": metric.threshold_operator,
            }
    
    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def list(self) -> List[Metric]:
        """List all registered metrics."""
        return list(self._metrics.values())
    
    def get_breached(self) -> List[Metric]:
        """Get all metrics that have breached their thresholds."""
        return [m for m in self._metrics.values() if m.breached()]
    
    def set_threshold(self, name: str, threshold: float, operator: str = "<=") -> None:
        """Set or update a threshold for a metric."""
        self._thresholds[name] = {
            "threshold": threshold,
            "operator": operator,
        }
        
        # Update existing metric if present
        if name in self._metrics:
            self._metrics[name].threshold = threshold
            self._metrics[name].threshold_operator = operator
    
    def get_threshold(self, name: str) -> Optional[Dict[str, Any]]:
        """Get threshold configuration for a metric."""
        return self._thresholds.get(name)
    
    def to_mlflow_tags(self) -> Dict[str, str]:
        """Convert thresholds to MLflow tags for storage."""
        tags = {}
        for name, config in self._thresholds.items():
            tags[f"metric_threshold_{name}"] = f"{config['operator']}{config['threshold']}"
        return tags
    
    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
        self._thresholds.clear()


# Global registry instance
_global_registry = MetricRegistry()


def get_metric_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _global_registry

