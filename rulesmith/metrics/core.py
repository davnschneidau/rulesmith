"""Unified metrics system with MLflow integration."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional
from datetime import datetime


@dataclass
class Threshold:
    """Threshold configuration for a metric."""
    
    value: float
    operator: str = "<="  # <=, >=, ==, <, >
    alert_action: Optional[str] = None  # "log", "route_to_hitl", "hook"
    alert_hook: Optional[Callable] = None
    
    def check(self, metric_value: float) -> bool:
        """Check if metric value breaches threshold."""
        if self.operator == "<=":
            return metric_value > self.value
        elif self.operator == ">=":
            return metric_value < self.value
        elif self.operator == "==":
            return abs(metric_value - self.value) > 1e-6
        elif self.operator == "<":
            return metric_value >= self.value
        elif self.operator == ">":
            return metric_value <= self.value
        else:
            return False


@dataclass
class Metric:
    """
    Unified metric representation.
    
    All metrics in Rulesmith are represented as Metric objects that can be
    logged to MLflow and tracked over time. Categories help organize metrics
    by purpose (operational, business, model_risk, guardrail).
    """
    
    name: str
    value: float
    category: Literal["operational", "business", "model_risk", "guardrail"] = "operational"
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[Threshold] = None
    timestamp: Optional[float] = None
    node_name: Optional[str] = None  # Which node produced this metric
    
    def breached(self) -> bool:
        """Check if metric has breached its threshold."""
        if self.threshold is None:
            return False
        return self.threshold.check(self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "value": self.value,
            "category": self.category,
            "unit": self.unit,
            "tags": self.tags,
            "breached": self.breached(),
        }
        if self.threshold:
            result["threshold"] = {
                "value": self.threshold.value,
                "operator": self.threshold.operator,
                "alert_action": self.threshold.alert_action,
            }
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.node_name:
            result["node_name"] = self.node_name
        return result


class MetricRegistry:
    """
    Unified registry for tracking all metrics across a rulebook execution.
    
    Supports all metric categories: operational, business, model_risk, guardrail.
    Integrates with MLflow for persistence and querying.
    """
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._thresholds: Dict[str, Threshold] = {}
    
    def register(self, metric: Metric) -> None:
        """Register a metric."""
        self._metrics[metric.name] = metric
        
        # Store threshold if set
        if metric.threshold:
            self._thresholds[metric.name] = metric.threshold
    
    def record(
        self,
        name: str,
        value: float,
        category: Literal["operational", "business", "model_risk", "guardrail"] = "operational",
        unit: str = "",
        node_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Metric:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            category: Metric category
            unit: Unit of measurement
            node_name: Optional node name that produced this metric
            tags: Optional tags
        
        Returns:
            Created Metric object
        """
        # Check if threshold exists for this metric
        threshold = self._thresholds.get(name)
        
        metric = Metric(
            name=name,
            value=value,
            category=category,
            unit=unit,
            tags=tags or {},
            threshold=threshold,
            timestamp=datetime.utcnow().timestamp(),
            node_name=node_name,
        )
        
        self.register(metric)
        return metric
    
    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def list(self, category: Optional[str] = None) -> List[Metric]:
        """
        List all registered metrics, optionally filtered by category.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of metrics
        """
        metrics = list(self._metrics.values())
        if category:
            metrics = [m for m in metrics if m.category == category]
        return metrics
    
    def get_breached(self) -> List[Metric]:
        """Get all metrics that have breached their thresholds."""
        return [m for m in self._metrics.values() if m.breached()]
    
    def set_threshold(
        self,
        name: str,
        threshold: float,
        operator: str = "<=",
        alert_action: Optional[str] = None,
        alert_hook: Optional[Callable] = None,
    ) -> None:
        """
        Set or update a threshold for a metric.
        
        Args:
            name: Metric name
            threshold: Threshold value
            operator: Comparison operator (<=, >=, ==, <, >)
            alert_action: Action on breach ("log", "route_to_hitl", "hook")
            alert_hook: Optional callable for custom actions
        """
        threshold = Threshold(
            value=threshold,
            operator=operator,
            alert_action=alert_action,
            alert_hook=alert_hook,
        )
        self._thresholds[name] = threshold
        
        # Update existing metric if present
        if name in self._metrics:
            self._metrics[name].threshold = threshold
    
    def get_threshold(self, name: str) -> Optional[Threshold]:
        """Get threshold configuration for a metric."""
        return self._thresholds.get(name)
    
    def check_threshold(self, name: str, value: float) -> bool:
        """
        Check if a metric value breaches its threshold.
        
        Args:
            name: Metric name
            value: Metric value to check
        
        Returns:
            True if threshold is breached
        """
        threshold = self._thresholds.get(name)
        if threshold is None:
            return False
        
        breached = threshold.check(value)
        
        if breached and threshold.alert_hook:
            try:
                threshold.alert_hook(name, value, threshold.value)
            except Exception:
                pass  # Ignore hook errors
        
        return breached
    
    def to_mlflow_tags(self) -> Dict[str, str]:
        """Convert thresholds to MLflow tags for storage."""
        tags = {}
        for name, threshold in self._thresholds.items():
            tags[f"metric_threshold_{name}"] = f"{threshold.operator}{threshold.value}"
            if threshold.alert_action:
                tags[f"metric_alert_{name}"] = threshold.alert_action
        return tags
    
    def clear(self) -> None:
        """Clear all metrics (but keep thresholds)."""
        self._metrics.clear()
    
    def clear_all(self) -> None:
        """Clear all metrics and thresholds."""
        self._metrics.clear()
        self._thresholds.clear()


# Global registry instance
_global_registry = MetricRegistry()


def get_metric_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _global_registry
