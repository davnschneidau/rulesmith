"""Metric thresholds with MLflow integration."""

from typing import Any, Dict, List, Optional

import mlflow

from rulesmith.metrics.core import Metric, MetricRegistry, get_metric_registry


class MetricThreshold:
    """Threshold configuration for a metric."""
    
    def __init__(
        self,
        metric_name: str,
        threshold: float,
        operator: str = "<=",
        alert_action: Optional[str] = None,
        alert_hook: Optional[Any] = None,
    ):
        """
        Initialize metric threshold.
        
        Args:
            metric_name: Name of the metric
            threshold: Threshold value
            operator: Comparison operator (<=, >=, ==, <, >)
            alert_action: Action to take on breach ("log", "route_to_hitl", "hook")
            alert_hook: Optional callable to call on breach (if alert_action="hook")
        """
        self.metric_name = metric_name
        self.threshold = threshold
        self.operator = operator
        self.alert_action = alert_action
        self.alert_hook = alert_hook
    
    def check(self, value: float) -> bool:
        """Check if value breaches threshold."""
        if self.operator == "<=":
            return value > self.threshold
        elif self.operator == ">=":
            return value < self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) > 1e-6
        elif self.operator == "<":
            return value >= self.threshold
        elif self.operator == ">":
            return value <= self.threshold
        else:
            return False
    
    def to_mlflow_tag(self) -> tuple[str, str]:
        """Convert to MLflow tag for storage."""
        return (f"metric_threshold_{self.metric_name}", f"{self.operator}{self.threshold}")


class MetricThresholdManager:
    """Manage metric thresholds with MLflow integration."""
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize threshold manager.
        
        Args:
            experiment_name: Optional MLflow experiment name for storing thresholds
        """
        self.experiment_name = experiment_name
        self._thresholds: Dict[str, MetricThreshold] = {}
        self._registry = get_metric_registry()
    
    def add_threshold(
        self,
        metric_name: str,
        threshold: float,
        operator: str = "<=",
        alert_action: Optional[str] = None,
        alert_hook: Optional[Any] = None,
    ) -> None:
        """
        Add or update a metric threshold.
        
        Args:
            metric_name: Name of the metric
            threshold: Threshold value
            operator: Comparison operator
            alert_action: Action to take on breach
            alert_hook: Optional callable for custom actions
        """
        threshold_obj = MetricThreshold(
            metric_name=metric_name,
            threshold=threshold,
            operator=operator,
            alert_action=alert_action,
            alert_hook=alert_hook,
        )
        self._thresholds[metric_name] = threshold_obj
        
        # Register with metric registry
        self._registry.set_threshold(metric_name, threshold, operator)
        
        # Store in MLflow if experiment is set
        if self.experiment_name:
            try:
                mlflow.set_experiment(self.experiment_name)
                tag_key, tag_value = threshold_obj.to_mlflow_tag()
                mlflow.set_tag(tag_key, tag_value)
                
                if alert_action:
                    mlflow.set_tag(f"metric_alert_{metric_name}", alert_action)
            except Exception:
                # Silently fail if MLflow not available
                pass
    
    def check_threshold(self, metric_name: str, value: float) -> bool:
        """Check if a metric value breaches its threshold."""
        if metric_name not in self._thresholds:
            return False
        
        threshold = self._thresholds[metric_name]
        breached = threshold.check(value)
        
        if breached:
            # Log breach to MLflow
            try:
                mlflow.log_metric(f"{metric_name}_breach", 1.0)
                mlflow.log_metric(f"{metric_name}_breach_value", value)
            except Exception:
                pass
            
            # Execute alert action
            self._execute_alert(metric_name, value, threshold)
        
        return breached
    
    def _execute_alert(self, metric_name: str, value: float, threshold: MetricThreshold) -> None:
        """Execute alert action for breached threshold."""
        if not threshold.alert_action:
            return
        
        if threshold.alert_action == "log":
            # Already logged to MLflow above
            pass
        elif threshold.alert_action == "hook" and threshold.alert_hook:
            # Call custom hook
            try:
                threshold.alert_hook(metric_name, value, threshold.threshold)
            except Exception:
                pass
        elif threshold.alert_action == "route_to_hitl":
            # This would need to be handled at the rulebook level
            # For now, just log it
            try:
                mlflow.set_tag(f"{metric_name}_route_to_hitl", "true")
            except Exception:
                pass
    
    def get_threshold(self, metric_name: str) -> Optional[MetricThreshold]:
        """Get threshold for a metric."""
        return self._thresholds.get(metric_name)
    
    def list_thresholds(self) -> List[MetricThreshold]:
        """List all configured thresholds."""
        return list(self._thresholds.values())
    
    def load_from_mlflow(self, experiment_name: str) -> None:
        """
        Load thresholds from MLflow experiment tags.
        
        Args:
            experiment_name: MLflow experiment name
        """
        try:
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                return
            
            # Get tags from experiment (MLflow doesn't have experiment-level tags directly)
            # Would need to check runs or use a different approach
            # For now, this is a placeholder
            pass
        except Exception:
            pass

