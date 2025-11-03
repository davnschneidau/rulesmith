"""Blue/green deployment for rulebooks."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from rulesmith.io.decision_result import DecisionResult


@dataclass
class DeploymentConfig:
    """Configuration for blue/green deployment."""
    
    blue_version: str  # Current production version
    green_version: str  # New version to test
    traffic_split: float = 0.0  # 0.0 = all blue, 1.0 = all green
    canary_percentage: float = 0.0  # Percentage of traffic to green
    enable_auto_rollback: bool = True
    rollback_thresholds: Dict[str, float] = None  # Metric thresholds for auto-rollback
    
    def __post_init__(self):
        if self.rollback_thresholds is None:
            self.rollback_thresholds = {
                "error_rate": 0.05,  # 5% error rate triggers rollback
                "latency_p95_ms": 1000.0,  # 1s p95 latency triggers rollback
            }


class BlueGreenDeployment:
    """Manages blue/green deployment of rulebooks."""
    
    def __init__(
        self,
        rulebook_name: str,
        blue_rulebook: Any,  # Rulebook instance
        green_rulebook: Any,  # Rulebook instance
        config: DeploymentConfig,
    ):
        """
        Initialize blue/green deployment.
        
        Args:
            rulebook_name: Name of rulebook
            blue_rulebook: Blue (current production) rulebook
            green_rulebook: Green (new) rulebook
            config: Deployment configuration
        """
        self.rulebook_name = rulebook_name
        self.blue_rulebook = blue_rulebook
        self.green_rulebook = green_rulebook
        self.config = config
        
        # Track metrics for comparison
        self.blue_metrics: Dict[str, Any] = {}
        self.green_metrics: Dict[str, Any] = {}
        self.comparisons: List[Dict[str, Any]] = []
    
    def execute(
        self,
        payload: Dict[str, Any],
        context: Optional[Any] = None,
        force_version: Optional[str] = None,  # "blue" or "green" to force
    ) -> DecisionResult:
        """
        Execute with blue/green routing.
        
        Args:
            payload: Input payload
            context: Optional execution context
            force_version: Force blue or green (for testing)
        
        Returns:
            DecisionResult from selected version
        """
        import random
        
        # Determine which version to use
        if force_version:
            use_green = force_version == "green"
        elif self.config.canary_percentage > 0:
            # Canary rollout based on percentage
            use_green = random.random() < self.config.canary_percentage
        else:
            # Traffic split based on split ratio
            use_green = random.random() < self.config.traffic_split
        
        # Execute selected version
        if use_green:
            rulebook = self.green_rulebook
            version_label = "green"
        else:
            rulebook = self.blue_rulebook
            version_label = "blue"
        
        result = rulebook.run(payload, context=context, enable_mlflow=True)
        
        # Track metrics
        self._track_metrics(version_label, result)
        
        # Check for auto-rollback
        if use_green and self.config.enable_auto_rollback:
            should_rollback = self._should_rollback(result)
            if should_rollback:
                # Switch to blue for this request
                result = self.blue_rulebook.run(payload, context=context, enable_mlflow=True)
                self._track_metrics("blue", result)
                # TODO: Could trigger full rollback here
        
        return result
    
    def _track_metrics(self, version: str, result: DecisionResult) -> None:
        """Track metrics for a version."""
        metrics = self.green_metrics if version == "green" else self.blue_metrics
        
        # Update metrics
        if "executions" not in metrics:
            metrics["executions"] = 0
            metrics["errors"] = 0
            metrics["latencies"] = []
            metrics["costs"] = []
        
        metrics["executions"] += 1
        
        if result.warnings:
            metrics["errors"] += len(result.warnings)
        
        if "total_duration_ms" in result.metrics:
            metrics["latencies"].append(result.metrics["total_duration_ms"])
        
        if result.costs:
            total_cost = sum(result.costs.values())
            metrics["costs"].append(total_cost)
    
    def _should_rollback(self, result: DecisionResult) -> bool:
        """Check if auto-rollback should trigger."""
        # Calculate error rate
        total_executions = self.green_metrics.get("executions", 1)
        total_errors = self.green_metrics.get("errors", 0)
        error_rate = total_errors / total_executions if total_executions > 0 else 0.0
        
        # Check error rate threshold
        if error_rate > self.config.rollback_thresholds.get("error_rate", 0.05):
            return True
        
        # Check latency threshold
        latencies = self.green_metrics.get("latencies", [])
        if latencies:
            sorted_latencies = sorted(latencies)
            p95_index = int(0.95 * len(sorted_latencies))
            if p95_index < len(sorted_latencies):
                p95_latency = sorted_latencies[p95_index]
                if p95_latency > self.config.rollback_thresholds.get("latency_p95_ms", 1000.0):
                    return True
        
        return False
    
    def compare_versions(
        self,
        sample_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Compare blue and green versions on sample inputs.
        
        Args:
            sample_size: Number of sample inputs to test
        
        Returns:
            Comparison report
        """
        # This would typically use a test dataset
        # For now, return summary of tracked metrics
        
        blue_executions = self.blue_metrics.get("executions", 0)
        green_executions = self.green_metrics.get("executions", 0)
        
        blue_error_rate = (
            self.blue_metrics.get("errors", 0) / blue_executions
            if blue_executions > 0
            else 0.0
        )
        green_error_rate = (
            self.green_metrics.get("errors", 0) / green_executions
            if green_executions > 0
            else 0.0
        )
        
        blue_latencies = self.blue_metrics.get("latencies", [])
        green_latencies = self.green_metrics.get("latencies", [])
        
        blue_p95 = self._calculate_percentile(blue_latencies, 95) if blue_latencies else 0.0
        green_p95 = self._calculate_percentile(green_latencies, 95) if green_latencies else 0.0
        
        return {
            "blue_version": self.config.blue_version,
            "green_version": self.config.green_version,
            "blue_metrics": {
                "executions": blue_executions,
                "error_rate": blue_error_rate,
                "latency_p95_ms": blue_p95,
            },
            "green_metrics": {
                "executions": green_executions,
                "error_rate": green_error_rate,
                "latency_p95_ms": green_p95,
            },
            "comparison": {
                "error_rate_delta": green_error_rate - blue_error_rate,
                "latency_delta_ms": green_p95 - blue_p95,
            },
        }
    
    def _calculate_percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile of a list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    def promote_green(self) -> None:
        """Promote green to blue (complete rollout)."""
        self.config.blue_version = self.config.green_version
        self.config.traffic_split = 1.0
        self.config.canary_percentage = 1.0
    
    def rollback_to_blue(self) -> None:
        """Rollback to blue (full rollback)."""
        self.config.traffic_split = 0.0
        self.config.canary_percentage = 0.0

