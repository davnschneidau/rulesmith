"""A/B testing metrics and comparative analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import statistics


@dataclass
class ArmMetrics:
    """Metrics for a single A/B test arm."""
    
    arm_name: str
    sample_size: int = 0
    conversion_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    cost_per_approval: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparativeMetrics:
    """Comparative metrics between two arms."""
    
    arm_a: str
    arm_b: str
    conversion_rate_delta: float = 0.0
    false_positive_rate_delta: float = 0.0
    latency_delta_p95: float = 0.0
    cost_delta: float = 0.0
    statistical_significance: Optional[float] = None  # p-value
    confidence_interval: Optional[Tuple[float, float]] = None
    custom_deltas: Dict[str, float] = field(default_factory=dict)


class ABMetricsCollector:
    """Collects and analyzes A/B testing metrics."""
    
    def __init__(self):
        self.arm_data: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_arm_execution(
        self,
        arm_name: str,
        outcome: Optional[Any] = None,
        latency_ms: float = 0.0,
        cost: float = 0.0,
        error: Optional[str] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record an execution for an arm.
        
        Args:
            arm_name: Name of the arm
            outcome: Outcome value (e.g., approved=True, conversion=True)
            latency_ms: Execution latency in milliseconds
            cost: Cost in USD
            error: Error message if execution failed
            custom_metrics: Custom metrics dictionary
        """
        if arm_name not in self.arm_data:
            self.arm_data[arm_name] = []
        
        record = {
            "outcome": outcome,
            "latency_ms": latency_ms,
            "cost": cost,
            "error": error,
            "custom_metrics": custom_metrics or {},
        }
        
        self.arm_data[arm_name].append(record)
    
    def compute_arm_metrics(self, arm_name: str, conversion_key: Optional[str] = None) -> ArmMetrics:
        """
        Compute metrics for an arm.
        
        Args:
            arm_name: Name of the arm
            conversion_key: Optional key to extract conversion from outcome (e.g., "approved")
        
        Returns:
            ArmMetrics object
        """
        if arm_name not in self.arm_data or not self.arm_data[arm_name]:
            return ArmMetrics(arm_name=arm_name)
        
        records = self.arm_data[arm_name]
        sample_size = len(records)
        
        # Extract conversions
        conversions = []
        if conversion_key:
            conversions = [
                1 if (isinstance(r["outcome"], dict) and r["outcome"].get(conversion_key)) 
                   or (r["outcome"] is True) 
                   else 0
                for r in records
            ]
        else:
            # Assume outcome is boolean
            conversions = [1 if r["outcome"] else 0 for r in records]
        
        conversion_rate = sum(conversions) / sample_size if sample_size > 0 else 0.0
        
        # Compute latencies
        latencies = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
        latency_p50 = statistics.median(latencies) if latencies else 0.0
        latency_p95 = self._percentile(latencies, 95) if latencies else 0.0
        latency_p99 = self._percentile(latencies, 99) if latencies else 0.0
        
        # Compute costs
        costs = [r["cost"] for r in records if r["cost"] > 0]
        total_cost = sum(costs)
        cost_per_approval = total_cost / sum(conversions) if sum(conversions) > 0 else 0.0
        
        # Error rate
        errors = [1 for r in records if r["error"]]
        error_rate = len(errors) / sample_size if sample_size > 0 else 0.0
        
        # Custom metrics (aggregate)
        custom_metrics = {}
        if records and records[0].get("custom_metrics"):
            for key in records[0]["custom_metrics"].keys():
                values = [r["custom_metrics"].get(key, 0) for r in records if r.get("custom_metrics")]
                if values:
                    custom_metrics[key] = statistics.mean(values)
        
        return ArmMetrics(
            arm_name=arm_name,
            sample_size=sample_size,
            conversion_rate=conversion_rate,
            cost_per_approval=cost_per_approval,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_rate=error_rate,
            custom_metrics=custom_metrics,
        )
    
    def compare_arms(
        self,
        arm_a: str,
        arm_b: str,
        conversion_key: Optional[str] = None,
    ) -> ComparativeMetrics:
        """
        Compare two arms and compute deltas.
        
        Args:
            arm_a: First arm name
            arm_b: Second arm name
            conversion_key: Optional key to extract conversion from outcome
        
        Returns:
            ComparativeMetrics object
        """
        metrics_a = self.compute_arm_metrics(arm_a, conversion_key)
        metrics_b = self.compute_arm_metrics(arm_b, conversion_key)
        
        # Compute deltas
        conversion_delta = metrics_b.conversion_rate - metrics_a.conversion_rate
        latency_delta = metrics_b.latency_p95 - metrics_a.latency_p95
        cost_delta = metrics_b.cost_per_approval - metrics_a.cost_per_approval
        
        # Compute statistical significance (simplified t-test approximation)
        p_value = None
        if metrics_a.sample_size > 30 and metrics_b.sample_size > 30:
            try:
                p_value = self._compute_p_value(metrics_a, metrics_b, conversion_key)
            except Exception:
                pass  # If computation fails, leave as None
        
        return ComparativeMetrics(
            arm_a=arm_a,
            arm_b=arm_b,
            conversion_rate_delta=conversion_delta,
            latency_delta_p95=latency_delta,
            cost_delta=cost_delta,
            statistical_significance=p_value,
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Compute percentile of a list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    def _compute_p_value(
        self,
        metrics_a: ArmMetrics,
        metrics_b: ArmMetrics,
        conversion_key: Optional[str] = None,
    ) -> Optional[float]:
        """
        Compute p-value for conversion rate difference (simplified two-proportion z-test).
        
        This is a simplified implementation. For production, use scipy.stats.
        """
        try:
            # Two-proportion z-test approximation
            n1, p1 = metrics_a.sample_size, metrics_a.conversion_rate
            n2, p2 = metrics_b.sample_size, metrics_b.conversion_rate
            
            if n1 == 0 or n2 == 0:
                return None
            
            # Pooled proportion
            pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
            
            # Standard error
            se = (pooled_p * (1 - pooled_p) * (1/n1 + 1/n2)) ** 0.5
            
            if se == 0:
                return None
            
            # Z-score
            z = (p2 - p1) / se
            
            # Approximate p-value (two-tailed)
            # Simplified: p-value ≈ 2 * (1 - norm.cdf(abs(z)))
            # For simplicity, return z-score as proxy (actual p-value requires scipy)
            return abs(z)
        except Exception:
            return None
    
    def get_all_metrics(self, conversion_key: Optional[str] = None) -> Dict[str, ArmMetrics]:
        """Get metrics for all arms."""
        return {
            arm_name: self.compute_arm_metrics(arm_name, conversion_key)
            for arm_name in self.arm_data.keys()
        }


# Global metrics collector
metrics_collector = ABMetricsCollector()

