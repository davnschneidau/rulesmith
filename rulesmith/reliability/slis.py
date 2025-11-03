"""Service Level Indicators (SLIs) and Service Level Objectives (SLOs)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
import time


@dataclass
class SLI:
    """Service Level Indicator definition."""
    
    name: str
    metric_name: str
    window_seconds: int = 300  # 5-minute default window
    description: Optional[str] = None


@dataclass
class SLO:
    """Service Level Objective definition."""
    
    name: str
    sli_name: str
    target: float  # Target value (e.g., 0.99 for 99%)
    window_seconds: int = 3600  # 1-hour default window
    description: Optional[str] = None


class SLICollector:
    """Collects SLI metrics from executions."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = {}  # Metric name -> time-series data
        self.sli_definitions: Dict[str, SLI] = {}
        self.slo_definitions: Dict[str, SLO] = {}
    
    def register_sli(self, sli: SLI) -> None:
        """Register an SLI definition."""
        self.sli_definitions[sli.name] = sli
        if sli.metric_name not in self.metrics:
            self.metrics[sli.metric_name] = deque()
    
    def register_slo(self, slo: SLO) -> None:
        """Register an SLO definition."""
        self.slo_definitions[slo.name] = slo
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a metric value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            timestamp: Optional timestamp (defaults to current time)
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque()
        
        ts = timestamp or time.time()
        self.metrics[metric_name].append((ts, value))
        
        # Clean old data (keep only last hour)
        cutoff = ts - 3600
        while self.metrics[metric_name] and self.metrics[metric_name][0][0] < cutoff:
            self.metrics[metric_name].popleft()
    
    def compute_sli(
        self,
        sli_name: str,
        window_seconds: Optional[int] = None,
    ) -> float:
        """
        Compute SLI value.
        
        Args:
            sli_name: Name of SLI
            window_seconds: Optional window override
        
        Returns:
            SLI value (0.0-1.0 for rates, absolute for counts)
        """
        sli = self.sli_definitions.get(sli_name)
        if not sli:
            raise ValueError(f"SLI '{sli_name}' not found")
        
        window = window_seconds or sli.window_seconds
        cutoff = time.time() - window
        
        metric_data = self.metrics.get(sli.metric_name, deque())
        if not metric_data:
            return 0.0
        
        # Filter to window
        window_data = [(ts, val) for ts, val in metric_data if ts >= cutoff]
        
        if not window_data:
            return 0.0
        
        # Compute based on metric type
        if "rate" in sli_name.lower() or "success" in sli_name.lower():
            # Success rate
            total = len(window_data)
            success = sum(1 for _, val in window_data if val > 0)
            return success / total if total > 0 else 0.0
        elif "latency" in sli_name.lower():
            # Latency percentile
            values = sorted([val for _, val in window_data])
            p95_index = int(0.95 * len(values))
            if p95_index < len(values):
                return values[p95_index]
            return values[-1] if values else 0.0
        else:
            # Count or average
            return sum(val for _, val in window_data) / len(window_data) if window_data else 0.0
    
    def evaluate_slo(
        self,
        slo_name: str,
        window_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate SLO compliance.
        
        Args:
            slo_name: Name of SLO
            window_seconds: Optional window override
        
        Returns:
            SLO evaluation result
        """
        slo = self.slo_definitions.get(slo_name)
        if not slo:
            raise ValueError(f"SLO '{slo_name}' not found")
        
        sli_value = self.compute_sli(slo.sli_name, window_seconds or slo.window_seconds)
        compliant = sli_value >= slo.target
        
        return {
            "slo_name": slo_name,
            "sli_name": slo.sli_name,
            "sli_value": sli_value,
            "target": slo.target,
            "compliant": compliant,
            "error_budget": slo.target - sli_value if not compliant else 0.0,
        }
    
    def export_opentelemetry(
        self,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export metrics in OpenTelemetry format.
        
        Args:
            metric_name: Optional metric name to export (exports all if None)
        
        Returns:
            List of OpenTelemetry metric data points
        """
        try:
            # Try to import OpenTelemetry
            from opentelemetry import metrics as otel_metrics
            from opentelemetry.sdk.metrics import MeterProvider
        except ImportError:
            # Return simplified format if OpenTelemetry not available
            return self._export_simplified(metric_name)
        
        # Export in OpenTelemetry format
        data_points = []
        metrics_to_export = [metric_name] if metric_name else list(self.metrics.keys())
        
        for m_name in metrics_to_export:
            if m_name not in self.metrics:
                continue
            
            metric_data = self.metrics[m_name]
            for timestamp, value in metric_data:
                data_points.append({
                    "metric_name": m_name,
                    "value": value,
                    "timestamp": timestamp,
                    "labels": {},  # Could add labels for rulebook/version/etc.
                })
        
        return data_points
    
    def _export_simplified(
        self,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Export simplified format when OpenTelemetry not available."""
        data_points = []
        metrics_to_export = [metric_name] if metric_name else list(self.metrics.keys())
        
        for m_name in metrics_to_export:
            if m_name not in self.metrics:
                continue
            
            metric_data = self.metrics[m_name]
            for timestamp, value in metric_data:
                data_points.append({
                    "metric_name": m_name,
                    "value": value,
                    "timestamp": timestamp,
                })
        
        return data_points
    
    def get_sli_summary(self) -> Dict[str, Any]:
        """Get summary of all SLIs."""
        summary = {}
        for sli_name, sli in self.sli_definitions.items():
            try:
                value = self.compute_sli(sli_name)
                summary[sli_name] = {
                    "value": value,
                    "metric_name": sli.metric_name,
                    "window_seconds": sli.window_seconds,
                }
            except Exception as e:
                summary[sli_name] = {"error": str(e)}
        
        return summary
    
    def get_slo_summary(self) -> Dict[str, Any]:
        """Get summary of all SLOs."""
        summary = {}
        for slo_name, slo in self.slo_definitions.items():
            try:
                evaluation = self.evaluate_slo(slo_name)
                summary[slo_name] = evaluation
            except Exception as e:
                summary[slo_name] = {"error": str(e)}
        
        return summary


# Global SLI collector
sli_collector = SLICollector()

