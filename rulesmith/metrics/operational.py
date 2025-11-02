"""Operational KPI metrics tracking."""

from datetime import datetime
from typing import Any, Dict, List, Optional


class OperationalMetric:
    """
    Track operational KPIs for rules.
    
    Metrics: False-positive rate, manual review throughput/time,
    SLAs, queue depth, review time tracking.
    """

    def __init__(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize operational metric.

        Args:
            metric_name: Metric name (e.g., "false_positive_rate", "review_time_seconds")
            value: Metric value
            timestamp: Optional timestamp
            metadata: Additional metadata
        """
        self.metric_name = metric_name
        self.value = value
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class OperationalMetricsTracker:
    """
    Track and aggregate operational metrics over time.
    """

    def __init__(self):
        self.metrics: List[OperationalMetric] = []

    def record(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an operational metric.

        Args:
            metric_name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        metric = OperationalMetric(metric_name, value, metadata=metadata)
        self.metrics.append(metric)

    def record_false_positive_rate(
        self,
        fpr: float,
        rulebook_name: Optional[str] = None,
    ) -> None:
        """Record false positive rate."""
        metadata = {}
        if rulebook_name:
            metadata["rulebook"] = rulebook_name
        self.record("false_positive_rate", fpr, metadata=metadata)

    def record_review_time(
        self,
        seconds: float,
        review_type: Optional[str] = None,
    ) -> None:
        """Record manual review time in seconds."""
        metadata = {}
        if review_type:
            metadata["review_type"] = review_type
        self.record("review_time_seconds", seconds, metadata=metadata)

    def record_queue_depth(
        self,
        depth: int,
        queue_name: Optional[str] = None,
    ) -> None:
        """Record review queue depth."""
        metadata = {}
        if queue_name:
            metadata["queue_name"] = queue_name
        self.record("queue_depth", float(depth), metadata=metadata)

    def record_review_throughput(
        self,
        reviews_per_hour: float,
        queue_name: Optional[str] = None,
    ) -> None:
        """Record review throughput (reviews per hour)."""
        metadata = {}
        if queue_name:
            metadata["queue_name"] = queue_name
        self.record("review_throughput_per_hour", reviews_per_hour, metadata=metadata)

    def record_sla_compliance(
        self,
        compliant: bool,
        sla_name: Optional[str] = None,
    ) -> None:
        """Record SLA compliance (1.0 for compliant, 0.0 for non-compliant)."""
        metadata = {}
        if sla_name:
            metadata["sla_name"] = sla_name
        self.record("sla_compliant", 1.0 if compliant else 0.0, metadata=metadata)

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[OperationalMetric]:
        """
        Get metrics, optionally filtered.

        Args:
            metric_name: Optional metric name filter
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of matching metrics
        """
        filtered = self.metrics.copy()

        if metric_name:
            filtered = [m for m in filtered if m.metric_name == metric_name]

        if start_time:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        return filtered

    def get_average_false_positive_rate(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> float:
        """Get average false positive rate over time period."""
        metrics = self.get_metrics("false_positive_rate", start_time, end_time)
        if not metrics:
            return 0.0
        return sum(m.value for m in metrics) / len(metrics)

    def get_average_review_time(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> float:
        """Get average review time over time period."""
        metrics = self.get_metrics("review_time_seconds", start_time, end_time)
        if not metrics:
            return 0.0
        return sum(m.value for m in metrics) / len(metrics)


# Global operational metrics tracker
operational_metrics_tracker = OperationalMetricsTracker()

