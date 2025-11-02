"""Business KPI metrics tracking."""

from datetime import datetime
from typing import Any, Dict, List, Optional


class BusinessMetric:
    """
    Track business KPIs for rules.
    
    Metrics: Prevented loss ($), bad debt %, recoveries,
    conversion lift/drop, NPS impact.
    """

    def __init__(
        self,
        metric_name: str,
        value: float,
        unit: str = "USD",
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize business metric.

        Args:
            metric_name: Metric name (e.g., "prevented_loss", "conversion_lift")
            value: Metric value
            unit: Unit of measurement (e.g., "USD", "percentage")
            timestamp: Optional timestamp (defaults to now)
            metadata: Additional metadata
        """
        self.metric_name = metric_name
        self.value = value
        self.unit = unit
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class BusinessMetricsTracker:
    """
    Track and aggregate business metrics over time.
    
    Provides programmatic access to business KPIs.
    """

    def __init__(self):
        self.metrics: List[BusinessMetric] = []

    def record(
        self,
        metric_name: str,
        value: float,
        unit: str = "USD",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a business metric.

        Args:
            metric_name: Metric name
            value: Metric value
            unit: Unit of measurement
            metadata: Optional metadata
        """
        metric = BusinessMetric(metric_name, value, unit, metadata=metadata)
        self.metrics.append(metric)

    def record_prevented_loss(self, amount: float, rulebook_name: Optional[str] = None) -> None:
        """Record prevented loss (in dollars)."""
        metadata = {}
        if rulebook_name:
            metadata["rulebook"] = rulebook_name
        self.record("prevented_loss", amount, unit="USD", metadata=metadata)

    def record_bad_debt_percentage(self, percentage: float, rulebook_name: Optional[str] = None) -> None:
        """Record bad debt percentage."""
        metadata = {}
        if rulebook_name:
            metadata["rulebook"] = rulebook_name
        self.record("bad_debt_percentage", percentage, unit="percentage", metadata=metadata)

    def record_conversion_lift(self, lift: float, rulebook_name: Optional[str] = None) -> None:
        """Record conversion lift (positive improvement)."""
        metadata = {}
        if rulebook_name:
            metadata["rulebook"] = rulebook_name
        self.record("conversion_lift", lift, unit="percentage", metadata=metadata)

    def record_conversion_drop(self, drop: float, rulebook_name: Optional[str] = None) -> None:
        """Record conversion drop (negative impact)."""
        metadata = {}
        if rulebook_name:
            metadata["rulebook"] = rulebook_name
        self.record("conversion_drop", drop, unit="percentage", metadata=metadata)

    def record_recoveries(self, amount: float, rulebook_name: Optional[str] = None) -> None:
        """Record recoveries (in dollars)."""
        metadata = {}
        if rulebook_name:
            metadata["rulebook"] = rulebook_name
        self.record("recoveries", amount, unit="USD", metadata=metadata)

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[BusinessMetric]:
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

    def aggregate(
        self,
        metric_name: str,
        aggregation: str = "sum",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> float:
        """
        Aggregate metrics.

        Args:
            metric_name: Metric name
            aggregation: Aggregation type ("sum", "avg", "max", "min", "count")
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            Aggregated value
        """
        metrics = self.get_metrics(metric_name, start_time, end_time)

        if not metrics:
            return 0.0

        values = [m.value for m in metrics]

        if aggregation == "sum":
            return sum(values)
        elif aggregation == "avg":
            return sum(values) / len(values) if values else 0.0
        elif aggregation == "max":
            return max(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "count":
            return float(len(values))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


# Global business metrics tracker
business_metrics_tracker = BusinessMetricsTracker()

