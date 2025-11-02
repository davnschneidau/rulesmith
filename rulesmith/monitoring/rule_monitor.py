"""Rule execution monitoring and metrics tracking."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class RuleMetrics:
    """Metrics for a single rule."""

    def __init__(self, rule_name: str):
        self.rule_name = rule_name
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.total_latency_ms = 0.0
        self.last_executed: Optional[datetime] = None

    @property
    def precision(self) -> float:
        """Calculate precision."""
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall."""
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate."""
        fp_tn = self.false_positives + self.true_negatives
        return self.false_positives / fp_tn if fp_tn > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.execution_count if self.execution_count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.success_count / self.execution_count if self.execution_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "precision": self.precision,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
        }


class MonitoringDashboard:
    """
    Aggregated metrics dashboard (data structure, not UI).
    
    Provides programmatic access to all monitoring metrics.
    """

    def __init__(self):
        self.rule_metrics: Dict[str, RuleMetrics] = {}
        self.rulebook_metrics: Dict[str, Dict[str, Any]] = {}
        self.timeline: List[Dict[str, Any]] = []

    def record_execution(
        self,
        rule_name: str,
        success: bool = True,
        latency_ms: Optional[float] = None,
        tp: int = 0,
        fp: int = 0,
        tn: int = 0,
        fn: int = 0,
    ) -> None:
        """
        Record a rule execution.

        Args:
            rule_name: Rule name
            success: Whether execution succeeded
            latency_ms: Execution latency in milliseconds
            tp: True positives (if ground truth available)
            fp: False positives
            tn: True negatives
            fn: False negatives
        """
        if rule_name not in self.rule_metrics:
            self.rule_metrics[rule_name] = RuleMetrics(rule_name)

        metrics = self.rule_metrics[rule_name]
        metrics.execution_count += 1
        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1

        metrics.true_positives += tp
        metrics.false_positives += fp
        metrics.true_negatives += tn
        metrics.false_negatives += fn

        if latency_ms:
            metrics.total_latency_ms += latency_ms

        metrics.last_executed = datetime.utcnow()

        # Add to timeline
        self.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "rule_name": rule_name,
            "success": success,
            "latency_ms": latency_ms,
        })

    def get_rule_metrics(self, rule_name: str) -> Optional[RuleMetrics]:
        """Get metrics for a specific rule."""
        return self.rule_metrics.get(rule_name)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all rule metrics as dictionary."""
        return {name: metrics.to_dict() for name, metrics in self.rule_metrics.items()}

    def get_time_window_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get metrics for a time window."""
        if not HAS_PANDAS:
            # Fallback to simple filtering
            window_events = [
                event
                for event in self.timeline
                if start_time <= datetime.fromisoformat(event["timestamp"]) <= end_time
            ]
            return {"events": window_events, "count": len(window_events)}

        # Use pandas for efficient filtering
        df = pd.DataFrame(self.timeline)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        window_df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

        return {
            "events": window_df.to_dict("records"),
            "count": len(window_df),
            "success_rate": window_df["success"].mean() if len(window_df) > 0 else 0.0,
            "average_latency_ms": window_df["latency_ms"].mean() if len(window_df) > 0 else 0.0,
        }


class RuleMonitor:
    """
    Monitor rule execution in real-time.
    
    Tracks metrics, coverage, precision/recall, conversion rates.
    """

    def __init__(self):
        self.dashboard = MonitoringDashboard()
        self.rulebook_coverage: Dict[str, float] = {}  # Rulebook name -> coverage %

    def track_execution(
        self,
        rule_name: str,
        success: bool = True,
        latency_ms: Optional[float] = None,
        ground_truth: Optional[bool] = None,
        prediction: Optional[bool] = None,
    ) -> None:
        """
        Track a rule execution.

        Args:
            rule_name: Rule name
            success: Whether execution succeeded
            latency_ms: Execution latency
            ground_truth: Ground truth label (for TP/FP calculation)
            prediction: Rule prediction
        """
        # Calculate TP/FP/TN/FN if ground truth available
        tp, fp, tn, fn = 0, 0, 0, 0
        if ground_truth is not None and prediction is not None:
            if prediction and ground_truth:
                tp = 1
            elif prediction and not ground_truth:
                fp = 1
            elif not prediction and not ground_truth:
                tn = 1
            else:  # not prediction and ground_truth
                fn = 1

        self.dashboard.record_execution(
            rule_name=rule_name,
            success=success,
            latency_ms=latency_ms,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    def update_coverage(self, rulebook_name: str, coverage: float) -> None:
        """Update rulebook coverage percentage."""
        self.rulebook_coverage[rulebook_name] = coverage

    def get_metrics(self, rule_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a rule or all rules.

        Args:
            rule_name: Optional rule name (if None, returns all)

        Returns:
            Metrics dictionary
        """
        if rule_name:
            metrics = self.dashboard.get_rule_metrics(rule_name)
            return metrics.to_dict() if metrics else {}
        else:
            return self.dashboard.get_all_metrics()

    def get_dashboard(self) -> MonitoringDashboard:
        """Get the monitoring dashboard."""
        return self.dashboard


# Global rule monitor instance
rule_monitor = RuleMonitor()

