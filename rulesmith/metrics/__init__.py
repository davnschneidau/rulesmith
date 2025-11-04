"""Unified metrics system with MLflow integration."""

from rulesmith.metrics.aggregation import compare_mlflow_metrics, query_mlflow_metrics
from rulesmith.metrics.core import (
    Metric,
    MetricRegistry,
    Threshold,
    get_metric_registry,
)

__all__ = [
    # Core metrics
    "Metric",
    "MetricRegistry",
    "Threshold",
    "get_metric_registry",
    # MLflow queries
    "query_mlflow_metrics",
    "compare_mlflow_metrics",
]
