"""Rulesmith: A production-grade rulebook/DAG execution engine with MLflow integration."""

from rulesmith.dag.decorators import rule, rulebook
from rulesmith.dag.graph import Rulebook
from rulesmith.dag.registry import rule_registry, rulebook_registry
from rulesmith.governance import compare_rulebooks
from rulesmith.metrics import (
    Metric,
    MetricRegistry,
    compare_mlflow_metrics,
    get_metric_registry,
    query_mlflow_metrics,
)
from rulesmith.mlflow import MLflowLogger, MLflowRunContext, log_decision_to_mlflow, log_rulebook_model
from rulesmith.testing import RulebookTester

__version__ = "0.1.0"

__all__ = [
    "rule",
    "rulebook",
    "Rulebook",
    "rule_registry",
    "rulebook_registry",
    "MLflowLogger",
    "log_decision_to_mlflow",
    "Metric",
    "MetricRegistry",
    "get_metric_registry",
    "query_mlflow_metrics",
    "compare_mlflow_metrics",
    "MLflowLogger",
    "MLflowRunContext",
    "log_decision_to_mlflow",
    "log_rulebook_model",
    "RulebookTester",
    "compare_rulebooks",
]

