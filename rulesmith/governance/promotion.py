"""Promotion system with SLO checks."""

from typing import Any, Dict, List, Optional

import mlflow
from mlflow.exceptions import MlflowException


class SLO:
    """Service Level Objective definition."""

    def __init__(
        self,
        metric_name: str,
        threshold: float,
        operator: str = ">=",
        window_hours: int = 24,
    ):
        """
        Initialize SLO.

        Args:
            metric_name: Metric name to check
            threshold: Threshold value
            operator: Comparison operator (">=", "<=", "==", ">", "<")
            window_hours: Time window in hours to evaluate
        """
        self.metric_name = metric_name
        self.threshold = threshold
        self.operator = operator
        self.window_hours = window_hours

    def evaluate(self, value: float) -> bool:
        """Evaluate if value meets SLO."""
        if self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 1e-6
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "<":
            return value < self.threshold
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class PromotionPolicy:
    """Policy defining promotion requirements."""

    def __init__(
        self,
        name: str,
        slos: List[SLO],
        require_all: bool = True,
        min_samples: int = 100,
    ):
        """
        Initialize promotion policy.

        Args:
            name: Policy name
            slos: List of SLOs to check
            require_all: If True, all SLOs must pass; if False, any SLO passing is enough
            min_samples: Minimum number of samples required for evaluation
        """
        self.name = name
        self.slos = slos
        self.require_all = require_all
        self.min_samples = min_samples


def get_model_metrics(
    model_uri: str,
    metric_names: List[str],
    window_hours: int = 24,
) -> Dict[str, float]:
    """
    Get model metrics from MLflow.

    Args:
        model_uri: Model URI (e.g., "models:/my_model/1")
        metric_names: List of metric names to retrieve
        window_hours: Time window in hours

    Returns:
        Dictionary mapping metric names to values
    """
    try:
        from datetime import datetime, timedelta

        # Get model version
        if model_uri.startswith("models:/"):
            parts = model_uri.split("/")
            model_name = parts[1]
            version_or_stage = parts[2] if len(parts) > 2 else "latest"

            # Get model version info
            client = mlflow.tracking.MlflowClient()
            if version_or_stage.startswith("@") or version_or_stage in ["None", "latest"]:
                # Get by stage
                stage = version_or_stage.replace("@", "")
                versions = client.search_model_versions(f"name='{model_name}'")
                version = next((v for v in versions if v.current_stage == stage), None)
                if not version:
                    return {}
            else:
                version = client.get_model_version(model_name, version_or_stage)

            # Get runs for this model version
            run_id = version.run_id

            # Get metrics from run
            run = client.get_run(run_id)
            metrics = {}

            for metric_name in metric_names:
                if metric_name in run.data.metrics:
                    metrics[metric_name] = run.data.metrics[metric_name]
                else:
                    # Try to get latest value from run history
                    metric_history = client.get_metric_history(run_id, metric_name)
                    if metric_history:
                        metrics[metric_name] = metric_history[-1].value

            return metrics

        else:
            # Try to get from run URI
            run_id = model_uri.split("/")[-1] if "/" in model_uri else model_uri
            run = mlflow.get_run(run_id)
            metrics = {}

            for metric_name in metric_names:
                if metric_name in run.data.metrics:
                    metrics[metric_name] = run.data.metrics[metric_name]

            return metrics

    except Exception as e:
        # Return empty dict on error
        return {}


def check_slos(
    model_uri: str,
    policy: PromotionPolicy,
) -> Dict[str, Any]:
    """
    Check if model meets promotion SLOs.

    Args:
        model_uri: Model URI
        policy: Promotion policy

    Returns:
        Dictionary with check results
    """
    # Get metric names from SLOs
    metric_names = [slo.metric_name for slo in policy.slos]

    # Get metrics
    metrics = get_model_metrics(model_uri, metric_names, window_hours=max(slo.window_hours for slo in policy.slos))

    # Check each SLO
    slo_results = []
    all_passed = True
    any_passed = False

    for slo in policy.slos:
        if slo.metric_name not in metrics:
            result = {
                "slo": slo.metric_name,
                "passed": False,
                "reason": "metric_not_found",
                "value": None,
                "threshold": slo.threshold,
            }
            slo_results.append(result)
            all_passed = False
        else:
            value = metrics[slo.metric_name]
            passed = slo.evaluate(value)
            result = {
                "slo": slo.metric_name,
                "passed": passed,
                "value": value,
                "threshold": slo.threshold,
                "operator": slo.operator,
            }
            slo_results.append(result)

            if passed:
                any_passed = True
            else:
                all_passed = False

    # Determine overall result
    if policy.require_all:
        overall_passed = all_passed
    else:
        overall_passed = any_passed

    return {
        "model_uri": model_uri,
        "policy": policy.name,
        "overall_passed": overall_passed,
        "slo_results": slo_results,
        "metrics": metrics,
    }


def promote_model(
    model_name: str,
    from_stage: str,
    to_stage: str,
    policy: Optional[PromotionPolicy] = None,
) -> Dict[str, Any]:
    """
    Promote a model between stages with optional SLO checks.

    Args:
        model_name: Model name
        from_stage: Source stage
        to_stage: Target stage
        policy: Optional promotion policy with SLOs

    Returns:
        Dictionary with promotion result
    """
    try:
        client = mlflow.tracking.MlflowClient()

        # Get current model version in source stage
        versions = client.search_model_versions(f"name='{model_name}'")
        source_version = next((v for v in versions if v.current_stage == from_stage.replace("@", "")), None)

        if not source_version:
            return {
                "success": False,
                "error": f"No model found in stage '{from_stage}'",
            }

        model_uri = f"models:/{model_name}/{source_version.version}"

        # Check SLOs if policy provided
        if policy:
            slo_check = check_slos(model_uri, policy)
            if not slo_check["overall_passed"]:
                return {
                    "success": False,
                    "error": "SLO checks failed",
                    "slo_results": slo_check["slo_results"],
                }

        # Promote model
        client.transition_model_version_stage(
            name=model_name,
            version=source_version.version,
            stage=to_stage.replace("@", ""),
            archive_existing_versions=True,
        )

        return {
            "success": True,
            "model_name": model_name,
            "version": source_version.version,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "slo_check": slo_check if policy else None,
        }

    except MlflowException as e:
        return {
            "success": False,
            "error": str(e),
        }
