"""MLflow-based metric aggregation and comparison."""

from typing import Any, Dict, List, Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None


def query_mlflow_metrics(
    experiment_name: str,
    metric_names: List[str],
    filter_string: Optional[str] = None,
    aggregation: List[str] = None,
    time_range: Optional[str] = None,
    max_results: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Query MLflow for aggregated metrics.
    
    Args:
        experiment_name: MLflow experiment name
        metric_names: List of metric names to query
        filter_string: Optional MLflow filter string (e.g., "tags.rulebook_version='1.0.0'")
        aggregation: List of aggregation functions (e.g., ["mean", "p95", "max"])
        time_range: Optional time range (e.g., "last_7_days", "last_24h")
        max_results: Maximum number of runs to query
    
    Returns:
        Dictionary mapping metric names to aggregated values
    
    Examples:
        # Query metrics for a specific rulebook version
        agg = query_mlflow_metrics(
            experiment_name="rulesmith/credit_decision",
            metric_names=["latency_ms", "guard_violations"],
            filter_string="tags.rulebook_version='1.0.0'",
            aggregation=["mean", "p95", "max"]
        )
        # Returns: {"latency_ms": {"mean": 45.2, "p95": 120.5, "max": 250.0}, ...}
    """
    if mlflow is None:
        raise ImportError("MLflow is required for metric aggregation. Install with: pip install mlflow")
    
    if aggregation is None:
        aggregation = ["mean", "p95", "max"]
    
    try:
        mlflow.set_experiment(experiment_name)
        client = MlflowClient()
        
        # Build filter string
        filters = []
        if filter_string:
            filters.append(filter_string)
        
        # Query runs
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return {name: {} for name in metric_names}
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=" AND ".join(filters) if filters else "",
            max_results=max_results,
        )
        
        if not runs:
            return {name: {} for name in metric_names}
        
        # Aggregate metrics
        results = {}
        for metric_name in metric_names:
            values = []
            for run in runs:
                if metric_name in run.data.metrics:
                    values.append(run.data.metrics[metric_name])
            
            if not values:
                results[metric_name] = {}
                continue
            
            # Calculate aggregations
            agg_results = {}
            if "mean" in aggregation:
                agg_results["mean"] = sum(values) / len(values)
            if "min" in aggregation:
                agg_results["min"] = min(values)
            if "max" in aggregation:
                agg_results["max"] = max(values)
            if "p50" in aggregation or "median" in aggregation:
                sorted_values = sorted(values)
                agg_results["p50"] = sorted_values[len(sorted_values) // 2]
            if "p95" in aggregation:
                sorted_values = sorted(values)
                agg_results["p95"] = sorted_values[int(len(sorted_values) * 0.95)]
            if "p99" in aggregation:
                sorted_values = sorted(values)
                agg_results["p99"] = sorted_values[int(len(sorted_values) * 0.99)]
            
            results[metric_name] = agg_results
        
        return results
    
    except Exception as e:
        raise RuntimeError(f"Failed to query MLflow metrics: {str(e)}")


def compare_mlflow_metrics(
    experiment_name: str,
    baseline_filter: str,
    comparison_filter: str,
    metrics: List[str],
    max_results: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare metrics between two rulebook versions or configurations.
    
    Args:
        experiment_name: MLflow experiment name
        baseline_filter: MLflow filter string for baseline (e.g., "tags.rulebook_version='1.0.0'")
        comparison_filter: MLflow filter string for comparison (e.g., "tags.rulebook_version='1.1.0'")
        metrics: List of metric names to compare
        max_results: Maximum number of runs to query per group
    
    Returns:
        Dictionary with comparison results for each metric
    
    Examples:
        # Compare two rulebook versions
        comparison = compare_mlflow_metrics(
            experiment_name="rulesmith/credit_decision",
            baseline_filter="tags.rulebook_version='1.0.0'",
            comparison_filter="tags.rulebook_version='1.1.0'",
            metrics=["latency_ms", "guard_violations", "cost_usd"]
        )
        # Returns: {
        #     "latency_ms": {
        #         "baseline_mean": 45.2,
        #         "comparison_mean": 42.1,
        #         "delta": -3.1,
        #         "delta_percent": -6.86
        #     },
        #     ...
        # }
    """
    if mlflow is None:
        raise ImportError("MLflow is required for metric comparison. Install with: pip install mlflow")
    
    # Query baseline metrics
    baseline_agg = query_mlflow_metrics(
        experiment_name=experiment_name,
        metric_names=metrics,
        filter_string=baseline_filter,
        aggregation=["mean"],
        max_results=max_results,
    )
    
    # Query comparison metrics
    comparison_agg = query_mlflow_metrics(
        experiment_name=experiment_name,
        metric_names=metrics,
        filter_string=comparison_filter,
        aggregation=["mean"],
        max_results=max_results,
    )
    
    # Calculate deltas
    results = {}
    for metric in metrics:
        baseline_mean = baseline_agg.get(metric, {}).get("mean")
        comparison_mean = comparison_agg.get(metric, {}).get("mean")
        
        if baseline_mean is None or comparison_mean is None:
            results[metric] = {
                "baseline_mean": baseline_mean,
                "comparison_mean": comparison_mean,
                "delta": None,
                "delta_percent": None,
            }
            continue
        
        delta = comparison_mean - baseline_mean
        delta_percent = (delta / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        results[metric] = {
            "baseline_mean": baseline_mean,
            "comparison_mean": comparison_mean,
            "delta": delta,
            "delta_percent": delta_percent,
        }
    
    return results
