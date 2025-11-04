"""MLflow integration hooks for execution engine."""

from typing import Any, Dict, Optional


def log_node_execution_to_mlflow(
    node_name: str,
    node_kind: str,
    execution_time_ms: float,
    node_outputs: Dict[str, Any],
    context: Any,
) -> None:
    """
    Log node execution to MLflow if context supports it.
    
    Args:
        node_name: Name of the node
        node_kind: Kind of node (rule, model, llm, etc.)
        execution_time_ms: Execution time in milliseconds
        node_outputs: Output dictionary from node
        context: Execution context
    """
    # Check if context supports MLflow logging
    if not hasattr(context, "enable_mlflow") or not context.enable_mlflow:
        return
    
    try:
        import mlflow
        
        # Log execution time
        mlflow.log_metric(f"{node_name}_duration_ms", execution_time_ms)
        
        # Log node-specific metrics
        if node_kind == "llm" and isinstance(node_outputs, dict):
            if "tokens" in node_outputs and isinstance(node_outputs["tokens"], dict):
                tokens = node_outputs["tokens"]
                if "total_tokens" in tokens:
                    mlflow.log_metric(f"{node_name}_tokens", float(tokens["total_tokens"]))
        
        # Log guardrail metrics if present
        if "_metrics" in node_outputs:
            metrics_data = node_outputs["_metrics"]
            for metric_name, metric_data in metrics_data.items():
                if isinstance(metric_data, dict):
                    metric_value = metric_data.get("value", True)
                    if isinstance(metric_value, bool):
                        metric_value = 1.0 if metric_value else 0.0
                    elif metric_value is None:
                        metric_value = 0.0
                    else:
                        metric_value = float(metric_value)
                    mlflow.log_metric(f"{node_name}_{metric_name}", metric_value)
    except Exception:
        # Silently fail if MLflow is not available
        pass


def should_log_to_mlflow(context: Any, is_error: bool = False) -> bool:
    """
    Determine if execution should be logged to MLflow.
    
    Args:
        context: Execution context
        is_error: Whether this is an error case (always log errors)
    
    Returns:
        True if should log, False otherwise
    """
    if not hasattr(context, "enable_mlflow") or not context.enable_mlflow:
        return False
    
    # Always log errors
    if is_error:
        return True
    
    # Check if there's a logger with sampling configured
    if hasattr(context, "mlflow_logger"):
        logger = context.mlflow_logger
        if hasattr(logger, "should_log"):
            return logger.should_log()
    
    return True

