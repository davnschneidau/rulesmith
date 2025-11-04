"""Extract and process metrics from node outputs."""

from typing import Any, Dict


def extract_metrics_from_node_output(
    node_name: str,
    node_outputs: Dict[str, Any],
    metrics_dict: Dict[str, float],
    state: Dict[str, Any],
) -> None:
    """
    Extract metrics from node outputs and add to metrics dictionary.
    
    Metrics are stored in node_outputs["_metrics"] and are extracted
    for DecisionResult and MLflow logging.
    
    Args:
        node_name: Name of the node
        node_outputs: Output dictionary from node execution
        metrics_dict: Dictionary to add extracted metrics to
        state: State dictionary to store full metric data
    """
    if "_metrics" not in node_outputs:
        return
    
    metrics_data = node_outputs["_metrics"]
    for metric_name, metric_data in metrics_data.items():
        # Store full metric data for artifacts
        state[f"_metrics_{node_name}_{metric_name}"] = metric_data
        
        # Extract metric value for metrics dict
        if isinstance(metric_data, dict):
            metric_value = metric_data.get("value", True)
            # Convert bool to numeric for MLflow
            if isinstance(metric_value, bool):
                metric_value = 1.0 if metric_value else 0.0
            elif metric_value is None:
                metric_value = 0.0
            else:
                metric_value = float(metric_value)
            
            # Add to metrics dict with guard prefix
            metrics_dict[f"guard_{node_name}_{metric_name}"] = metric_value
            
            # Track violations
            if not metric_data.get("value", True):
                if "guard_violations" not in metrics_dict:
                    metrics_dict["guard_violations"] = 0
                metrics_dict["guard_violations"] += 1
        else:
            # Simple value
            metric_value = float(bool(metric_data)) if metric_data is not None else 0.0
            metrics_dict[f"guard_{node_name}_{metric_name}"] = metric_value


def extract_llm_metrics_from_output(
    node_name: str,
    node_outputs: Dict[str, Any],
    metrics_dict: Dict[str, float],
    costs_dict: Dict[str, float],
) -> None:
    """
    Extract LLM-specific metrics (tokens, costs) from node outputs.
    
    Args:
        node_name: Name of the LLM node
        node_outputs: Output dictionary from node execution
        metrics_dict: Dictionary to add metrics to
        costs_dict: Dictionary to add costs to
    """
    if not isinstance(node_outputs, dict):
        return
    
    # Extract token information
    if "tokens" in node_outputs and isinstance(node_outputs["tokens"], dict):
        tokens_dict = node_outputs["tokens"]
        total_tokens = tokens_dict.get("total_tokens", 0)
        prompt_tokens = tokens_dict.get("prompt_tokens", tokens_dict.get("input_tokens", 0))
        completion_tokens = tokens_dict.get("completion_tokens", tokens_dict.get("output_tokens", 0))
        
        if total_tokens > 0:
            metrics_dict["llm_tokens"] = float(total_tokens)
        if prompt_tokens > 0:
            metrics_dict["llm_tokens_in"] = float(prompt_tokens)
        if completion_tokens > 0:
            metrics_dict["llm_tokens_out"] = float(completion_tokens)
    
    # Extract cost information
    if "cost" in node_outputs:
        cost_val = float(node_outputs["cost"])
        costs_dict[f"{node_name}_cost"] = cost_val
        costs_dict["usd"] = costs_dict.get("usd", 0.0) + cost_val
    elif "cost_usd" in node_outputs:
        cost_val = float(node_outputs["cost_usd"])
        costs_dict[f"{node_name}_cost"] = cost_val
        costs_dict["usd"] = costs_dict.get("usd", 0.0) + cost_val

