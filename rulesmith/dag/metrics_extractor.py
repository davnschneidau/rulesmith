"""Extract metrics from node outputs for logging."""

from typing import Any, Dict

from rulesmith.metrics.core import MetricRegistry, get_metric_registry


def extract_metrics_from_state(
    state: Dict[str, Any],
    node_name: str,
    metrics_dict: Dict[str, float],
) -> None:
    """
    Extract metrics from execution state.
    
    Metrics are now just regular outputs in state - no special "_metrics" dict.
    Look for numeric values that might be metrics and register them.
    
    Args:
        state: Execution state dictionary
        node_name: Name of the node that produced these outputs
        metrics_dict: Dictionary to add extracted metrics to
    """
    registry = get_metric_registry()
    
    # Look for numeric values in state that might be metrics
    # Skip internal fields (starting with _)
    for key, value in state.items():
        if key.startswith("_"):
            continue
        
        # If it's a numeric value, it might be a metric
        if isinstance(value, (int, float)):
            metrics_dict[f"{node_name}_{key}"] = float(value)
            
            # Register in metric registry with category="guardrail" if from guardrail node
            # For now, we'll use "operational" as default - can be refined later
            registry.record(
                name=f"{node_name}_{key}",
                value=float(value),
                category="guardrail" if key.endswith("_score") or "toxicity" in key.lower() or "pii" in key.lower() else "operational",
                node_name=node_name,
            )


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
    
    registry = get_metric_registry()
    
    # Extract token information
    if "tokens" in node_outputs and isinstance(node_outputs["tokens"], dict):
        tokens_dict = node_outputs["tokens"]
        total_tokens = tokens_dict.get("total_tokens", 0)
        prompt_tokens = tokens_dict.get("prompt_tokens", tokens_dict.get("input_tokens", 0))
        completion_tokens = tokens_dict.get("completion_tokens", tokens_dict.get("output_tokens", 0))
        
        if total_tokens > 0:
            metrics_dict["llm_tokens"] = float(total_tokens)
            registry.record("llm_tokens", float(total_tokens), category="operational", node_name=node_name)
        if prompt_tokens > 0:
            metrics_dict["llm_tokens_in"] = float(prompt_tokens)
            registry.record("llm_tokens_in", float(prompt_tokens), category="operational", node_name=node_name)
        if completion_tokens > 0:
            metrics_dict["llm_tokens_out"] = float(completion_tokens)
            registry.record("llm_tokens_out", float(completion_tokens), category="operational", node_name=node_name)
    
    # Extract cost information
    if "cost" in node_outputs:
        cost_val = float(node_outputs["cost"])
        costs_dict[f"{node_name}_cost"] = cost_val
        costs_dict["usd"] = costs_dict.get("usd", 0.0) + cost_val
        registry.record("cost_usd", cost_val, category="business", unit="USD", node_name=node_name)
    elif "cost_usd" in node_outputs:
        cost_val = float(node_outputs["cost_usd"])
        costs_dict[f"{node_name}_cost"] = cost_val
        costs_dict["usd"] = costs_dict.get("usd", 0.0) + cost_val
        registry.record("cost_usd", cost_val, category="business", unit="USD", node_name=node_name)
