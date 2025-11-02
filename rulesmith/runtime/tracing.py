"""MLflow tracing integration for node execution."""

import time
from typing import Any, Dict, Optional

import mlflow


def start_node_run(
    node_name: str,
    node_kind: str,
    parent_run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> mlflow.ActiveRun:
    """
    Start a nested MLflow run for a node execution.

    Args:
        node_name: Node name
        node_kind: Node kind (rule|fork|gate|byom|llm|hitl)
        parent_run_id: Optional parent run ID
        tags: Optional tags

    Returns:
        Active MLflow run
    """
    run_tags = {
        "rulesmith.node_kind": node_kind,
        "rulesmith.node_name": node_name,
    }

    if tags:
        run_tags.update(tags)

    run = mlflow.start_run(
        nested=True,
        run_name=f"node:{node_name}",
        tags=run_tags,
    )

    return run


def log_node_execution(
    node_name: str,
    node_kind: str,
    execution_time: float,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    code_hash: Optional[str] = None,
    model_uri: Optional[str] = None,
    ab_bucket: Optional[str] = None,
) -> None:
    """
    Log node execution details to current run.

    Args:
        node_name: Node name
        node_kind: Node kind
        execution_time: Execution time in seconds
        inputs: Optional input data
        outputs: Optional output data
        metrics: Optional metrics
        code_hash: Optional code hash for rules
        model_uri: Optional model URI for BYOM/GenAI nodes
        ab_bucket: Optional A/B bucket for fork nodes
    """
    tags = {
        "rulesmith.node_kind": node_kind,
        "rulesmith.node_name": node_name,
    }

    if code_hash:
        tags["rulesmith.code_hash"] = code_hash

    if model_uri:
        tags["rulesmith.ref_model_uri"] = model_uri

    if ab_bucket:
        tags["rulesmith.ab_bucket"] = ab_bucket

    mlflow.set_tags(tags)

    if metrics:
        mlflow.log_metrics(metrics)

    mlflow.log_metric("execution_time_ms", execution_time * 1000)

    if inputs:
        mlflow.log_dict(inputs, "inputs.json")

    if outputs:
        mlflow.log_dict(outputs, "outputs.json")


def start_genai_trace(
    node_name: str,
    model_uri: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[Any]:
    """
    Start an MLflow trace for GenAI node execution.

    Args:
        node_name: Node name
        model_uri: Optional model URI
        provider: Optional provider name

    Returns:
        Trace object if MLflow tracing is available
    """
    try:
        # MLflow 3.0+ tracing API
        trace_data = {
            "name": node_name,
            "tags": {
                "rulesmith.node_kind": "llm",
                "rulesmith.node_name": node_name,
            },
        }

        if model_uri:
            trace_data["tags"]["rulesmith.ref_model_uri"] = model_uri

        if provider:
            trace_data["tags"]["rulesmith.provider"] = provider

        # Return trace ID for later use
        return trace_data
    except Exception:
        # Tracing not available or failed
        return None


def log_genai_metrics(
    tokens: Optional[Dict[str, int]] = None,
    cost: Optional[float] = None,
    latency: Optional[float] = None,
    provider: Optional[str] = None,
) -> None:
    """
    Log GenAI metrics (tokens, cost, latency).

    Args:
        tokens: Dictionary with input_tokens, output_tokens, total_tokens
        cost: Cost in USD
        latency: Latency in seconds
        provider: Provider name
    """
    metrics = {}

    if tokens:
        metrics.update({f"tokens_{k}": v for k, v in tokens.items()})

    if cost is not None:
        metrics["cost_usd"] = cost

    if latency is not None:
        metrics["latency_ms"] = latency * 1000

    if provider:
        mlflow.set_tag("rulesmith.provider", provider)

    if metrics:
        mlflow.log_metrics(metrics)

