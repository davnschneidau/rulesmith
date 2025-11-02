"""MLflow integration for HITL tracking."""

from typing import Any, Dict, Optional

import mlflow

from rulesmith.hitl.base import ReviewDecision, ReviewRequest


def log_hitl_request(request: ReviewRequest, node_name: str) -> None:
    """
    Log HITL request to MLflow.

    Args:
        request: ReviewRequest instance
        node_name: Node name
    """
    try:
        mlflow.set_tag("rulesmith.hitl_node", node_name)
        mlflow.set_tag("rulesmith.hitl_request_id", request.id)
        mlflow.log_param(f"hitl_{node_name}_request_id", request.id)

        # Log payload size
        payload_size = len(str(request.payload))
        mlflow.log_metric(f"hitl_{node_name}_payload_size", float(payload_size))

        # Log if expired
        if request.expires_at:
            mlflow.set_tag("rulesmith.hitl_expires_at", request.expires_at.isoformat())

    except Exception:
        pass  # Ignore MLflow errors


def log_hitl_decision(decision: ReviewDecision, node_name: str, request_id: str) -> None:
    """
    Log HITL decision to MLflow.

    Args:
        decision: ReviewDecision instance
        node_name: Node name
        request_id: Request ID
    """
    try:
        mlflow.log_metric(f"hitl_{node_name}_approved", 1.0 if decision.approved else 0.0)
        mlflow.set_tag("rulesmith.hitl_approved", str(decision.approved))

        if decision.reviewer:
            mlflow.set_tag("rulesmith.hitl_reviewer", decision.reviewer)

        if decision.comment:
            mlflow.log_text(decision.comment, f"hitl_{node_name}_comment.txt")

        # Log if edited output was provided
        if decision.edited_output:
            mlflow.log_dict(decision.edited_output, f"hitl_{node_name}_edited_output.json")
            mlflow.log_metric(f"hitl_{node_name}_edited", 1.0)

        # Log review time if available
        mlflow.log_param(f"hitl_{node_name}_request_id", request_id)

    except Exception:
        pass  # Ignore MLflow errors


def log_hitl_metrics(
    node_name: str,
    total_requests: int,
    approved: int,
    rejected: int,
    pending: int,
    avg_review_time: Optional[float] = None,
) -> None:
    """
    Log aggregate HITL metrics.

    Args:
        node_name: Node name
        total_requests: Total number of requests
        approved: Number of approved requests
        rejected: Number of rejected requests
        pending: Number of pending requests
        avg_review_time: Average review time in seconds
    """
    try:
        mlflow.log_metric(f"hitl_{node_name}_total_requests", float(total_requests))
        mlflow.log_metric(f"hitl_{node_name}_approved", float(approved))
        mlflow.log_metric(f"hitl_{node_name}_rejected", float(rejected))
        mlflow.log_metric(f"hitl_{node_name}_pending", float(pending))

        if total_requests > 0:
            mlflow.log_metric(f"hitl_{node_name}_approval_rate", approved / total_requests)

        if avg_review_time is not None:
            mlflow.log_metric(f"hitl_{node_name}_avg_review_time_seconds", avg_review_time)

    except Exception:
        pass  # Ignore MLflow errors

