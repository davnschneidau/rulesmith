"""MLflow integration for A/B testing metrics and policy snapshots."""

from typing import Any, Dict, List, Optional

import mlflow

from rulesmith.ab.policies import TrafficPolicy
from rulesmith.io.ser import ABArm


def log_ab_policy_snapshot(
    fork_name: str,
    arms: List[ABArm],
    policy: str,
    arms_history: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Log A/B policy snapshot to MLflow.

    Args:
        fork_name: Fork node name
        arms: List of A/B arms
        policy: Policy name
        arms_history: Optional arms history for bandit policies

    Returns:
        Artifact path
    """
    snapshot = {
        "fork_name": fork_name,
        "policy": policy,
        "arms": [
            {
                "node": arm.node,
                "weight": arm.weight,
            }
            for arm in arms
        ],
    }

    if arms_history:
        snapshot["arms_history"] = arms_history

    mlflow.log_dict(snapshot, f"ab_policy_{fork_name}.json")
    return f"ab_policy_{fork_name}.json"


def log_ab_metrics(
    fork_name: str,
    arm_name: str,
    metrics: Dict[str, float],
) -> None:
    """
    Log A/B testing metrics for a specific arm.

    Args:
        fork_name: Fork node name
        arm_name: Arm name
        metrics: Dictionary of metrics (e.g., {"conversion_rate": 0.15, "revenue": 100.0})
    """
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"ab_{fork_name}_{arm_name}_{metric_name}", value)


def log_ab_outcome(
    fork_name: str,
    arm_name: str,
    success: bool,
    reward: Optional[float] = None,
) -> None:
    """
    Log A/B test outcome for bandit policy updates.

    Args:
        fork_name: Fork node name
        arm_name: Arm name
        success: Whether the outcome was successful
        reward: Optional reward value for bandit policies
    """
    mlflow.log_metric(f"ab_{fork_name}_{arm_name}_success", 1.0 if success else 0.0)

    if reward is not None:
        mlflow.log_metric(f"ab_{fork_name}_{arm_name}_reward", reward)


def update_bandit_history(
    policy: TrafficPolicy,
    arm_name: str,
    success: Optional[bool] = None,
    reward: Optional[float] = None,
) -> None:
    """
    Update bandit policy history after observing outcome.

    Args:
        policy: TrafficPolicy instance
        arm_name: Arm name
        success: Success indicator (for Thompson Sampling)
        reward: Reward value (for UCB, Epsilon-Greedy)
    """
    if isinstance(policy, (ThompsonSamplingPolicy,)) and success is not None:
        policy.update_history(arm_name, success)
    elif isinstance(policy, (UCBPolicy, EpsilonGreedyPolicy,)) and reward is not None:
        policy.update_history(arm_name, reward)

