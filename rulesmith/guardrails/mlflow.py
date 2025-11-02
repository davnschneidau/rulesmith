"""MLflow integration for guardrails."""

from typing import Any, Dict, List

import mlflow

from rulesmith.guardrails.execution import GuardResult


def log_guard_results(guard_results: List[GuardResult], node_name: str) -> None:
    """
    Log guard evaluation results to MLflow.

    Args:
        guard_results: List of GuardResult objects
        node_name: Node name where guards were evaluated
    """
    try:
        # Log individual guard results
        for result in guard_results:
            mlflow.log_metric(
                f"guard_{node_name}_{result.guard_name}_passed",
                1.0 if result.passed else 0.0,
            )

            if result.score is not None:
                mlflow.log_metric(
                    f"guard_{node_name}_{result.guard_name}_score",
                    result.score,
                )

            # Log tags
            mlflow.set_tag(
                f"rulesmith.guard_{node_name}_{result.guard_name}",
                "passed" if result.passed else "failed",
            )
            mlflow.set_tag(
                f"rulesmith.guard_{node_name}_{result.guard_name}_action",
                result.action.value,
            )

        # Log summary metrics
        total_guards = len(guard_results)
        passed_guards = sum(1 for r in guard_results if r.passed)
        failed_guards = total_guards - passed_guards

        mlflow.log_metric(f"guard_{node_name}_total", float(total_guards))
        mlflow.log_metric(f"guard_{node_name}_passed", float(passed_guards))
        mlflow.log_metric(f"guard_{node_name}_failed", float(failed_guards))

        if total_guards > 0:
            mlflow.log_metric(
                f"guard_{node_name}_pass_rate",
                passed_guards / total_guards,
            )

        # Log guard results as artifact
        results_dict = [r.to_dict() for r in guard_results]
        mlflow.log_dict(results_dict, f"guard_results_{node_name}.json")

    except Exception:
        pass  # Ignore MLflow errors


def log_guard_policy(policy_name: str, checks: List[str], node_name: str) -> None:
    """
    Log guard policy configuration to MLflow.

    Args:
        policy_name: Policy name
        checks: List of guard check names
        node_name: Node name
    """
    try:
        mlflow.set_tag(f"rulesmith.guard_policy_{node_name}", policy_name)
        mlflow.log_param(f"guard_policy_{node_name}_checks", len(checks))

        for i, check in enumerate(checks):
            mlflow.set_tag(f"rulesmith.guard_{node_name}_check_{i}", check)

    except Exception:
        pass  # Ignore MLflow errors

