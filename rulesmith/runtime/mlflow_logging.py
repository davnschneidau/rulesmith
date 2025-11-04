"""Comprehensive MLflow logging for rulebook executions.

This module implements detailed MLflow logging following best practices:
- Parent/child run structure for replays and simulations
- Comprehensive tags for reproducibility and comparability
- Metrics at decision and aggregate levels
- Artifacts for auditability and explainability
- Sampling for production volume control
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union

import mlflow
from rulesmith.io.decision_result import DecisionResult, FiredRule


class MLflowLogger:
    """Comprehensive MLflow logger for rulebook executions."""

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        parent_run_name: Optional[str] = None,
        sample_rate: float = 1.0,  # 1.0 = log all, 0.01 = log 1%
        enable_artifacts: bool = True,
        redact_pii: bool = True,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: MLflow experiment name (defaults to active experiment)
            parent_run_name: Name for parent run (if None, uses child runs only)
            sample_rate: Sampling rate for production (0.0-1.0)
            enable_artifacts: Whether to log artifacts
            redact_pii: Whether to redact PII from artifacts
        """
        self.experiment_name = experiment_name
        self.parent_run_name = parent_run_name
        self.sample_rate = sample_rate
        self.enable_artifacts = enable_artifacts
        self.redact_pii = redact_pii
        self._parent_run = None
        self._child_runs = []

    def start_parent_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> mlflow.entities.Run:
        """Start a parent run for batch operations."""
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        run_name = run_name or self.parent_run_name or "rulebook_batch"
        tags = tags or {}

        self._parent_run = mlflow.start_run(run_name=run_name, tags=tags)
        return self._parent_run

    def end_parent_run(self):
        """End the parent run."""
        if self._parent_run:
            mlflow.end_run()
            self._parent_run = None

    def should_log(self, trace_id: Optional[str] = None, is_error: bool = False) -> bool:
        """
        Determine if a decision should be logged based on sampling.

        Args:
            trace_id: Optional trace ID for deterministic sampling
            is_error: If True, always log (don't sample errors)

        Returns:
            True if should log, False otherwise
        """
        if is_error:
            return True  # Always log errors

        if self.sample_rate >= 1.0:
            return True  # Log all

        if self.sample_rate <= 0.0:
            return False  # Log nothing

        # Deterministic sampling based on trace_id
        if trace_id:
            hash_val = int(hashlib.sha256(trace_id.encode()).hexdigest(), 16)
            return (hash_val % 10000) < (self.sample_rate * 10000)

        # Random sampling if no trace_id
        import random
        return random.random() < self.sample_rate

    def log_decision(
        self,
        decision: DecisionResult,
        context: Dict[str, Any],
        run_name: Optional[str] = None,
        nested: bool = True,
    ) -> Optional[str]:
        """
        Log a single decision to MLflow.

        Args:
            decision: DecisionResult from rulebook execution
            context: Execution context with tags/metadata
            run_name: Optional run name (defaults to case_id or trace_id)
            nested: If True, create nested run under parent

        Returns:
            Run ID if logged, None if skipped due to sampling
        """
        # Check if we should log this decision
        trace_id = context.get("trace_id") or context.get("run_id")
        is_error = decision.metrics.get("error", 0) > 0 or len(decision.warnings) > 0

        if not self.should_log(trace_id, is_error):
            return None

        # Start nested run if parent exists, otherwise regular run
        if nested and self._parent_run:
            mlflow.start_run(run_name=run_name or f"case_{trace_id or 'unknown'}", nested=True)
        elif not nested:
            mlflow.start_run(run_name=run_name or f"case_{trace_id or 'unknown'}")

        try:
            # Set comprehensive tags
            tags = self._build_tags(context, decision)
            mlflow.set_tags(tags)

            # Log metrics
            metrics = self._build_metrics(decision)
            mlflow.log_metrics(metrics)

            # Log artifacts if enabled
            if self.enable_artifacts:
                self._log_artifacts(decision, context)

            return mlflow.active_run().info.run_id

        finally:
            mlflow.end_run()

    def _build_tags(self, context: Dict[str, Any], decision: DecisionResult) -> Dict[str, str]:
        """Build comprehensive tags for the run."""
        tags = {
            "rulebook": context.get("rulebook_name", "unknown"),
            "rulebook_version": decision.version or context.get("rulebook_version", "unknown"),
            "stage": context.get("stage", context.get("env", "unknown")),
            "alias": context.get("alias", "unknown"),  # champion/challenger
            "source": context.get("source", "unknown"),  # http/stream/spark
            "tenant": context.get("tenant", "default"),
            "run_kind": context.get("run_kind", "decision"),
            "trace_id": context.get("trace_id", context.get("run_id", "unknown")),
            "engine": context.get("engine", "dag"),
        }

        # Optional tags
        if context.get("input_schema_hash"):
            tags["input_schema_hash"] = context["input_schema_hash"]
        if context.get("git_commit"):
            tags["git_commit"] = context["git_commit"]
        if context.get("guard_policy_version"):
            tags["guard_policy_version"] = context["guard_policy_version"]

        return tags

    def _build_metrics(self, decision: DecisionResult) -> Dict[str, Union[int, float]]:
        """Build metrics dictionary from DecisionResult."""
        metrics = {
            "latency_ms": decision.metrics.get("total_duration_ms", 0.0),
            "rules_fired": len(decision.fired),
            "error": 1 if decision.metrics.get("error", 0) > 0 or decision.warnings else 0,
        }

        # LLM metrics
        if "llm_tokens_in" in decision.metrics:
            metrics["llm_tokens_in"] = int(decision.metrics["llm_tokens_in"])
        if "llm_tokens_out" in decision.metrics:
            metrics["llm_tokens_out"] = int(decision.metrics["llm_tokens_out"])
        if "llm_tokens" in decision.metrics:
            tokens = decision.metrics["llm_tokens"]
            if isinstance(tokens, dict):
                metrics["llm_tokens_in"] = int(tokens.get("prompt_tokens", tokens.get("input_tokens", 0)))
                metrics["llm_tokens_out"] = int(tokens.get("completion_tokens", tokens.get("output_tokens", 0)))

        # Cost metrics
        if decision.costs:
            total_cost = sum(decision.costs.values())
            metrics["cost_usd"] = float(total_cost)
        elif "cost_usd" in decision.metrics:
            metrics["cost_usd"] = float(decision.metrics["cost_usd"])

        # Guardrail metrics (from _metrics in node outputs)
        guard_violations = 0
        guard_metrics = {}
        for key, value in decision.metrics.items():
            if key.startswith("_metrics_"):
                # Extract metric name and check if it failed
                metric_name = key.replace("_metrics_", "")
                if isinstance(value, dict) and not value.get("value", True):
                    guard_violations += 1
                    guard_metrics[f"guard_{metric_name}_fail"] = 1
                else:
                    guard_metrics[f"guard_{metric_name}_fail"] = 0

        metrics["guard_violations"] = guard_violations
        metrics.update(guard_metrics)

        # Action count (if available)
        if "action_count" in decision.metrics:
            metrics["action_count"] = int(decision.metrics["action_count"])

        # Rule-level counts
        metrics["rule_fire_count"] = len(decision.fired)
        metrics["rule_skip_count"] = len(decision.skipped)
        metrics["rule_error_count"] = len([w for w in decision.warnings if "Error" in w])

        # Business metrics (if labels available)
        if "label" in decision.metrics:
            metrics["label"] = int(decision.metrics["label"])
        if "tp" in decision.metrics:
            metrics["tp"] = int(decision.metrics["tp"])
        if "fp" in decision.metrics:
            metrics["fp"] = int(decision.metrics["fp"])
        if "tn" in decision.metrics:
            metrics["tn"] = int(decision.metrics["tn"])
        if "fn" in decision.metrics:
            metrics["fn"] = int(decision.metrics["fn"])
        if "score" in decision.metrics:
            metrics["score"] = float(decision.metrics["score"])

        return metrics

    def _log_artifacts(self, decision: DecisionResult, context: Dict[str, Any]):
        """Log artifacts for auditability and explainability."""
        # 1. Decision result (compact, redacted)
        decision_dict = self._decision_to_dict(decision)
        mlflow.log_text(
            json.dumps(decision_dict, indent=2),
            "artifacts/decision_result.json"
        )

        # 2. Explanation (why.json)
        explain_dict = self._build_explanation(decision)
        mlflow.log_text(
            json.dumps(explain_dict, indent=2),
            "artifacts/explain/why.json"
        )

        # 3. Rules breakdown
        rules_dict = self._build_rules_breakdown(decision)
        mlflow.log_text(
            json.dumps(rules_dict, indent=2),
            "artifacts/rules/fired_breakdown.json"
        )

        # 4. Guardrails report (from metrics)
        guard_report = self._build_guardrails_report(decision)
        if guard_report:
            mlflow.log_text(
                json.dumps(guard_report, indent=2),
                "artifacts/guardrails/report.json"
            )

        # 5. Redacted inputs (if available)
        if "redacted_inputs" in context:
            mlflow.log_text(
                json.dumps(context["redacted_inputs"], indent=2),
                "artifacts/inputs/redacted.json"
            )

    def _decision_to_dict(self, decision: DecisionResult) -> Dict[str, Any]:
        """Convert DecisionResult to compact dictionary."""
        # Extract value without PII if redaction enabled
        value = decision.value
        if self.redact_pii:
            value = self._redact_dict(value)

        return {
            "value": value,
            "fired": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "duration_ms": rule.duration_ms,
                }
                for rule in decision.fired
            ],
            "skipped": decision.skipped,
            "metrics": {
                k: v for k, v in decision.metrics.items()
                if not k.startswith("_")  # Exclude internal metrics
            },
            "costs": decision.costs,
            "warnings": decision.warnings,
        }

    def _build_explanation(self, decision: DecisionResult) -> Dict[str, Any]:
        """Build human-friendly explanation."""
        summary_parts = []
        proof = []

        for rule in decision.fired:
            summary_parts.append(f"{rule.name} (reason: {rule.reason})")
            proof.append({
                "rule": rule.id,
                "name": rule.name,
                "inputs": rule.inputs,
                "outputs": rule.outputs,
                "reason": rule.reason,
            })

        return {
            "summary": "; ".join(summary_parts) if summary_parts else "No rules fired",
            "proof": proof,
            "skipped": decision.skipped,
        }

    def _build_rules_breakdown(self, decision: DecisionResult) -> Dict[str, Any]:
        """Build rules breakdown artifact."""
        return {
            "fired": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "salience": rule.salience,
                    "duration_ms": rule.duration_ms,
                }
                for rule in decision.fired
            ],
            "skipped": [
                {"id": node_id, "reason": "not_executed"}
                for node_id in decision.skipped
            ],
            "errors": [
                {"message": warning}
                for warning in decision.warnings
                if "Error" in warning
            ],
        }

    def _build_guardrails_report(self, decision: DecisionResult) -> Optional[Dict[str, Any]]:
        """Build guardrails report from metrics."""
        # Extract metrics from node outputs
        guard_metrics = {}
        violations = []

        for key, value in decision.metrics.items():
            if key.startswith("_metrics_"):
                metric_name = key.replace("_metrics_", "")
                if isinstance(value, dict):
                    guard_metrics[metric_name] = {
                        "value": value.get("value", True),
                        "message": value.get("message"),
                    }
                    if not value.get("value", True):
                        violations.append(metric_name)

        if not guard_metrics:
            return None

        return {
            "checks": guard_metrics,
            "violations": violations,
            "violation_count": len(violations),
            "action": "block" if violations else "allow",
        }

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from dictionary."""
        redacted = {}
        for key, value in data.items():
            if isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            elif isinstance(value, str):
                # Simple email redaction
                if "@" in value and "." in value:
                    redacted[key] = "[EMAIL_REDACTED]"
                else:
                    redacted[key] = value
            else:
                redacted[key] = value
        return redacted

    def log_aggregate_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        artifact: Optional[Dict[str, Any]] = None,
    ):
        """
        Log aggregate metrics to parent run.

        Args:
            metrics: Dictionary of aggregate metrics (e.g., latency_p50_ms, cost_usd_sum)
            artifact: Optional artifact to log (e.g., compare/summary.json)
        """
        if not self._parent_run:
            raise ValueError("No parent run active. Call start_parent_run() first.")

        mlflow.log_metrics(metrics)

        if artifact and self.enable_artifacts:
            artifact_path = artifact.get("path", "artifacts/aggregate.json")
            mlflow.log_text(
                json.dumps(artifact.get("data", {}), indent=2),
                artifact_path
            )


def log_decision_to_mlflow(
    decision: DecisionResult,
    context: Dict[str, Any],
    logger: Optional[MLflowLogger] = None,
    **kwargs,
) -> Optional[str]:
    """
    Convenience function to log a decision to MLflow.

    Args:
        decision: DecisionResult to log
        context: Execution context
        logger: Optional MLflowLogger instance (creates one if None)
        **kwargs: Additional arguments for MLflowLogger

    Returns:
        Run ID if logged, None otherwise
    """
    if logger is None:
        logger = MLflowLogger(**kwargs)

    return logger.log_decision(decision, context)

