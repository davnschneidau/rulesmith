"""MLflow-aware runtime context with automatic run management."""

import time
from typing import Any, Dict, Optional

import mlflow
from mlflow.entities import RunStatus

from rulesmith.io.mlflow_io import end_run, start_run
from rulesmith.io.ser import RulebookSpec
from rulesmith.runtime.context import RunContext
from rulesmith.runtime.tracing import log_genai_metrics, log_node_execution, start_genai_trace, start_node_run


class MLflowRunContext(RunContext):
    """RunContext enhanced with MLflow integration."""

    def __init__(
        self,
        rulebook_spec: Optional[RulebookSpec] = None,
        run_id: Optional[str] = None,
        identity: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        enable_mlflow: bool = True,
        parent_run_id: Optional[str] = None,
    ):
        """
        Initialize MLflow-aware context.

        Args:
            rulebook_spec: Rulebook specification for lineage
            run_id: Optional run ID (auto-generated if None)
            identity: Identity for deterministic hashing
            tags: Optional tags
            seed: Optional random seed
            params: Optional parameters
            enable_mlflow: Whether to enable MLflow logging
            parent_run_id: Optional parent MLflow run ID
        """
        super().__init__(run_id, identity, tags, seed, params)
        self.rulebook_spec = rulebook_spec
        self.enable_mlflow = enable_mlflow
        self.parent_run_id = parent_run_id
        self._mlflow_run = None
        self._node_runs: Dict[str, mlflow.ActiveRun] = {}

        if self.enable_mlflow and self.rulebook_spec:
            try:
                self._start_mlflow_run()
            except Exception:
                # If MLflow is not available, disable it
                self.enable_mlflow = False

    def _start_mlflow_run(self) -> None:
        """Start MLflow run for this rulebook execution."""
        try:
            self._mlflow_run = start_run(
                self.rulebook_spec,
                params=self.params,
                tags=self.tags,
            )

            # Log rulebook spec as artifact
            from rulesmith.io.mlflow_io import log_rulebook, log_lineage

            log_rulebook(self.rulebook_spec)
            log_lineage(self.rulebook_spec)

            # Set additional context tags
            mlflow.set_tag("rulesmith.run_id", self.run_id)
            if self.identity:
                mlflow.set_tag("rulesmith.identity", self.identity)
            if self.seed:
                mlflow.set_tag("rulesmith.seed", str(self.seed))

        except Exception as e:
            # If MLflow fails, continue without it
            self.enable_mlflow = False
            self._mlflow_run = None

    def start_node_execution(
        self,
        node_name: str,
        node_kind: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> "NodeExecutionContext":
        """
        Start execution context for a node.

        Args:
            node_name: Node name
            node_kind: Node kind
            inputs: Optional input data

        Returns:
            NodeExecutionContext for tracking execution
        """
        return NodeExecutionContext(
            parent_context=self,
            node_name=node_name,
            node_kind=node_kind,
            inputs=inputs,
        )

    def end_mlflow_run(
        self,
        status: RunStatus = RunStatus.FINISHED,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """End the MLflow run."""
        if self._mlflow_run and self.enable_mlflow:
            try:
                end_run(status=status, metrics=metrics)
            except Exception:
                pass  # Ignore errors on cleanup

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = RunStatus.FAILED if exc_type else RunStatus.FINISHED
        self.end_mlflow_run(status=status)


class NodeExecutionContext:
    """Context for tracking a single node execution with MLflow."""

    def __init__(
        self,
        parent_context: MLflowRunContext,
        node_name: str,
        node_kind: str,
        inputs: Optional[Dict[str, Any]] = None,
    ):
        self.parent_context = parent_context
        self.node_name = node_name
        self.node_kind = node_kind
        self.inputs = inputs or {}
        self.start_time = time.time()
        self._mlflow_run = None
        self.outputs: Optional[Dict[str, Any]] = None
        self.execution_time: Optional[float] = None
        self.code_hash: Optional[str] = None
        self.model_uri: Optional[str] = None
        self.ab_bucket: Optional[str] = None

        if parent_context.enable_mlflow:
            self._start_node_run()

    def _start_node_run(self) -> None:
        """Start nested MLflow run for this node."""
        try:
            self._mlflow_run = start_node_run(
                self.node_name,
                self.node_kind,
                parent_run_id=self.parent_context.parent_run_id,
            )
        except Exception:
            self._mlflow_run = None

    def set_code_hash(self, code_hash: str) -> None:
        """Set code hash for rule nodes."""
        self.code_hash = code_hash

    def set_model_uri(self, model_uri: str) -> None:
        """Set model URI for BYOM/GenAI nodes."""
        self.model_uri = model_uri

    def set_ab_bucket(self, bucket: str) -> None:
        """Set A/B bucket for fork nodes."""
        self.ab_bucket = bucket

    def finish(self, outputs: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Finish node execution and log to MLflow.

        Args:
            outputs: Node output dictionary
            metrics: Optional metrics to log
        """
        self.outputs = outputs
        self.execution_time = time.time() - self.start_time

        if self.parent_context.enable_mlflow:
            try:
                log_node_execution(
                    node_name=self.node_name,
                    node_kind=self.node_kind,
                    execution_time=self.execution_time,
                    inputs=self.inputs,
                    outputs=outputs,
                    metrics=metrics,
                    code_hash=self.code_hash,
                    model_uri=self.model_uri,
                    ab_bucket=self.ab_bucket,
                )

                if self._mlflow_run:
                    mlflow.end_run()
            except Exception:
                pass  # Ignore MLflow errors

    def finish_genai(
        self,
        outputs: Dict[str, Any],
        tokens: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        latency: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> None:
        """
        Finish GenAI node execution with token/cost tracking.

        Args:
            outputs: Node output dictionary
            tokens: Token counts
            cost: Cost in USD
            latency: Latency in seconds
            provider: Provider name
        """
        self.outputs = outputs
        self.execution_time = time.time() - self.start_time

        if self.parent_context.enable_mlflow:
            try:
                # Start trace if available
                trace = start_genai_trace(
                    self.node_name,
                    model_uri=self.model_uri,
                    provider=provider,
                )

                # Log metrics
                log_genai_metrics(
                    tokens=tokens,
                    cost=cost,
                    latency=latency or self.execution_time,
                    provider=provider,
                )

                # Log standard execution
                log_node_execution(
                    node_name=self.node_name,
                    node_kind=self.node_kind,
                    execution_time=self.execution_time,
                    inputs=self.inputs,
                    outputs=outputs,
                    code_hash=self.code_hash,
                    model_uri=self.model_uri,
                )

                if self._mlflow_run:
                    mlflow.end_run()
            except Exception:
                pass  # Ignore MLflow errors

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        if exc_type and self.parent_context.enable_mlflow:
            try:
                if self._mlflow_run:
                    mlflow.end_run(status=RunStatus.FAILED)
            except Exception:
                pass

