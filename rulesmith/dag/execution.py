"""Core execution engine with resolution policy."""

import time
from typing import Any, Dict, List, Optional

from rulesmith.dag.nodes import Node
from rulesmith.dag.scheduler import topological_sort
from rulesmith.io.ser import Edge, RulebookSpec
from rulesmith.runtime.context import RunContext


class ExecutionEngine:
    """Core execution engine for rulebooks."""

    def __init__(self, spec: RulebookSpec):
        self.spec = spec
        self.nodes: Dict[str, Node] = {}

    def register_node(self, name: str, node: Node) -> None:
        """Register a node instance."""
        self.nodes[name] = node

    def execute(
        self,
        payload: Dict[str, Any],
        context: Any,
        nodes: Optional[Dict[str, Node]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the rulebook DAG.

        Args:
            payload: Initial input payload
            context: Execution context (RunContext or MLflowRunContext)
            nodes: Optional node instances (if None, uses registered nodes)

        Returns:
            Final state after execution
        """
        node_instances = nodes or self.nodes

        # Build execution order
        edges = [(edge.source, edge.target) for edge in self.spec.edges]
        node_names = [node.name for node in self.spec.nodes]
        execution_order = topological_sort(node_names, edges)

        # Initialize state with payload
        state = payload.copy()

        # Check if context supports MLflow tracking
        supports_mlflow = hasattr(context, "enable_mlflow") and context.enable_mlflow
        supports_node_tracking = hasattr(context, "start_node_execution")

        # Execute nodes in topological order
        for node_name in execution_order:
            if node_name not in node_instances:
                raise ValueError(f"Node '{node_name}' not found in node instances")

            node = node_instances[node_name]

            # Apply field mapping from upstream nodes
            mapped_inputs = self._map_inputs(node_name, state, execution_order)

            # Merge mapped inputs into state (for node input tracking)
            node_inputs = state.copy()
            node_inputs.update(mapped_inputs)

            # Start node execution tracking if supported
            node_ctx = None
            if supports_node_tracking:
                node_ctx = context.start_node_execution(node_name, node.kind, inputs=node_inputs)

            # Merge mapped inputs into state
            state.update(mapped_inputs)

            # Execute node
            try:
                start_time = time.time()
                node_outputs = node.execute(state, context)

                # Apply guardrails if attached
                if hasattr(node, "_guard_policies") and node._guard_policies:
                    from rulesmith.guardrails.execution import guard_executor
                    from rulesmith.guardrails.mlflow import log_guard_results, log_guard_policy

                    # Evaluate all guard policies
                    for guard_policy in node._guard_policies:
                        # Log policy configuration
                        if supports_mlflow:
                            log_guard_policy(
                                guard_policy.name,
                                guard_policy.checks,
                                node_name,
                            )

                        # Apply guard policy
                        node_outputs = guard_executor.apply_policy(
                            guard_policy,
                            inputs=node_inputs,
                            outputs=node_outputs,
                        )

                        # Log guard results to MLflow
                        if supports_mlflow:
                            # Get guard results from output
                            guard_results = node_outputs.get("_guard_results", [])
                            if guard_results:
                                from rulesmith.guardrails.execution import GuardResult

                                results = [
                                    GuardResult(**r) if isinstance(r, dict) else r
                                    for r in guard_results
                                ]
                                log_guard_results(results, node_name)

                        # Check if blocked
                        if node_outputs.get("_guard_blocked"):
                            raise ValueError(
                                f"Guard blocked execution: {node_outputs.get('_guard_message', 'Unknown reason')}"
                            )

                # Set execution metadata if context supports it
                if node_ctx:
                    # Try to get code hash from rule nodes
                    if node.kind == "rule" and hasattr(node, "rule_func"):
                        if hasattr(node.rule_func, "_rule_spec"):
                            rule_spec = node.rule_func._rule_spec
                            if rule_spec.code_hash:
                                node_ctx.set_code_hash(rule_spec.code_hash)

                    # Try to get model URI from BYOM/GenAI nodes
                    if node.kind in ("byom", "llm") and hasattr(node, "model_uri"):
                        if node.model_uri:
                            node_ctx.set_model_uri(node.model_uri)

                    # Finish node execution tracking
                    execution_time = time.time() - start_time
                    node_ctx.finish(node_outputs, metrics={"execution_time_seconds": execution_time})

            except Exception as e:
                # Error handling - could be enhanced with retries
                state[f"_error_{node_name}"] = str(e)

                # Log error in MLflow if supported
                if node_ctx:
                    try:
                        node_ctx.finish({}, metrics={"error": 1.0})
                    except Exception:
                        pass

                raise

            # Merge outputs with resolution policy
            state = self._merge_outputs(state, node_outputs, node_name)

        return state

    def _map_inputs(
        self,
        node_name: str,
        state: Dict[str, Any],
        execution_order: List[str],
    ) -> Dict[str, Any]:
        """
        Map inputs from upstream nodes based on edge mappings.

        Args:
            node_name: Target node name
            state: Current state
            execution_order: Execution order list

        Returns:
            Mapped inputs dictionary
        """
        mapped = {}

        # Find all edges targeting this node
        for edge in self.spec.edges:
            if edge.target == node_name:
                source_outputs = state

                # Apply field mapping
                if edge.mapping:
                    for target_field, source_field in edge.mapping.items():
                        if source_field in source_outputs:
                            mapped[target_field] = source_outputs[source_field]
                else:
                    # If no explicit mapping, copy all fields from source
                    mapped.update(source_outputs)

        return mapped

    def _merge_outputs(
        self,
        state: Dict[str, Any],
        node_outputs: Dict[str, Any],
        node_name: str,
    ) -> Dict[str, Any]:
        """
        Merge node outputs into state with resolution policy.

        Resolution policy priority:
        1. Human decisions (HITL)
        2. Guard overrides
        3. Upstream values (existing state)
        4. New node outputs

        Args:
            state: Current state
            node_outputs: Outputs from the node
            node_name: Node name for tracking

        Returns:
            Merged state
        """
        merged = state.copy()

        # Check for human override
        if "_human_override" in state:
            human_overrides = state.get("_human_override", {})
            if node_name in human_overrides:
                merged.update(human_overrides[node_name])
                return merged

        # Check for guard override
        if "_guard_override" in state:
            guard_overrides = state.get("_guard_override", {})
            if node_name in guard_overrides:
                merged.update(guard_overrides[node_name])
                return merged

        # Merge node outputs (they take precedence over upstream for same fields)
        merged.update(node_outputs)

        # Track node execution
        merged[f"_node_{node_name}"] = True

        return merged

