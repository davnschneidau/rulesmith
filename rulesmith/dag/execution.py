"""Core execution engine with resolution policy."""

import time
from typing import Any, Dict, List, Optional

from rulesmith.dag.nodes import Node
from rulesmith.dag.scheduler import topological_sort
from rulesmith.io.ser import Edge, RulebookSpec
from rulesmith.runtime.context import RunContext
from rulesmith.runtime.hooks import hook_registry


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

        # Check context capabilities (MLflow integration is optional and handled gracefully)
        supports_node_tracking = hasattr(context, "start_node_execution")

        # Execute nodes in topological order
        for node_name in execution_order:
            if node_name not in node_instances:
                raise ValueError(f"Node '{node_name}' not found in node instances")

            node = node_instances[node_name]

            # All nodes receive the full state - simple and predictable
            # Field mapping is handled at connection time if needed, but by default
            # all fields are available to all nodes (simplifies common use cases)

            # Start node execution tracking if supported
            node_ctx = None
            if supports_node_tracking:
                node_ctx = context.start_node_execution(node_name, node.kind, inputs=state)

            # Call before_node hooks
            hook_registry.before_node(node_name, state, context)

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
                        # Log to MLflow if available (gracefully handles if MLflow not enabled)
                        try:
                            log_guard_policy(guard_policy.name, guard_policy.checks, node_name)
                        except Exception:
                            pass  # MLflow not available - that's okay

                        # Apply guard policy
                        node_outputs = guard_executor.apply_policy(
                            guard_policy,
                            inputs=state,
                            outputs=node_outputs,
                        )

                        # Log guard results to MLflow if available
                        guard_results = node_outputs.get("_guard_results", [])
                        if guard_results:
                            try:
                                from rulesmith.guardrails.execution import GuardResult
                                results = [
                                    GuardResult(**r) if isinstance(r, dict) else r
                                    for r in guard_results
                                ]
                                log_guard_results(results, node_name)
                                # Call hooks for each result
                                for result in results:
                                    hook_registry.on_guard(node_name, state, context, result)
                            except Exception:
                                pass  # MLflow not available - that's okay

                        # Stop execution if guard blocked
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

                # Call after_node hooks
                hook_registry.after_node(node_name, state, context, node_outputs)

            except Exception as e:
                # Error handling - could be enhanced with retries
                state[f"_error_{node_name}"] = str(e)

                # Call on_error hooks
                hook_registry.on_error(node_name, state, context, e)

                # Log error in MLflow if supported
                if node_ctx:
                    try:
                        node_ctx.finish({}, metrics={"error": 1.0})
                    except Exception:
                        pass

                raise

            # Simply merge outputs into state (new values override old ones)
            # This is the default behavior - simple and predictable
            state.update(node_outputs)
            
            # Track that this node executed (for debugging)
            state[f"_executed_{node_name}"] = True

        return state

    # Simplified: Removed _map_inputs and _merge_outputs
    # - All nodes receive the full state dictionary (simple)
    # - Node outputs simply update the state dictionary (predictable)
    # - Field mapping can be handled at the rule level if needed
    # This makes the system much easier to understand and debug

