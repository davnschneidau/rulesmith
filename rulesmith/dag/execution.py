"""Core execution engine with resolution policy."""

import time
from typing import Any, Dict, List, Optional, Union

from rulesmith.dag.nodes import Node
from rulesmith.dag.scheduler import topological_sort
from rulesmith.io.decision_result import DecisionResult, FiredRule
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
        return_decision_result: bool = True,
        enable_memoization: bool = False,
    ) -> Union[DecisionResult, Dict[str, Any]]:
        """
        Execute the rulebook DAG.

        Args:
            payload: Initial input payload
            context: Execution context (RunContext or MLflowRunContext)
            nodes: Optional node instances (if None, uses registered nodes)
            return_decision_result: If True, return DecisionResult; if False, return Dict (legacy)

        Returns:
            DecisionResult with execution trace and metadata
        """
        from rulesmith.performance.memoization import memoization_cache
        
        node_instances = nodes or self.nodes

        # Check memoization cache
        if enable_memoization:
            cached_result = memoization_cache.get(
                rulebook_version=self.spec.version,
                inputs=payload,
            )
            if cached_result is not None:
                return cached_result

        # Build execution order
        edges = [(edge.source, edge.target) for edge in self.spec.edges]
        node_names = [node.name for node in self.spec.nodes]
        execution_order = topological_sort(node_names, edges)

        # Initialize state with payload
        state = payload.copy()

        # Track execution metadata for DecisionResult
        fired_rules: List[FiredRule] = []
        skipped_nodes: List[str] = []
        metrics: Dict[str, float] = {}
        costs: Dict[str, float] = {}
        warnings: List[str] = []
        total_start_time = time.time()

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
                
                # Apply auto-mapping if enabled
                from rulesmith.dx.auto_mapping import auto_mapper
                if hasattr(node, "_rule_func") and node._rule_func:
                    # Try to auto-map inputs for rule nodes
                    try:
                        from rulesmith.dx.typing import type_validator
                        # Validate and map inputs
                        mapped_state = type_validator.validate_inputs(node._rule_func, state)
                        node_outputs = node.execute(mapped_state, context)
                    except Exception:
                        # Fall back to normal execution if mapping fails
                        node_outputs = node.execute(state, context)
                else:
                    node_outputs = node.execute(state, context)
                
                execution_time_ms = (time.time() - start_time) * 1000

                # Extract metrics from node outputs (guardrails, LLM metrics, etc.)
                from rulesmith.dag.metrics_extractor import (
                    extract_llm_metrics_from_output,
                    extract_metrics_from_node_output,
                )
                from rulesmith.runtime.mlflow_hooks import log_node_execution_to_mlflow
                
                # Extract guardrail metrics
                extract_metrics_from_node_output(node_name, node_outputs, metrics, state)
                
                # Extract LLM-specific metrics
                if node.kind == "llm":
                    extract_llm_metrics_from_output(node_name, node_outputs, metrics, costs)
                
                # Log to MLflow if enabled
                log_node_execution_to_mlflow(
                    node_name, node.kind, execution_time_ms, node_outputs, context
                )

                # Track fired rules for rule nodes
                if node.kind == "rule":
                    try:
                        rule_id = node_name
                        rule_name = node_name
                        salience = 0
                        
                        # Try to get rule spec if available
                        if hasattr(node, "rule_func") and hasattr(node.rule_func, "_rule_spec"):
                            rule_spec = node.rule_func._rule_spec
                            rule_id = rule_spec.id or node_name
                            rule_name = rule_spec.name or node_name
                            salience = getattr(rule_spec, "salience", 0)
                        
                        # Generate reason (simple explanation)
                        reason = f"Rule '{rule_name}' executed successfully"
                        if node_outputs:
                            output_keys = list(node_outputs.keys())
                            if output_keys:
                                reason = f"Rule '{rule_name}' produced outputs: {', '.join(output_keys)}"
                        
                        fired_rule = FiredRule(
                            id=rule_id,
                            name=rule_name,
                            salience=salience,
                            inputs=state.copy(),  # Snapshot of state at rule execution
                            outputs=node_outputs.copy(),
                            reason=reason,
                            duration_ms=execution_time_ms,
                        )
                        fired_rules.append(fired_rule)
                    except Exception as e:
                        # If tracking fails, continue but add warning
                        warnings.append(f"Failed to track rule {node_name}: {str(e)}")

                # LLM metrics extraction is now handled in metrics_extractor module above

                # Track metrics
                metrics[f"{node_name}_duration_ms"] = execution_time_ms
                if node_outputs and isinstance(node_outputs, dict):
                    # Extract any numeric metrics from outputs
                    for key, value in node_outputs.items():
                        if isinstance(value, (int, float)) and not key.startswith("_"):
                            metrics[f"{node_name}_{key}"] = float(value)

                # Set execution metadata if context supports it
                if node_ctx:
                    # Try to get code hash from rule nodes
                    if node.kind == "rule" and hasattr(node, "rule_func"):
                        if hasattr(node.rule_func, "_rule_spec"):
                            rule_spec = node.rule_func._rule_spec
                            if rule_spec.code_hash:
                                node_ctx.set_code_hash(rule_spec.code_hash)

                    # Try to get model URI from Model/LLM nodes
                    if node.kind in ("model", "byom", "llm") and hasattr(node, "model_uri"):
                        if node.model_uri:
                            node_ctx.set_model_uri(node.model_uri)

                    # Finish node execution tracking
                    execution_time = time.time() - start_time
                    node_ctx.finish(node_outputs, metrics={"execution_time_seconds": execution_time})

                # Call after_node hooks
                hook_registry.after_node(node_name, state, context, node_outputs)

            except Exception as e:
                # Enhanced error handling with context
                from rulesmith.dx.errors import error_handler
                
                rule_name = None
                if hasattr(node, "rule_func") and hasattr(node.rule_func, "_rule_spec"):
                    rule_name = node.rule_func._rule_spec.name
                
                error_info = error_handler.handle_error(
                    error=e,
                    node_name=node_name,
                    rule_name=rule_name,
                    input_data=state,
                )
                
                # Log error with context
                error_msg = str(error_info)
                state[f"_error_{node_name}"] = error_msg
                warnings.append(f"Error in {node_name}: {error_msg}")

                # Call on_error hooks
                hook_registry.on_error(node_name, state, context, e)

                # Log error in MLflow if supported
                if node_ctx:
                    try:
                        context.end_node_execution(node_ctx, error=error_msg)
                    except Exception:
                        try:
                            node_ctx.finish({}, metrics={"error": 1.0})
                        except Exception:
                            pass

                # Continue execution (fail gracefully) - don't raise
                continue

            # Simply merge outputs into state (new values override old ones)
            # This is the default behavior - simple and predictable
            state.update(node_outputs)
            
            # Track that this node executed (for debugging)
            state[f"_executed_{node_name}"] = True

        # Build DecisionResult
        total_duration_ms = (time.time() - total_start_time) * 1000
        metrics["total_duration_ms"] = total_duration_ms
        
        # Extract primary value (default to state, but can be customized)
        # For now, we'll use the entire state as value, but remove internal tracking fields
        value = {
            k: v for k, v in state.items()
            if not k.startswith("_") or k in ["_executed_", "_ab_selection", "_fork_selection"]
        }
        
        # Get trace URI from context if available
        trace_uri = None
        if hasattr(context, "get_trace_uri"):
            try:
                trace_uri = context.get_trace_uri()
            except Exception:
                pass
        elif hasattr(context, "run_id"):
            # MLflow run ID
            try:
                import mlflow
                if hasattr(mlflow, "active_run") and mlflow.active_run():
                    trace_uri = mlflow.active_run().info.run_id
            except Exception:
                pass
        
        # Identify skipped nodes (nodes in spec but not in execution order due to conditional routing)
        all_node_names = set(node.name for node in self.spec.nodes)
        executed_node_names = set(execution_order)
        skipped_nodes.extend(sorted(all_node_names - executed_node_names))
        
        if return_decision_result:
            decision_result = DecisionResult(
                value=value,
                version=self.spec.version,
                trace_uri=trace_uri,
                fired=fired_rules,
                skipped=skipped_nodes,
                metrics=metrics,
                costs=costs,
                warnings=warnings,
            )
            
            # Cache result if memoization enabled
            if enable_memoization:
                try:
                    memoization_cache.set(
                        rulebook_version=self.spec.version,
                        inputs=payload,
                        result=decision_result,
                    )
                except Exception:
                    # If caching fails, continue without it
                    pass
            
            return decision_result
        else:
            # Legacy mode: return state dict
            return state

    # Simplified: Removed _map_inputs and _merge_outputs
    # - All nodes receive the full state dictionary (simple)
    # - Node outputs simply update the state dictionary (predictable)
    # - Field mapping can be handled at the rule level if needed
    # This makes the system much easier to understand and debug

