"""Rulebook builder and DAG construction."""

from typing import Any, Callable, Dict, List, Optional

from rulesmith.dag.execution import ExecutionEngine
from rulesmith.dag.nodes import (
    LLMNode,
    ModelNode,
    Node,
    RuleNode,
)
from rulesmith.dag.registry import rule_registry
from rulesmith.io.ser import ABArm, Edge, NodeSpec, RulebookSpec


class Rulebook:
    """Rulebook builder and executor."""

    def __init__(self, name: str, version: str = "1.0.0", metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.version = version
        self.metadata = metadata or {}
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []

    def add_rule(
        self,
        rule_func: Callable,
        as_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a rule node to the rulebook.

        Args:
            rule_func: Rule function (must be decorated with @rule)
            as_name: Optional node name (defaults to rule name)
            params: Optional parameters to override defaults

        Returns:
            Self for chaining
        """
        # Get rule spec
        if hasattr(rule_func, "_rule_spec"):
            spec = rule_func._rule_spec
        else:
            # Try to get from registry by function name
            spec = rule_registry.get_spec(rule_func.__name__)
            if spec is None:
                raise ValueError(f"Rule function '{rule_func.__name__}' not registered")

        node_name = as_name or spec.name

        # Merge params
        final_params = spec.params.copy()
        if params:
            final_params.update(params)

        node = RuleNode(node_name, rule_func, params=final_params)
        self._nodes[node_name] = node

        return self

    def add_split(
        self,
        name: str,
        variants: Dict[str, float],
        policy: Optional[str] = "hash",
        policy_instance: Optional[Any] = None,
    ) -> "Rulebook":
        """
        Add an A/B test split node. This is the simple, intuitive way to do A/B testing.
        
        Args:
            name: Split node name (e.g., "experiment_1")
            variants: Dictionary of variant names to their weights/percentages
                     (e.g., {"control": 0.5, "treatment": 0.5})
            policy: Traffic splitting policy - simple names:
                   - "hash" (default): Deterministic, same user gets same variant
                   - "random": Random allocation
                   - "thompson": Thompson Sampling bandit (adaptive)
                   - "ucb": Upper Confidence Bound bandit
                   - "epsilon": Epsilon-greedy bandit
            policy_instance: Advanced: Custom policy instance (rarely needed)
        
        Examples:
            # Simple 50/50 split
            rb.add_split("experiment", {"control": 0.5, "treatment": 0.5})
            
            # 70/30 split
            rb.add_split("test", {"variant_a": 0.7, "variant_b": 0.3})
            
            # Three-way test
            rb.add_split("multi_test", {"a": 0.33, "b": 0.33, "c": 0.34})
            
            # With adaptive bandit
            rb.add_split("bandit_test", {"a": 1.0, "b": 1.0}, policy="thompson")
        
        Returns:
            Self for chaining
        """
        # Note: Fork functionality is now handled via fork() function in execution
        # This method is kept for API compatibility but fork routing happens at execution time
        # Store fork configuration in metadata for execution engine to use
        from rulesmith.io.ser import ABArm
        
        arms = [ABArm(node=variant_name, weight=weight) for variant_name, weight in variants.items()]
        
        # Store fork configuration in metadata - execution engine will use fork() function
        self.metadata[f"_fork_{name}"] = {
            "arms": [{"node": arm.node, "weight": arm.weight} for arm in arms],
            "policy": policy,
            "policy_instance": policy_instance,
        }
        
        # Create a placeholder node that will be handled by fork() function during execution
        # We'll use a simple rule node as placeholder
        from rulesmith.dag.nodes import RuleNode
        
        def fork_placeholder(state, context=None):
            # This won't actually execute - fork() function handles routing
            from rulesmith.dag.functions import fork
            return fork(arms, policy=policy, policy_instance=policy_instance, context=context)
        
        # Mark as fork kind
        placeholder = RuleNode(name, fork_placeholder)
        placeholder.kind = "fork"  # Override kind for execution engine
        self._nodes[name] = placeholder
        return self
    


    
    def add_model(
        self,
        name: str,
        model_uri: Optional[str] = None,
        langchain_model: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a model node (MLflow or LangChain model).

        Args:
            name: Node name
            model_uri: Optional MLflow model URI
            langchain_model: Optional direct LangChain model/chain instance
            params: Optional parameters

        Returns:
            Self for chaining
            
        Examples:
            # MLflow model
            rb.add_model("my_model", model_uri="models:/my_model/1")
            
            # LangChain model directly
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4")
            rb.add_model("llm_node", langchain_model=llm)
        """
        node = ModelNode(name, model_uri=model_uri, langchain_model=langchain_model, params=params)
        self._nodes[name] = node
        return self

    def add_llm(
        self,
        name: str,
        model_uri: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        gateway_uri: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[Any]] = None,
    ) -> "Rulebook":
        """
        Add an LLM node.

        Args:
            name: Node name
            model_uri: Optional MLflow model URI
            provider: Optional provider name (openai, anthropic, etc.)
            model_name: Optional model name
            gateway_uri: Optional MLflow AI Gateway URI
            params: Optional parameters
            metrics: Optional list of rule functions to evaluate on output (guardrails)

        Returns:
            Self for chaining
            
        Examples:
            # Basic LLM node
            rb.add_llm("gpt4", provider="openai", model_name="gpt-4")
            
            # With custom metrics (guardrails)
            @rule(name="check_toxicity", inputs=["output"], outputs=["is_toxic"])
            def check_toxicity(output: str) -> dict:
                toxic_words = ["hate", "violence"]
                return {"is_toxic": any(word in output.lower() for word in toxic_words)}
            
            rb.add_llm("gpt4", provider="openai", model_name="gpt-4", metrics=[check_toxicity])
        """
        node = LLMNode(
            name,
            model_uri=model_uri,
            provider=provider,
            model_name=model_name,
            gateway_uri=gateway_uri,
            params=params,
            metrics=metrics,
        )
        self._nodes[name] = node
        return self

    def add_hitl(
        self,
        name: str,
        queue: Any,
        timeout: Optional[float] = None,
        async_mode: bool = False,
        active_learning_threshold: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a Human-in-the-Loop node.

        Args:
            name: Node name
            queue: HITL queue instance (HITLQueue)
            timeout: Optional timeout in seconds
            async_mode: If True, don't block execution
            active_learning_threshold: Optional confidence threshold for active learning
            params: Optional parameters

        Returns:
            Self for chaining
        """
        node = HITLNode(
            name,
            queue,
            timeout=timeout,
            async_mode=async_mode,
            active_learning_threshold=active_learning_threshold,
            params=params,
        )
        self._nodes[name] = node
        return self


    def connect(
        self,
        source: str,
        target: str,
        mapping: Optional[Dict[str, str]] = None,
        auto_map: bool = True,
    ) -> "Rulebook":
        """
        Connect two nodes. By default, all fields flow through automatically.
        
        Args:
            source: Source node name
            target: Target node name
            mapping: Optional field mapping (rarely needed - only if you want to rename fields)
            auto_map: If True, automatically infer missing mappings
        
        Examples:
            # Simple connection - all fields flow through
            rb.connect("node1", "node2")
            
            # With field mapping (only if you need to rename)
            rb.connect("node1", "node2", mapping={"new_name": "old_name"})
            
            # Auto-map will infer mappings based on rule inputs/outputs
            rb.connect("node1", "node2", auto_map=True)
        
        Returns:
            Self for chaining
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")

        # Auto-map if enabled and no explicit mapping provided
        if auto_map and mapping is None:
            try:
                from rulesmith.dx.auto_mapping import auto_mapper
                
                # Get source node outputs (from rule spec if available)
                source_node = self._nodes[source]
                source_outputs = {}
                if hasattr(source_node, "rule_func") and hasattr(source_node.rule_func, "_rule_spec"):
                    source_outputs = {out: None for out in source_node.rule_func._rule_spec.outputs}
                
                # Get target node spec (for rule nodes)
                target_node = self._nodes[target]
                if hasattr(target_node, "rule_func") and hasattr(target_node.rule_func, "_rule_spec"):
                    target_spec = target_node.rule_func._rule_spec
                    # Infer mapping
                    mapping = auto_mapper.infer_mapping(
                        source_outputs,
                        target_spec.inputs,
                        existing_mapping=mapping,
                    )
            except Exception:
                # If auto-mapping fails, use empty mapping (all fields flow through)
                mapping = None

        edge = Edge(source=source, target=target, mapping=mapping)
        self._edges.append(edge)
        return self

    def add_metrics(
        self,
        node_name: str,
        metrics: List[Callable],
    ) -> "Rulebook":
        """
        Add metrics (guardrails) to a node. Metrics are rule functions that evaluate the node's output.

        Args:
            node_name: Node name (must be LLM or Model node)
            metrics: List of rule functions to evaluate on output

        Returns:
            Self for chaining
            
        Examples:
            @rule(name="check_pii", inputs=["output"], outputs=["has_pii"])
            def check_pii(output: str) -> dict:
                return {"has_pii": "@" in output}
            
            rb.add_llm("gpt4", provider="openai", model_name="gpt-4")
            rb.add_metrics("gpt4", [check_pii])
        """
        if node_name not in self._nodes:
            raise ValueError(f"Node '{node_name}' not found")

        node = self._nodes[node_name]

        # Only LLM and Model nodes support metrics
        if node.kind not in ("llm", "model"):
            raise ValueError(f"Metrics can only be added to LLM or Model nodes, not {node.kind}")

        # Add metrics to node
        if not hasattr(node, "metrics"):
            node.metrics = []
        node.metrics.extend(metrics)

        return self
    
    def set_metric_threshold(
        self,
        metric_name: str,
        threshold: float,
        operator: str = "<=",
        alert_action: Optional[str] = None,
        alert_hook: Optional[Callable] = None,
    ) -> "Rulebook":
        """
        Set a threshold for a metric.
        
        Args:
            metric_name: Name of the metric
            threshold: Threshold value
            operator: Comparison operator (<=, >=, ==, <, >)
            alert_action: Action on breach ("log", "route_to_hitl", "hook")
            alert_hook: Optional callable for custom actions
        
        Returns:
            Self for chaining
            
        Examples:
            rb.set_metric_threshold("toxicity_score", threshold=0.5, operator="<=")
            rb.set_metric_threshold("latency_ms", threshold=100, operator="<=")
        """
        from rulesmith.metrics.core import get_metric_registry
        
        registry = get_metric_registry()
        registry.set_threshold(
            metric_name,
            threshold,
            operator=operator,
            alert_action=alert_action,
            alert_hook=alert_hook,
        )
        
        return self

    def to_spec(self) -> RulebookSpec:
        """Serialize rulebook to RulebookSpec."""
        nodes = []
        for name, node in self._nodes.items():
            node_spec = NodeSpec(
                name=name,
                kind=node.kind,
                params=getattr(node, "params", {}),
            )

            # Fill in node-specific fields
            if isinstance(node, RuleNode):
                # Get rule spec
                if hasattr(node.rule_func, "_rule_spec"):
                    rule_spec = node.rule_func._rule_spec
                    node_spec.rule_ref = rule_spec.name
            elif isinstance(node, ModelNode):
                node_spec.model_uri = node.model_uri
                # Note: langchain_model cannot be serialized, must be provided at runtime
                if node.langchain_model:
                    node_spec.params["_has_langchain_model"] = True
            elif isinstance(node, LLMNode):
                node_spec.model_uri = node.model_uri
                if hasattr(node, "provider") and node.provider:
                    node_spec.params["provider"] = node.provider
                if hasattr(node, "model_name") and node.model_name:
                    node_spec.params["model_name"] = node.model_name
                if hasattr(node, "gateway_uri") and node.gateway_uri:
                    node_spec.params["gateway_uri"] = node.gateway_uri
                # Store all params for provider-specific config
                if node.params:
                    node_spec.params.update(node.params)
            # LangChain and LangGraph models are now handled by ModelNode
            # No need for separate node type checks

            nodes.append(node_spec)

        return RulebookSpec(
            name=self.name,
            version=self.version,
            nodes=nodes,
            edges=self._edges.copy(),
            metadata=self.metadata,
        )

    def run(
        self,
        payload: Dict[str, Any],
        context: Optional[Any] = None,
        enable_mlflow: bool = True,
        return_decision_result: bool = True,
        mlflow_logger: Optional[Any] = None,
            debug: bool = False,
    ):
        """
        Execute the rulebook with a payload.

        Args:
            payload: Input payload dictionary
            context: Optional execution context (RunContext or MLflowRunContext). 
                    If None, automatically created based on enable_mlflow flag.
            enable_mlflow: Whether to enable MLflow logging (default: True)
            return_decision_result: If True, return DecisionResult; if False, return Dict (legacy, default: True)
            mlflow_logger: Optional MLflowLogger instance. If None and enable_mlflow=True, 
                          automatically creates one with sensible defaults.
            debug: If True, include debug information in result (default: False)

        Returns:
            DecisionResult (default) or Dict (if return_decision_result=False)
        
        Examples:
            # Simple execution
            result = rb.run({"age": 30, "income": 50000})
            
            # With debug mode
            result = rb.run({"age": 30, "income": 50000}, debug=True)
            # Access debug info: result.debug_info
        """
        from rulesmith.io.decision_result import DecisionResult
        
        spec = self.to_spec()
        engine = ExecutionEngine(spec)
        # Register all nodes
        for name, node in self._nodes.items():
            engine.register_node(name, node)

        # Create context automatically if not provided
        # For most users, MLflow integration "just works" - no need to understand contexts
        if context is None:
            if enable_mlflow:
                try:
                    from rulesmith.mlflow import MLflowRunContext
                    context = MLflowRunContext(rulebook_spec=spec, enable_mlflow=True)
                except Exception:
                    # MLflow not available, fall back to basic context
                    from rulesmith.runtime.context import RunContext
                    context = RunContext()
                    enable_mlflow = False  # Disable MLflow if not available
            else:
                from rulesmith.runtime.context import RunContext
                context = RunContext()
        
        # Auto-create MLflow logger if not provided but MLflow is enabled
        if enable_mlflow and mlflow_logger is None:
            try:
                from rulesmith.mlflow import MLflowLogger
                # Create logger with sensible defaults
                experiment_name = f"rulesmith/{self.name}"
                mlflow_logger = MLflowLogger(
                    experiment_name=experiment_name,
                    sample_rate=0.01,  # Sample 1% in production by default
                    enable_artifacts=True,
                    redact_pii=True,
                )
            except Exception:
                # MLflow not available, disable logging
                mlflow_logger = None
                enable_mlflow = False

        # Execute - context manager handles setup/teardown automatically
        if hasattr(context, "__enter__"):
            with context:
                result = engine.execute(
                    payload,
                    context,
                    nodes=self._nodes,
                    return_decision_result=return_decision_result,
                    enable_memoization=True,  # Enable memoization by default
                )
        else:
            result = engine.execute(
                payload,
                context,
                nodes=self._nodes,
                return_decision_result=return_decision_result,
                enable_memoization=True,  # Enable memoization by default
            )
        
        # Log to MLflow using comprehensive logger if provided
        if enable_mlflow and mlflow_logger and return_decision_result and isinstance(result, DecisionResult):
            try:
                # Build context dict for logging
                log_context = {
                    "rulebook_name": self.name,
                    "rulebook_version": self.version,
                    "trace_id": getattr(context, "run_id", None) or getattr(context, "trace_id", None),
                    "stage": getattr(context, "stage", None) or getattr(context, "env", "unknown"),
                    "tenant": getattr(context, "tenant_id", None) or getattr(context, "tenant", "default"),
                    "source": getattr(context, "source", "unknown"),
                    "run_kind": getattr(context, "run_kind", "decision"),
                    "engine": "dag",
                    "alias": getattr(context, "alias", "unknown"),
                }
                
                # Add optional context fields
                if hasattr(context, "input_schema_hash"):
                    log_context["input_schema_hash"] = context.input_schema_hash
                if hasattr(context, "git_commit"):
                    log_context["git_commit"] = context.git_commit
                if hasattr(context, "guard_policy_version"):
                    log_context["guard_policy_version"] = context.guard_policy_version
                
                # Add redacted inputs if available
                if hasattr(context, "redacted_inputs"):
                    log_context["redacted_inputs"] = context.redacted_inputs
                
                # Log decision
                mlflow_logger.log_decision(result, log_context)
            except Exception as e:
                # Don't fail execution if MLflow logging fails
                import warnings
                warnings.warn(f"MLflow logging failed: {str(e)}")
        
        # Return DecisionResult or Dict based on flag
        if return_decision_result:
            return result
        else:
            # Legacy mode: return the value dict
            if isinstance(result, DecisionResult):
                return result.value
            return result

