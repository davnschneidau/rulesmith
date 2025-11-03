"""Rulebook builder and DAG construction."""

from typing import Any, Callable, Dict, List, Optional

from rulesmith.dag.execution import ExecutionEngine
from rulesmith.dag.langchain_node import LangChainNode
from rulesmith.dag.langgraph_node import LangGraphNode
from rulesmith.dag.nodes import (
    BYOMNode,  # Deprecated alias
    ForkNode,  # Deprecated - use fork() function
    GateNode,  # Deprecated - use gate() function
    GenAINode,  # Deprecated alias
    LLMNode,
    ModelNode,
    Node,
    RuleNode,
)
from rulesmith.hitl.node import HITLNode
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
        # Convert simple dict to ABArm list for backward compatibility
        from rulesmith.io.ser import ABArm
        
        arms = [ABArm(node=variant_name, weight=weight) for variant_name, weight in variants.items()]
        
        node = ForkNode(
            name,
            arms,
            policy=policy,
            policy_instance=policy_instance,
            track_metrics=True,
        )
        self._nodes[name] = node
        return self
    
    def add_fork(
        self,
        name: str,
        arms: List[ABArm],
        policy: Optional[str] = "hash",
        policy_instance: Optional[Any] = None,
        track_metrics: bool = True,
    ) -> "Rulebook":
        """
        Add a fork node for A/B testing (advanced API).
        
        Note: For most users, use add_split() instead - it's simpler!
        
        Args:
            name: Fork node name
            arms: List of A/B arms (ABArm objects)
            policy: Traffic splitting policy
            policy_instance: Optional TrafficPolicy instance
            track_metrics: Whether to track metrics
        
        Returns:
            Self for chaining
        """
        node = ForkNode(
            name,
            arms,
            policy=policy,
            policy_instance=policy_instance,
            track_metrics=track_metrics,
        )
        self._nodes[name] = node
        return self

    def add_gate(self, name: str, condition: str) -> "Rulebook":
        """
        Add a gate node for conditional routing.

        Args:
            name: Gate node name
            condition: Expression to evaluate (e.g., "age >= 18")

        Returns:
            Self for chaining
        """
        node = GateNode(name, condition)
        self._nodes[name] = node
        return self

    def add_byom(
        self,
        name: str,
        model_uri: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a BYOM (Bring Your Own Model) node.
        
        DEPRECATED: Use add_model() instead.

        Args:
            name: Node name
            model_uri: MLflow model URI
            params: Optional parameters

        Returns:
            Self for chaining
        """
        import warnings
        warnings.warn(
            "add_byom() is deprecated. Use add_model() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        node = ModelNode(name, model_uri=model_uri, params=params)
        self._nodes[name] = node
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

    def add_genai(
        self,
        name: str,
        model_uri: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        gateway_uri: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a GenAI/LLM node.
        
        DEPRECATED: Use add_llm() instead.

        Args:
            name: Node name
            model_uri: Optional MLflow model URI
            provider: Optional provider name (openai, anthropic, etc.)
            model_name: Optional model name
            gateway_uri: Optional MLflow AI Gateway URI
            params: Optional parameters

        Returns:
            Self for chaining
        """
        import warnings
        warnings.warn(
            "add_genai() is deprecated. Use add_llm() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        node = LLMNode(
            name,
            model_uri=model_uri,
            provider=provider,
            model_name=model_name,
            gateway_uri=gateway_uri,
            params=params,
        )
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

        Returns:
            Self for chaining
        """
        node = LLMNode(
            name,
            model_uri=model_uri,
            provider=provider,
            model_name=model_name,
            gateway_uri=gateway_uri,
            params=params,
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

    def add_langchain(
        self,
        name: str,
        chain_model_uri: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a LangChain chain node.

        Args:
            name: Node name
            chain_model_uri: MLflow model URI for LangChain chain
            params: Optional parameters

        Returns:
            Self for chaining
        """
        node = LangChainNode(name, chain_model_uri, params=params)
        self._nodes[name] = node
        return self

    def add_langgraph(
        self,
        name: str,
        graph_model_uri: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a LangGraph graph node.

        Args:
            name: Node name
            graph_model_uri: MLflow model URI for LangGraph graph
            params: Optional parameters

        Returns:
            Self for chaining
        """
        node = LangGraphNode(name, graph_model_uri, params=params)
        self._nodes[name] = node
        return self

    def connect(self, source: str, target: str, mapping: Optional[Dict[str, str]] = None) -> "Rulebook":
        """
        Connect two nodes. By default, all fields flow through automatically.
        
        Args:
            source: Source node name
            target: Target node name
            mapping: Optional field mapping (rarely needed - only if you want to rename fields)
        
        Examples:
            # Simple connection - all fields flow through
            rb.connect("node1", "node2")
            
            # With field mapping (only if you need to rename)
            rb.connect("node1", "node2", mapping={"new_name": "old_name"})
        
        Returns:
            Self for chaining
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")

        edge = Edge(source=source, target=target, mapping=mapping)
        self._edges.append(edge)
        return self

    def attach_guard(
        self,
        node_name: str,
        guard_policy: Any,
        guard_fn: Optional[Callable] = None,
    ) -> "Rulebook":
        """
        Attach a guardrail policy to a node.

        Args:
            node_name: Node name
            guard_policy: GuardPolicy instance or GuardPack
            guard_fn: Optional guard function (deprecated, use guard_policy)

        Returns:
            Self for chaining
        """
        if node_name not in self._nodes:
            raise ValueError(f"Node '{node_name}' not found")

        node = self._nodes[node_name]

        # Support GuardPack
        from rulesmith.guardrails.packs import GuardPack

        if isinstance(guard_policy, GuardPack):
            guard_policy = guard_policy.to_policy(when_node=node_name)

        # Store guard policy
        if not hasattr(node, "_guard_policies"):
            node._guard_policies = []
        node._guard_policies.append(guard_policy)

        # Register default guards if not already done (deprecated - use LangChain guardrails)
        try:
            register_default_guards(guard_executor)
        except Exception:
            # If register_default_guards fails (e.g., deprecated), that's okay
            # Users should migrate to LangChain guardrails
            pass

        # Legacy support for guard functions
        if guard_fn:
            if not hasattr(node, "_guards"):
                node._guards = []
            node._guards.append(guard_fn)

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
            elif isinstance(node, (ModelNode, BYOMNode)):  # BYOMNode is alias
                node_spec.model_uri = node.model_uri
                # Note: langchain_model cannot be serialized, must be provided at runtime
                if node.langchain_model:
                    node_spec.params["_has_langchain_model"] = True
            elif isinstance(node, (LLMNode, GenAINode)):  # GenAINode is alias
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
            elif isinstance(node, LangChainNode):
                node_spec.model_uri = node.chain_model_uri
            elif isinstance(node, LangGraphNode):
                node_spec.model_uri = node.graph_model_uri
            elif isinstance(node, ForkNode):
                node_spec.ab_arms = node.arms
            elif isinstance(node, GateNode):
                node_spec.condition = node.condition

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
    ):
        """
        Execute the rulebook with a payload.

        Args:
            payload: Input payload dictionary
            context: Optional execution context (RunContext or MLflowRunContext)
            enable_mlflow: Whether to enable MLflow logging (default: True)
            return_decision_result: If True, return DecisionResult; if False, return Dict (legacy)

        Returns:
            DecisionResult (default) or Dict (if return_decision_result=False)
        """
        from rulesmith.io.decision_result import DecisionResult
        
        spec = self.to_spec()
        engine = ExecutionEngine(spec)
        # Register all nodes
        for name, node in self._nodes.items():
            engine.register_node(name, node)

        # Create simple context if not provided
        # For most users, MLflow integration "just works" - no need to understand contexts
        if context is None:
            if enable_mlflow:
                from rulesmith.runtime.mlflow_context import MLflowRunContext
                context = MLflowRunContext(rulebook_spec=spec, enable_mlflow=True)
            else:
                from rulesmith.runtime.context import RunContext
                context = RunContext()

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
        
        # Return DecisionResult or Dict based on flag
        if return_decision_result:
            return result
        else:
            # Legacy mode: return the value dict
            if isinstance(result, DecisionResult):
                return result.value
            return result

