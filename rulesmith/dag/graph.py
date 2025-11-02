"""Rulebook builder and DAG construction."""

from typing import Any, Callable, Dict, List, Optional

from rulesmith.dag.execution import ExecutionEngine
from rulesmith.dag.langchain_node import LangChainNode
from rulesmith.dag.langgraph_node import LangGraphNode
from rulesmith.dag.nodes import (
    BYOMNode,
    ForkNode,
    GateNode,
    GenAINode,
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

    def add_fork(
        self,
        name: str,
        arms: List[ABArm],
        policy: Optional[str] = "hash",
        policy_instance: Optional[Any] = None,
        track_metrics: bool = True,
    ) -> "Rulebook":
        """
        Add a fork node for A/B testing.

        Args:
            name: Fork node name
            arms: List of A/B arms
            policy: Traffic splitting policy ("hash", "random", "thompson_sampling", "ucb1", "epsilon_greedy")
            policy_instance: Optional TrafficPolicy instance (overrides policy string)
            track_metrics: Whether to track A/B metrics in MLflow

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

        Args:
            name: Node name
            model_uri: MLflow model URI
            params: Optional parameters

        Returns:
            Self for chaining
        """
        node = BYOMNode(name, model_uri, params=params)
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
        node = GenAINode(
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

        # Register default guards if not already done
        register_default_guards(guard_executor)

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
            elif isinstance(node, BYOMNode):
                node_spec.model_uri = node.model_uri
            elif isinstance(node, GenAINode):
                node_spec.model_uri = node.model_uri
                if hasattr(node, "provider"):
                    node_spec.params["provider"] = node.provider
                if hasattr(node, "model_name"):
                    node_spec.params["model_name"] = node.model_name
                if hasattr(node, "gateway_uri"):
                    node_spec.params["gateway_uri"] = node.gateway_uri
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
    ) -> Dict[str, Any]:
        """
        Execute the rulebook with a payload.

        Args:
            payload: Input payload dictionary
            context: Optional execution context (RunContext or MLflowRunContext)
            enable_mlflow: Whether to enable MLflow logging (default: True)

        Returns:
            Final output state
        """
        spec = self.to_spec()
        engine = ExecutionEngine(spec)
        engine.register_node(self._nodes)

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
                return engine.execute(payload, context, nodes=self._nodes)
        else:
            return engine.execute(payload, context, nodes=self._nodes)

