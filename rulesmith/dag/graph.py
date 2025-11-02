"""Rulebook builder and DAG construction."""

from typing import Any, Callable, Dict, List, Optional

from rulesmith.dag.execution import ExecutionEngine
from rulesmith.dag.nodes import (
    BYOMNode,
    ForkNode,
    GateNode,
    GenAINode,
    HITLNode,
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

    def add_fork(
        self,
        name: str,
        arms: List[ABArm],
        policy: Optional[str] = "hash",
    ) -> "Rulebook":
        """
        Add a fork node for A/B testing.

        Args:
            name: Fork node name
            arms: List of A/B arms
            policy: Traffic splitting policy ("hash" or "random")

        Returns:
            Self for chaining
        """
        node = ForkNode(name, arms, policy=policy)
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
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a GenAI/LLM node.

        Args:
            name: Node name
            model_uri: Optional MLflow model URI
            provider: Optional provider name
            params: Optional parameters

        Returns:
            Self for chaining
        """
        node = GenAINode(name, model_uri=model_uri, provider=provider, params=params)
        self._nodes[name] = node
        return self

    def add_hitl(
        self,
        name: str,
        queue: Any,
        timeout: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Rulebook":
        """
        Add a Human-in-the-Loop node.

        Args:
            name: Node name
            queue: HITL queue instance
            timeout: Optional timeout in seconds
            params: Optional parameters

        Returns:
            Self for chaining
        """
        node = HITLNode(name, queue, timeout=timeout, params=params)
        self._nodes[name] = node
        return self

    def connect(
        self,
        source: str,
        target: str,
        mapping: Optional[Dict[str, str]] = None,
    ) -> "Rulebook":
        """
        Connect two nodes with an edge.

        Args:
            source: Source node name
            target: Target node name
            mapping: Optional field mapping {target_field: source_field}

        Returns:
            Self for chaining
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")

        edge = Edge(source=source, target=target, mapping=mapping or {})
        self._edges.append(edge)
        return self

    def attach_guard(self, node_name: str, guard_fn: Callable) -> "Rulebook":
        """
        Attach a guardrail to a node.

        Args:
            node_name: Node name
            guard_fn: Guard function

        Returns:
            Self for chaining
        """
        # This will be enhanced in Phase 5
        # For now, just store reference
        if node_name not in self._nodes:
            raise ValueError(f"Node '{node_name}' not found")

        if not hasattr(self._nodes[node_name], "_guards"):
            self._nodes[node_name]._guards = []
        self._nodes[node_name]._guards.append(guard_fn)

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

    def run(self, payload: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute the rulebook with a payload.

        Args:
            payload: Input payload dictionary
            context: Optional execution context

        Returns:
            Final output state
        """
        spec = self.to_spec()
        engine = ExecutionEngine(spec)
        engine.register_node(self._nodes)

        # Create minimal context if not provided
        if context is None:
            from types import SimpleNamespace

            context = SimpleNamespace(identity=None, run_id=None)

        return engine.execute(payload, context, nodes=self._nodes)

