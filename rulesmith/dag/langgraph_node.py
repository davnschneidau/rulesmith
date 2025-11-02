"""LangGraph-specific node for integrating LangGraph graphs."""

from typing import Any, Dict, Optional

from rulesmith.dag.nodes import Node
from rulesmith.models.langgraph_adapter import LGNode


class LangGraphNode(Node):
    """Node that executes a LangGraph compiled graph."""

    def __init__(self, name: str, graph_model_uri: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LangGraph node.

        Args:
            name: Node name
            graph_model_uri: MLflow model URI for LangGraph graph
            params: Optional parameters
        """
        super().__init__(name, "langgraph")
        self.graph_model_uri = graph_model_uri
        self.params = params or {}
        self._lg_node = None

    def _get_lg_node(self):
        """Lazy load LangGraph node."""
        if self._lg_node is None:
            self._lg_node = LGNode(self.graph_model_uri)
        return self._lg_node

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the LangGraph graph."""
        lg_node = self._get_lg_node()

        # Merge state with params
        inputs = state.copy()
        inputs.update(self.params)

        # Invoke graph
        result = lg_node.invoke(inputs)

        # Set model URI in context if supported
        if hasattr(context, "set_model_uri"):
            context.set_model_uri(self.graph_model_uri)

        return result

