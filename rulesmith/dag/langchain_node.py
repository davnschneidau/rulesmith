"""LangChain-specific node for integrating LangChain chains."""

from typing import Any, Dict, Optional

from rulesmith.dag.nodes import Node
from rulesmith.models.langchain_adapter import LCNode


class LangChainNode(Node):
    """Node that executes a LangChain chain."""

    def __init__(self, name: str, chain_model_uri: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LangChain node.

        Args:
            name: Node name
            chain_model_uri: MLflow model URI for LangChain chain
            params: Optional parameters
        """
        super().__init__(name, "langchain")
        self.chain_model_uri = chain_model_uri
        self.params = params or {}
        self._lc_node = None

    def _get_lc_node(self):
        """Lazy load LangChain node."""
        if self._lc_node is None:
            self._lc_node = LCNode(self.chain_model_uri)
        return self._lc_node

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the LangChain chain."""
        lc_node = self._get_lc_node()

        # Merge state with params
        inputs = state.copy()
        inputs.update(self.params)

        # Invoke chain
        result = lc_node.invoke(inputs)

        # Set model URI in context if supported
        if hasattr(context, "set_model_uri"):
            context.set_model_uri(self.chain_model_uri)

        return result

