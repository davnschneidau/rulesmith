"""LangChain adapter for integrating LangChain chains as nodes."""

from typing import Any, Dict, Optional

import mlflow.pyfunc


class LCNode:
    """LangChain chain node adapter."""

    def __init__(self, chain_model_uri: str):
        """
        Initialize LangChain node.

        Args:
            chain_model_uri: MLflow model URI for LangChain chain
        """
        self.chain_model_uri = chain_model_uri
        self._chain = None

    def _load_chain(self):
        """Lazy load the LangChain chain."""
        if self._chain is None:
            self._chain = mlflow.pyfunc.load_model(self.chain_model_uri)
        return self._chain

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the LangChain chain."""
        chain = self._load_chain()
        result = chain.invoke(inputs)
        if isinstance(result, dict):
            return result
        return {"output": result}


def log_langchain_model(
    chain: Any,
    name: str,
    prompt_templates: Optional[Dict[str, str]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Log a LangChain chain as an MLflow model.

    Args:
        chain: LangChain chain or runnable
        name: Model name
        prompt_templates: Optional prompt templates
        model_config: Optional model configuration

    Returns:
        Model URI
    """
    try:
        import mlflow.langchain

        mlflow.langchain.log_model(
            lc_model=chain,
            artifact_path="model",
            registered_model_name=name,
        )
        return f"models:/{name}/latest"
    except ImportError:
        raise ImportError("LangChain support requires mlflow[langchain] or langchain package")

