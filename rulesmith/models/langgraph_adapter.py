"""LangGraph adapter for integrating LangGraph graphs as nodes."""

from typing import Any, Dict, Optional


class LGNode:
    """LangGraph compiled graph node adapter."""

    def __init__(self, graph_model_uri: str):
        """
        Initialize LangGraph node.

        Args:
            graph_model_uri: MLflow model URI for LangGraph graph
        """
        self.graph_model_uri = graph_model_uri
        self._graph = None

    def _load_graph(self):
        """Lazy load the LangGraph graph."""
        if self._graph is None:
            try:
                import mlflow.pyfunc

                self._graph = mlflow.pyfunc.load_model(self.graph_model_uri)
            except ImportError:
                raise ImportError("MLflow is required for LangGraph adapter")
            except Exception as e:
                raise ValueError(f"Could not load LangGraph model from {self.graph_model_uri}: {str(e)}")
        return self._graph

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the LangGraph graph."""
        graph = self._load_graph()

        # Handle LangGraph compiled graph
        if hasattr(graph, "invoke"):
            result = graph.invoke(inputs)
        elif hasattr(graph, "predict"):
            result = graph.predict(inputs)
        elif hasattr(graph, "__call__"):
            result = graph(inputs)
        else:
            raise ValueError("LangGraph model does not support invoke, predict, or __call__")

        # Normalize output
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            return {"output": result}
        else:
            return {"output": result}

