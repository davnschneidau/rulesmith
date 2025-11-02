"""Node types for the DAG execution engine."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from rulesmith.io.ser import ABArm


class Node(ABC):
    """Base class for all node types."""

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind

    @abstractmethod
    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Execute the node and return outputs.

        Args:
            state: Current state dictionary (shared payload)
            context: Execution context (RunContext)

        Returns:
            Dictionary of output fields
        """
        pass


class RuleNode(Node):
    """Node that executes a registered rule function."""

    def __init__(self, name: str, rule_func, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, "rule")
        self.rule_func = rule_func
        self.params = params or {}

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the rule function with inputs from state."""
        # Extract inputs from state based on function signature
        import inspect

        sig = inspect.signature(self.rule_func)
        kwargs = {}
        for param_name in sig.parameters:
            if param_name in state:
                kwargs[param_name] = state[param_name]

        # Merge with params
        kwargs.update(self.params)

        # Call the function
        result = self.rule_func(**kwargs)

        # Ensure result is a dict
        if not isinstance(result, dict):
            return {"result": result}

        return result


class ForkNode(Node):
    """Node that performs A/B testing traffic splitting."""

    def __init__(
        self,
        name: str,
        arms: List[ABArm],
        policy: Optional[str] = "hash",
    ):
        super().__init__(name, "fork")
        self.arms = arms
        self.policy = policy

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Select an arm and route traffic."""
        from rulesmith.ab.traffic import pick_arm

        identity = getattr(context, "identity", None) or state.get("identity")
        selected_arm = pick_arm(self.arms, identity, policy=self.policy)

        # Store selection in state
        state["_fork_selection"] = {
            "fork_name": self.name,
            "selected_arm": selected_arm.node,
        }

        # Return empty dict - fork doesn't modify state, just routes
        return {}


class GateNode(Node):
    """Node that conditionally routes based on an expression."""

    def __init__(self, name: str, condition: str):
        super().__init__(name, "gate")
        self.condition = condition

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Evaluate condition and route accordingly."""
        # Safe expression evaluation
        try:
            from asteval import Interpreter

            aeval = Interpreter()
            # Make state variables available in expression
            for key, value in state.items():
                if not key.startswith("_"):  # Skip internal state
                    aeval.symtable[key] = value

            result = aeval(self.condition)
            state["_gate_result"] = {self.name: bool(result)}
            return {"passed": bool(result)}
        except ImportError:
            # Fallback to basic eval if asteval not available (not recommended for production)
            try:
                # Create a safe namespace from state
                safe_dict = {k: v for k, v in state.items() if not k.startswith("_")}
                result = eval(self.condition, {"__builtins__": {}}, safe_dict)
                state["_gate_result"] = {self.name: bool(result)}
                return {"passed": bool(result)}
            except Exception as e:
                state["_gate_result"] = {self.name: False}
                state["_gate_error"] = {self.name: str(e)}
                return {"passed": False, "error": str(e)}
        except Exception as e:
            # On error, fail closed
            state["_gate_result"] = {self.name: False}
            state["_gate_error"] = {self.name: str(e)}
            return {"passed": False, "error": str(e)}


class BYOMNode(Node):
    """Node that loads and executes an MLflow model (Bring Your Own Model)."""

    def __init__(self, name: str, model_uri: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, "byom")
        self.model_uri = model_uri
        self.params = params or {}
        self._model = None

    def _load_model(self):
        """Lazy load the MLflow model."""
        if self._model is None:
            import mlflow.pyfunc

            self._model = mlflow.pyfunc.load_model(self.model_uri)
        return self._model

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the MLflow model."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for BYOMNode. Install with: pip install pandas")

        model = self._load_model()
        # Prepare input - model expects pandas DataFrame or dict
        if isinstance(state, dict):
            # Convert dict to DataFrame row
            input_df = pd.DataFrame([state])
        else:
            input_df = state

        result = model.predict(input_df)
        # Convert result to dict if needed
        if isinstance(result, pd.DataFrame):
            return result.iloc[0].to_dict()
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            return {"prediction": result[0]}
        else:
            return {"result": result}


class GenAINode(Node):
    """Node that executes a GenAI/LLM call."""

    def __init__(
        self,
        name: str,
        model_uri: Optional[str] = None,
        provider: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, "llm")
        self.model_uri = model_uri
        self.provider = provider
        self.params = params or {}
        self._chain = None

    def _load_chain(self):
        """Lazy load the LangChain/LangGraph chain."""
        if self._chain is None and self.model_uri:
            import mlflow.pyfunc

            self._chain = mlflow.pyfunc.load_model(self.model_uri)
        return self._chain

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the GenAI model."""
        # This will be enhanced in Phase 3 with actual GenAI integration
        # For now, placeholder implementation
        chain = self._load_chain()
        if chain:
            result = chain.predict(state)
            if isinstance(result, dict):
                return result
            return {"output": result}
        return {"output": "GenAI node not implemented yet"}


class HITLNode(Node):
    """Node that submits a request for human review."""

    def __init__(
        self,
        name: str,
        queue,
        timeout: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, "hitl")
        self.queue = queue
        self.timeout = timeout
        self.params = params or {}

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Submit HITL request and await decision."""
        # This will be enhanced in Phase 6 with actual HITL integration
        # For now, placeholder implementation
        return {"status": "pending", "message": "HITL node not implemented yet"}

