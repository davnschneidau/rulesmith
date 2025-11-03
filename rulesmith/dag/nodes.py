"""Node types for the DAG execution engine."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from rulesmith.guardrails.execution import guard_executor
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
        """
        Execute the rule function. 
        
        Simple approach: Extract function parameters from state and call the function.
        If a parameter is missing from state, it uses the default from params or function signature.
        """
        import inspect

        sig = inspect.signature(self.rule_func)
        kwargs = {}
        
        # Extract function parameters from state
        for param_name in sig.parameters:
            if param_name in state:
                kwargs[param_name] = state[param_name]
            elif param_name in self.params:
                kwargs[param_name] = self.params[param_name]
            # If not in state or params, let Python handle the default/missing error

        # Call the function
        result = self.rule_func(**kwargs)

        # Always return a dict for consistency
        if not isinstance(result, dict):
            return {"result": result}

        return result


class ForkNode(Node):
    """
    DEPRECATED: Use fork() function instead of ForkNode.
    
    Node that performs A/B testing traffic splitting.
    
    This class is deprecated. Use the fork() function from rulesmith.dag.functions
    instead for better integration with the execution engine.
    """

    def __init__(
        self,
        name: str,
        arms: List[ABArm],
        policy: Optional[str] = "hash",
        policy_instance: Optional[Any] = None,
        track_metrics: bool = True,
    ):
        """
        Initialize ForkNode.

        Args:
            name: Node name
            arms: List of A/B arms
            policy: Policy name (hash, random, thompson_sampling, ucb1, epsilon_greedy)
            policy_instance: Optional TrafficPolicy instance
            track_metrics: Whether to track A/B metrics for MLflow
        """
        warnings.warn(
            "ForkNode is deprecated. Use fork() function from rulesmith.dag.functions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(name, "fork")
        self.arms = arms
        self.policy = policy or "hash"
        self.policy_instance = policy_instance
        self.track_metrics = track_metrics

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Select an arm and route traffic."""
        from rulesmith.dag.functions import fork

        identity = getattr(context, "identity", None) or state.get("identity")
        seed = getattr(context, "seed", None)

        # Use fork function internally
        result = fork(
            self.arms,
            policy=self.policy,
            policy_instance=self.policy_instance,
            identity=identity,
            seed=seed,
            track_metrics=self.track_metrics,
            context=context,
        )
        
        # Update result with node name
        result["ab_test"] = self.name
        result["_ab_selection"] = {
            "fork": self.name,
            "arm": result["selected_variant"],
            "policy": self.policy,
        }
        
        return result


class GateNode(Node):
    """
    DEPRECATED: Use gate() function instead of GateNode.
    
    Node that conditionally routes based on an expression.
    
    This class is deprecated. Use the gate() function from rulesmith.dag.functions
    instead for better integration with the execution engine.
    """

    def __init__(self, name: str, condition: str):
        warnings.warn(
            "GateNode is deprecated. Use gate() function from rulesmith.dag.functions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(name, "gate")
        self.condition = condition

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Evaluate condition and route accordingly."""
        from rulesmith.dag.functions import gate

        # Use gate function internally
        result = gate(self.condition, state, context)
        
        # Store result in state for backward compatibility
        state["_gate_result"] = {self.name: result.get("passed", False)}
        if "error" in result:
            state["_gate_error"] = {self.name: result["error"]}
        
        return result


class ModelNode(Node):
    """Node that loads and executes an MLflow model or LangChain model."""

    def __init__(self, name: str, model_uri: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, "model")
        self.model_uri = model_uri
        self.params = params or {}
        self._model_ref = None

    def _get_model_ref(self):
        """Lazy load model reference."""
        if self._model_ref is None:
            from rulesmith.models.mlflow_byom import BYOMRef

            self._model_ref = BYOMRef(self.model_uri)
        return self._model_ref

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the MLflow model."""
        model_ref = self._get_model_ref()

        # Prepare inputs by merging state with params
        inputs = state.copy()
        inputs.update(self.params)

        # Execute prediction
        result = model_ref.predict(inputs)

        # Log model URI to context if supported
        if hasattr(context, "set_model_uri"):
            context.set_model_uri(self.model_uri)

        return result


# Backward compatibility alias
BYOMNode = ModelNode


class LLMNode(Node):
    """Node that executes an LLM call via LangChain or direct provider."""

    def __init__(
        self,
        name: str,
        model_uri: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        gateway_uri: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, "llm")
        self.model_uri = model_uri
        self.provider = provider or "openai"
        self.model_name = model_name
        self.gateway_uri = gateway_uri
        self.params = params or {}
        self._llm_wrapper = None
        self._chain = None

    def _get_llm_wrapper(self):
        """Lazy load LLM wrapper."""
        if self._llm_wrapper is None:
            from rulesmith.models.genai import GenAIWrapper

            self._llm_wrapper = GenAIWrapper(
                provider=self.provider,
                model_name=self.model_name,
                model_uri=self.model_uri,
                gateway_uri=self.gateway_uri,
            )
        return self._llm_wrapper

    def _load_chain(self):
        """Lazy load LangChain chain from MLflow if model_uri provided."""
        if self._chain is None and self.model_uri:
            try:
                import mlflow.pyfunc

                self._chain = mlflow.pyfunc.load_model(self.model_uri)
            except Exception:
                # If MLflow model loading fails, fall back to LLM wrapper
                self._chain = None
        return self._chain

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the LLM model."""
        import time

        start_time = time.time()

        # Try LangChain chain first if model_uri is provided
        chain = self._load_chain()
        if chain:
            try:
                # Use LangChain chain
                result = chain.invoke(state) if hasattr(chain, "invoke") else chain.predict(state)
                if isinstance(result, dict):
                    output = result
                else:
                    output = {"output": result}
            except Exception:
                # Fall back to LLM wrapper
                chain = None

        # Use LLM wrapper if no chain or chain failed
        if chain is None:
            llm_wrapper = self._get_llm_wrapper()

            # Extract prompt from state
            prompt = state.get("prompt") or state.get("input") or str(state)
            if isinstance(prompt, dict):
                prompt = str(prompt)

            # Merge with params
            llm_params = self.params.copy()
            llm_params.update({k: v for k, v in state.items() if k not in ["prompt", "input"]})

            # Invoke LLM
            output = llm_wrapper.invoke(prompt, **llm_params)

        # Track execution with MLflow if context supports it
        latency = time.time() - start_time

        if hasattr(context, "set_model_uri") and self.model_uri:
            context.set_model_uri(self.model_uri)

        # Log LLM metrics if context supports it
        if hasattr(context, "finish_genai"):
            # Extract token/cost info from output if available
            tokens = output.get("tokens", {}) if isinstance(output.get("tokens"), dict) else None
            cost = output.get("cost", output.get("cost_usd"))
            provider = self.provider

            context.finish_genai(
                outputs=output,
                tokens=tokens,
                cost=cost,
                latency=latency,
                provider=provider,
            )

        return output


# Backward compatibility alias
GenAINode = LLMNode


# HITLNode is now in rulesmith.hitl.node module
# Import it here for backwards compatibility
# Note: HITLNode is deprecated - use hitl() function from rulesmith.dag.functions instead
from rulesmith.hitl.node import HITLNode as _HITLNodeBase


class HITLNode(_HITLNodeBase):
    """DEPRECATED: Use hitl() function instead of HITLNode."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "HITLNode is deprecated. Use hitl() function from rulesmith.dag.functions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

