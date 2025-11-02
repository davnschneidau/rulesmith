"""Node types for the DAG execution engine."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from rulesmith.guardrails.execution import guard_executor
from rulesmith.guardrails.packs import register_default_guards
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
        super().__init__(name, "fork")
        self.arms = arms
        self.policy = policy or "hash"
        self.policy_instance = policy_instance
        self.track_metrics = track_metrics

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Select an arm and route traffic."""
        from rulesmith.ab.traffic import pick_arm

        identity = getattr(context, "identity", None) or state.get("identity")
        seed = getattr(context, "seed", None)

        # Build context for policy
        policy_context = {
            "identity": identity,
            "seed": seed,
        }

        # Try to get arms_history from context if available
        if hasattr(context, "arms_history"):
            policy_context["arms_history"] = context.arms_history

        # Select arm
        selected_arm = pick_arm(
            self.arms,
            identity=identity,
            policy=self.policy,
            policy_instance=self.policy_instance,
            context=policy_context,
        )

        # Store selection in state
        state["_fork_selection"] = {
            "fork_name": self.name,
            "selected_arm": selected_arm.node,
            "policy": self.policy,
            "arm_weight": selected_arm.weight,
        }

        # Track A/B bucket in context if supported
        bucket = f"{self.name}:{selected_arm.node}"
        if hasattr(context, "set_ab_bucket"):
            context.set_ab_bucket(bucket)

        # Log metrics to MLflow if supported
        if self.track_metrics and hasattr(context, "enable_mlflow") and context.enable_mlflow:
            try:
                import mlflow

                mlflow.set_tag("rulesmith.ab_fork", self.name)
                mlflow.set_tag("rulesmith.ab_arm", selected_arm.node)
                mlflow.set_tag("rulesmith.ab_policy", self.policy)
                mlflow.log_metric("ab_arm_weight", selected_arm.weight)
            except Exception:
                pass  # Ignore MLflow errors

        # Return selection info
        return {
            "_ab_selection": {
                "fork": self.name,
                "arm": selected_arm.node,
                "policy": self.policy,
            }
        }


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
        self._byom_ref = None

    def _get_byom_ref(self):
        """Lazy load BYOM reference."""
        if self._byom_ref is None:
            from rulesmith.models.mlflow_byom import BYOMRef

            self._byom_ref = BYOMRef(self.model_uri)
        return self._byom_ref

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the MLflow model."""
        byom_ref = self._get_byom_ref()

        # Prepare inputs by merging state with params
        inputs = state.copy()
        inputs.update(self.params)

        # Execute prediction
        result = byom_ref.predict(inputs)

        # Log model URI to context if supported
        if hasattr(context, "set_model_uri"):
            context.set_model_uri(self.model_uri)

        return result


class GenAINode(Node):
    """Node that executes a GenAI/LLM call."""

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
        self._genai_wrapper = None
        self._chain = None

    def _get_genai_wrapper(self):
        """Lazy load GenAI wrapper."""
        if self._genai_wrapper is None:
            from rulesmith.models.genai import GenAIWrapper

            self._genai_wrapper = GenAIWrapper(
                provider=self.provider,
                model_name=self.model_name,
                model_uri=self.model_uri,
                gateway_uri=self.gateway_uri,
            )
        return self._genai_wrapper

    def _load_chain(self):
        """Lazy load LangChain chain from MLflow if model_uri provided."""
        if self._chain is None and self.model_uri:
            try:
                import mlflow.pyfunc

                self._chain = mlflow.pyfunc.load_model(self.model_uri)
            except Exception:
                # If MLflow model loading fails, fall back to GenAI wrapper
                self._chain = None
        return self._chain

    def execute(self, state: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute the GenAI model."""
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
                # Fall back to GenAI wrapper
                chain = None

        # Use GenAI wrapper if no chain or chain failed
        if chain is None:
            genai_wrapper = self._get_genai_wrapper()

            # Extract prompt from state
            prompt = state.get("prompt") or state.get("input") or str(state)
            if isinstance(prompt, dict):
                prompt = str(prompt)

            # Merge with params
            genai_params = self.params.copy()
            genai_params.update({k: v for k, v in state.items() if k not in ["prompt", "input"]})

            # Invoke GenAI
            output = genai_wrapper.invoke(prompt, **genai_params)

        # Track execution with MLflow if context supports it
        latency = time.time() - start_time

        if hasattr(context, "set_model_uri") and self.model_uri:
            context.set_model_uri(self.model_uri)

        # Log GenAI metrics if context supports it
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


# HITLNode is now in rulesmith.hitl.node module
# Import it here for backwards compatibility
from rulesmith.hitl.node import HITLNode

