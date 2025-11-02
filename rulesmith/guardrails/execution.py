"""Guard execution and evaluation."""

from typing import Any, Callable, Dict, List, Optional

from rulesmith.guardrails.policy import GuardAction, GuardPolicy


class GuardResult:
    """Result of guard evaluation."""

    def __init__(
        self,
        guard_name: str,
        passed: bool,
        action: GuardAction = GuardAction.ALLOW,
        message: Optional[str] = None,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.guard_name = guard_name
        self.passed = passed
        self.action = action
        self.message = message or ("Guard passed" if passed else "Guard failed")
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "guard_name": self.guard_name,
            "passed": self.passed,
            "action": self.action.value,
            "message": self.message,
            "score": self.score,
            "metadata": self.metadata,
        }


class GuardExecutor:
    """Executes guard functions and applies policies."""

    def __init__(self):
        self._guards: Dict[str, Callable] = {}

    def register_guard(self, name: str, guard_func: Callable) -> None:
        """Register a guard function."""
        self._guards[name] = guard_func

    def evaluate(
        self,
        guard_name: str,
        inputs: Dict[str, Any],
        policy: Optional[GuardPolicy] = None,
    ) -> GuardResult:
        """
        Evaluate a guard function.

        Args:
            guard_name: Name of the guard
            inputs: Input data to check
            policy: Optional guard policy

        Returns:
            GuardResult
        """
        if guard_name not in self._guards:
            raise ValueError(f"Guard '{guard_name}' not registered")

        guard_func = self._guards[guard_name]

        try:
            # Call guard function
            result = guard_func(inputs)

            # Normalize result
            if isinstance(result, dict):
                passed = result.get("passed", result.get("allow", True))
                message = result.get("message")
                score = result.get("score")
                metadata = result.get("metadata", {})
            elif isinstance(result, bool):
                passed = result
                message = None
                score = None
                metadata = {}
            else:
                # Assume passed if truthy
                passed = bool(result)
                message = None
                score = None
                metadata = {}

            # Determine action from policy
            action = GuardAction.ALLOW
            if not passed and policy:
                action = policy.on_fail

            return GuardResult(
                guard_name=guard_name,
                passed=passed,
                action=action,
                message=message,
                score=score,
                metadata=metadata,
            )

        except Exception as e:
            # On error, fail closed
            action = GuardAction.BLOCK
            if policy:
                action = policy.on_fail

            return GuardResult(
                guard_name=guard_name,
                passed=False,
                action=action,
                message=f"Guard evaluation error: {str(e)}",
                metadata={"error": str(e)},
            )

    def evaluate_policy(
        self,
        policy: GuardPolicy,
        inputs: Dict[str, Any],
    ) -> List[GuardResult]:
        """
        Evaluate all guards in a policy.

        Args:
            policy: Guard policy
            inputs: Input data to check

        Returns:
            List of GuardResult objects
        """
        results = []

        for check_name in policy.checks:
            result = self.evaluate(check_name, inputs, policy=policy)
            results.append(result)

            # Short-circuit if blocking guard fails
            if not result.passed and result.action == GuardAction.BLOCK:
                break

        return results

    def apply_policy(
        self,
        policy: GuardPolicy,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply guard policy to inputs/outputs and return modified state.

        Args:
            policy: Guard policy
            inputs: Input data
            outputs: Optional output data

        Returns:
            Modified state dictionary with guard overrides if applicable
        """
        results = self.evaluate_policy(policy, inputs)

        # Check if any guards failed
        failed_guards = [r for r in results if not r.passed]

        if not failed_guards:
            return outputs or {}

        # Apply policy actions
        final_outputs = (outputs or {}).copy()
        override_applied = False

        for result in failed_guards:
            if result.action == GuardAction.BLOCK:
                # Block: return empty or error state
                return {"_guard_blocked": True, "_guard_message": result.message}

            elif result.action == GuardAction.OVERRIDE:
                # Override: apply override template
                if policy.override_template:
                    final_outputs.update(policy.override_template)
                    override_applied = True

            elif result.action == GuardAction.FLAG:
                # Flag: add flag metadata but allow
                if "_guard_flags" not in final_outputs:
                    final_outputs["_guard_flags"] = []
                final_outputs["_guard_flags"].append({
                    "guard": result.guard_name,
                    "message": result.message,
                })

        # Store guard results
        final_outputs["_guard_results"] = [r.to_dict() for r in results]

        return final_outputs


# Global guard executor
guard_executor = GuardExecutor()
