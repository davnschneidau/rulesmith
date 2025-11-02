"""Decorators for registering rules and rulebooks."""

import functools
from typing import Any, Callable, Dict, List, Optional

from rulesmith.dag.registry import rule_registry
from rulesmith.io.ser import RuleSpec
from rulesmith.utils import hash_code


def rule(
    name: Optional[str] = None,
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    default_params: Optional[Dict[str, Any]] = None,
    version: str = "1.0.0",
):
    """
    Decorator to register a function as a rule.

    Args:
        name: Rule name (defaults to function name)
        inputs: List of input field names (auto-inferred from function signature if None)
        outputs: List of output field names (defaults to ["result"] if None)
        default_params: Default parameters for the rule
        version: Rule version

    Returns:
        Decorated function (registered in global registry)

    Example:
        @rule(name="check_age", inputs=["age"], outputs=["eligible"])
        def check_age(age: int) -> dict:
            return {"eligible": age >= 18}
    """

    def decorator(func: Callable) -> Callable:
        rule_name = name or func.__name__

        # Infer inputs from function signature if not provided
        if inputs is None:
            import inspect

            sig = inspect.signature(func)
            rule_inputs = [param for param in sig.parameters.keys() if param != "self"]
        else:
            rule_inputs = inputs

        # Default outputs if not provided
        rule_outputs = outputs or ["result"]

        # Create rule spec
        spec = RuleSpec(
            name=rule_name,
            version=version,
            inputs=rule_inputs,
            outputs=rule_outputs,
            code_hash=hash_code(func),
            params=default_params or {},
        )

        # Register in global registry
        rule_registry.register(spec, func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach metadata
        wrapper._rule_spec = spec
        wrapper._is_rule = True

        return wrapper

    return decorator


def rulebook(name: str, version: str = "1.0.0"):
    """
    Decorator to mark a function as a rulebook builder.

    Args:
        name: Rulebook name
        version: Rulebook version

    Returns:
        Decorated function

    Example:
        @rulebook(name="credit_decision", version="1.0.0")
        def build_credit_rulebook():
            rb = Rulebook(name="credit_decision", version="1.0.0")
            rb.add_rule(check_age, as_name="age_check")
            return rb
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._rulebook_name = name
        wrapper._rulebook_version = version
        wrapper._is_rulebook = True

        return wrapper

    return decorator

