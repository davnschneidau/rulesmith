"""LangChain guardrails adapter for Rulesmith.

This module provides integration with LangChain's guardrail ecosystem,
allowing users to use LangChain-compatible guardrails instead of built-in ones.
"""

import warnings
from typing import Any, Dict, Optional

from rulesmith.guardrails.execution import GuardResult
from rulesmith.guardrails.policy import GuardAction


def create_langchain_guard(guard_name: str, langchain_guard: Any, config: Optional[Dict[str, Any]] = None) -> callable:
    """
    Create a Rulesmith guard function from a LangChain guardrail.
    
    Args:
        guard_name: Name for the guard in Rulesmith
        langchain_guard: LangChain guardrail instance or callable
        config: Optional configuration for the guard
    
    Returns:
        Guard function compatible with Rulesmith's GuardExecutor
    """
    config = config or {}
    
    def guard_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LangChain guardrail and return Rulesmith-compatible result.
        
        Args:
            inputs: Input dictionary (may contain 'text', 'input', 'prompt', 'output', etc.)
        
        Returns:
            Dictionary with 'passed', 'message', 'score', 'metadata'
        """
        # Extract text from inputs (try common field names)
        text = inputs.get("text") or inputs.get("input") or inputs.get("prompt") or inputs.get("output") or ""
        
        # Convert to string if needed
        if isinstance(text, dict):
            text = str(text)
        elif not isinstance(text, str):
            text = str(text)
        
        try:
            # Try different LangChain guardrail interfaces
            if hasattr(langchain_guard, "invoke"):
                # LangChain Runnable interface
                result = langchain_guard.invoke(text)
            elif hasattr(langchain_guard, "validate"):
                # LangChain Guardrails interface
                result = langchain_guard.validate(text)
            elif hasattr(langchain_guard, "check"):
                # Generic check interface
                result = langchain_guard.check(text)
            elif callable(langchain_guard):
                # Direct callable
                result = langchain_guard(text)
            else:
                raise ValueError(f"LangChain guardrail {guard_name} does not have a recognized interface")
            
            # Normalize result to Rulesmith format
            if isinstance(result, dict):
                # Already a dict - check for common LangChain guardrail result formats
                passed = result.get("valid", result.get("passed", result.get("is_valid", True)))
                message = result.get("message") or result.get("reason")
                score = result.get("score") or result.get("confidence")
                metadata = result.get("metadata", {})
                if "violations" in result:
                    metadata["violations"] = result["violations"]
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
            
            return {
                "passed": passed,
                "message": message,
                "score": score,
                "metadata": metadata,
            }
        
        except Exception as e:
            # On error, fail closed
            return {
                "passed": False,
                "message": f"LangChain guardrail error: {str(e)}",
                "score": 0.0,
                "metadata": {"error": str(e), "guard_name": guard_name},
            }
    
    # Set guard name for registration
    guard_func._guard_name = guard_name
    guard_func._is_guard = True
    guard_func._langchain_guard = langchain_guard
    
    return guard_func


def register_langchain_guards(
    executor: Any,
    guards: Dict[str, Any],
    config: Optional[Dict[str, str]] = None,
) -> None:
    """
    Register multiple LangChain guardrails with Rulesmith executor.
    
    Args:
        executor: GuardExecutor instance
        guards: Dictionary mapping guard names to LangChain guardrail instances
        config: Optional configuration per guard
    
    Example:
        from langchain_guardrails import PIIGuard, ToxicityGuard
        from rulesmith.guardrails.langchain_adapter import register_langchain_guards
        from rulesmith.guardrails.execution import guard_executor
        
        guards = {
            "pii_check": PIIGuard(),
            "toxicity_check": ToxicityGuard(),
        }
        register_langchain_guards(guard_executor, guards)
    """
    config = config or {}
    
    for guard_name, langchain_guard in guards.items():
        guard_config = config.get(guard_name, {})
        guard_func = create_langchain_guard(guard_name, langchain_guard, guard_config)
        executor.register_guard(guard_name, guard_func)


# Pre-defined guardrail mappings for common LangChain guardrail packages
def create_pii_guard_from_langchain(executor: Any, langchain_pii_guard: Any = None) -> None:
    """
    Create and register a PII guard from LangChain.
    
    Args:
        executor: GuardExecutor instance
        langchain_pii_guard: Optional LangChain PII guard instance
                           If None, will try to create one from common packages
    """
    if langchain_pii_guard is None:
        # Try to import from common LangChain guardrail packages
        try:
            # Try langchain-core guardrails
            from langchain_core.guardrails import PIIGuard
            langchain_pii_guard = PIIGuard()
        except ImportError:
            try:
                # Try langchain-guardrails package
                from langchain_guardrails import PIIGuard
                langchain_pii_guard = PIIGuard()
            except ImportError:
                warnings.warn(
                    "LangChain PII guardrail not found. Install langchain-core or langchain-guardrails.",
                    UserWarning,
                )
                return
    
    guard_func = create_langchain_guard("pii_langchain", langchain_pii_guard)
    executor.register_guard("pii_langchain", guard_func)


def create_toxicity_guard_from_langchain(executor: Any, langchain_toxicity_guard: Any = None) -> None:
    """
    Create and register a toxicity guard from LangChain.
    
    Args:
        executor: GuardExecutor instance
        langchain_toxicity_guard: Optional LangChain toxicity guard instance
                                 If None, will try to create one from common packages
    """
    if langchain_toxicity_guard is None:
        try:
            from langchain_core.guardrails import ToxicityGuard
            langchain_toxicity_guard = ToxicityGuard()
        except ImportError:
            try:
                from langchain_guardrails import ToxicityGuard
                langchain_toxicity_guard = ToxicityGuard()
            except ImportError:
                warnings.warn(
                    "LangChain toxicity guardrail not found. Install langchain-core or langchain-guardrails.",
                    UserWarning,
                )
                return
    
    guard_func = create_langchain_guard("toxicity_langchain", langchain_toxicity_guard)
    executor.register_guard("toxicity_langchain", guard_func)


def create_moderation_guard_from_langchain(executor: Any, langchain_moderation_guard: Any = None) -> None:
    """
    Create and register a moderation guard from LangChain.
    
    Args:
        executor: GuardExecutor instance
        langchain_moderation_guard: Optional LangChain moderation guard instance
                                   If None, will try to create one from common packages
    """
    if langchain_moderation_guard is None:
        try:
            from langchain_core.guardrails import ModerationGuard
            langchain_moderation_guard = ModerationGuard()
        except ImportError:
            try:
                from langchain_guardrails import ModerationGuard
                langchain_moderation_guard = ModerationGuard()
            except ImportError:
                warnings.warn(
                    "LangChain moderation guardrail not found. Install langchain-core or langchain-guardrails.",
                    UserWarning,
                )
                return
    
    guard_func = create_langchain_guard("moderation_langchain", langchain_moderation_guard)
    executor.register_guard("moderation_langchain", guard_func)

