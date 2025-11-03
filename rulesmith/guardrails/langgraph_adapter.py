"""LangGraph guardrails adapter for Rulesmith.

This module provides integration with LangGraph guardrail patterns,
allowing users to use LangGraph guardrail nodes within their workflows.
"""

import warnings
from typing import Any, Dict, Optional

from rulesmith.guardrails.execution import GuardResult
from rulesmith.guardrails.policy import GuardAction


def create_langgraph_guard(guard_name: str, langgraph_guard_node: Any, config: Optional[Dict[str, Any]] = None) -> callable:
    """
    Create a Rulesmith guard function from a LangGraph guardrail node.
    
    Args:
        guard_name: Name for the guard in Rulesmith
        langgraph_guard_node: LangGraph guardrail node or function
        config: Optional configuration for the guard
    
    Returns:
        Guard function compatible with Rulesmith's GuardExecutor
    """
    config = config or {}
    
    def guard_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LangGraph guardrail and return Rulesmith-compatible result.
        
        Args:
            inputs: Input dictionary (may contain 'text', 'input', 'prompt', 'output', etc.)
        
        Returns:
            Dictionary with 'passed', 'message', 'score', 'metadata'
        """
        # Extract text from inputs
        text = inputs.get("text") or inputs.get("input") or inputs.get("prompt") or inputs.get("output") or ""
        
        # Convert to string if needed
        if isinstance(text, dict):
            text = str(text)
        elif not isinstance(text, str):
            text = str(text)
        
        try:
            # LangGraph nodes typically use state dictionaries
            state = {"text": text, **inputs}
            
            # Try LangGraph node interface
            if hasattr(langgraph_guard_node, "invoke"):
                result = langgraph_guard_node.invoke(state)
            elif hasattr(langgraph_guard_node, "__call__"):
                result = langgraph_guard_node(state)
            elif callable(langgraph_guard_node):
                result = langgraph_guard_node(text)
            else:
                raise ValueError(f"LangGraph guardrail {guard_name} does not have a recognized interface")
            
            # Normalize result
            if isinstance(result, dict):
                # Check for LangGraph state format
                passed = result.get("valid", result.get("passed", result.get("is_valid", True)))
                message = result.get("message") or result.get("reason")
                score = result.get("score") or result.get("confidence")
                metadata = result.get("metadata", {})
            elif isinstance(result, bool):
                passed = result
                message = None
                score = None
                metadata = {}
            else:
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
            return {
                "passed": False,
                "message": f"LangGraph guardrail error: {str(e)}",
                "score": 0.0,
                "metadata": {"error": str(e), "guard_name": guard_name},
            }
    
    guard_func._guard_name = guard_name
    guard_func._is_guard = True
    guard_func._langgraph_guard = langgraph_guard_node
    
    return guard_func


def register_langgraph_guards(
    executor: Any,
    guards: Dict[str, Any],
    config: Optional[Dict[str, str]] = None,
) -> None:
    """
    Register multiple LangGraph guardrails with Rulesmith executor.
    
    Args:
        executor: GuardExecutor instance
        guards: Dictionary mapping guard names to LangGraph guardrail nodes
        config: Optional configuration per guard
    """
    config = config or {}
    
    for guard_name, langgraph_guard in guards.items():
        guard_config = config.get(guard_name, {})
        guard_func = create_langgraph_guard(guard_name, langgraph_guard, guard_config)
        executor.register_guard(guard_name, guard_func)

