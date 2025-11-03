"""Routing functions for DAG execution (Fork, Gate, HITL).

These functions replace the node-based implementations to allow for
more flexible routing and better integration with the execution engine.
"""

import warnings
from typing import Any, Dict, List, Optional

from rulesmith.ab.traffic import pick_arm
from rulesmith.io.ser import ABArm


def fork(
    arms: List[ABArm],
    policy: str = "hash",
    policy_instance: Optional[Any] = None,
    identity: Optional[str] = None,
    seed: Optional[Any] = None,
    track_metrics: bool = True,
    context: Optional[Any] = None,
    ghost_fork: bool = False,
) -> Dict[str, Any]:
    """
    Fork function for A/B testing traffic splitting.
    
    This function selects an arm based on the policy and returns routing metadata.
    When used in a rulebook, the execution engine uses this metadata to route
    execution to the appropriate branch.
    
    Args:
        arms: List of A/B arms to choose from
        policy: Policy name (hash, random, thompson_sampling, ucb1, epsilon_greedy)
        policy_instance: Optional TrafficPolicy instance (overrides policy string)
        identity: Optional identity for deterministic hashing
        seed: Optional seed for random policies
        track_metrics: Whether to track A/B metrics
        context: Optional execution context
        ghost_fork: If True, run both arms but only execute actions on primary (shadow testing)
    
    Returns:
        Dictionary with routing metadata:
        - selected_variant: Name of selected arm node
        - ab_test: Fork name
        - ab_policy: Policy used
        - _ghost_fork: True if ghost fork mode
        - _fork_selection: Internal selection details
    """
    # Build context for policy
    policy_context = {
        "identity": identity,
        "seed": seed,
    }
    
    # Try to get arms_history from context if available
    if context and hasattr(context, "arms_history"):
        policy_context["arms_history"] = context.arms_history
    elif context and hasattr(context, "state"):
        # Fallback to state if available
        state = getattr(context, "state", {})
        if "arms_history" in state:
            policy_context["arms_history"] = state["arms_history"]
    
    # Select arm
    selected_arm = pick_arm(
        arms,
        identity=identity,
        policy=policy,
        policy_instance=policy_instance,
        context=policy_context,
    )
    
    result = {
        "selected_variant": selected_arm.node,
        "ab_test": "fork",  # Will be set by rulebook if name provided
        "ab_policy": policy,
        "_fork_selection": {
            "selected_arm": selected_arm.node,
            "policy": policy,
            "arm_weight": selected_arm.weight,
        },
    }
    
    if ghost_fork:
        result["_ghost_fork"] = True
        result["_ghost_arms"] = [arm.node for arm in arms]
    
    # Track A/B bucket in context if supported
    if context and hasattr(context, "set_ab_bucket"):
        bucket = f"fork:{selected_arm.node}"
        context.set_ab_bucket(bucket)
    
    # Log metrics if requested and context supports it
    if track_metrics and context and hasattr(context, "enable_mlflow") and context.enable_mlflow:
        try:
            import mlflow
            
            mlflow.set_tag("rulesmith.ab_fork", "fork")
            mlflow.set_tag("rulesmith.ab_arm", selected_arm.node)
            mlflow.set_tag("rulesmith.ab_policy", policy)
            mlflow.log_metric("ab_arm_weight", selected_arm.weight)
        except Exception:
            pass  # Ignore MLflow errors
    
    return result


def gate(
    condition: str,
    state: Dict[str, Any],
    context: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Gate function for conditional routing.
    
    Evaluates a condition expression against the current state and returns
    a boolean result. The execution engine uses this to determine which
    edges to follow.
    
    Args:
        condition: Expression to evaluate (e.g., "age >= 18")
        state: Current state dictionary
        context: Optional execution context
    
    Returns:
        Dictionary with:
        - passed: Boolean indicating if condition passed
        - error: Optional error message if evaluation failed
    """
    # Safe expression evaluation
    try:
        from asteval import Interpreter
        
        aeval = Interpreter()
        # Make state variables available in expression
        for key, value in state.items():
            if not key.startswith("_"):  # Skip internal state
                aeval.symtable[key] = value
        
        result = aeval(condition)
        return {"passed": bool(result)}
    except ImportError:
        # Fallback to basic eval if asteval not available (not recommended for production)
        try:
            # Create a safe namespace from state
            safe_dict = {k: v for k, v in state.items() if not k.startswith("_")}
            result = eval(condition, {"__builtins__": {}}, safe_dict)
            return {"passed": bool(result)}
        except Exception as e:
            # On error, fail closed
            return {"passed": False, "error": str(e)}
    except Exception as e:
        # On error, fail closed
        return {"passed": False, "error": str(e)}


def hitl(
    queue: Any,
    state: Dict[str, Any],
    timeout: Optional[float] = None,
    async_mode: bool = False,
    active_learning_threshold: Optional[float] = None,
    context: Optional[Any] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Human-in-the-Loop function for manual review workflows.
    
    Submits a review request to the HITL queue and optionally waits for a decision.
    In async mode, returns immediately with a pending state.
    
    Args:
        queue: HITL queue instance (HITLQueue)
        state: Current state dictionary
        timeout: Optional timeout in seconds (None = wait indefinitely)
        async_mode: If True, don't block execution (returns pending state)
        active_learning_threshold: Optional confidence threshold for active learning
        context: Optional execution context
        params: Optional parameters
    
    Returns:
        Dictionary with:
        - output: Review result or original output
        - _hitl_pending: True if async and still pending
        - _hitl_approved: True if approved
        - _hitl_rejected: True if rejected
        - _hitl_skipped: True if skipped due to active learning
        - _hitl_request_id: Request ID for tracking
    """
    from uuid import uuid4
    from rulesmith.hitl.base import ReviewRequest
    
    params = params or {}
    
    # Check if review is needed (active learning)
    if active_learning_threshold is not None:
        confidence = state.get("confidence", state.get("score", 1.0))
        if isinstance(confidence, dict):
            confidence = confidence.get("value", 1.0)
        
        # Skip review if confidence above threshold
        if confidence >= active_learning_threshold:
            return {
                "output": state.get("output", state.get("result", {})),
                "_hitl_skipped": True,
                "_hitl_reason": "confidence_above_threshold",
            }
    
    # Create review request
    request_id = str(uuid4())
    
    # Prepare payload for review
    review_payload = {
        "inputs": {k: v for k, v in state.items() if not k.startswith("_")},
        "model_output": state.get("output", state.get("result", {})),
    }
    
    # Add suggestions if available
    suggestions = params.get("suggestions", {})
    
    # Create request (need node name - will be set by rulebook)
    request = ReviewRequest(
        id=request_id,
        node="hitl",  # Will be overridden by rulebook
        payload=review_payload,
        suggestions=suggestions,
    )
    
    # Submit to queue
    try:
        submitted_id = queue.submit(request)
        
        # Log HITL request if context supports it
        if context and hasattr(context, "on_hitl"):
            context.on_hitl("hitl", state, context, submitted_id)
        
        if async_mode:
            # Return immediately with pending state
            return {
                "_hitl_pending": True,
                "_hitl_request_id": submitted_id,
                "output": state.get("output", {}),
            }
        
        # Wait for decision (blocking)
        decision = queue.get_decision(submitted_id, timeout=timeout)
        
        if decision is None:
            # Timeout - return timeout state
            return {
                "_hitl_timeout": True,
                "_hitl_request_id": submitted_id,
                "output": state.get("output", {}),
            }
        
        # Process decision
        if decision.approved:
            # Use edited output if provided, otherwise use original
            output = decision.edited_output if decision.edited_output else state.get("output", {})
            return {
                "output": output,
                "_hitl_approved": True,
                "_hitl_reviewer": decision.reviewer,
                "_hitl_comment": decision.comment,
            }
        else:
            # Rejected - return rejection state
            return {
                "_hitl_rejected": True,
                "_hitl_reviewer": decision.reviewer,
                "_hitl_comment": decision.comment,
                "output": state.get("output", {}),
            }
    
    except Exception as e:
        # Error submitting or getting decision
        return {
            "_hitl_error": True,
            "_hitl_error_message": str(e),
            "output": state.get("output", {}),
        }

