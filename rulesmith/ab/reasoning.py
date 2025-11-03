"""A/B testing reasoning and explainability."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ForkReason:
    """Reasoning explanation for fork/arm selection."""
    
    fork_name: str
    selected_arm: str
    policy: str
    policy_explanation: str
    decision_factors: Dict[str, Any]
    arm_weights: Dict[str, float]
    historical_performance: Optional[Dict[str, Dict[str, float]]] = None


def explain_fork_selection(
    fork_name: str,
    selected_arm: str,
    policy: str,
    arms: List[Any],
    context: Optional[Dict[str, Any]] = None,
    arms_history: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ForkReason:
    """
    Generate explanation for why a specific arm was selected.
    
    Args:
        fork_name: Name of the fork
        selected_arm: Selected arm name
        policy: Policy used
        arms: List of arms
        context: Execution context
        arms_history: Historical performance data
    
    Returns:
        ForkReason object with explanation
    """
    # Build arm weights
    arm_weights = {arm.node: arm.weight for arm in arms}
    
    # Build decision factors
    decision_factors = {
        "policy": policy,
        "arm_weights": arm_weights,
    }
    
    if context:
        if "identity" in context:
            decision_factors["identity"] = context["identity"]
        if "seed" in context:
            decision_factors["seed"] = context["seed"]
    
    # Generate policy explanation
    policy_explanation = _explain_policy(policy, selected_arm, arms, context, arms_history)
    
    # Extract historical performance if available
    historical_performance = None
    if arms_history:
        historical_performance = {}
        for arm in arms:
            if arm.node in arms_history:
                historical_performance[arm.node] = arms_history[arm.node]
    
    return ForkReason(
        fork_name=fork_name,
        selected_arm=selected_arm,
        policy=policy,
        policy_explanation=policy_explanation,
        decision_factors=decision_factors,
        arm_weights=arm_weights,
        historical_performance=historical_performance,
    )


def _explain_policy(
    policy: str,
    selected_arm: str,
    arms: List[Any],
    context: Optional[Dict[str, Any]],
    arms_history: Optional[Dict[str, Dict[str, Any]]],
) -> str:
    """Generate human-readable explanation for policy selection."""
    policy_lower = policy.lower()
    
    if policy_lower in ("hash", "deterministic", "consistent"):
        identity = context.get("identity", "unknown") if context else "unknown"
        return (
            f"Selected '{selected_arm}' using deterministic hash policy. "
            f"Same identity ({identity}) will always get same arm for consistency."
        )
    
    elif policy_lower == "random":
        return (
            f"Selected '{selected_arm}' using random weighted allocation. "
            f"Selection is random but weighted by arm weights."
        )
    
    elif policy_lower in ("thompson", "thompson_sampling", "thompsonsampling"):
        if arms_history and selected_arm in arms_history:
            history = arms_history[selected_arm]
            successes = history.get("successes", 0)
            failures = history.get("failures", 0)
            return (
                f"Selected '{selected_arm}' using Thompson Sampling. "
                f"Arm has {successes} successes and {failures} failures. "
                f"Selection based on Beta distribution sampling."
            )
        return (
            f"Selected '{selected_arm}' using Thompson Sampling. "
            f"Arm selected based on Beta-Binomial model."
        )
    
    elif policy_lower in ("ucb", "ucb1"):
        if arms_history and selected_arm in arms_history:
            history = arms_history[selected_arm]
            plays = history.get("plays", 0)
            total_reward = history.get("total_reward", 0.0)
            avg_reward = total_reward / plays if plays > 0 else 0.0
            return (
                f"Selected '{selected_arm}' using Upper Confidence Bound (UCB1). "
                f"Arm has been played {plays} times with average reward {avg_reward:.3f}. "
                f"Selection balances exploration and exploitation."
            )
        return (
            f"Selected '{selected_arm}' using Upper Confidence Bound (UCB1). "
            f"Selection balances exploration (trying new arms) and exploitation (using best known arm)."
        )
    
    elif policy_lower.startswith("epsilon") or policy_lower == "epsilon_greedy":
        epsilon = context.get("epsilon", 0.1) if context else 0.1
        if arms_history and selected_arm in arms_history:
            history = arms_history[selected_arm]
            plays = history.get("plays", 0)
            total_reward = history.get("total_reward", 0.0)
            avg_reward = total_reward / plays if plays > 0 else 0.0
            return (
                f"Selected '{selected_arm}' using Epsilon-Greedy (epsilon={epsilon}). "
                f"Arm has been played {plays} times with average reward {avg_reward:.3f}. "
                f"With probability {epsilon}, explore randomly; otherwise exploit best arm."
            )
        return (
            f"Selected '{selected_arm}' using Epsilon-Greedy (epsilon={epsilon}). "
            f"Selection balances exploration ({epsilon*100}% random) and exploitation ({100-epsilon*100}% best arm)."
        )
    
    else:
        return f"Selected '{selected_arm}' using policy '{policy}'."


def explain_why_not(
    fork_name: str,
    selected_arm: str,
    other_arm: str,
    policy: str,
    arms_history: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Explain why a different arm was NOT selected.
    
    Args:
        fork_name: Name of the fork
        selected_arm: Arm that was selected
        other_arm: Arm that was not selected
        policy: Policy used
        arms_history: Historical performance data
    
    Returns:
        Human-readable explanation
    """
    if policy.lower() in ("hash", "deterministic"):
        return (
            f"'{other_arm}' was not selected because the deterministic hash policy "
            f"assigned this identity to '{selected_arm}' based on consistent hashing."
        )
    
    elif policy.lower() in ("thompson", "thompson_sampling"):
        if arms_history:
            selected_history = arms_history.get(selected_arm, {})
            other_history = arms_history.get(other_arm, {})
            selected_success_rate = (
                selected_history.get("successes", 1) / 
                (selected_history.get("successes", 1) + selected_history.get("failures", 1))
            )
            other_success_rate = (
                other_history.get("successes", 1) / 
                (other_history.get("successes", 1) + other_history.get("failures", 1))
            )
            return (
                f"'{other_arm}' was not selected because Thompson Sampling sampled "
                f"a higher success rate for '{selected_arm}' ({selected_success_rate:.3f}) "
                f"than for '{other_arm}' ({other_success_rate:.3f})."
            )
        return (
            f"'{other_arm}' was not selected because Thompson Sampling "
            f"sampled a higher success probability for '{selected_arm}'."
        )
    
    elif policy.lower() in ("ucb", "ucb1"):
        if arms_history:
            selected_history = arms_history.get(selected_arm, {})
            other_history = arms_history.get(other_arm, {})
            return (
                f"'{other_arm}' was not selected because '{selected_arm}' had a higher UCB value, "
                f"indicating either better performance or more exploration needed."
            )
        return (
            f"'{other_arm}' was not selected because '{selected_arm}' had a higher "
            f"Upper Confidence Bound (UCB) value."
        )
    
    else:
        return (
            f"'{other_arm}' was not selected because '{selected_arm}' was chosen "
            f"by the {policy} policy."
        )

