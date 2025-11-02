"""Traffic management for A/B testing."""

import hashlib
import random
from typing import Any, Dict, List, Optional

from rulesmith.ab.policies import (
    EpsilonGreedyPolicy,
    HashPolicy,
    RandomPolicy,
    ThompsonSamplingPolicy,
    TrafficPolicy,
    UCBPolicy,
)
from rulesmith.io.ser import ABArm


def hash_bucket(key: str, buckets: int) -> int:
    """
    Deterministic hash-based bucketing.

    Args:
        key: Key to hash (e.g., user_id, session_id)
        buckets: Number of buckets

    Returns:
        Bucket index (0 to buckets-1)
    """
    hash_val = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return hash_val % buckets


def pick_arm(
    arms: List[ABArm],
    identity: Optional[str] = None,
    policy: str = "hash",
    policy_instance: Optional[TrafficPolicy] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ABArm:
    """
    Select an A/B arm based on policy.

    Args:
        arms: List of A/B arms
        identity: Optional identity for deterministic hashing
        policy: Policy name ("hash", "random", "thompson_sampling", "ucb1", "epsilon_greedy")
        policy_instance: Optional TrafficPolicy instance (if provided, overrides policy string)
        context: Optional context dictionary for policies

    Returns:
        Selected ABArm
    """
    if not arms:
        raise ValueError("No arms provided")

    # Use provided policy instance if available
    if policy_instance is not None:
        ctx = context or {}
        if identity:
            ctx["identity"] = identity
        return policy_instance.select_arm(arms, ctx)

    # Create policy instance based on policy string
    ctx = context or {}
    if identity:
        ctx["identity"] = identity

    if policy == "hash" or policy == "deterministic":
        policy_instance = HashPolicy()
    elif policy == "random":
        seed = ctx.get("seed")
        policy_instance = RandomPolicy(seed=seed)
    elif policy == "thompson_sampling":
        arms_history = ctx.get("arms_history")
        policy_instance = ThompsonSamplingPolicy(arms_history=arms_history)
    elif policy == "ucb1" or policy == "ucb":
        arms_history = ctx.get("arms_history")
        policy_instance = UCBPolicy(arms_history=arms_history)
    elif policy == "epsilon_greedy" or policy.startswith("epsilon"):
        epsilon = ctx.get("epsilon", 0.1)
        arms_history = ctx.get("arms_history")
        policy_instance = EpsilonGreedyPolicy(epsilon=epsilon, arms_history=arms_history)
    else:
        raise ValueError(
            f"Unknown policy: {policy}. "
            "Supported: 'hash', 'random', 'thompson_sampling', 'ucb1', 'epsilon_greedy'"
        )

    return policy_instance.select_arm(arms, ctx)

