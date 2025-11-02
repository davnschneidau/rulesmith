"""Traffic management for A/B testing."""

import hashlib
import random
from typing import List, Optional

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
) -> ABArm:
    """
    Select an A/B arm based on policy.

    Args:
        arms: List of A/B arms
        identity: Optional identity for deterministic hashing
        policy: "hash" for deterministic or "random" for random selection

    Returns:
        Selected ABArm
    """
    if not arms:
        raise ValueError("No arms provided")

    if policy == "random":
        # Random selection weighted by arm weights
        weights = [arm.weight for arm in arms]
        total_weight = sum(weights)
        if total_weight == 0:
            # Equal probability if all weights are 0
            return random.choice(arms)

        r = random.random() * total_weight
        cumulative = 0
        for arm in arms:
            cumulative += arm.weight
            if r <= cumulative:
                return arm

        # Fallback to last arm
        return arms[-1]

    elif policy == "hash":
        # Deterministic hash-based selection
        if identity is None:
            # Fallback to random if no identity
            return pick_arm(arms, identity, policy="random")

        # Hash the identity to get a value in [0, 1)
        hash_val = int(hashlib.md5(identity.encode("utf-8")).hexdigest(), 16)
        normalized = (hash_val % 10000) / 10000.0

        # Select arm based on cumulative weights
        cumulative = 0
        for arm in arms:
            cumulative += arm.weight
            if normalized <= cumulative:
                return arm

        # Fallback to last arm
        return arms[-1]

    else:
        raise ValueError(f"Unknown policy: {policy}")

