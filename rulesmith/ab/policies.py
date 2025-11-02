"""A/B testing policies including bandits."""

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from rulesmith.io.ser import ABArm


class TrafficPolicy(ABC):
    """Abstract base class for traffic allocation policies."""

    @abstractmethod
    def select_arm(self, arms: List[ABArm], context: Dict) -> ABArm:
        """
        Select an arm based on the policy.

        Args:
            arms: List of available arms
            context: Context dictionary (identity, seed, history, etc.)

        Returns:
            Selected ABArm
        """
        pass


class HashPolicy(TrafficPolicy):
    """Deterministic hash-based policy."""

    def __init__(self):
        self.name = "hash"

    def select_arm(self, arms: List[ABArm], context: Dict) -> ABArm:
        """Select arm based on deterministic hash of identity."""
        import hashlib

        identity = context.get("identity") or str(context.get("seed", random.random()))
        hash_val = int(hashlib.md5(identity.encode("utf-8")).hexdigest(), 16)
        normalized = (hash_val % 10000) / 10000.0

        cumulative = 0.0
        for arm in arms:
            cumulative += arm.weight
            if normalized <= cumulative:
                return arm

        return arms[-1]


class RandomPolicy(TrafficPolicy):
    """Random weighted policy."""

    def __init__(self, seed: Optional[int] = None):
        self.name = "random"
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def select_arm(self, arms: List[ABArm], context: Dict) -> ABArm:
        """Select arm randomly weighted by arm weights."""
        weights = [arm.weight for arm in arms]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.choice(arms)

        r = random.random() * total_weight
        cumulative = 0.0
        for arm in arms:
            cumulative += arm.weight
            if r <= cumulative:
                return arm

        return arms[-1]


class ThompsonSamplingPolicy(TrafficPolicy):
    """Thompson Sampling multi-armed bandit policy."""

    def __init__(self, arms_history: Optional[Dict[str, Dict[str, int]]] = None):
        """
        Initialize Thompson Sampling policy.

        Args:
            arms_history: Dictionary mapping arm names to {"successes": int, "failures": int}
        """
        self.name = "thompson_sampling"
        self.arms_history = arms_history or {}

    def select_arm(self, arms: List[ABArm], context: Dict) -> ABArm:
        """
        Select arm using Thompson Sampling.

        Assumes Beta-Binomial model where each arm's success rate
        follows a Beta distribution.
        """
        try:
            import numpy as np

            use_numpy = True
        except ImportError:
            # Fallback to Python's random.gammavariate for Beta distribution
            use_numpy = False
            import random as py_random

        # Initialize history for arms without it
        for arm in arms:
            if arm.node not in self.arms_history:
                self.arms_history[arm.node] = {"successes": 1, "failures": 1}  # Beta(1,1) prior

        # Sample from Beta distribution for each arm
        samples = {}
        for arm in arms:
            history = self.arms_history[arm.node]
            alpha = history["successes"]
            beta = history["failures"]

            if use_numpy:
                samples[arm.node] = np.random.beta(alpha, beta)
            else:
                # Beta distribution using Gamma: Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
                # Simplified approximation using gammavariate
                gamma_a = py_random.gammavariate(alpha, 1)
                gamma_b = py_random.gammavariate(beta, 1)
                samples[arm.node] = gamma_a / (gamma_a + gamma_b) if (gamma_a + gamma_b) > 0 else 0.5

        # Select arm with highest sample
        best_arm_name = max(samples, key=samples.get)
        for arm in arms:
            if arm.node == best_arm_name:
                return arm

        # Fallback
        return arms[0]

    def update_history(self, arm_name: str, success: bool) -> None:
        """Update arm history after observing outcome."""
        if arm_name not in self.arms_history:
            self.arms_history[arm_name] = {"successes": 1, "failures": 1}

        if success:
            self.arms_history[arm_name]["successes"] += 1
        else:
            self.arms_history[arm_name]["failures"] += 1


class UCBPolicy(TrafficPolicy):
    """Upper Confidence Bound (UCB1) multi-armed bandit policy."""

    def __init__(self, arms_history: Optional[Dict[str, Dict[str, int]]] = None):
        """
        Initialize UCB1 policy.

        Args:
            arms_history: Dictionary mapping arm names to {
                "plays": int, "rewards": float, "total_reward": float
            }
        """
        self.name = "ucb1"
        self.arms_history = arms_history or {}
        self.total_plays = sum(
            h.get("plays", 0) for h in self.arms_history.values()
        ) or 1  # Avoid division by zero

    def select_arm(self, arms: List[ABArm], context: Dict) -> ABArm:
        """
        Select arm using UCB1 algorithm.

        UCB1 = mean_reward + c * sqrt(ln(total_plays) / arm_plays)
        where c is typically 2.0 (exploration constant)
        """
        import math

        exploration_constant = 2.0

        # Initialize history for arms without it
        for arm in arms:
            if arm.node not in self.arms_history:
                self.arms_history[arm.node] = {
                    "plays": 0,
                    "rewards": [],
                    "total_reward": 0.0,
                }

        # Calculate UCB for each arm
        ucbs = {}
        for arm in arms:
            history = self.arms_history[arm.node]
            plays = history["plays"] or 1  # Avoid division by zero

            if plays == 0:
                # Play unplayed arms first
                mean_reward = 0.0
                ucb_value = float("inf")
            else:
                mean_reward = history["total_reward"] / plays
                # UCB1 formula
                ucb_value = mean_reward + exploration_constant * math.sqrt(
                    math.log(self.total_plays) / plays
                )

            ucbs[arm.node] = ucb_value

        # Select arm with highest UCB
        best_arm_name = max(ucbs, key=ucbs.get)
        for arm in arms:
            if arm.node == best_arm_name:
                return arm

        # Fallback
        return arms[0]

    def update_history(self, arm_name: str, reward: float) -> None:
        """Update arm history after observing reward."""
        if arm_name not in self.arms_history:
            self.arms_history[arm_name] = {"plays": 0, "rewards": [], "total_reward": 0.0}

        self.arms_history[arm_name]["plays"] += 1
        self.arms_history[arm_name]["rewards"].append(reward)
        self.arms_history[arm_name]["total_reward"] += reward
        self.total_plays += 1


class EpsilonGreedyPolicy(TrafficPolicy):
    """Epsilon-greedy multi-armed bandit policy."""

    def __init__(
        self,
        epsilon: float = 0.1,
        arms_history: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize epsilon-greedy policy.

        Args:
            epsilon: Exploration probability (0.0-1.0)
            arms_history: Dictionary mapping arm names to {
                "plays": int, "total_reward": float
            }
        """
        self.name = "epsilon_greedy"
        self.epsilon = epsilon
        self.arms_history = arms_history or {}

    def select_arm(self, arms: List[ABArm], context: Dict) -> ABArm:
        """
        Select arm using epsilon-greedy strategy.

        With probability epsilon, explore (random).
        With probability 1-epsilon, exploit (best arm).
        """
        # Initialize history for arms without it
        for arm in arms:
            if arm.node not in self.arms_history:
                self.arms_history[arm.node] = {"plays": 0, "total_reward": 0.0}

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(arms)

        # Exploit: select arm with highest average reward
        avg_rewards = {}
        for arm in arms:
            history = self.arms_history[arm.node]
            plays = history["plays"] or 1
            avg_rewards[arm.node] = history["total_reward"] / plays

        best_arm_name = max(avg_rewards, key=avg_rewards.get)
        for arm in arms:
            if arm.node == best_arm_name:
                return arm

        return arms[0]

    def update_history(self, arm_name: str, reward: float) -> None:
        """Update arm history after observing reward."""
        if arm_name not in self.arms_history:
            self.arms_history[arm_name] = {"plays": 0, "total_reward": 0.0}

        self.arms_history[arm_name]["plays"] += 1
        self.arms_history[arm_name]["total_reward"] += reward

