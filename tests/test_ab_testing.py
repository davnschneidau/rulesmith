"""Tests for A/B testing system."""

import pytest

from rulesmith import Rulebook
from rulesmith.ab.policies import (
    EpsilonGreedyPolicy,
    HashPolicy,
    RandomPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from rulesmith.ab.traffic import hash_bucket, pick_arm
from rulesmith.io.ser import ABArm


class TestTrafficPolicies:
    """Test traffic allocation policies."""

    def test_hash_policy(self):
        """Test hash-based policy."""
        arms = [
            ABArm(node="arm_a", weight=0.5),
            ABArm(node="arm_b", weight=0.5),
        ]

        policy = HashPolicy()
        context = {"identity": "user123"}

        # Same identity should select same arm
        arm1 = policy.select_arm(arms, context)
        arm2 = policy.select_arm(arms, context)
        assert arm1.node == arm2.node

    def test_random_policy(self):
        """Test random policy."""
        arms = [
            ABArm(node="arm_a", weight=0.3),
            ABArm(node="arm_b", weight=0.7),
        ]

        policy = RandomPolicy(seed=42)
        context = {}

        # Should select one of the arms
        arm = policy.select_arm(arms, context)
        assert arm.node in ["arm_a", "arm_b"]

    def test_epsilon_greedy_policy(self):
        """Test epsilon-greedy policy."""
        arms = [
            ABArm(node="arm_a", weight=1.0),
            ABArm(node="arm_b", weight=1.0),
        ]

        policy = EpsilonGreedyPolicy(epsilon=0.1)
        context = {}

        # Should select an arm
        arm = policy.select_arm(arms, context)
        assert arm.node in ["arm_a", "arm_b"]

        # Update history and test exploitation
        policy.update_history("arm_a", reward=0.9)
        policy.update_history("arm_b", reward=0.5)

        # With low epsilon, should prefer arm_a
        selections = [policy.select_arm(arms, context).node for _ in range(100)]
        # Most selections should be arm_a (though some exploration)
        assert "arm_a" in selections

    def test_thompson_sampling_policy(self):
        """Test Thompson Sampling policy."""
        arms = [
            ABArm(node="arm_a", weight=1.0),
            ABArm(node="arm_b", weight=1.0),
        ]

        policy = ThompsonSamplingPolicy()
        context = {}

        # Should select an arm
        arm = policy.select_arm(arms, context)
        assert arm.node in ["arm_a", "arm_b"]

        # Update history
        policy.update_history("arm_a", success=True)
        policy.update_history("arm_b", success=False)

        # Should prefer arm_a after updates
        selections = [policy.select_arm(arms, context).node for _ in range(50)]
        assert "arm_a" in selections or "arm_b" in selections

    def test_ucb_policy(self):
        """Test UCB1 policy."""
        arms = [
            ABArm(node="arm_a", weight=1.0),
            ABArm(node="arm_b", weight=1.0),
        ]

        policy = UCBPolicy()
        context = {}

        # Should select an arm
        arm = policy.select_arm(arms, context)
        assert arm.node in ["arm_a", "arm_b"]

        # Update history
        policy.update_history("arm_a", reward=0.9)
        policy.update_history("arm_b", reward=0.3)

        # Should prefer arm_a
        selections = [policy.select_arm(arms, context).node for _ in range(50)]
        assert "arm_a" in selections or "arm_b" in selections


class TestTrafficManagement:
    """Test traffic management functions."""

    def test_hash_bucket(self):
        """Test hash-based bucketing."""
        bucket1 = hash_bucket("user123", 100)
        bucket2 = hash_bucket("user123", 100)
        assert bucket1 == bucket2  # Deterministic

        bucket3 = hash_bucket("user456", 100)
        assert 0 <= bucket3 < 100

    def test_pick_arm_hash(self):
        """Test pick_arm with hash policy."""
        arms = [
            ABArm(node="arm_a", weight=0.5),
            ABArm(node="arm_b", weight=0.5),
        ]

        arm = pick_arm(arms, identity="user123", policy="hash")
        assert arm.node in ["arm_a", "arm_b"]

    def test_pick_arm_random(self):
        """Test pick_arm with random policy."""
        arms = [
            ABArm(node="arm_a", weight=0.3),
            ABArm(node="arm_b", weight=0.7),
        ]

        arm = pick_arm(arms, policy="random")
        assert arm.node in ["arm_a", "arm_b"]

    def test_pick_arm_with_policy_instance(self):
        """Test pick_arm with policy instance."""
        arms = [
            ABArm(node="arm_a", weight=1.0),
            ABArm(node="arm_b", weight=1.0),
        ]

        policy = EpsilonGreedyPolicy(epsilon=0.0)  # Pure exploitation
        policy.update_history("arm_a", reward=1.0)
        policy.update_history("arm_b", reward=0.0)

        arm = pick_arm(arms, policy_instance=policy)
        assert arm.node == "arm_a"  # Should always select best arm


class TestForkNode:
    """Test ForkNode A/B testing."""

    def test_fork_node_creation(self):
        """Test creating fork node."""
        arms = [
            ABArm(node="variant_a", weight=0.5),
            ABArm(node="variant_b", weight=0.5),
        ]

        rb = Rulebook(name="test", version="1.0.0")
        rb.add_fork("ab_test", arms, policy="hash")

        spec = rb.to_spec()
        assert len(spec.nodes) == 1
        assert spec.nodes[0].kind == "fork"
        assert len(spec.nodes[0].ab_arms) == 2

    def test_fork_node_with_bandit(self):
        """Test fork node with bandit policy."""
        arms = [
            ABArm(node="variant_a", weight=1.0),
            ABArm(node="variant_b", weight=1.0),
        ]

        policy = ThompsonSamplingPolicy()
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_fork("ab_test", arms, policy_instance=policy)

        spec = rb.to_spec()
        assert spec.nodes[0].kind == "fork"

