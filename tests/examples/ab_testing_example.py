"""Examples demonstrating A/B testing capabilities."""

from rulesmith import rule, Rulebook
from rulesmith.ab.policies import EpsilonGreedyPolicy, ThompsonSamplingPolicy, UCBPolicy
from rulesmith.io.ser import ABArm


@rule(name="prepare_test", inputs=["user_id"], outputs=["test_input"])
def prepare_test(user_id: str) -> dict:
    """Prepare input for A/B test."""
    return {"test_input": f"test_data_{user_id}"}


@rule(name="variant_a_logic", inputs=["test_input"], outputs=["result"])
def variant_a_logic(test_input: str) -> dict:
    """Logic for variant A."""
    return {"result": f"A: {test_input}", "variant": "A"}


@rule(name="variant_b_logic", inputs=["test_input"], outputs=["result"])
def variant_b_logic(test_input: str) -> dict:
    """Logic for variant B."""
    return {"result": f"B: {test_input}", "variant": "B"}


def example_hash_policy():
    """Example with hash-based deterministic policy."""
    print("=" * 60)
    print("Example: Hash-Based A/B Testing")
    print("=" * 60)

    rb = Rulebook(name="hash_ab_test", version="1.0.0")

    # Create A/B arms
    arms = [
        ABArm(node="variant_a", weight=0.5),
        ABArm(node="variant_b", weight=0.5),
    ]

    # Add fork with hash policy
    rb.add_fork("ab_test", arms, policy="hash")

    # Add variant nodes
    rb.add_rule(variant_a_logic, as_name="variant_a")
    rb.add_rule(variant_b_logic, as_name="variant_b")

    # Connect (in practice, you'd route based on fork selection)
    rb.connect("ab_test", "variant_a")  # Simplified - actual routing needs enhancement

    print("Hash-based A/B test rulebook:")
    print(f"  - Name: {rb.name}")
    print("  - Policy: hash (deterministic)")
    print("  - Arms: variant_a (50%), variant_b (50%)")
    print()
    print("Features:")
    print("  - Deterministic: same user_id always gets same variant")
    print("  - Consistent: no variation across runs")
    print()


def example_random_policy():
    """Example with random policy."""
    print("=" * 60)
    print("Example: Random A/B Testing")
    print("=" * 60)

    rb = Rulebook(name="random_ab_test", version="1.0.0")

    arms = [
        ABArm(node="variant_a", weight=0.3),
        ABArm(node="variant_b", weight=0.7),
    ]

    rb.add_fork("ab_test", arms, policy="random")

    print("Random A/B test rulebook:")
    print("  - Policy: random")
    print("  - Arms: variant_a (30%), variant_b (70%)")
    print()
    print("Features:")
    print("  - Random allocation per request")
    print("  - Weighted by arm weights")
    print()


def example_thompson_sampling():
    """Example with Thompson Sampling bandit."""
    print("=" * 60)
    print("Example: Thompson Sampling Bandit")
    print("=" * 60)

    # Create policy with history
    policy = ThompsonSamplingPolicy()

    rb = Rulebook(name="thompson_ab_test", version="1.0.0")

    arms = [
        ABArm(node="variant_a", weight=1.0),
        ABArm(node="variant_b", weight=1.0),
    ]

    rb.add_fork("ab_test", arms, policy_instance=policy)

    print("Thompson Sampling bandit:")
    print("  - Policy: thompson_sampling")
    print("  - Adaptive: learns from outcomes")
    print()
    print("Usage:")
    print("  1. Policy tracks successes/failures per arm")
    print("  2. Samples from Beta distribution")
    print("  3. Selects arm with highest sample")
    print("  4. Update with policy.update_history(arm_name, success=True)")
    print()


def example_ucb_policy():
    """Example with UCB1 bandit."""
    print("=" * 60)
    print("Example: UCB1 Bandit")
    print("=" * 60)

    policy = UCBPolicy()

    rb = Rulebook(name="ucb_ab_test", version="1.0.0")

    arms = [
        ABArm(node="variant_a", weight=1.0),
        ABArm(node="variant_b", weight=1.0),
        ABArm(node="variant_c", weight=1.0),
    ]

    rb.add_fork("ab_test", arms, policy_instance=policy)

    print("UCB1 bandit:")
    print("  - Policy: ucb1")
    print("  - Exploration/Exploitation balance")
    print()
    print("Usage:")
    print("  1. Tracks plays and rewards per arm")
    print("  2. Calculates UCB = mean + confidence_bound")
    print("  3. Selects arm with highest UCB")
    print("  4. Update with policy.update_history(arm_name, reward=0.9)")
    print()


def example_epsilon_greedy():
    """Example with epsilon-greedy bandit."""
    print("=" * 60)
    print("Example: Epsilon-Greedy Bandit")
    print("=" * 60)

    policy = EpsilonGreedyPolicy(epsilon=0.1)

    rb = Rulebook(name="epsilon_ab_test", version="1.0.0")

    arms = [
        ABArm(node="variant_a", weight=1.0),
        ABArm(node="variant_b", weight=1.0),
    ]

    rb.add_fork("ab_test", arms, policy_instance=policy)

    print("Epsilon-Greedy bandit:")
    print("  - Policy: epsilon_greedy")
    print("  - Epsilon: 0.1 (10% exploration)")
    print()
    print("Usage:")
    print("  - 10% of time: explore (random)")
    print("  - 90% of time: exploit (best arm)")
    print("  - Update with policy.update_history(arm_name, reward=0.8)")
    print()


def example_multi_armed_bandit():
    """Example of multi-armed bandit A/B testing workflow."""
    print("=" * 60)
    print("Example: Multi-Armed Bandit Workflow")
    print("=" * 60)

    policy = ThompsonSamplingPolicy()

    rb = Rulebook(name="bandit_workflow", version="1.0.0")
    rb.add_rule(prepare_test, as_name="prepare")

    arms = [
        ABArm(node="variant_a", weight=1.0),
        ABArm(node="variant_b", weight=1.0),
    ]

    rb.add_fork("ab_test", arms, policy_instance=policy)
    rb.add_rule(variant_a_logic, as_name="variant_a")
    rb.add_rule(variant_b_logic, as_name="variant_b")

    print("Complete bandit workflow:")
    print("  1. Prepare test input")
    print("  2. Fork to select variant (Thompson Sampling)")
    print("  3. Execute variant logic")
    print("  4. Measure outcome")
    print("  5. Update bandit history")
    print("  6. Next request uses updated policy")
    print()
    print("Benefits:")
    print("  - Automatic traffic allocation")
    print("  - Learning from outcomes")
    print("  - Optimal exploitation over time")
    print()


if __name__ == "__main__":
    example_hash_policy()
    example_random_policy()
    example_thompson_sampling()
    example_ucb_policy()
    example_epsilon_greedy()
    example_multi_armed_bandit()

