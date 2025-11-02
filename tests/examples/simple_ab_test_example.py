"""Simple A/B testing example showing the simplified API."""

from rulesmith import rule, Rulebook


@rule(name="score_user", inputs=["user_id"], outputs=["score"])
def score_user(user_id: str) -> dict:
    """Score a user."""
    return {"score": len(user_id) * 10}


@rule(name="variant_control", inputs=["user_id"], outputs=["recommendation"])
def variant_control(user_id: str) -> dict:
    """Control variant - original algorithm."""
    return {"recommendation": "original_algorithm", "variant": "control"}


@rule(name="variant_new", inputs=["user_id"], outputs=["recommendation"])
def variant_new(user_id: str) -> dict:
    """New variant - improved algorithm."""
    return {"recommendation": "improved_algorithm", "variant": "new"}


def example_simple_ab_test():
    """Simplest possible A/B test."""
    print("=" * 60)
    print("Simple A/B Test Example")
    print("=" * 60)
    
    rb = Rulebook(name="simple_ab", version="1.0.0")
    
    # Add a 50/50 split - super simple!
    rb.add_split("experiment", {"control": 0.5, "new": 0.5})
    
    # Add the variants
    rb.add_rule(variant_control, as_name="control")
    rb.add_rule(variant_new, as_name="new")
    
    # Connect - the split automatically routes to the selected variant
    rb.connect("experiment", "control")
    rb.connect("experiment", "new")
    
    # Run it
    result = rb.run({"user_id": "user123"}, enable_mlflow=False)
    
    print(f"Selected variant: {result.get('selected_variant', 'unknown')}")
    print(f"Recommendation: {result.get('recommendation', 'none')}")
    print()
    print("That's it! The split node automatically:")
    print("  - Selects a variant (50/50 in this case)")
    print("  - Routes to the correct variant")
    print("  - Shows which variant was selected in the result")


def example_weighted_split():
    """Example with different weights."""
    print("\n" + "=" * 60)
    print("Weighted A/B Test")
    print("=" * 60)
    
    rb = Rulebook(name="weighted_ab", version="1.0.0")
    
    # 70% to treatment, 30% to control
    rb.add_split("test", {"control": 0.3, "treatment": 0.7})
    
    rb.add_rule(variant_control, as_name="control")
    rb.add_rule(variant_new, as_name="treatment")
    
    rb.connect("test", "control")
    rb.connect("test", "treatment")
    
    print("70/30 split between control and treatment")
    print("Use case: Gradually roll out new features")


def example_three_way_test():
    """Example with multiple variants."""
    print("\n" + "=" * 60)
    print("Three-Way A/B/C Test")
    print("=" * 60)
    
    rb = Rulebook(name="three_way", version="1.0.0")
    
    # Three equal variants
    rb.add_split("multi_test", {"variant_a": 0.33, "variant_b": 0.33, "variant_c": 0.34})
    
    print("Test three different variants at once")
    print("Use case: Comparing multiple approaches")


def example_adaptive_bandit():
    """Example with adaptive bandit policy."""
    print("\n" + "=" * 60)
    print("Adaptive Bandit (Thompson Sampling)")
    print("=" * 60)
    
    rb = Rulebook(name="bandit_test", version="1.0.0")
    
    # Use Thompson Sampling to automatically optimize
    rb.add_split("bandit", {"variant_a": 1.0, "variant_b": 1.0}, policy="thompson")
    
    print("Adaptive A/B test:")
    print("  - Starts equal (50/50)")
    print("  - Learns which variant performs better")
    print("  - Automatically sends more traffic to better variant")
    print("  - Optimal balance of exploration vs exploitation")


if __name__ == "__main__":
    example_simple_ab_test()
    example_weighted_split()
    example_three_way_test()
    example_adaptive_bandit()

