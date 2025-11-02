"""Example credit decision rulebook."""

from rulesmith import rule, Rulebook
from rulesmith.io.ser import ABArm


@rule(name="check_eligibility", inputs=["age", "income"], outputs=["eligible"])
def check_eligibility(age: int, income: float) -> dict:
    """Check if applicant meets basic eligibility criteria."""
    return {"eligible": age >= 18 and income > 30000}


@rule(name="calculate_score", inputs=["age", "income", "credit_score"], outputs=["score"])
def calculate_score(age: int, income: float, credit_score: int) -> dict:
    """Calculate credit score."""
    base_score = credit_score
    income_bonus = min(income / 10000, 50)  # Cap at 50 points
    age_bonus = min(age / 2, 30)  # Cap at 30 points
    total_score = base_score + income_bonus + age_bonus
    return {"score": int(total_score)}


@rule(name="make_decision", inputs=["eligible", "score"], outputs=["approved", "reason"])
def make_decision(eligible: bool, score: float) -> dict:
    """Make final credit decision."""
    if not eligible:
        return {"approved": False, "reason": "Does not meet eligibility criteria"}
    elif score >= 700:
        return {"approved": True, "reason": "High credit score"}
    elif score >= 600:
        return {"approved": True, "reason": "Moderate credit score"}
    else:
        return {"approved": False, "reason": "Low credit score"}


def build_credit_rulebook() -> Rulebook:
    """Build the credit decision rulebook."""
    rb = Rulebook(name="credit_decision", version="1.0.0")

    # Add rules
    rb.add_rule(check_eligibility, as_name="eligibility_check")
    rb.add_rule(calculate_score, as_name="scoring")
    rb.add_rule(make_decision, as_name="decision")

    # Connect nodes
    rb.connect("eligibility_check", "scoring")
    rb.connect("scoring", "decision")

    return rb


if __name__ == "__main__":
    # Example execution
    rulebook = build_credit_rulebook()

    # Test case 1: Approved applicant
    result1 = rulebook.run({
        "age": 30,
        "income": 75000,
        "credit_score": 720,
    })
    print("Test 1:", result1)
    assert result1["approved"] is True

    # Test case 2: Rejected applicant (low score)
    result2 = rulebook.run({
        "age": 25,
        "income": 40000,
        "credit_score": 550,
    })
    print("Test 2:", result2)
    assert result2["approved"] is False

    # Test case 3: Rejected applicant (not eligible)
    result3 = rulebook.run({
        "age": 17,
        "income": 50000,
        "credit_score": 800,
    })
    print("Test 3:", result3)
    assert result3["approved"] is False

    print("All tests passed!")

