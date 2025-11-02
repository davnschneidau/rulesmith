"""Example demonstrating MLflow integration with Rulesmith."""

from rulesmith import rule, Rulebook
from rulesmith.mlflow_flavor import log_rulebook_model


@rule(name="check_age", inputs=["age"], outputs=["eligible"])
def check_age(age: int) -> dict:
    """Check if age meets minimum requirement."""
    return {"eligible": age >= 18}


@rule(name="calculate_score", inputs=["age", "income"], outputs=["score"])
def calculate_score(age: int, income: float) -> dict:
    """Calculate credit score based on age and income."""
    score = (age * 10) + (income / 1000)
    return {"score": int(score)}


@rule(name="make_decision", inputs=["eligible", "score"], outputs=["approved"])
def make_decision(eligible: bool, score: float) -> dict:
    """Make approval decision."""
    return {"approved": eligible and score >= 500}


def build_example_rulebook() -> Rulebook:
    """Build example rulebook."""
    rb = Rulebook(name="credit_check", version="1.0.0")
    rb.add_rule(check_age, as_name="age_check")
    rb.add_rule(calculate_score, as_name="scoring")
    rb.add_rule(make_decision, as_name="decision")

    rb.connect("age_check", "scoring")
    rb.connect("scoring", "decision")

    return rb


def example_without_mlflow():
    """Example execution without MLflow."""
    print("=" * 60)
    print("Example: Execution without MLflow")
    print("=" * 60)

    rb = build_example_rulebook()
    result = rb.run({"age": 25, "income": 50000}, enable_mlflow=False)

    print(f"Result: {result}")
    print(f"Approved: {result.get('approved', False)}")
    print()


def example_with_mlflow():
    """Example execution with MLflow (requires MLflow server)."""
    print("=" * 60)
    print("Example: Execution with MLflow")
    print("=" * 60)

    rb = build_example_rulebook()

    # Note: This requires MLflow to be running
    # Uncomment to test with actual MLflow:
    # result = rb.run({"age": 25, "income": 50000}, enable_mlflow=True)
    # print(f"Result: {result}")

    print("MLflow integration is enabled by default")
    print("To use MLflow, ensure MLflow tracking server is running")
    print()


def example_log_to_mlflow():
    """Example logging rulebook as MLflow model."""
    print("=" * 60)
    print("Example: Log Rulebook as MLflow Model")
    print("=" * 60)

    rb = build_example_rulebook()

    # Note: This requires MLflow to be running
    # Uncomment to test:
    # import mlflow
    # with mlflow.start_run():
    #     model = log_rulebook_model(
    #         rulebook=rb,
    #         artifact_path="credit_check_model",
    #         registered_model_name="credit_check",
    #     )
    #     print(f"Logged model: {model.model_uri}")

    print("To log rulebook as MLflow model:")
    print("  1. Start MLflow tracking server")
    print("  2. Use log_rulebook_model() function")
    print("  3. Model will be registered and versioned")
    print()


if __name__ == "__main__":
    example_without_mlflow()
    example_with_mlflow()
    example_log_to_mlflow()

