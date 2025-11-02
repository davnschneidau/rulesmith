"""Examples demonstrating phases 8-11 features."""

# Phase 8: Evaluation
from rulesmith.evaluation import evaluate_rulebook, accuracy_scorer
from rulesmith.evaluation.datasets import load_dataset

# Phase 9: Reliability
from rulesmith.reliability import RateLimiter, RetryConfig, retry, CircuitBreaker

# Phase 10: Security
from rulesmith.security import RedactionConfig, redact_dict, validate_no_secrets

# Phase 11: Schemas
from rulesmith.io.schemas import FieldSchema, SchemaContract, validate_with_schema


def example_evaluation():
    """Example evaluation system."""
    print("=" * 60)
    print("Phase 8: Evaluation System")
    print("=" * 60)

    print("Evaluation Features:")
    print("  - MLflow.evaluate() integration")
    print("  - Custom scorers (accuracy, precision, recall, F1)")
    print("  - Business KPI scorers")
    print("  - LLM-as-judge evaluation")
    print("  - Dataset loading and slicing")
    print()
    print("Usage:")
    print("  result = evaluate_rulebook('models:/rulebook/1', data)")
    print("  score = accuracy_scorer(data, predictions, targets='label')")
    print()


def example_reliability():
    """Example reliability features."""
    print("=" * 60)
    print("Phase 9: Reliability & Performance")
    print("=" * 60)

    print("Reliability Features:")
    print("  - Retry with exponential backoff and jitter")
    print("  - Rate limiting with token buckets")
    print("  - Circuit breakers for fault tolerance")
    print("  - Shadow mode execution")
    print()
    print("Usage:")
    print("  @retry(config=RetryConfig(max_retries=3))")
    print("  def flaky_api_call():")
    print("      ...")
    print()
    print("  limiter = RateLimiter(requests_per_second=10)")
    print("  with limiter:")
    print("      make_request()")
    print()
    print("  breaker = CircuitBreaker(failure_threshold=5)")
    print("  result = breaker.call(risky_function)")
    print()


def example_security():
    """Example security features."""
    print("=" * 60)
    print("Phase 10: Security & Privacy")
    print("=" * 60)

    print("Security Features:")
    print("  - PII redaction (email, phone, SSN)")
    print("  - Field suppression")
    print("  - Secret detection")
    print("  - Secret validation")
    print()
    print("Usage:")
    print("  config = RedactionConfig(redact_emails=True)")
    print("  safe_data = redact_dict(data, config)")
    print()
    print("  is_valid, issues = validate_no_secrets(data)")
    print()


def example_schemas():
    """Example schema validation."""
    print("=" * 60)
    print("Phase 11: Schema & Contracts")
    print("=" * 60)

    schema = SchemaContract(
        name="user_schema",
        fields=[
            FieldSchema(name="age", type="int", min_value=0, max_value=150),
            FieldSchema(name="email", type="str", required=True),
        ],
    )

    print("Schema Features:")
    print("  - Type validation")
    print("  - Range constraints")
    print("  - Length constraints")
    print("  - Enum validation")
    print("  - Schema inference")
    print()
    print("Usage:")
    print("  is_valid, errors = validate_with_schema(data, schema)")
    print("  schema = infer_schema(sample_data)")
    print()


if __name__ == "__main__":
    example_evaluation()
    example_reliability()
    example_security()
    example_schemas()

