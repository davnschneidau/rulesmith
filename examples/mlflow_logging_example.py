"""Example: Comprehensive MLflow logging for rulebook executions."""

from rulesmith import rule, Rulebook, MLflowLogger

# Define rules
@rule(name="check_age", inputs=["age"], outputs=["eligible"])
def check_age(age: int) -> dict:
    return {"eligible": age >= 18}

# Define guardrail metrics
@rule(name="check_pii", inputs=["output"], outputs=["has_pii"])
def check_pii(output: str) -> dict:
    """Check if output contains PII."""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    has_email = bool(re.search(email_pattern, output))
    return {"has_pii": has_email, "name": "check_pii"}

@rule(name="check_toxicity", inputs=["output"], outputs=["is_toxic"])
def check_toxicity(output: str) -> dict:
    """Check if output contains toxic content."""
    toxic_words = ["hate", "violence"]
    return {"is_toxic": any(word in output.lower() for word in toxic_words), "name": "check_toxicity"}

# Build rulebook
rb = Rulebook(name="credit_decision", version="1.0.0")
rb.add_rule(check_age, as_name="age_check")

# Add LLM node with metrics (guardrails)
rb.add_llm(
    "llm_node",
    provider="openai",
    model_name="gpt-4",
    metrics=[check_pii, check_toxicity]  # Guardrails as metrics
)

rb.connect("age_check", "llm_node")

# Example 1: Single decision logging
logger = MLflowLogger(
    experiment_name="rulesmith/credit_decision",
    sample_rate=1.0,  # Log all for testing
    enable_artifacts=True,
    redact_pii=True,
)

result = rb.run(
    {"age": 25, "prompt": "Hello"},
    mlflow_logger=logger,
    enable_mlflow=True,
)

# Example 2: Batch/replay with parent run
logger = MLflowLogger(
    experiment_name="rulesmith/credit_decision",
    parent_run_name="credit@1.0 replay 2025-01-15",
    sample_rate=0.01,  # Sample 1% in production
)

logger.start_parent_run(tags={
    "rulebook": "credit_decision",
    "rulebook_version": "1.0.0",
    "stage": "staging",
    "alias": "champion",
    "run_kind": "replay",
})

# Log multiple decisions
for i, case in enumerate(test_cases):
    result = rb.run(
        case,
        context=MLflowRunContext(
            rulebook_spec=rb.to_spec(),
            tags={"case_id": f"case_{i}"},
        ),
        mlflow_logger=logger,
    )
    
    # Decision is automatically logged as nested run

# Log aggregate metrics
logger.log_aggregate_metrics({
    "decisions_total": len(test_cases),
    "errors_total": error_count,
    "latency_p50_ms": p50_latency,
    "latency_p95_ms": p95_latency,
    "cost_usd_sum": total_cost,
})

logger.end_parent_run()

