# Rulesmith Quickstart Guide

## Installation

```bash
pip install rulesmith
```

## Basic Usage

### 1. Define Rules

```python
from rulesmith import rule

@rule(name="check_age", inputs=["age"], outputs=["eligible"])
def check_age(age: int) -> dict:
    """Check if age meets requirement."""
    return {"eligible": age >= 18}
```

### 2. Build a Rulebook

```python
from rulesmith import rule, Rulebook

@rule(name="calculate_score", inputs=["age", "income"], outputs=["score"])
def calculate_score(age: int, income: float) -> dict:
    return {"score": (age * 10) + (income / 1000)}

# Build rulebook
rb = Rulebook(name="credit_check", version="1.0.0")
rb.add_rule(check_age, as_name="age_check")
rb.add_rule(calculate_score, as_name="scoring")
rb.connect("age_check", "scoring")
```

### 3. Execute

```python
result = rb.run({"age": 25, "income": 50000})
print(result)  # {"eligible": True, "score": 550.0}
```

## Advanced Features

### A/B Testing

```python
# Simple 50/50 split
rb.add_split("experiment", {"variant_a": 0.5, "variant_b": 0.5})

# Adaptive bandit (learns which variant is better)
rb.add_split("bandit", {"a": 1.0, "b": 1.0}, policy="thompson")
```

### Guardrails (Rule-based Metrics)

Guardrails are defined as rule functions that evaluate node outputs:

```python
from rulesmith import rule

@rule(name="check_pii", inputs=["output"], outputs=["has_pii"])
def check_pii(output: str) -> dict:
    return {"has_pii": "@" in output}

rb.add_llm("gpt4", provider="openai", model_name="gpt-4", metrics=[check_pii])
```

### MLflow Integration

MLflow logging is enabled by default. Simply run your rulebook and metrics are automatically logged:

```python
# MLflow logging happens automatically
result = rb.run({"age": 25, "income": 50000})
# Metrics, costs, and artifacts are logged to MLflow
```

Configure MLflow explicitly:

```python
from rulesmith import MLflowLogger

logger = MLflowLogger(
    experiment_name="rulesmith/credit_check",
    sample_rate=0.01,  # Sample 1% in production
    enable_artifacts=True,
    redact_pii=True,
)

result = rb.run({"age": 25, "income": 50000}, mlflow_logger=logger)
```

### Metric Thresholds

Set thresholds for metrics and get automatic alerts:

```python
# Set threshold for guard violations
rb.add_metric_threshold("guard_violations", threshold=0, operator="<=")

# Set threshold for latency
rb.add_metric_threshold("latency_ms", threshold=100, operator="<=")

# Configure alert action
rb.on_metric_breach("guard_violations", action="log")  # Log to MLflow
```

### Testing Rulebooks

Write test suites for your rulebooks:

```python
from rulesmith import RulebookTester

tester = RulebookTester(rb)

tester.add_test_case(
    name="approved_high_score",
    inputs={"age": 30, "income": 75000},
    expected={"approved": True}
)

tester.add_test_case(
    name="rejected_low_score",
    inputs={"age": 25, "income": 40000},
    expected={"approved": False}
)

results = tester.run_all()
print(results.summary())
```

### Debugging

Enable debug mode to get step-by-step execution information:

```python
result = rb.run({"age": 25, "income": 50000}, debug=True)
# Access debug info: result.debug_info
# Contains: step_by_step, state_snapshots, execution_path
```

### Comparing Rulebook Versions

Compare two rulebook versions to see what changed:

```python
from rulesmith import compare_rulebooks

diff = compare_rulebooks(
    old_spec=rb1.to_spec(),
    new_spec=rb2.to_spec()
)

print(diff.summary())
# Shows: nodes added/removed, edges changed, etc.
```

### Querying MLflow Metrics

Query aggregated metrics from MLflow:

```python
from rulesmith.metrics import query_mlflow_metrics, compare_mlflow_metrics

# Query aggregated metrics
agg = query_mlflow_metrics(
    experiment_name="rulesmith/credit_check",
    metric_names=["latency_ms", "guard_violations"],
    filter_string="tags.rulebook_version='1.0.0'",
    aggregation=["mean", "p95", "max"]
)

# Compare two versions
comparison = compare_mlflow_metrics(
    experiment_name="rulesmith/credit_check",
    baseline_filter="tags.rulebook_version='1.0.0'",
    comparison_filter="tags.rulebook_version='1.1.0'",
    metrics=["latency_ms", "guard_violations"]
)
```

## Common Patterns

### Auto-Configuration

Use auto-configuration for quick setup:

```python
# Automatically configures MLflow and metrics
rb = Rulebook(name="credit_check", version="1.0.0", auto_configure=True)
```

### Smart Defaults

MLflow logger is auto-created with sensible defaults:

```python
# No logger needed - auto-created with defaults
result = rb.run({"age": 25, "income": 50000})
```

## Troubleshooting

### MLflow Not Available

If MLflow is not installed, execution continues without logging:

```python
# Execution works, just without MLflow logging
result = rb.run({"age": 25, "income": 50000}, enable_mlflow=False)
```

### Error Messages

Enhanced error messages provide helpful suggestions:

```python
# If you get a KeyError, suggestions will show available fields
# If you get a TypeError, suggestions will show expected types
```

## Best Practices

1. **Always version your rulebooks**: Use semantic versioning (e.g., "1.0.0")
2. **Use metrics for guardrails**: Define custom metrics as rule functions
3. **Set thresholds**: Configure metric thresholds for production monitoring
4. **Test your rulebooks**: Write test suites for critical decision paths
5. **Query MLflow**: Use MLflow query helpers to analyze metrics over time
6. **Use debug mode**: Enable debug mode when developing new rulebooks

## Next Steps

- See `examples/` for more examples
- Check `docs/API.md` for detailed API documentation
- Read `docs/ARCHITECTURE.md` for system architecture

