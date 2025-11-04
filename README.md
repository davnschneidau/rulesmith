# Rulesmith

A production-grade rulebook/DAG execution engine with deep MLflow integration for building, deploying, and governing ML/AI decision systems.

## Quick Start

```python
from rulesmith import rule, Rulebook

@rule(name="check_eligibility", inputs=["age", "income"], outputs=["eligible"])
def check_eligibility(age: int, income: float) -> dict:
    return {"eligible": age >= 18 and income > 30000}

# Build rulebook
rb = Rulebook(name="credit_decision", version="1.0.0", auto_configure=True)
rb.add_rule(check_eligibility, as_name="eligibility_check")

# Execute - MLflow logging happens automatically
result = rb.run({"age": 25, "income": 50000})
print(result.value)  # {"eligible": True}
```

## Installation

```bash
pip install rulesmith
```

For optional dependencies:

```bash
pip install rulesmith[all]  # Includes LangChain, LangGraph, Redis, Postgres
```

## Features

### Core Engine
- **DAG-based Rulebooks**: Build complex decision flows with rules, models, and conditional routing
- **Automatic MLflow Integration**: Metrics, costs, and artifacts logged by default
- **Smart Defaults**: Auto-configuration for quick setup
- **Standardized Results**: Rich `DecisionResult` objects with full execution traces

### Model Integration
- **MLflow Models**: Integrate any MLflow model (sklearn, PyTorch, XGBoost, custom pyfunc)
- **LangChain/LangGraph**: Native support for LangChain chains and LangGraph graphs
- **Multi-Provider LLMs**: OpenAI, Anthropic, Azure, AWS Bedrock, and more via unified interface

### Guardrails & Metrics
- **Rule-based Metrics**: Define custom guardrails as rule functions
- **Metric Thresholds**: Set thresholds with automatic MLflow alerting
- **MLflow Query Helpers**: Query and compare metrics across rulebook versions

### A/B Testing
- **Traffic Splitting**: Hash-based bucketing for consistent user assignment
- **Bandit Policies**: Thompson Sampling, UCB, epsilon-greedy, and more
- **Version Comparison**: Compare metrics between rulebook versions

### Developer Experience
- **Testing Framework**: Write and run test suites for rulebooks
- **Debug Mode**: Step-by-step execution with state snapshots
- **Enhanced Errors**: Context-aware error messages with helpful suggestions
- **Version Comparison**: See what changed between rulebook versions

### Governance & Reliability
- **SLO-gated Promotion**: Promote rulebooks based on service level objectives
- **Audit Logs**: Complete audit trail of all rulebook executions
- **Lineage Tracking**: Full dependency graph with code hashes and model URIs
- **Reliability Features**: Caching, retries, rate limiting, circuit breakers, shadow mode

## Core Concepts

### Rules

Rules are pure Python functions that define business logic:

```python
from rulesmith import rule

@rule(name="calculate_score", inputs=["age", "income"], outputs=["score"])
def calculate_score(age: int, income: float) -> dict:
    return {"score": (age * 10) + (income / 1000)}
```

### Rulebooks

Rulebooks are DAGs that connect rules, models, and conditional logic:

```python
from rulesmith import Rulebook

rb = Rulebook(name="loan_approval", version="1.0.0", auto_configure=True)
rb.add_rule(check_eligibility, as_name="eligibility")
rb.add_rule(calculate_score, as_name="scoring")
rb.connect("eligibility", "scoring")
```

### Nodes

- **RuleNode**: Executes a decorated rule function
- **ModelNode**: Loads and executes MLflow models or LangChain models
- **LLMNode**: LLM inference with provider abstraction (multi-provider support)
- **HITLNode**: Human review and approval workflows

**Note:** Fork and gate functionality is handled via functions (`fork()`, `gate()`) rather than node classes for better flexibility.

## MLflow Integration

Rulesmith integrates deeply with MLflow. Metrics, costs, and artifacts are automatically logged:

```python
# MLflow logging happens automatically
result = rb.run({"age": 25, "income": 50000})

# Configure MLflow explicitly
from rulesmith import MLflowLogger

logger = MLflowLogger(
    experiment_name="rulesmith/credit_check",
    sample_rate=0.01,  # Sample 1% in production
    enable_artifacts=True,
    redact_pii=True,
)

result = rb.run({"age": 25, "income": 50000}, mlflow_logger=logger)
```

### MLflow Features

- **Nested Runs**: Each node execution creates a nested run with full lineage
- **Traces**: GenAI nodes emit MLflow traces with token/cost tracking
- **Lineage**: Complete dependency graph with code hashes and model URIs
- **Query Helpers**: Query aggregated metrics and compare versions

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

## Advanced Features

### A/B Testing

```python
# Simple 50/50 split
rb.add_split("experiment", {"control": 0.5, "treatment": 0.5})

# Weighted split (70/30)
rb.add_split("test", {"variant_a": 0.7, "variant_b": 0.3})

# Adaptive bandit (automatically optimizes)
rb.add_split("bandit_test", {"a": 1.0, "b": 1.0}, policy="thompson")
```

### Guardrails (Rule-based Metrics)

Guardrails are defined as rule functions that evaluate the output of LLM/Model nodes:

```python
from rulesmith import rule

# Define custom guardrails as rules
@rule(name="check_pii", inputs=["output"], outputs=["has_pii"])
def check_pii(output: str) -> dict:
    """Check if output contains PII."""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    has_email = bool(re.search(email_pattern, output))
    return {"has_pii": has_email}

@rule(name="check_toxicity", inputs=["output"], outputs=["is_toxic"])
def check_toxicity(output: str) -> dict:
    """Check if output contains toxic content."""
    toxic_words = ["hate", "violence", "harassment"]
    return {"is_toxic": any(word in output.lower() for word in toxic_words)}

# Add metrics to LLM node (can use functions or rule names)
rb.add_llm("gpt4", provider="openai", model_name="gpt-4", metrics=[check_pii, check_toxicity])

# Or add metrics separately
rb.add_llm("gpt4", provider="openai", model_name="gpt-4")
rb.add_metrics("gpt4", [check_pii, check_toxicity])
# Or use rule names: rb.add_metrics("gpt4", ["check_pii", "check_toxicity"])
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

# Custom alert hook
def alert_on_breach(metric_name, value, threshold):
    send_slack_alert(f"{metric_name} breached: {value}")

rb.add_metric_threshold("cost_usd", threshold=0.01, alert_action="hook", alert_hook=alert_on_breach)
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
# Contains: step_by_step, state_snapshots, execution_path, metrics_per_node
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

### Human-in-the-Loop

```python
from rulesmith.hitl.adapters import LocalFileQueue

queue = LocalFileQueue(queue_dir="./reviews")
rb.add_hitl("review", queue, active_learning_threshold=0.7)
```

### Model Promotion with SLOs

```python
from rulesmith.governance.promotion import PromotionPolicy, SLO, promote_model

policy = PromotionPolicy(
    name="prod_policy",
    slos=[
        SLO("accuracy", 0.95),
        SLO("latency_ms", 100, operator="<="),
    ],
)
result = promote_model("my_model", "@staging", "@prod", policy=policy)
```

### Evaluation

```python
from rulesmith.evaluation import evaluate_rulebook, accuracy_scorer

result = evaluate_rulebook("models:/rulebook/1", evaluation_data)
score = accuracy_scorer(data, predictions, targets="label")
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

### Adding Models

```python
# MLflow model
rb.add_model("my_model", model_uri="models:/my_model/1")

# LangChain model directly
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
rb.add_model("llm_node", langchain_model=llm)

# LLM with provider
rb.add_llm("gpt4", provider="openai", model_name="gpt-4")
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
7. **Auto-configure**: Use `auto_configure=True` for quick setup

## Architecture

Rulesmith provides:

- **DAG Execution**: Topological sorting, parallel execution support
- **MLflow Integration**: Native pyfunc flavor, nested runs, traces
- **Model Support**: MLflow models, LangChain, LangGraph, GenAI providers
- **A/B Testing**: Multiple bandit policies, traffic management
- **Guardrails**: Rule-based metrics with threshold alerting
- **HITL**: Queue adapters, active learning
- **Governance**: Promotion, diff, lineage, audit
- **Evaluation**: MLflow integration, custom scorers
- **Reliability**: Retry, rate limiting, circuit breakers, shadow mode
- **Security**: PII redaction, secret detection
- **Schemas**: Input/output validation, contracts

## Documentation

- [Quickstart Guide](docs/QUICKSTART.md) - Get started quickly
- [Architecture](docs/ARCHITECTURE.md) - System architecture details
- [API Reference](docs/API.md) - Complete API documentation

## Examples

See the `examples/` directory for more examples:

- `mlflow_logging_example.py` - MLflow logging integration

## License

Apache 2.0
