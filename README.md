# Rulesmith

A production-grade rulebook/DAG execution engine with deep MLflow integration for building, deploying, and governing ML/AI decision systems.

## Quick Start

```python
from rulesmith import rule, rulebook, Rulebook

@rule(name="check_eligibility", inputs=["age", "income"], outputs=["eligible"])
def check_eligibility(age: int, income: float) -> dict:
    return {"eligible": age >= 18 and income > 30000}

@rulebook(name="credit_decision", version="1.0.0")
def build_credit_rulebook():
    rb = Rulebook(name="credit_decision", version="1.0.0")
    rb.add_rule(check_eligibility, as_name="eligibility_check")
    return rb

# Execute
rulebook = build_credit_rulebook()
result = rulebook.run({"age": 25, "income": 50000})
print(result)  # {"eligible": True}
```

## Features

- **DAG-based Rulebooks**: Build complex decision flows with rules, forks, gates, and conditional routing
- **MLflow Integration**: Native MLflow 3 support with nested runs, traces, and lineage tracking
- **Model Support**: Integrate any MLflow model (sklearn, PyTorch, XGBoost, custom pyfunc) or LangChain models
- **GenAI Integration**: LangChain, LangGraph, and provider-agnostic LLM wrappers
- **A/B Testing**: Built-in traffic splitting, hash bucketing, and bandit policies
- **Guardrails**: PII detection, toxicity checks, hallucination validation, and custom guards
- **Human-in-the-Loop**: Queue-based HITL nodes for manual review workflows
- **Governance**: SLO-gated promotion, change manifests, audit logs, and full lineage tracking
- **Evaluation**: Business KPI scorers, LLM-as-judge, and comprehensive evaluation workflows
- **Reliability**: Caching, retries, rate limiting, circuit breakers, and shadow mode

## Installation

```bash
pip install rulesmith
```

For optional dependencies:

```bash
pip install rulesmith[all]  # Includes LangChain, LangGraph, Redis, Postgres
```

## Core Concepts

### Rules
Rules are pure Python functions that define business logic:

```python
@rule(name="calculate_score", inputs=["age", "income"], outputs=["score"])
def calculate_score(age: int, income: float) -> dict:
    return {"score": (age * 10) + (income / 1000)}
```

### Rulebooks
Rulebooks are DAGs that connect rules, models, and conditional logic:

```python
@rulebook(name="loan_approval", version="1.0.0")
def build_loan_rulebook():
    rb = Rulebook(name="loan_approval", version="1.0.0")
    rb.add_rule(check_eligibility, as_name="eligibility")
    rb.add_rule(calculate_score, as_name="scoring")
    rb.connect("eligibility", "scoring", mapping={"eligible": "eligible"})
    return rb
```

### Nodes
- **RuleNode**: Executes a decorated rule function
- **ModelNode**: Loads and executes MLflow models or LangChain models
- **LLMNode**: LLM inference with provider abstraction (multi-provider support)
- **HITLNode**: Human review and approval workflows

**Note:** Fork and gate functionality is handled via functions (`fork()`, `gate()`) rather than node classes for better flexibility.

## MLflow Integration

Rulesmith integrates deeply with MLflow 3:

- **Model Registry**: Rulebooks are registered as MLflow models
- **Nested Runs**: Each node execution creates a nested run with full lineage
- **Traces**: GenAI nodes emit MLflow traces with token/cost tracking
- **Lineage**: Complete dependency graph with code hashes and model URIs
- **Evaluation**: Native `mlflow.evaluate()` integration with custom scorers

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

# Add metrics to LLM node
rb.add_llm("gpt4", provider="openai", model_name="gpt-4", metrics=[check_pii, check_toxicity])

# Or add metrics separately
rb.add_llm("gpt4", provider="openai", model_name="gpt-4")
rb.add_metrics("gpt4", [check_pii, check_toxicity])
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

## Architecture

Rulesmith provides:

- **DAG Execution**: Topological sorting, parallel execution support
- **MLflow Integration**: Native pyfunc flavor, nested runs, traces
- **Model Support**: BYOM, LangChain, LangGraph, GenAI providers
- **A/B Testing**: 5 bandit policies, traffic management
- **Guardrails**: 8 built-in guards, 4 action types
- **HITL**: 5 queue adapters, active learning
- **Governance**: Promotion, diff, lineage, audit
- **Evaluation**: MLflow integration, custom scorers
- **Reliability**: Retry, rate limiting, circuit breakers, shadow mode
- **Security**: PII redaction, secret detection
- **Schemas**: Input/output validation, contracts

## Documentation

- [Quickstart Guide](docs/QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)

## License

Apache 2.0

