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
- **BYOM Support**: Integrate any MLflow model (sklearn, PyTorch, XGBoost, custom pyfunc)
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
- **ForkNode**: A/B testing and traffic splitting
- **GateNode**: Conditional routing based on expressions
- **BYOMNode**: Loads and executes MLflow models
- **GenAINode**: LLM inference with provider abstraction
- **HITLNode**: Human review and approval workflows

## MLflow Integration

Rulesmith integrates deeply with MLflow 3:

- **Model Registry**: Rulebooks are registered as MLflow models
- **Nested Runs**: Each node execution creates a nested run with full lineage
- **Traces**: GenAI nodes emit MLflow traces with token/cost tracking
- **Lineage**: Complete dependency graph with code hashes and model URIs
- **Evaluation**: Native `mlflow.evaluate()` integration with custom scorers

## Documentation

See `docs/` for detailed guides and examples.

## License

Apache 2.0

