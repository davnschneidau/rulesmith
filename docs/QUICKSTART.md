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

### Guardrails

```python
from rulesmith.guardrails.langchain_adapter import create_pii_guard_from_langchain

rb.attach_guard("node_name", create_pii_guard_from_langchain())
```

### MLflow Integration

```python
from rulesmith.mlflow_flavor import log_rulebook_model
import mlflow

with mlflow.start_run():
    log_rulebook_model(rb, registered_model_name="credit_check")
```

## Next Steps

- See `examples/` for more examples
- Check `docs/` for detailed documentation
- Read API docs in code

