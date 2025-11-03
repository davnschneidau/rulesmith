# Rulesmith API Reference

Complete API reference for the Rulesmith platform.

## Table of Contents

- [Core DAG Engine](#core-dag-engine)
- [MLflow Integration](#mlflow-integration)
- [Model Integration](#model-integration)
- [A/B Testing](#ab-testing)
- [Guardrails](#guardrails)
- [Human-in-the-Loop](#human-in-the-loop)
- [Governance](#governance)
- [Evaluation](#evaluation)
- [Reliability](#reliability)
- [Security](#security)
- [Schemas](#schemas)
- [Runtime](#runtime)

---

## Core DAG Engine

### `rule`

Decorator for registering functions as rules.

```python
from rulesmith import rule

@rule(name="check_age", inputs=["age"], outputs=["eligible"], version="1.0.0")
def check_age(age: int) -> dict:
    """Check if age meets requirement."""
    return {"eligible": age >= 18}
```

**Parameters:**
- `name` (str, optional): Rule name (defaults to function name)
- `inputs` (List[str], optional): Input field names (auto-inferred if None)
- `outputs` (List[str], optional): Output field names (defaults to ["result"])
- `default_params` (Dict[str, Any], optional): Default parameters
- `version` (str): Rule version (default: "1.0.0")

**Returns:** Decorated function (registered in global registry)

---

### `Rulebook`

Main class for building and executing rulebook DAGs.

```python
from rulesmith import Rulebook

rb = Rulebook(name="credit_check", version="1.0.0")
```

#### Methods

##### `add_rule(func, as_name=None)`

Add a rule node to the rulebook.

**Parameters:**
- `func`: Decorated rule function
- `as_name` (str, optional): Node name (defaults to rule name)

**Returns:** Self for chaining

##### `add_model(name, model_uri=None, langchain_model=None, params=None)`

Add a model node (MLflow or LangChain model).

**Parameters:**
- `name` (str): Node name
- `model_uri` (str, optional): MLflow model URI
- `langchain_model` (Any, optional): Direct LangChain model/chain instance
- `params` (Dict[str, Any], optional): Additional parameters

**Returns:** Self for chaining

**Note:** Gate functionality is now handled via the `gate()` function during execution, not as a node class.

##### `add_llm(name, model_uri=None, provider=None, model_name=None, gateway_uri=None, params=None)`

Add an LLM node with multi-provider support.

**Parameters:**
- `name` (str): Node name
- `model_uri` (str, optional): MLflow model URI
- `provider` (str, optional): Provider name (auto-detected from model_name if not provided)
- `model_name` (str, optional): Model name (e.g., "gpt-4", "claude-3", "gemini-pro")
- `gateway_uri` (str, optional): MLflow Gateway URI
- `params` (Dict[str, Any], optional): Provider-specific parameters

**Returns:** Self for chaining

**Note:** `add_byom()` and `add_genai()` are deprecated. Use `add_model()` and `add_llm()` instead.

##### `add_split(name, variants, policy="hash", policy_instance=None)`

Add an A/B test split node (simplified API for A/B testing).

**Parameters:**
- `name` (str): Split node name
- `variants` (Dict[str, float]): Dictionary of variant names to weights (e.g., {"control": 0.5, "treatment": 0.5})
- `policy` (str): Policy name ("hash", "random", "thompson", "ucb", "epsilon")
- `policy_instance` (TrafficPolicy, optional): Custom policy instance

**Returns:** Self for chaining

##### `add_hitl(name, queue, timeout=None, async_mode=False, active_learning_threshold=None)`

Add a Human-in-the-Loop node.

**Parameters:**
- `name` (str): Node name
- `queue` (HITLQueue): Queue adapter instance
- `timeout` (float, optional): Timeout in seconds
- `async_mode` (bool): If True, non-blocking execution
- `active_learning_threshold` (float, optional): Confidence threshold for skipping review

**Returns:** Self for chaining

##### `add_metrics(node_name, metrics)`

Add metrics (guardrails) to a node. Metrics are rule functions that evaluate the node's output.

**Parameters:**
- `node_name` (str): Node name (must be LLM or Model node)
- `metrics` (List[Callable]): List of rule functions to evaluate on output

**Returns:** Self for chaining

**Examples:**
```python
@rule(name="check_pii", inputs=["output"], outputs=["has_pii"])
def check_pii(output: str) -> dict:
    return {"has_pii": "@" in output}

rb.add_llm("gpt4", provider="openai", model_name="gpt-4")
rb.add_metrics("gpt4", [check_pii])
```

##### `connect(source, target, mapping=None)`

Connect two nodes with optional field mapping.

**Parameters:**
- `source` (str): Source node name
- `target` (str): Target node name
- `mapping` (Dict[str, str], optional): Field name mapping

**Returns:** Self for chaining

##### `run(payload, context=None, enable_mlflow=True)`

Execute the rulebook.

**Parameters:**
- `payload` (Dict[str, Any]): Input payload
- `context` (RunContext, optional): Execution context
- `enable_mlflow` (bool): Whether to enable MLflow logging

**Returns:** Final output state dictionary

##### `to_spec()`

Convert rulebook to specification.

**Returns:** RulebookSpec object

---

### `ExecutionEngine`

Core execution engine for rulebooks.

```python
from rulesmith.dag.execution import ExecutionEngine

engine = ExecutionEngine(spec)
```

#### Methods

##### `register_node(name, node)`

Register a node instance.

**Parameters:**
- `name` (str): Node name
- `node` (Node): Node instance

##### `execute(payload, context, nodes=None)`

Execute the rulebook DAG.

**Parameters:**
- `payload` (Dict[str, Any]): Initial input payload
- `context` (RunContext): Execution context
- `nodes` (Dict[str, Node], optional): Node instances

**Returns:** Final state dictionary

---

### Node Types

#### `Node`

Base class for all node types.

**Methods:**
- `execute(state, context) -> Dict[str, Any]`: Execute node

#### `RuleNode`

Executes a registered rule function.

#### `ModelNode`

Executes MLflow models or LangChain models.

#### `LLMNode`

Executes LLM calls with multi-provider support (OpenAI, Anthropic, Google, etc.).

#### `HITLNode`

Human-in-the-Loop review node.

**Note:** `GateNode`, `ForkNode`, `BYOMNode`, and `GenAINode` are deprecated. Gate and fork functionality is handled via functions (`gate()`, `fork()`) during execution. Use `ModelNode` and `LLMNode` instead of `BYOMNode` and `GenAINode`.

---

## MLflow Integration

### `log_rulebook_model`

Log a rulebook as an MLflow model.

```python
from rulesmith.mlflow_flavor import log_rulebook_model
import mlflow

with mlflow.start_run():
    model = log_rulebook_model(
        rulebook=rb,
        artifact_path="model",
        registered_model_name="credit_check",
    )
```

**Parameters:**
- `rulebook` (Rulebook): Rulebook instance
- `artifact_path` (str): Artifact path within run
- `registered_model_name` (str, optional): Model name in registry
- `extra_pip_requirements` (List[str], optional): Additional dependencies
- `signature` (ModelSignature, optional): Model signature
- `input_example` (ModelInputExample, optional): Input example
- `metadata` (Dict[str, Any], optional): Metadata

**Returns:** MLflow Model object

---

### `MLflowRunContext`

MLflow-aware execution context.

```python
from rulesmith.runtime.mlflow_context import MLflowRunContext

context = MLflowRunContext(
    rulebook_spec=spec,
    enable_mlflow=True,
    identity="user123",
)
```

**Parameters:**
- `rulebook_spec` (RulebookSpec): Rulebook specification
- `run_id` (str, optional): Existing run ID
- `identity` (str, optional): User/request identity
- `tags` (Dict[str, str], optional): Tags
- `seed` (int, optional): Random seed
- `params` (Dict[str, Any], optional): Parameters
- `enable_mlflow` (bool): Enable MLflow logging
- `parent_run_id` (str, optional): Parent run ID

**Methods:**
- `start_node_execution(node_name, node_kind, inputs=None) -> NodeExecutionContext`
- `set_ab_bucket(bucket)`: Set A/B bucket tag
- `set_model_uri(uri)`: Set model URI
- `finish_genai(outputs, tokens, cost, latency, provider)`: Log GenAI metrics

---

### `NodeExecutionContext`

Context for tracking individual node executions.

**Methods:**
- `set_code_hash(hash)`: Set code hash
- `set_model_uri(uri)`: Set model URI
- `finish(outputs, metrics=None)`: Finish node execution

---

## Model Integration

### `BYOMRef` / `ModelNode`

Reference to an MLflow model or LangChain model.

```python
from rulesmith.models.mlflow_byom import BYOMRef

# For direct model loading
model_ref = BYOMRef("models:/my_model/1")
result = model_ref.predict({"feature": 123})

# Or use ModelNode in rulebook
rb.add_model("my_model", model_uri="models:/my_model/1")
```

**Methods:**
- `load()`: Load the model
- `predict(inputs) -> Dict[str, Any]`: Run prediction

**Note:** `BYOMRef` is still available for direct model access, but rulebooks should use `add_model()` instead of `add_byom()`.

---

### `GenAIWrapper`

Provider-agnostic LLM wrapper.

```python
from rulesmith.models.genai import GenAIWrapper

wrapper = GenAIWrapper(
    provider="openai",
    model_name="gpt-4",
)
result = wrapper.invoke("Hello, world!")
```

**Parameters:**
- `provider` (str): Provider name
- `model_name` (str, optional): Model name
- `model_uri` (str, optional): MLflow model URI
- `gateway_uri` (str, optional): MLflow Gateway URI

**Methods:**
- `invoke(prompt, **kwargs) -> Dict[str, Any]`: Invoke LLM

---

## A/B Testing

### Traffic Policies

#### `HashPolicy`

Deterministic hash-based allocation.

```python
from rulesmith.ab.policies import HashPolicy

policy = HashPolicy()
```

#### `RandomPolicy`

Random allocation.

```python
from rulesmith.ab.policies import RandomPolicy

policy = RandomPolicy(seed=42)
```

#### `ThompsonSamplingPolicy`

Thompson Sampling bandit.

```python
from rulesmith.ab.policies import ThompsonSamplingPolicy

policy = ThompsonSamplingPolicy(arms_history={"arm1": {"successes": 10, "failures": 2}})
```

#### `UCBPolicy`

Upper Confidence Bound bandit.

```python
from rulesmith.ab.policies import UCBPolicy

policy = UCBPolicy(arms_history={"arm1": {"successes": 10, "failures": 2}})
```

#### `EpsilonGreedyPolicy`

Epsilon-greedy bandit.

```python
from rulesmith.ab.policies import EpsilonGreedyPolicy

policy = EpsilonGreedyPolicy(epsilon=0.1, arms_history={...})
```

---

### `pick_arm`

Select an A/B arm based on policy.

```python
from rulesmith.ab.traffic import pick_arm
from rulesmith.io.ser import ABArm

arms = [ABArm(node="variant_a", weight=1.0), ABArm(node="variant_b", weight=1.0)]
selected = pick_arm(arms, identity="user123", policy="thompson_sampling")
```

**Parameters:**
- `arms` (List[ABArm]): List of arms
- `identity` (str, optional): User identity
- `policy` (str): Policy name
- `policy_instance` (TrafficPolicy, optional): Policy instance
- `context` (Dict[str, Any], optional): Policy context

**Returns:** Selected ABArm

---

## Guardrails (Rule-based Metrics)

Guardrails are defined as rule functions that evaluate the output of LLM/Model nodes. They are added as metrics to nodes and automatically evaluated after node execution.

### Defining Metrics

Metrics are rule functions that take the node output and return evaluation results:

```python
from rulesmith import rule

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
    toxic_words = ["hate", "violence"]
    return {"is_toxic": any(word in output.lower() for word in toxic_words)}
```

### Adding Metrics to Nodes

Metrics can be added when creating a node or separately:

```python
# Add metrics when creating node
rb.add_llm("gpt4", provider="openai", model_name="gpt-4", metrics=[check_pii, check_toxicity])

# Or add metrics separately
rb.add_llm("gpt4", provider="openai", model_name="gpt-4")
rb.add_metrics("gpt4", [check_pii, check_toxicity])
```

### Metric Results

Metric results are stored in the node's output under `_metrics`:

```python
result = rb.run({"prompt": "Hello"})
# result["_metrics"] contains:
# {
#     "check_pii": {"value": False, "message": None},
#     "check_toxicity": {"value": False, "message": None}
# }
```

Metrics are automatically logged to MLflow if enabled.

---

### `GuardExecutor`

Executes guard functions and applies policies.

```python
from rulesmith.guardrails.execution import guard_executor

guard_executor.register_guard("custom_guard", guard_function)
result = guard_executor.evaluate_guard("custom_guard", inputs, outputs)
```

**Methods:**
- `register_guard(name, func)`: Register guard function
- `evaluate_guard(name, inputs, outputs) -> GuardResult`: Evaluate guard
- `apply_policy(policy, inputs, outputs) -> Dict[str, Any]`: Apply policy

---

### `GuardResult`

Result of guard evaluation.

**Fields:**
- `guard_name` (str): Guard name
- `passed` (bool): Whether guard passed
- `message` (str, optional): Result message
- `score` (float, optional): Confidence score

---

## Human-in-the-Loop

### Queue Adapters

#### `LocalFileQueue`

File-based queue (dev/testing).

```python
from rulesmith.hitl.adapters import LocalFileQueue

queue = LocalFileQueue(queue_dir="./reviews")
```

#### `InMemoryQueue`

In-memory queue (testing).

```python
from rulesmith.hitl.adapters import InMemoryQueue

queue = InMemoryQueue()
```

#### `PostgresQueue`

PostgreSQL queue (production).

```python
from rulesmith.hitl.adapters import PostgresQueue

queue = PostgresQueue("postgresql://user:pass@host/db")
```

#### `RedisQueue`

Redis queue (high-throughput).

```python
from rulesmith.hitl.adapters import RedisQueue

queue = RedisQueue("redis://localhost:6379")
```

#### `SlackQueue`

Slack notifications.

```python
from rulesmith.hitl.adapters import SlackQueue

queue = SlackQueue("https://hooks.slack.com/services/...")
```

---

### `ReviewRequest`

Human review request.

**Fields:**
- `id` (str): Request ID
- `node` (str): Node name
- `payload` (Dict[str, Any]): Request payload
- `suggestions` (Dict[str, Any], optional): Model suggestions
- `expires_at` (datetime, optional): Expiration time

---

### `ReviewDecision`

Human review decision.

**Fields:**
- `id` (str): Request ID
- `approved` (bool): Approval status
- `edited_output` (Dict[str, Any], optional): Edited output
- `comment` (str, optional): Review comment
- `reviewer` (str, optional): Reviewer ID

---

## Governance

### `SLO`

Service Level Objective.

```python
from rulesmith.governance.promotion import SLO

slo = SLO(
    metric_name="accuracy",
    threshold=0.95,
    operator=">=",
    window_hours=24,
)
```

**Methods:**
- `evaluate(value) -> bool`: Evaluate if value meets SLO

---

### `PromotionPolicy`

Model promotion policy.

```python
from rulesmith.governance.promotion import PromotionPolicy, SLO

policy = PromotionPolicy(
    name="prod_policy",
    slos=[SLO("accuracy", 0.95), SLO("latency_ms", 100, operator="<=")],
    require_all=True,
    min_samples=100,
)
```

---

### `promote_model`

Promote a model between stages.

```python
from rulesmith.governance.promotion import promote_model

result = promote_model(
    model_name="credit_check",
    from_stage="@staging",
    to_stage="@prod",
    policy=policy,
)
```

**Returns:** Dictionary with promotion result

---

### `RulebookDiff`

Difference between two rulebook specifications.

```python
from rulesmith.governance.diff import diff_rulebooks

diff = diff_rulebooks(spec1, spec2)
print(diff.to_string())
```

**Properties:**
- `added_nodes`: List of added nodes
- `removed_nodes`: List of removed nodes
- `modified_nodes`: List of modified nodes
- `added_edges`: List of added edges
- `removed_edges`: List of removed edges

**Methods:**
- `has_changes() -> bool`: Check if changes exist
- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `to_string() -> str`: Convert to human-readable string

---

### `build_lineage`

Build lineage graph from rulebook spec.

```python
from rulesmith.governance.lineage import build_lineage

lineage = build_lineage(rulebook_spec)
node_ref = lineage.get_node_lineage("node1")
```

---

### `AuditLogger`

Audit log manager.

```python
from rulesmith.governance.audit import AuditLogger, log_promotion

logger = AuditLogger()
entry = log_promotion(logger, "model", "@staging", "@prod", actor="user")
```

**Methods:**
- `log(action, entity_type, entity_id, actor=None, metadata=None) -> AuditLogEntry`
- `sign_entry(entry) -> str`: Generate signed hash
- `verify_entry(entry, hash) -> bool`: Verify hash
- `get_entries(...) -> List[AuditLogEntry]`: Query entries
- `export() -> List[Dict[str, Any]]`: Export all entries

---

## Evaluation

### `evaluate_rulebook`

Evaluate a rulebook using MLflow.

```python
from rulesmith.evaluation import evaluate_rulebook

result = evaluate_rulebook(
    model_uri="models:/rulebook/1",
    data=evaluation_data,
    targets="label",
)
```

---

### Scorers

#### `accuracy_scorer`

Compute accuracy score.

```python
from rulesmith.evaluation import accuracy_scorer

score = accuracy_scorer(data, predictions, targets="label")
```

#### `precision_scorer`

Compute precision score.

```python
from rulesmith.evaluation import precision_scorer

score = precision_scorer(data, predictions, targets="label", positive_label=1)
```

#### `recall_scorer`

Compute recall score.

#### `f1_scorer`

Compute F1 score.

#### `business_kpi_scorer`

Compute custom business KPI.

```python
from rulesmith.evaluation import business_kpi_scorer

score = business_kpi_scorer(data, predictions, kpi_function=custom_kpi)
```

---

### Dataset Utilities

#### `load_dataset`

Load dataset from file.

```python
from rulesmith.evaluation import load_dataset

df = load_dataset("data.csv", format="csv")
```

#### `slice_dataset`

Slice dataset based on filters.

```python
from rulesmith.evaluation import slice_dataset

sliced = slice_dataset(data, {"category": "A"})
```

#### `create_evaluation_slices`

Create slices for fairness analysis.

```python
from rulesmith.evaluation import create_evaluation_slices

slices = create_evaluation_slices(data, slice_columns=["category", "region"])
```

---

## Reliability

### `retry`

Retry decorator with exponential backoff.

```python
from rulesmith.reliability import retry, RetryConfig

@retry(config=RetryConfig(max_retries=3, initial_backoff=1.0))
def flaky_api_call():
    ...
```

---

### `RateLimiter`

Rate limiter with token bucket.

```python
from rulesmith.reliability import RateLimiter

limiter = RateLimiter(requests_per_second=10.0)

with limiter:
    make_request()
```

---

### `CircuitBreaker`

Circuit breaker for fault tolerance.

```python
from rulesmith.reliability import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)

result = breaker.call(risky_function)
```

**States:**
- `CLOSED`: Normal operation
- `OPEN`: Failing, reject requests
- `HALF_OPEN`: Testing recovery

---

### `shadow_mode`

Shadow mode execution decorator.

```python
from rulesmith.reliability import shadow_mode

@shadow_mode(primary_func, shadow_func, shadow_probability=1.0)
def function():
    ...
```

---

## Security

### `RedactionConfig`

PII redaction configuration.

```python
from rulesmith.security import RedactionConfig, redact_dict

config = RedactionConfig(
    redact_emails=True,
    redact_phones=True,
    suppress_fields=["ssn"],
)
safe_data = redact_dict(data, config)
```

---

### `detect_secrets`

Detect potential secrets in data.

```python
from rulesmith.security import detect_secrets

secrets = detect_secrets(data)
```

---

### `validate_no_secrets`

Validate data contains no secrets.

```python
from rulesmith.security import validate_no_secrets

is_valid, issues = validate_no_secrets(data)
```

---

## Schemas

### `SchemaContract`

Schema contract for validation.

```python
from rulesmith.io.schemas import SchemaContract, FieldSchema

schema = SchemaContract(
    name="user_schema",
    fields=[
        FieldSchema(name="age", type="int", min_value=0, max_value=150),
        FieldSchema(name="email", type="str", required=True),
    ],
    strict=True,
)
```

---

### `validate_with_schema`

Validate data against schema.

```python
from rulesmith.io.schemas import validate_with_schema

is_valid, errors = validate_with_schema(data, schema)
```

---

### `infer_schema`

Infer schema from sample data.

```python
from rulesmith.io.schemas import infer_schema

schema = infer_schema(sample_data, name="inferred")
```

---

## Runtime

### `RunContext`

Basic execution context.

```python
from rulesmith.runtime import RunContext

context = RunContext(
    run_id="run123",
    identity="user123",
    tags={"env": "prod"},
)
```

---

### `Hooks`

Protocol for execution hooks.

```python
from rulesmith.runtime import Hooks

class MyHooks(Hooks):
    def before_node(self, node_name, state, context):
        ...

    def after_node(self, node_name, state, context, outputs):
        ...

    def on_error(self, node_name, state, context, error):
        ...
```

---

### `hook_registry`

Global hook registry.

```python
from rulesmith.runtime import hook_registry

hook_registry.register(my_hooks)
```

---

### `Plugin`

Base class for plugins.

```python
from rulesmith.runtime import Plugin

class MyPlugin(Plugin, Hooks):
    def __init__(self):
        super().__init__(name="my_plugin", version="1.0.0")

    def activate(self):
        ...
```

---

### `plugin_registry`

Global plugin registry.

```python
from rulesmith.runtime import plugin_registry

plugin_registry.register(MyPlugin())
```

**Methods:**
- `register(plugin)`: Register plugin
- `unregister(plugin_name)`: Unregister plugin
- `get(plugin_name) -> Plugin`: Get plugin
- `list() -> List[str]`: List all plugins

---

## Data Models

### `ABArm`

A/B testing arm.

```python
from rulesmith.io.ser import ABArm

arm = ABArm(node="variant_a", weight=0.5)
```

**Fields:**
- `node` (str): Target node name
- `weight` (float): Arm weight

---

### `RulebookSpec`

Rulebook specification (serializable).

**Fields:**
- `name` (str): Rulebook name
- `version` (str): Version
- `nodes` (List[NodeSpec]): Node specifications
- `edges` (List[Edge]): Edge list
- `metadata` (Dict[str, Any]): Metadata

---

## Error Classes

### `CircuitBreakerOpenError`

Raised when circuit breaker is open.

```python
from rulesmith.reliability import CircuitBreakerOpenError
```

---

## Constants

### `GuardAction`

Guardrail action types.

- `ALLOW`: Allow execution
- `BLOCK`: Block execution
- `OVERRIDE`: Override output
- `FLAG`: Flag for review

---

### `CircuitState`

Circuit breaker states.

- `CLOSED`: Normal operation
- `OPEN`: Failing
- `HALF_OPEN`: Testing recovery

---

## Utilities

### `hash_code`

Compute SHA256 hash of function code.

```python
from rulesmith.utils import hash_code

code_hash = hash_code(my_function)
```

---

## CLI Commands

### `rulesmith init`

Initialize a new Rulesmith project.

```bash
rulesmith init --name my_project --output ./project_dir
```

### `rulesmith run`

Run a rulebook from file.

```bash
rulesmith run payload.json --identity user123
```

---

## Examples

See `tests/examples/` for complete working examples:
- `ab_testing_example.py`: A/B testing
- `byom_genai_example.py`: BYOM and GenAI integration
- `guardrails_example.py`: Guardrails
- `hitl_example.py`: Human-in-the-Loop
- `governance_example.py`: Governance
- `phases8-11_example.py`: Evaluation, Reliability, Security, Schemas

