# Rulesmith Simplification and Enhancement Plan

## Executive Summary

This plan focuses on 3 major improvements:
1. **API Simplification**: Reduce complexity, remove redundancy, clarify "happy paths"
2. **Actionable Metrics via MLflow**: Transform logged metrics into actionable insights using MLflow's native capabilities
3. **Developer Experience**: Add debugging tools and better error handling

**Estimated Impact**: 
- Reduce API surface by ~30%
- Make metrics actionable through MLflow integration
- Improve onboarding time by 50%

---

## Part 1: API Simplification

### 1.1 Consolidate Context Types

**Problem**: Two context types (`RunContext`, `MLflowRunContext`) confuse users. Most users just want "it to work."

**Solution**: 
- Make `Rulebook.run()` auto-create appropriate context based on `enable_mlflow` flag
- Hide context creation from users by default
- Keep contexts for advanced users, but make them optional

**Files**: `rulesmith/dag/graph.py` (lines 479-487), `rulesmith/runtime/context.py`, `rulesmith/runtime/mlflow_context.py`

**Changes**:
```python
# Current (confusing):
if context is None:
    if enable_mlflow:
        context = MLflowRunContext(...)
    else:
        context = RunContext()

# Simplified (automatic):
# Context creation is internal, users don't see it
# MLflow integration happens automatically if MLflow is available
```

### 1.2 Remove Deprecated Methods

**Problem**: `add_langchain()`, `add_langgraph()` still exist with deprecation warnings, causing confusion.

**Solution**: Remove entirely. Users should use `add_model()` which handles both.

**Files**: `rulesmith/dag/graph.py` (lines 249-300)

**Changes**: Delete `add_langchain()` and `add_langgraph()` methods entirely.

### 1.3 Simplify Metrics API

**Problem**: Metrics can be added in two ways: at node creation or via `add_metrics()`. This is confusing.

**Solution**: 
- Keep both but make it clearer with better documentation
- Make `metrics` parameter accept both functions and rule names (strings)
- Add helper to convert rule names to functions

**Files**: `rulesmith/dag/graph.py` (lines 369-406)

**Changes**:
```python
# Current: metrics must be functions
rb.add_llm("gpt4", metrics=[check_pii, check_toxicity])

# Enhanced: metrics can be functions or rule names
rb.add_llm("gpt4", metrics=[check_pii, "check_toxicity"])  # Auto-resolve from registry
```

### 1.4 Reduce Execution Engine Complexity

**Problem**: `rulesmith/dag/execution.py` (343 lines) mixes concerns: execution, metrics extraction, MLflow logging, error handling, auto-mapping.

**Solution**: 
- Extract metrics extraction to `rulesmith/dag/metrics_extractor.py`
- Extract MLflow logging hook to `rulesmith/runtime/mlflow_hooks.py`
- Keep execution engine focused on DAG traversal only

**Files**: `rulesmith/dag/execution.py`

**Changes**: Split into:
- `execution.py` - Core DAG execution (200 lines)
- `metrics_extractor.py` - Extract metrics from node outputs (50 lines)
- `mlflow_hooks.py` - MLflow integration hooks (50 lines)

### 1.5 Simplify Rulebook.run() Signature

**Problem**: Too many optional parameters: `context`, `enable_mlflow`, `return_decision_result`, `mlflow_logger`

**Solution**: 
- Use sensible defaults: `enable_mlflow=True`, `return_decision_result=True`
- Make `mlflow_logger` optional - auto-create default if `enable_mlflow=True` and logger not provided
- Hide `context` parameter from typical users

**Files**: `rulesmith/dag/graph.py` (lines 452-551)

**Changes**:
```python
# Current:
rb.run(payload, context=None, enable_mlflow=True, return_decision_result=True, mlflow_logger=None)

# Simplified:
rb.run(payload, mlflow_logger=None)  # All else has sensible defaults
# MLflow logging happens automatically if MLflow is available
```

---

## Part 2: Making Metrics Actionable via MLflow

### 2.1 Unified Metrics System

**Problem**: Metrics scattered across `guardrails`, `metrics/business.py`, `metrics/operational.py`, `ab/metrics.py`. No single place to track all metrics.

**Solution**: Create `rulesmith/metrics/core.py` with unified `Metric` class that integrates with MLflow.

**Files**: Create `rulesmith/metrics/core.py`, update `rulesmith/metrics/__init__.py`

**Features**:
- Single `Metric` class with name, value, threshold, unit, tags
- `MetricRegistry` for tracking all metrics
- Automatic MLflow logging integration
- Store metrics in MLflow for querying later

### 2.2 Metric Thresholds with MLflow Integration

**Problem**: Metrics are logged to MLflow but there's no way to set thresholds or get alerts when metrics breach.

**Solution**: Add threshold system that stores thresholds in MLflow and checks them during execution.

**Files**: Create `rulesmith/metrics/thresholds.py`, integrate with MLflow

**Features**:
```python
# Define thresholds (stored in MLflow experiment tags)
rb.add_metric_threshold("guard_violations", max=0)
rb.add_metric_threshold("latency_ms", max=100)
rb.add_metric_threshold("cost_usd", max=0.01)

# Thresholds are checked during execution
# Breaches are logged to MLflow as metrics with "_breach" suffix
# MLflow can be queried for breaches: mlflow.search_runs(filter_string="metrics.guard_violations_breach > 0")
```

### 2.3 MLflow-Based Alerting

**Problem**: When metrics breach, nothing happens automatically.

**Solution**: Use MLflow's built-in capabilities + simple alert hooks that can be configured.

**Files**: Create `rulesmith/metrics/alerts.py`

**Features**:
```python
# Define alert actions (stored in MLflow experiment tags)
rb.on_metric_breach("guard_violations", action="log")  # Just log to MLflow
rb.on_metric_breach("latency_ms", action="hook", hook=my_alert_function)  # Call custom function
rb.on_metric_breach("cost_usd", action="route_to_hitl")  # Route to HITL queue

# Breaches are logged to MLflow
# Users can query MLflow for breaches and set up external alerting
# Or use hooks for custom actions
```

### 2.4 Metric Aggregation via MLflow Queries

**Problem**: Metrics are per-decision but no easy way to aggregate or analyze trends.

**Solution**: Add helper functions that query MLflow for aggregated metrics.

**Files**: Create `rulesmith/metrics/aggregation.py`

**Features**:
```python
from rulesmith.metrics import query_mlflow_metrics

# Query MLflow for aggregated metrics
agg = query_mlflow_metrics(
    experiment_name="rulesmith/credit_decision",
    metric_names=["latency_ms", "guard_violations"],
    filter_string="tags.rulebook_version='1.0.0'",
    aggregation=["mean", "p95", "max"],
    time_range="last_7_days"
)

# Returns dict with aggregated values
# Uses MLflow's native search_runs() API
```

### 2.5 MLflow Metric Comparison

**Problem**: No easy way to compare metrics across rulebook versions or A/B tests.

**Solution**: Add helper functions that use MLflow to compare metrics.

**Files**: Create `rulesmith/metrics/comparison.py`

**Features**:
```python
from rulesmith.metrics import compare_mlflow_metrics

# Compare metrics between versions
comparison = compare_mlflow_metrics(
    experiment_name="rulesmith/credit_decision",
    baseline_filter="tags.rulebook_version='1.0.0'",
    comparison_filter="tags.rulebook_version='1.1.0'",
    metrics=["latency_ms", "guard_violations", "cost_usd"]
)

# Returns dict with deltas, percent changes, significance
# Uses MLflow's native search_runs() API for comparison
```

---

## Part 3: Developer Experience Features

### 3.1 Rulebook Debugger

**Problem**: Hard to debug why a rule didn't fire or why execution took a path.

**Solution**: Add debug mode with step-by-step execution and state inspection.

**Files**: Create `rulesmith/dag/debugger.py`

**Features**:
```python
# Debug mode
result = rb.run(payload, debug=True)
# Returns DecisionResult with additional debug info:
# - debug_info.step_by_step: List of each node execution
# - debug_info.state_snapshots: State at each step
# - debug_info.execution_path: Which nodes executed and why
# - debug_info.metrics_per_node: Metrics for each node

# Or use debug context manager
with rb.debug(payload) as debug_ctx:
    result = debug_ctx.execute()
    # Inspect state at any point
    state = debug_ctx.get_state_at("node_name")
```

### 3.2 Rulebook Testing Framework

**Problem**: No built-in way to test rulebooks with test cases.

**Solution**: Add `RulebookTester` class for writing and running test suites.

**Files**: Create `rulesmith/testing/rulebook_tester.py`

**Features**:
```python
from rulesmith.testing import RulebookTester

tester = RulebookTester(rb)

tester.add_test_case(
    name="approved_high_score",
    inputs={"age": 30, "income": 75000, "credit_score": 720},
    expected={"approved": True}
)

tester.add_test_case(
    name="rejected_low_score",
    inputs={"age": 25, "income": 40000, "credit_score": 550},
    expected={"approved": False}
)

results = tester.run_all()
print(results.summary())
# Shows: passed/failed, actual vs expected, execution traces
```

### 3.3 Rulebook Version Comparison

**Problem**: No easy way to compare rulebook versions or see what changed.

**Solution**: Add `compare_rulebooks()` function to show diffs.

**Files**: Create `rulesmith/governance/comparison.py`

**Features**:
```python
from rulesmith.governance import compare_rulebooks

diff = compare_rulebooks(
    old_spec=rb1.to_spec(),
    new_spec=rb2.to_spec()
)

print(diff.summary())
# Shows: rules added/removed, nodes changed, edges changed
# Outputs structured diff that can be logged to MLflow
```

### 3.4 Smart Defaults and Auto-Configuration

**Problem**: Users have to configure many things manually (MLflow, metrics, etc.).

**Solution**: Add "smart defaults" that auto-configure based on environment detection.

**Files**: Create `rulesmith/dx/smart_defaults.py`

**Features**:
```python
# Auto-detect MLflow, create logger, configure sampling
rb = Rulebook(name="test", version="1.0.0", auto_configure=True)
# Automatically:
# - Detects MLflow availability
# - Creates MLflowLogger with sensible defaults (sample_rate=0.01 for prod)
# - Sets up experiment name based on rulebook name
# - Configures basic metric logging

# Or configure manually but with better defaults
rb = Rulebook(name="test", version="1.0.0")
rb.configure_mlflow(experiment_name="rulesmith/test")  # Simple one-liner
```

### 3.5 Better Error Messages

**Problem**: Error messages could be more helpful for debugging.

**Solution**: Enhance error messages with context and suggestions.

**Files**: `rulesmith/dx/errors.py` (already exists, enhance)

**Features**:
- Better error messages with node context
- Suggestions for common errors (e.g., "Did you mean to add the rule to the rulebook?")
- Links to relevant documentation
- Error codes for programmatic handling

---

## Part 4: Documentation Improvements

### 4.1 Improve Quickstart

**Problem**: Current quickstart is too simple, doesn't show real-world usage.

**Solution**: 
- Add "Common Patterns" section
- Add "Troubleshooting" section
- Add "Best Practices" section

**Files**: `docs/QUICKSTART.md`

### 4.2 Add Migration Guide

**Problem**: Users upgrading from old API don't know how to migrate.

**Solution**: Create `docs/MIGRATION.md` with before/after examples.

**Files**: Create `docs/MIGRATION.md`

### 4.3 Improve API Documentation

**Problem**: API docs could be clearer about when to use what.

**Solution**: 
- Add "When to use" sections for each major feature
- Add "Common pitfalls" sections
- Add more examples in docstrings

**Files**: `docs/API.md`, all docstrings

---

## Implementation Priority

### Phase 1: Quick Wins (High Impact, Low Effort)
1. Remove deprecated methods (`add_langchain`, `add_langgraph`)
2. Simplify `Rulebook.run()` signature with defaults
3. Auto-create MLflow logger if not provided
4. Consolidate context creation

### Phase 2: Core Improvements (High Impact, Medium Effort)
5. Split execution engine
6. Add unified metrics system
7. Add metric thresholds with MLflow integration
8. Add MLflow metric query helpers
9. Add rulebook tester

### Phase 3: Advanced Features (Medium Impact, Medium Effort)
10. Add rulebook debugger
11. Add rulebook version comparison
12. Add smart defaults
13. Add MLflow-based alerting hooks
14. Enhance error messages

---

## Success Metrics

- **API Simplicity**: Reduce `Rulebook` methods from 15 to 10
- **Metrics Actionability**: All metrics queryable via MLflow
- **Developer Experience**: 50% reduction in time to first working rulebook
- **MLflow Integration**: 100% of metrics accessible via MLflow queries

---

## Key Principles

1. **MLflow-First**: All metrics, logging, and analysis should leverage MLflow's native capabilities
2. **Developer-Friendly**: Simple, intuitive API with sensible defaults
3. **No Visualization**: Users use MLflow UI for visualization
4. **No Templates**: Users start from scratch but with better guidance
5. **Piggyback on MLflow**: Don't reinvent - use MLflow's search, filtering, comparison features

