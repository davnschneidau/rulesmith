# Rulesmith Architecture

## Overview

Rulesmith is a rulebook/DAG execution engine with deep MLflow integration for production ML/AI systems.

## Core Components

### 1. DAG Engine (`rulesmith/dag/`)
- **Rulebook**: DAG builder and executor
- **Nodes**: Rule, Fork, Gate, BYOM, GenAI, HITL, LangChain, LangGraph
- **Execution Engine**: Topological sorting, field mapping, state management
- **Scheduler**: Execution order calculation

### 2. MLflow Integration (`rulesmith/mlflow_flavor.py`, `rulesmith/io/mlflow_io.py`)
- **Pyfunc Flavor**: Rulebooks as MLflow models
- **Nested Runs**: Per-node execution tracking
- **Traces**: GenAI token/cost tracking
- **Lineage**: Code hashes, model URIs

### 3. Model Integration (`rulesmith/models/`)
- **BYOM**: Bring Your Own Model (MLflow models)
- **LangChain**: Chain integration
- **LangGraph**: Graph integration
- **GenAI**: Provider-agnostic LLM wrapper

### 4. A/B Testing (`rulesmith/ab/`)
- **Policies**: Hash, Random, Thompson Sampling, UCB, Epsilon-Greedy
- **Traffic Management**: Deterministic and adaptive allocation

### 5. Guardrails (`rulesmith/guardrails/`)
- **Guards**: PII, Toxicity, Hallucination, Output Validation
- **Policies**: ALLOW, BLOCK, OVERRIDE, FLAG
- **Execution**: Automatic guard evaluation

### 6. Human-in-the-Loop (`rulesmith/hitl/`)
- **Queues**: LocalFile, InMemory, Postgres, Redis, Slack
- **Active Learning**: Confidence-based review skipping

### 7. Governance (`rulesmith/governance/`)
- **Promotion**: SLO-gated model promotion
- **Diff**: Rulebook comparison
- **Lineage**: Provenance tracking
- **Audit**: Signed audit logs

### 8. Evaluation (`rulesmith/evaluation/`)
- **MLflow Integration**: Native `mlflow.evaluate()`
- **Scorers**: Accuracy, Precision, Recall, F1, Business KPI, LLM-as-judge
- **Datasets**: Loading, slicing, fairness analysis

### 9. Reliability (`rulesmith/reliability/`)
- **Retry**: Exponential backoff with jitter
- **Rate Limiting**: Token bucket algorithm
- **Circuit Breaker**: Fault tolerance
- **Shadow Mode**: Safe testing

### 10. Security (`rulesmith/security/`)
- **Redaction**: PII removal
- **Secret Detection**: Pattern matching
- **Field Suppression**: Complete removal

### 11. Schema (`rulesmith/io/schemas.py`)
- **Validation**: Type, range, length, enum
- **Inference**: Auto-generate from data
- **Contracts**: Input/output validation

### 12. Runtime (`rulesmith/runtime/`)
- **Context**: Execution context with MLflow
- **Hooks**: Plugin system
- **Caching**: In-memory cache

## Data Flow

1. **Input**: Payload dictionary
2. **Execution**: Nodes execute in topological order
3. **Guards**: Evaluate after node execution
4. **Resolution**: Merge outputs with priority policy
5. **Output**: Final state dictionary

## Extension Points

- **Hooks**: Custom execution hooks
- **Plugins**: Extensible plugin system
- **Custom Nodes**: Subclass `Node` base class
- **Custom Scorers**: Evaluation functions
- **Custom Guards**: Guard functions

