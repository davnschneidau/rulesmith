"""MLflow I/O utilities for rulebooks, lineage, and artifacts."""

import json
import os
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.entities import RunStatus

from rulesmith.io.ser import RulebookSpec


def start_run(
    rulebook_spec: RulebookSpec,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> mlflow.ActiveRun:
    """
    Start an MLflow run with rulebook lineage tags.

    Args:
        rulebook_spec: Rulebook specification
        params: Optional parameters
        tags: Optional tags

    Returns:
        Active MLflow run
    """
    run_tags = {
        "rulesmith.rulebook_name": rulebook_spec.name,
        "rulesmith.rulebook_version": rulebook_spec.version,
    }

    if tags:
        run_tags.update(tags)

    run = mlflow.start_run(tags=run_tags)
    if params:
        mlflow.log_params(params)

    return run


def end_run(
    status: RunStatus = RunStatus.FINISHED,
    metrics: Optional[Dict[str, float]] = None,
    artifacts: Optional[List[str]] = None,
) -> None:
    """
    End the current MLflow run.

    Args:
        status: Run status
        metrics: Optional metrics to log
        artifacts: Optional artifact paths to log
    """
    if metrics:
        mlflow.log_metrics(metrics)

    if artifacts:
        for artifact_path in artifacts:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

    mlflow.end_run(status=status)


def log_rulebook(spec: RulebookSpec, artifact_path: str = "rulebook_spec.json") -> str:
    """
    Log rulebook specification as artifact.

    Args:
        spec: Rulebook specification
        artifact_path: Artifact path

    Returns:
        Path to logged artifact
    """
    spec_dict = spec.model_dump()
    mlflow.log_dict(spec_dict, artifact_path)
    return artifact_path


def log_lineage(rulebook_spec: RulebookSpec, artifact_path: str = "lineage.json") -> str:
    """
    Create and log lineage artifact with all node references.

    Args:
        rulebook_spec: Rulebook specification
        artifact_path: Artifact path

    Returns:
        Path to logged artifact
    """
    lineage = {
        "rulebook": {
            "name": rulebook_spec.name,
            "version": rulebook_spec.version,
        },
        "nodes": [],
    }

    for node in rulebook_spec.nodes:
        node_ref = {
            "name": node.name,
            "kind": node.kind,
        }

        if node.rule_ref:
            node_ref["rule_ref"] = node.rule_ref

        if node.model_uri:
            node_ref["model_uri"] = node.model_uri

        lineage["nodes"].append(node_ref)

    mlflow.log_dict(lineage, artifact_path)
    return artifact_path

