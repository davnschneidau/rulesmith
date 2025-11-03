"""Lineage tracking and provenance."""

from typing import Any, Dict, List, Optional

from rulesmith.io.ser import NodeRef, RulebookSpec


class LineageGraph:
    """Lineage graph for a rulebook."""

    def __init__(self, rulebook_spec: RulebookSpec):
        self.spec = rulebook_spec
        self.nodes: Dict[str, NodeRef] = {}
        self._build_lineage()

    def _build_lineage(self):
        """Build lineage graph from rulebook spec."""
        from rulesmith.dag.registry import rule_registry

        for node in self.spec.nodes:
            node_ref = NodeRef(
                kind=node.kind,
                name=node.name,
            )

            if node.kind == "rule" and node.rule_ref:
                # Get rule spec
                rule_spec = rule_registry.get_spec(node.rule_ref)
                if rule_spec:
                    node_ref.code_hash = rule_spec.code_hash
                    node_ref.signature = f"{rule_spec.name}({', '.join(rule_spec.inputs)}) -> {', '.join(rule_spec.outputs)}"

            elif node.kind in ("model", "byom", "llm", "langchain", "langgraph") and node.model_uri:
                node_ref.uri = node.model_uri
                # Could fetch model signature from MLflow

            self.nodes[node.name] = node_ref

    def get_node_lineage(self, node_name: str) -> Optional[NodeRef]:
        """Get lineage information for a node."""
        return self.nodes.get(node_name)

    def get_all_lineage(self) -> Dict[str, NodeRef]:
        """Get all node lineage information."""
        return self.nodes.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert lineage to dictionary."""
        return {
            "rulebook": {
                "name": self.spec.name,
                "version": self.spec.version,
            },
            "nodes": {name: {
                "kind": ref.kind,
                "name": ref.name,
                "uri": ref.uri,
                "code_hash": ref.code_hash,
                "signature": ref.signature,
            } for name, ref in self.nodes.items()},
        }


def build_lineage(rulebook_spec: RulebookSpec) -> LineageGraph:
    """
    Build lineage graph from rulebook specification.

    Args:
        rulebook_spec: Rulebook specification

    Returns:
        LineageGraph object
    """
    return LineageGraph(rulebook_spec)


def get_model_lineage(model_uri: str) -> Dict[str, Any]:
    """
    Get lineage information for a model URI.

    Args:
        model_uri: Model URI

    Returns:
        Dictionary with lineage information
    """
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()

        # Get model version
        if model_uri.startswith("models:/"):
            parts = model_uri.split("/")
            model_name = parts[1]
            version_or_stage = parts[2] if len(parts) > 2 else "latest"

            if version_or_stage.startswith("@") or version_or_stage == "latest":
                stage = version_or_stage.replace("@", "")
                versions = client.search_model_versions(f"name='{model_name}'")
                version = next((v for v in versions if v.current_stage == stage), None)
            else:
                version = client.get_model_version(model_name, version_or_stage)

            if version:
                run = client.get_run(version.run_id)
                return {
                    "model_name": model_name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                    "tags": dict(version.tags),
                    "run_tags": dict(run.data.tags),
                    "run_params": dict(run.data.params),
                }

        return {}

    except Exception:
        return {}
