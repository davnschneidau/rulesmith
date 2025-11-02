"""Diff system for comparing rulebook specifications."""

from typing import Any, Dict, List, Optional

from rulesmith.io.ser import Edge, NodeSpec, RulebookSpec


class RulebookDiff:
    """Difference between two rulebook specifications."""

    def __init__(self, from_spec: RulebookSpec, to_spec: RulebookSpec):
        self.from_spec = from_spec
        self.to_spec = to_spec
        self.added_nodes: List[NodeSpec] = []
        self.removed_nodes: List[NodeSpec] = []
        self.modified_nodes: List[Dict[str, Any]] = []
        self.added_edges: List[Edge] = []
        self.removed_edges: List[Edge] = []
        self.modified_metadata: Dict[str, Any] = {}

        self._compute_diff()

    def _compute_diff(self):
        """Compute differences between specs."""
        from_nodes = {node.name: node for node in self.from_spec.nodes}
        to_nodes = {node.name: node for node in self.to_spec.nodes}

        # Find added nodes
        for name, node in to_nodes.items():
            if name not in from_nodes:
                self.added_nodes.append(node)

        # Find removed nodes
        for name, node in from_nodes.items():
            if name not in to_nodes:
                self.removed_nodes.append(node)

        # Find modified nodes
        for name, to_node in to_nodes.items():
            if name in from_nodes:
                from_node = from_nodes[name]
                changes = self._compare_nodes(from_node, to_node)
                if changes:
                    changes["name"] = name
                    self.modified_nodes.append(changes)

        # Compare edges
        from_edges = {(e.source, e.target): e for e in self.from_spec.edges}
        to_edges = {(e.source, e.target): e for e in self.to_spec.edges}

        for key, edge in to_edges.items():
            if key not in from_edges:
                self.added_edges.append(edge)

        for key, edge in from_edges.items():
            if key not in to_edges:
                self.removed_edges.append(edge)

        # Compare metadata
        for key in set(self.from_spec.metadata.keys()) | set(self.to_spec.metadata.keys()):
            from_val = self.from_spec.metadata.get(key)
            to_val = self.to_spec.metadata.get(key)
            if from_val != to_val:
                self.modified_metadata[key] = {"from": from_val, "to": to_val}

    def _compare_nodes(self, from_node: NodeSpec, to_node: NodeSpec) -> Dict[str, Any]:
        """Compare two nodes and return differences."""
        changes = {}

        if from_node.kind != to_node.kind:
            changes["kind"] = {"from": from_node.kind, "to": to_node.kind}

        if from_node.rule_ref != to_node.rule_ref:
            changes["rule_ref"] = {"from": from_node.rule_ref, "to": to_node.rule_ref}

        if from_node.model_uri != to_node.model_uri:
            changes["model_uri"] = {"from": from_node.model_uri, "to": to_node.model_uri}

        if from_node.condition != to_node.condition:
            changes["condition"] = {"from": from_node.condition, "to": to_node.condition}

        # Compare params
        if from_node.params != to_node.params:
            changes["params"] = {"from": from_node.params, "to": to_node.params}

        # Compare A/B arms
        if from_node.ab_arms != to_node.ab_arms:
            changes["ab_arms"] = {"from": from_node.ab_arms, "to": to_node.ab_arms}

        return changes

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (
            len(self.added_nodes) > 0
            or len(self.removed_nodes) > 0
            or len(self.modified_nodes) > 0
            or len(self.added_edges) > 0
            or len(self.removed_edges) > 0
            or len(self.modified_metadata) > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert diff to dictionary."""
        return {
            "from": {
                "name": self.from_spec.name,
                "version": self.from_spec.version,
            },
            "to": {
                "name": self.to_spec.name,
                "version": self.to_spec.version,
            },
            "added_nodes": [{"name": n.name, "kind": n.kind} for n in self.added_nodes],
            "removed_nodes": [{"name": n.name, "kind": n.kind} for n in self.removed_nodes],
            "modified_nodes": self.modified_nodes,
            "added_edges": [{"source": e.source, "target": e.target} for e in self.added_edges],
            "removed_edges": [{"source": e.source, "target": e.target} for e in self.removed_edges],
            "modified_metadata": self.modified_metadata,
        }

    def to_string(self) -> str:
        """Convert diff to human-readable string."""
        lines = [
            f"Diff: {self.from_spec.name} v{self.from_spec.version} -> {self.to_spec.name} v{self.to_spec.version}",
            "",
        ]

        if self.added_nodes:
            lines.append("Added nodes:")
            for node in self.added_nodes:
                lines.append(f"  + {node.name} ({node.kind})")

        if self.removed_nodes:
            lines.append("\nRemoved nodes:")
            for node in self.removed_nodes:
                lines.append(f"  - {node.name} ({node.kind})")

        if self.modified_nodes:
            lines.append("\nModified nodes:")
            for change in self.modified_nodes:
                lines.append(f"  ~ {change['name']}")
                for key, val in change.items():
                    if key != "name":
                        lines.append(f"    {key}: {val.get('from')} -> {val.get('to')}")

        if self.added_edges:
            lines.append("\nAdded edges:")
            for edge in self.added_edges:
                lines.append(f"  + {edge.source} -> {edge.target}")

        if self.removed_edges:
            lines.append("\nRemoved edges:")
            for edge in self.removed_edges:
                lines.append(f"  - {edge.source} -> {edge.target}")

        if not self.has_changes():
            lines.append("\nNo changes detected.")

        return "\n".join(lines)


def diff_rulebooks(
    from_spec: RulebookSpec,
    to_spec: RulebookSpec,
) -> RulebookDiff:
    """
    Compute diff between two rulebook specifications.

    Args:
        from_spec: Source rulebook spec
        to_spec: Target rulebook spec

    Returns:
        RulebookDiff object
    """
    return RulebookDiff(from_spec, to_spec)


def diff_rulebook_uris(
    from_uri: str,
    to_uri: str,
) -> RulebookDiff:
    """
    Compute diff between two rulebook model URIs.

    Args:
        from_uri: Source model URI
        to_uri: Target model URI

    Returns:
        RulebookDiff object
    """
    import json

    import mlflow.pyfunc

    # Load rulebook specs from MLflow models
    from_model = mlflow.pyfunc.load_model(from_uri)
    to_model = mlflow.pyfunc.load_model(to_uri)

    # Get specs from artifacts
    from_spec_dict = None
    to_spec_dict = None

    try:
        import mlflow

        from_run = mlflow.get_run(from_model.metadata.run_id)
        to_run = mlflow.get_run(to_model.metadata.run_id)

        # Try to load rulebook_spec.json from artifacts
        # This is a simplified version - in practice would use artifact API
        pass
    except Exception:
        pass

    # If can't load from MLflow, raise error
    raise NotImplementedError("Loading specs from MLflow URIs not yet fully implemented")

