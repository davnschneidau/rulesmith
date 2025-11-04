"""Rulebook version comparison."""

from typing import Any, Dict, List, Optional

from rulesmith.io.ser import Edge, NodeSpec, RulebookSpec


class RulebookDiff:
    """Represents differences between two rulebook versions."""
    
    def __init__(
        self,
        old_spec: RulebookSpec,
        new_spec: RulebookSpec,
    ):
        """
        Initialize rulebook diff.
        
        Args:
            old_spec: Old rulebook specification
            new_spec: New rulebook specification
        """
        self.old_spec = old_spec
        self.new_spec = new_spec
        self._compute_diff()
    
    def _compute_diff(self) -> None:
        """Compute differences between specs."""
        old_nodes = {node.name: node for node in self.old_spec.nodes}
        new_nodes = {node.name: node for node in self.new_spec.nodes}
        
        old_node_names = set(old_nodes.keys())
        new_node_names = set(new_nodes.keys())
        
        self.nodes_added = sorted(new_node_names - old_node_names)
        self.nodes_removed = sorted(old_node_names - new_node_names)
        self.nodes_changed = []
        
        # Check for changed nodes
        for node_name in old_node_names & new_node_names:
            old_node = old_nodes[node_name]
            new_node = new_nodes[node_name]
            if self._nodes_different(old_node, new_node):
                self.nodes_changed.append(node_name)
        
        # Compare edges
        old_edges = {(e.source, e.target) for e in self.old_spec.edges}
        new_edges = {(e.source, e.target) for e in self.new_spec.edges}
        
        self.edges_added = sorted(new_edges - old_edges)
        self.edges_removed = sorted(old_edges - new_edges)
        
        # Compare metadata
        self.metadata_changed = self.old_spec.metadata != self.new_spec.metadata
    
    def _nodes_different(self, old_node: NodeSpec, new_node: NodeSpec) -> bool:
        """Check if two nodes are different."""
        if old_node.kind != new_node.kind:
            return True
        if old_node.params != new_node.params:
            return True
        # Add more comparison logic as needed
        return False
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Rulebook Comparison: {self.old_spec.name}",
            f"  Old Version: {self.old_spec.version}",
            f"  New Version: {self.new_spec.version}",
            "",
        ]
        
        if self.nodes_added:
            lines.append(f"Nodes Added ({len(self.nodes_added)}):")
            for node in self.nodes_added:
                lines.append(f"  + {node}")
            lines.append("")
        
        if self.nodes_removed:
            lines.append(f"Nodes Removed ({len(self.nodes_removed)}):")
            for node in self.nodes_removed:
                lines.append(f"  - {node}")
            lines.append("")
        
        if self.nodes_changed:
            lines.append(f"Nodes Changed ({len(self.nodes_changed)}):")
            for node in self.nodes_changed:
                lines.append(f"  ~ {node}")
            lines.append("")
        
        if self.edges_added:
            lines.append(f"Edges Added ({len(self.edges_added)}):")
            for edge in self.edges_added:
                lines.append(f"  + {edge[0]} -> {edge[1]}")
            lines.append("")
        
        if self.edges_removed:
            lines.append(f"Edges Removed ({len(self.edges_removed)}):")
            for edge in self.edges_removed:
                lines.append(f"  - {edge[0]} -> {edge[1]}")
            lines.append("")
        
        if self.metadata_changed:
            lines.append("Metadata Changed: Yes")
        
        if not any([
            self.nodes_added,
            self.nodes_removed,
            self.nodes_changed,
            self.edges_added,
            self.edges_removed,
            self.metadata_changed,
        ]):
            lines.append("No changes detected.")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "old_version": self.old_spec.version,
            "new_version": self.new_spec.version,
            "nodes_added": self.nodes_added,
            "nodes_removed": self.nodes_removed,
            "nodes_changed": self.nodes_changed,
            "edges_added": [list(e) for e in self.edges_added],
            "edges_removed": [list(e) for e in self.edges_removed],
            "metadata_changed": self.metadata_changed,
        }


def compare_rulebooks(
    old_spec: RulebookSpec,
    new_spec: RulebookSpec,
) -> RulebookDiff:
    """
    Compare two rulebook specifications.
    
    Args:
        old_spec: Old rulebook specification
        new_spec: New rulebook specification
    
    Returns:
        RulebookDiff object with comparison results
    
    Examples:
        from rulesmith.governance import compare_rulebooks
        
        diff = compare_rulebooks(
            old_spec=rb1.to_spec(),
            new_spec=rb2.to_spec()
        )
        
        print(diff.summary())
        # Shows: nodes added/removed, edges changed, etc.
    """
    return RulebookDiff(old_spec, new_spec)

