"""Serialization models for rules, rulebooks, nodes, and edges."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeRef(BaseModel):
    """Reference to a node for lineage tracking."""

    kind: str = Field(..., description="Node kind: rule|fork|gate|model|byom|llm|hitl")
    name: str = Field(..., description="Node name")
    uri: Optional[str] = Field(None, description="Model URI or code location")
    code_hash: Optional[str] = Field(None, description="SHA256 hash of code")
    signature: Optional[str] = Field(None, description="Function signature or model signature")


class Edge(BaseModel):
    """Edge connecting two nodes in the DAG."""

    source: str = Field(..., description="Source node name")
    target: str = Field(..., description="Target node name")
    mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Field mapping from source output to target input",
    )


class ABArm(BaseModel):
    """A/B test arm definition."""

    node: str = Field(..., description="Node name for this arm")
    weight: float = Field(..., ge=0.0, le=1.0, description="Traffic weight (0.0-1.0)")


class RuleSpec(BaseModel):
    """Specification for a rule function."""

    name: str = Field(..., description="Rule name")
    version: str = Field(default="1.0.0", description="Rule version")
    inputs: List[str] = Field(default_factory=list, description="Input field names")
    outputs: List[str] = Field(default_factory=list, description="Output field names")
    code_hash: Optional[str] = Field(None, description="SHA256 hash of source code")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameters for the rule",
    )


class NodeSpec(BaseModel):
    """Specification for a node in the rulebook."""

    name: str = Field(..., description="Node name")
    kind: str = Field(
        ...,
        description="Node type: rule|fork|gate|model|byom|llm|hitl",
    )
    rule_ref: Optional[str] = Field(None, description="Reference to rule name if kind=rule")
    model_uri: Optional[str] = Field(None, description="Model URI if kind=model|byom|llm")
    ab_arms: Optional[List[ABArm]] = Field(None, description="A/B arms if kind=fork")
    condition: Optional[str] = Field(None, description="Condition expression if kind=gate")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node-specific parameters",
    )


class RulebookSpec(BaseModel):
    """Complete specification for a rulebook/DAG."""

    name: str = Field(..., description="Rulebook name")
    version: str = Field(..., description="Rulebook version")
    nodes: List[NodeSpec] = Field(default_factory=list, description="List of nodes")
    edges: List[Edge] = Field(default_factory=list, description="List of edges")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def get_node_by_name(self, name: str) -> Optional[NodeSpec]:
        """Get a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_edges_from(self, source: str) -> List[Edge]:
        """Get all edges from a source node."""
        return [edge for edge in self.edges if edge.source == source]

    def get_edges_to(self, target: str) -> List[Edge]:
        """Get all edges to a target node."""
        return [edge for edge in self.edges if edge.target == target]

