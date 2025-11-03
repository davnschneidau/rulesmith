"""Single-file rulebook format (.rsm) for defining rulebooks in a single file."""

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from rulesmith.dag.graph import Rulebook
from rulesmith.dag.nodes import LLMNode, ModelNode, RuleNode
from rulesmith.io.ser import ABArm, Edge, NodeSpec, RulebookSpec


class RulebookDSL:
    """DSL for defining rulebooks in a single file."""
    
    def __init__(self):
        self.name: Optional[str] = None
        self.version: str = "1.0.0"
        self.metadata: Dict[str, Any] = {}
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.rules: Dict[str, Callable] = {}
    
    def rulebook(self, name: str, version: str = "1.0.0", **metadata):
        """Define rulebook metadata."""
        self.name = name
        self.version = version
        self.metadata = metadata
        return self
    
    def rule(self, name: Optional[str] = None, inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None):
        """Decorator for defining a rule function."""
        def decorator(func: Callable) -> Callable:
            rule_name = name or func.__name__
            self.rules[rule_name] = func
            
            # Store metadata on function
            func._rule_name = rule_name
            func._rule_inputs = inputs
            func._rule_outputs = outputs
            return func
        return decorator
    
    def add_node(self, kind: str, name: str, **kwargs):
        """Add a node to the rulebook."""
        node = {"kind": kind, "name": name, **kwargs}
        self.nodes.append(node)
        return self
    
    def add_edge(self, source: str, target: str, mapping: Optional[Dict[str, str]] = None):
        """Add an edge between nodes."""
        edge = {"source": source, "target": target, "mapping": mapping or {}}
        self.edges.append(edge)
        return self
    
    def build(self) -> Rulebook:
        """Build a Rulebook instance from the DSL."""
        if not self.name:
            raise ValueError("Rulebook name must be set")
        
        rb = Rulebook(name=self.name, version=self.version, metadata=self.metadata)
        
        # Add all rules first
        for rule_name, rule_func in self.rules.items():
            rb.add_rule(rule_func, as_name=rule_name)
        
        # Add all nodes
        for node_def in self.nodes:
            kind = node_def.pop("kind")
            name = node_def.pop("name")
            
            if kind == "rule":
                rule_name = node_def.pop("rule_ref", name)
                if rule_name not in self.rules:
                    raise ValueError(f"Rule '{rule_name}' not found")
                # Rule already added above
                continue
            elif kind == "model":
                rb.add_model(
                    name=name,
                    model_uri=node_def.pop("model_uri", None),
                    langchain_model=node_def.pop("langchain_model", None),
                    params=node_def.pop("params", {}),
                )
            elif kind == "llm":
                rb.add_llm(
                    name=name,
                    model_name=node_def.pop("model_name", None),
                    provider=node_def.pop("provider", None),
                    model_uri=node_def.pop("model_uri", None),
                    gateway_uri=node_def.pop("gateway_uri", None),
                    params=node_def.pop("params", {}),
                )
            elif kind == "gate":
                condition = node_def.pop("condition")
                from rulesmith.dag.functions import gate
                # Gate is handled as a function, not a node
                # We'll store it for later edge evaluation
                node_def["condition"] = condition
                # For now, we'll add it as metadata
                rb.metadata[f"_gate_{name}"] = condition
            elif kind == "fork":
                arms = node_def.pop("arms", [])
                # Fork is also a function
                rb.metadata[f"_fork_{name}"] = arms
            else:
                raise ValueError(f"Unknown node kind: {kind}")
            
            # Store remaining params
            if node_def:
                rb.metadata[f"_node_{name}_params"] = node_def
        
        # Add all edges
        for edge_def in self.edges:
            source = edge_def["source"]
            target = edge_def["target"]
            mapping = edge_def.get("mapping", {})
            rb.connect(source, target, mapping=mapping)
        
        return rb


def load_rsm_file(file_path: Union[str, Path]) -> Rulebook:
    """
    Load a .rsm (Rulebook Single File) and build a Rulebook.
    
    Supports:
    - Python files with DSL definition
    - JSON/YAML files with rulebook spec
    
    Args:
        file_path: Path to .rsm file
    
    Returns:
        Built Rulebook instance
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Rulebook file not found: {file_path}")
    
    if file_path.suffix == ".py":
        return _load_python_rsm(file_path)
    elif file_path.suffix in [".json", ".yaml", ".yml"]:
        return _load_spec_rsm(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _load_python_rsm(file_path: Path) -> Rulebook:
    """Load a Python .rsm file."""
    # Read and execute the file
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Parse and execute in a controlled namespace
    namespace = {}
    exec(code, namespace)
    
    # Look for DSL instance
    dsl = None
    for value in namespace.values():
        if isinstance(value, RulebookDSL):
            dsl = value
            break
    
    if dsl is None:
        # Try to find a variable named 'rb' or 'rulebook'
        dsl = namespace.get("rb") or namespace.get("rulebook")
        if not isinstance(dsl, RulebookDSL):
            raise ValueError("No RulebookDSL instance found in file")
    
    return dsl.build()


def _load_spec_rsm(file_path: Path) -> Rulebook:
    """Load a JSON/YAML .rsm file."""
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif file_path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML rulebook files")
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    # Convert to RulebookSpec and build
    spec = RulebookSpec(**data)
    return _build_from_spec(spec)


def _build_from_spec(spec: RulebookSpec) -> Rulebook:
    """Build a Rulebook from a RulebookSpec."""
    from rulesmith.dag.registry import rule_registry
    
    rb = Rulebook(name=spec.name, version=spec.version, metadata=spec.metadata)
    
    # Add nodes
    for node_spec in spec.nodes:
        if node_spec.kind == "rule":
            # Look up rule in registry
            rule_func = rule_registry.get_rule(node_spec.rule_ref)
            if rule_func:
                rb.add_rule(rule_func, as_name=node_spec.name)
            else:
                raise ValueError(f"Rule '{node_spec.rule_ref}' not found in registry")
        elif node_spec.kind == "model":
            rb.add_model(
                name=node_spec.name,
                model_uri=node_spec.model_uri,
                params=node_spec.params,
            )
        elif node_spec.kind == "llm":
            rb.add_llm(
                name=node_spec.name,
                model_name=node_spec.params.get("model_name"),
                provider=node_spec.params.get("provider"),
                model_uri=node_spec.model_uri,
                gateway_uri=node_spec.params.get("gateway_uri"),
                params=node_spec.params,
            )
        # Other node types...
    
    # Add edges
    for edge in spec.edges:
        rb.connect(edge.source, edge.target, mapping=edge.mapping)
    
    return rb


def save_rsm_file(rulebook: Rulebook, file_path: Union[str, Path], format: str = "json"):
    """
    Save a Rulebook to a .rsm file.
    
    Args:
        rulebook: Rulebook instance
        file_path: Output file path
        format: Output format ("json", "yaml", or "python")
    """
    file_path = Path(file_path)
    spec = rulebook.to_spec()
    
    if format == "json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(spec.model_dump(), f, indent=2)
    elif format in ["yaml", "yml"]:
        try:
            import yaml
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(spec.model_dump(), f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML output")
    elif format == "python":
        # Generate Python DSL code
        code = _generate_python_dsl(rulebook)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_python_dsl(rulebook: Rulebook) -> str:
    """Generate Python DSL code from a Rulebook."""
    lines = [
        "from rulesmith.dx.single_file import RulebookDSL",
        "",
        "rb = RulebookDSL()",
        f'rb.rulebook(name="{rulebook.name}", version="{rulebook.version}")',
        "",
    ]
    
    # Add rules (would need to extract from rule functions)
    # This is simplified - in practice, you'd need to serialize rule code
    
    # Add nodes
    spec = rulebook.to_spec()
    for node in spec.nodes:
        if node.kind == "model":
            lines.append(f'rb.add_node(kind="model", name="{node.name}", model_uri="{node.model_uri}")')
        elif node.kind == "llm":
            lines.append(f'rb.add_node(kind="llm", name="{node.name}", model_name="{node.params.get("model_name")}")')
        # Other node types...
    
    # Add edges
    for edge in spec.edges:
        mapping_str = str(edge.mapping) if edge.mapping else "{}"
        lines.append(f'rb.add_edge("{edge.source}", "{edge.target}", mapping={mapping_str})')
    
    lines.append("")
    lines.append("rulebook = rb.build()")
    
    return "\n".join(lines)

