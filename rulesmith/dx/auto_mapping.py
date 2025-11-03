"""Automatic field mapping between nodes."""

from typing import Any, Dict, List, Optional, Set, Tuple

from rulesmith.dx.errors import MappingError, error_handler
from rulesmith.dx.typing import type_validator


class AutoMapper:
    """Automatic field mapping between nodes."""
    
    def __init__(self, strict: bool = False):
        """
        Initialize auto-mapper.
        
        Args:
            strict: If True, require exact field name matches (no inference)
        """
        self.strict = strict
    
    def infer_mapping(
        self,
        source_outputs: Dict[str, Any],
        target_inputs: List[str],
        existing_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Infer field mapping from source outputs to target inputs.
        
        Args:
            source_outputs: Dictionary of source node outputs
            target_inputs: List of target node input field names
            existing_mapping: Optional existing mapping to preserve
        
        Returns:
            Mapping dictionary {target_field: source_field}
        """
        mapping = existing_mapping or {}
        
        # Find unmapped target inputs
        unmapped_targets = [t for t in target_inputs if t not in mapping.values()]
        
        if not unmapped_targets:
            return mapping
        
        # Available source outputs (not already mapped)
        mapped_sources = set(mapping.keys())
        available_sources = {k: v for k, v in source_outputs.items() if k not in mapped_sources}
        
        # Try exact matches first
        for target in unmapped_targets:
            if target in available_sources:
                mapping[target] = target
                continue
        
        # Try case-insensitive matches
        if not self.strict:
            source_lower = {k.lower(): k for k in available_sources.keys()}
            for target in unmapped_targets:
                target_lower = target.lower()
                if target_lower in source_lower:
                    source_key = source_lower[target_lower]
                    mapping[target] = source_key
                    continue
        
        # Try prefix/suffix matches
        if not self.strict:
            for target in unmapped_targets:
                # Try various naming patterns
                candidates = self._find_candidates(target, available_sources.keys())
                if candidates:
                    # Use first candidate
                    mapping[target] = candidates[0]
        
        return mapping
    
    def _find_candidates(self, target: str, available: List[str]) -> List[str]:
        """Find candidate source fields for a target field."""
        candidates = []
        target_lower = target.lower()
        
        for source in available:
            source_lower = source.lower()
            
            # Exact match (already handled, but check)
            if source_lower == target_lower:
                candidates.insert(0, source)
                continue
            
            # Prefix match (e.g., "user_age" -> "age")
            if target_lower.endswith(source_lower) or source_lower.endswith(target_lower):
                candidates.append(source)
                continue
            
            # Suffix match (e.g., "age" -> "user_age")
            if target_lower.startswith(source_lower) or source_lower.startswith(target_lower):
                candidates.append(source)
                continue
        
        return candidates
    
    def validate_mapping(
        self,
        source_outputs: Dict[str, Any],
        target_inputs: List[str],
        mapping: Dict[str, str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a mapping is complete.
        
        Args:
            source_outputs: Source node outputs
            target_inputs: Target node input field names
            mapping: Mapping dictionary {target_field: source_field}
        
        Returns:
            Tuple of (is_valid, missing_fields)
        """
        missing_fields = []
        
        for target in target_inputs:
            if target not in mapping:
                missing_fields.append(target)
                continue
            
            source_field = mapping[target]
            if source_field not in source_outputs:
                missing_fields.append(f"{target} (source: {source_field})")
        
        return (len(missing_fields) == 0, missing_fields)
    
    def apply_mapping(
        self,
        source_outputs: Dict[str, Any],
        target_inputs: List[str],
        mapping: Dict[str, str],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply mapping to create target inputs from source outputs.
        
        Args:
            source_outputs: Source node outputs
            target_inputs: Target node input field names
            mapping: Mapping dictionary {target_field: source_field}
            strict: If True, raise error on missing fields
        
        Returns:
            Mapped inputs for target node
        
        Raises:
            MappingError: If mapping is incomplete and strict=True
        """
        mapped_inputs = {}
        
        # Apply mapping
        for target in target_inputs:
            if target in mapping:
                source_field = mapping[target]
                if source_field in source_outputs:
                    mapped_inputs[target] = source_outputs[source_field]
                elif strict:
                    raise MappingError(
                        message=f"Source field '{source_field}' not found in outputs",
                        missing_fields=[target],
                    )
            elif strict:
                raise MappingError(
                    message=f"No mapping for target field '{target}'",
                    missing_fields=[target],
                )
        
        return mapped_inputs
    
    def auto_map_edge(
        self,
        source_outputs: Dict[str, Any],
        target_node_spec: Any,
        existing_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Automatically map an edge from source outputs to target node.
        
        Args:
            source_outputs: Source node outputs
            target_node_spec: Target node specification (NodeSpec or similar)
            existing_mapping: Optional existing mapping
        
        Returns:
            Complete mapping dictionary
        """
        # Get target inputs (from rule spec or node spec)
        target_inputs = []
        
        if hasattr(target_node_spec, "inputs"):
            target_inputs = target_node_spec.inputs
        elif hasattr(target_node_spec, "rule_ref"):
            # Look up rule inputs
            from rulesmith.dag.registry import rule_registry
            rule_func = rule_registry.get_rule(target_node_spec.rule_ref)
            if rule_func and hasattr(rule_func, "_rule_spec"):
                target_inputs = rule_func._rule_spec.inputs
        
        if not target_inputs:
            # No inputs specified - return empty mapping (will use full state)
            return existing_mapping or {}
        
        # Infer mapping
        mapping = self.infer_mapping(source_outputs, target_inputs, existing_mapping)
        
        # Validate
        is_valid, missing = self.validate_mapping(source_outputs, target_inputs, mapping)
        
        if not is_valid and self.strict:
            raise MappingError(
                message=f"Incomplete mapping: missing fields {missing}",
                target_node=target_node_spec.name if hasattr(target_node_spec, "name") else None,
                missing_fields=missing,
            )
        
        return mapping


# Global auto-mapper
auto_mapper = AutoMapper(strict=False)

