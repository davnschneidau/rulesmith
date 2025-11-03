"""Admission controller for rulebook deployment validation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from rulesmith.io.ser import RulebookSpec


class CheckResult(Enum):
    """Result of an admission check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class AdmissionCheck:
    """Result of a single admission check."""
    
    name: str
    result: CheckResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdmissionResult:
    """Result of admission controller evaluation."""
    
    approved: bool
    checks: List[AdmissionCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approved": self.approved,
            "checks": [
                {
                    "name": c.name,
                    "result": c.result.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "warnings": self.warnings,
            "errors": self.errors,
        }


class AdmissionController:
    """Static checks for rulebook deployment safety."""
    
    def __init__(
        self,
        max_complexity: int = 1000,
        require_fallbacks: bool = True,
        block_cycles: bool = True,
        block_unbounded_recursion: bool = True,
        require_pii_guards: bool = False,
    ):
        """
        Initialize admission controller.
        
        Args:
            max_complexity: Maximum complexity score
            require_fallbacks: Require fallback routes
            block_cycles: Block DAG cycles
            block_unbounded_recursion: Block unbounded recursion
            require_pii_guards: Require PII guards on LLM nodes
        """
        self.max_complexity = max_complexity
        self.require_fallbacks = require_fallbacks
        self.block_cycles = block_cycles
        self.block_unbounded_recursion = block_unbounded_recursion
        self.require_pii_guards = require_pii_guards
    
    def check(
        self,
        spec: RulebookSpec,
        previous_spec: Optional[RulebookSpec] = None,
    ) -> AdmissionResult:
        """
        Run admission checks on a rulebook spec.
        
        Args:
            spec: Rulebook spec to check
            previous_spec: Optional previous spec for comparison
        
        Returns:
            AdmissionResult
        """
        checks = []
        warnings = []
        errors = []
        
        # Check for cycles
        if self.block_cycles:
            cycle_check = self._check_cycles(spec)
            checks.append(cycle_check)
            if cycle_check.result == CheckResult.FAIL:
                errors.append(cycle_check.message)
        
        # Check for unbounded recursion
        if self.block_unbounded_recursion:
            recursion_check = self._check_recursion(spec)
            checks.append(recursion_check)
            if recursion_check.result == CheckResult.FAIL:
                errors.append(recursion_check.message)
        
        # Check for fallbacks
        if self.require_fallbacks:
            fallback_check = self._check_fallbacks(spec)
            checks.append(fallback_check)
            if fallback_check.result == CheckResult.FAIL:
                errors.append(fallback_check.message)
        
        # Check complexity
        complexity_check = self._check_complexity(spec)
        checks.append(complexity_check)
        if complexity_check.result == CheckResult.FAIL:
            errors.append(complexity_check.message)
        elif complexity_check.result == CheckResult.WARNING:
            warnings.append(complexity_check.message)
        
        # Check PII guards
        if self.require_pii_guards:
            pii_check = self._check_pii_guards(spec)
            checks.append(pii_check)
            if pii_check.result == CheckResult.FAIL:
                errors.append(pii_check.message)
        
        # Check migration safety if previous spec provided
        if previous_spec:
            migration_check = self._check_migration_safety(spec, previous_spec)
            checks.append(migration_check)
            if migration_check.result == CheckResult.FAIL:
                errors.append(migration_check.message)
            elif migration_check.result == CheckResult.WARNING:
                warnings.append(migration_check.message)
        
        # Approved if no errors
        approved = len(errors) == 0
        
        return AdmissionResult(
            approved=approved,
            checks=checks,
            warnings=warnings,
            errors=errors,
        )
    
    def _check_cycles(self, spec: RulebookSpec) -> AdmissionCheck:
        """Check for cycles in DAG."""
        # Build adjacency list
        edges = {}
        for edge in spec.edges:
            if edge.source not in edges:
                edges[edge.source] = []
            edges[edge.source].append(edge.target)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in edges.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for node in [n.name for n in spec.nodes]:
            if node not in visited:
                if has_cycle(node):
                    return AdmissionCheck(
                        name="cycle_detection",
                        result=CheckResult.FAIL,
                        message=f"Cycle detected in DAG involving node '{node}'",
                        details={"cycle_node": node},
                    )
        
        return AdmissionCheck(
            name="cycle_detection",
            result=CheckResult.PASS,
            message="No cycles detected",
        )
    
    def _check_recursion(self, spec: RulebookSpec) -> AdmissionCheck:
        """Check for unbounded recursion."""
        # Simple check: look for nodes that might call themselves
        # This is a simplified check - real recursion detection would need more analysis
        
        # Check for self-loops
        for edge in spec.edges:
            if edge.source == edge.target:
                return AdmissionCheck(
                    name="recursion_detection",
                    result=CheckResult.FAIL,
                    message=f"Self-loop detected: '{edge.source}' -> '{edge.source}'",
                    details={"node": edge.source},
                )
        
        # More sophisticated recursion detection could check for cycles with no termination condition
        # For now, this is a basic check
        
        return AdmissionCheck(
            name="recursion_detection",
            result=CheckResult.PASS,
            message="No obvious unbounded recursion detected",
        )
    
    def _check_fallbacks(self, spec: RulebookSpec) -> AdmissionCheck:
        """Check for fallback routes."""
        # Check that gate nodes have else branches
        gate_nodes = [n.name for n in spec.nodes if n.kind == "gate"]
        missing_fallbacks = []
        
        for gate_name in gate_nodes:
            # Check if gate has an else edge
            has_else = any(
                e.source == gate_name and getattr(e, "condition", None) == "else"
                for e in spec.edges
            )
            
            if not has_else:
                missing_fallbacks.append(gate_name)
        
        if missing_fallbacks:
            return AdmissionCheck(
                name="fallback_check",
                result=CheckResult.FAIL,
                message=f"Missing fallback routes for gates: {', '.join(missing_fallbacks)}",
                details={"missing_fallbacks": missing_fallbacks},
            )
        
        return AdmissionCheck(
            name="fallback_check",
            result=CheckResult.PASS,
            message="All gates have fallback routes",
        )
    
    def _check_complexity(self, spec: RulebookSpec) -> AdmissionCheck:
        """Check complexity score."""
        # Simple complexity calculation: nodes + edges + predicates
        node_count = len(spec.nodes)
        edge_count = len(spec.edges)
        
        # Count predicates (rough estimate)
        predicate_count = 0
        for node in spec.nodes:
            if node.kind == "rule":
                # Estimate predicates from rule complexity
                predicate_count += 1  # At least one condition
            elif node.kind == "gate":
                predicate_count += 1  # Gate condition
        
        complexity_score = node_count * 10 + edge_count * 5 + predicate_count * 3
        
        if complexity_score > self.max_complexity:
            return AdmissionCheck(
                name="complexity_check",
                result=CheckResult.FAIL,
                message=f"Complexity score {complexity_score} exceeds maximum {self.max_complexity}",
                details={
                    "complexity_score": complexity_score,
                    "max_complexity": self.max_complexity,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "predicate_count": predicate_count,
                },
            )
        elif complexity_score > self.max_complexity * 0.8:
            return AdmissionCheck(
                name="complexity_check",
                result=CheckResult.WARNING,
                message=f"Complexity score {complexity_score} is high (80% of max)",
                details={
                    "complexity_score": complexity_score,
                    "max_complexity": self.max_complexity,
                },
            )
        
        return AdmissionCheck(
            name="complexity_check",
            result=CheckResult.PASS,
            message=f"Complexity score {complexity_score} is acceptable",
            details={"complexity_score": complexity_score},
        )
    
    def _check_pii_guards(self, spec: RulebookSpec) -> AdmissionCheck:
        """Check for PII guards on LLM nodes."""
        llm_nodes = [n.name for n in spec.nodes if n.kind in ("llm", "genai")]
        missing_guards = []
        
        # Check if LLM nodes have guard policies
        # This is a simplified check - at spec level we check metadata
        for node in spec.nodes:
            if node.kind in ("llm", "genai"):
                # Check if node has guard metadata in params
                # Real implementation would check actual guard attachments at runtime
                has_guard = False
                if hasattr(node, "params") and node.params:
                    # Check for guard-related metadata
                    has_guard = any(
                        "guard" in str(k).lower() or "pii" in str(k).lower()
                        for k in node.params.keys()
                    )
                
                if not has_guard:
                    missing_guards.append(node.name)
        
        if missing_guards:
            return AdmissionCheck(
                name="pii_guard_check",
                result=CheckResult.FAIL,
                message=f"LLM nodes missing PII guards: {', '.join(missing_guards)}",
                details={"missing_guards": missing_guards},
            )
        
        return AdmissionCheck(
            name="pii_guard_check",
            result=CheckResult.PASS,
            message="All LLM nodes have PII guards (or check disabled)",
        )
    
    def _check_migration_safety(
        self,
        new_spec: RulebookSpec,
        old_spec: RulebookSpec,
    ) -> AdmissionCheck:
        """Check migration safety between versions."""
        # Compare node counts
        old_node_count = len(old_spec.nodes)
        new_node_count = len(new_spec.nodes)
        node_delta = new_node_count - old_node_count
        
        # Check for removed nodes
        old_node_names = {n.name for n in old_spec.nodes}
        new_node_names = {n.name for n in new_spec.nodes}
        removed_nodes = old_node_names - new_node_names
        
        # Check for new nodes
        new_nodes = new_node_names - old_node_names
        
        details = {
            "node_count_delta": node_delta,
            "removed_nodes": list(removed_nodes),
            "new_nodes": list(new_nodes),
        }
        
        if removed_nodes:
            return AdmissionCheck(
                name="migration_safety",
                result=CheckResult.WARNING,
                message=f"Nodes removed: {', '.join(removed_nodes)}",
                details=details,
            )
        
        if abs(node_delta) > old_node_count * 0.2:  # More than 20% change
            return AdmissionCheck(
                name="migration_safety",
                result=CheckResult.WARNING,
                message=f"Significant node count change: {node_delta} nodes",
                details=details,
            )
        
        return AdmissionCheck(
            name="migration_safety",
            result=CheckResult.PASS,
            message="Migration appears safe",
            details=details,
        )


# Default admission controller
default_admission_controller = AdmissionController()

