"""Dead rule detection - identify rules that never fire or are unreachable."""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from rulesmith.dag.scheduler import topological_sort
from rulesmith.io.decision_result import DecisionResult, FiredRule
from rulesmith.io.ser import Edge, RulebookSpec


@dataclass
class DeadRuleReport:
    """Report on dead rules."""
    
    rule_name: str
    rule_id: str
    status: str  # "never_fired", "unreachable", "always_skipped", "low_usage"
    days_since_last_fire: Optional[int] = None
    execution_count: int = 0
    skip_count: int = 0
    last_fired: Optional[datetime] = None
    reachability_info: Optional[Dict[str, Any]] = None
    suggestions: List[str] = field(default_factory=list)


class DeadRuleDetector:
    """Detects dead rules in rulebooks."""
    
    def __init__(self, execution_history: Optional[List[DecisionResult]] = None):
        """
        Initialize dead rule detector.
        
        Args:
            execution_history: Optional list of DecisionResult objects for analysis
        """
        self.execution_history = execution_history or []
        self.rule_stats: Dict[str, Dict[str, Any]] = {}
    
    def analyze_rulebook(
        self,
        spec: RulebookSpec,
        execution_history: Optional[List[DecisionResult]] = None,
        min_executions: int = 10,
        days_threshold: int = 30,
    ) -> List[DeadRuleReport]:
        """
        Analyze rulebook for dead rules.
        
        Args:
            spec: Rulebook specification
            execution_history: Optional execution history (uses instance history if None)
            min_executions: Minimum executions to consider rule "alive"
            days_threshold: Days since last execution to consider rule "dead"
        
        Returns:
            List of DeadRuleReport objects
        """
        history = execution_history or self.execution_history
        
        # Build rule statistics
        self._build_rule_stats(history)
        
        # Get all rule nodes
        rule_nodes = [node for node in spec.nodes if node.kind == "rule"]
        
        reports = []
        
        for node in rule_nodes:
            rule_name = node.name
            rule_id = node.rule_ref or node.name
            
            # Check if rule is reachable
            is_reachable = self._check_reachability(spec, rule_name)
            
            # Get execution stats
            stats = self.rule_stats.get(rule_id, {})
            execution_count = stats.get("execution_count", 0)
            skip_count = stats.get("skip_count", 0)
            last_fired = stats.get("last_fired")
            
            # Determine status
            status = "active"
            days_since_last_fire = None
            
            if not is_reachable:
                status = "unreachable"
            elif execution_count == 0 and skip_count == 0:
                status = "never_fired"
            elif execution_count == 0 and skip_count > 0:
                status = "always_skipped"
            elif execution_count < min_executions:
                status = "low_usage"
            elif last_fired:
                days_since_last_fire = (datetime.utcnow() - last_fired).days
                if days_since_last_fire > days_threshold:
                    status = "stale"
            
            if status != "active":
                report = DeadRuleReport(
                    rule_name=rule_name,
                    rule_id=rule_id,
                    status=status,
                    days_since_last_fire=days_since_last_fire,
                    execution_count=execution_count,
                    skip_count=skip_count,
                    last_fired=last_fired,
                    reachability_info=self._get_reachability_info(spec, rule_name),
                    suggestions=self._generate_suggestions(status, rule_name, spec),
                )
                reports.append(report)
        
        return reports
    
    def _build_rule_stats(self, history: List[DecisionResult]) -> None:
        """Build statistics from execution history."""
        self.rule_stats = {}
        
        for result in history:
            # Track fired rules
            for fired_rule in result.fired:
                rule_id = fired_rule.id
                if rule_id not in self.rule_stats:
                    self.rule_stats[rule_id] = {
                        "execution_count": 0,
                        "skip_count": 0,
                        "last_fired": None,
                    }
                
                self.rule_stats[rule_id]["execution_count"] += 1
                # Use current time if last_fired not available
                self.rule_stats[rule_id]["last_fired"] = datetime.utcnow()
            
            # Track skipped rules
            for skipped in result.skipped:
                if skipped not in self.rule_stats:
                    self.rule_stats[skipped] = {
                        "execution_count": 0,
                        "skip_count": 0,
                        "last_fired": None,
                    }
                self.rule_stats[skipped]["skip_count"] += 1
    
    def _check_reachability(self, spec: RulebookSpec, target_node: str) -> bool:
        """
        Check if a node is reachable from entry points.
        
        Args:
            spec: Rulebook specification
            target_node: Node name to check
        
        Returns:
            True if node is reachable
        """
        # Build graph
        edges = [(edge.source, edge.target) for edge in spec.edges]
        node_names = [node.name for node in spec.nodes]
        
        # Find entry nodes (nodes with no incoming edges)
        targets = {edge.target for edge in spec.edges}
        entry_nodes = [node for node in node_names if node not in targets]
        
        if not entry_nodes:
            # No entry nodes - all nodes are potentially reachable
            return True
        
        # BFS from entry nodes to check reachability
        visited: Set[str] = set()
        queue = list(entry_nodes)
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            if node == target_node:
                return True
            
            # Add neighbors
            for edge in spec.edges:
                if edge.source == node and edge.target not in visited:
                    queue.append(edge.target)
        
        return target_node in visited
    
    def _get_reachability_info(self, spec: RulebookSpec, node_name: str) -> Dict[str, Any]:
        """Get reachability information for a node."""
        # Find incoming edges
        incoming = [edge for edge in spec.edges if edge.target == node_name]
        outgoing = [edge for edge in spec.edges if edge.source == node_name]
        
        return {
            "incoming_edges": len(incoming),
            "outgoing_edges": len(outgoing),
            "has_gate_conditions": any(
                n.kind == "gate" and any(e.target == node_name for e in spec.edges if e.source == n.name)
                for n in spec.nodes
            ),
        }
    
    def _generate_suggestions(
        self,
        status: str,
        rule_name: str,
        spec: RulebookSpec,
    ) -> List[str]:
        """Generate suggestions for dead rules."""
        suggestions = []
        
        if status == "unreachable":
            suggestions.append(f"Rule '{rule_name}' is not reachable from any entry point")
            suggestions.append("Check if there are missing edges or gate conditions that prevent execution")
            suggestions.append("Consider removing the rule if it's no longer needed")
        
        elif status == "never_fired":
            suggestions.append(f"Rule '{rule_name}' has never fired in execution history")
            suggestions.append("Check if rule conditions are too restrictive")
            suggestions.append("Verify that rule inputs are available in the state")
        
        elif status == "always_skipped":
            suggestions.append(f"Rule '{rule_name}' is always skipped")
            suggestions.append("Check gate conditions or fork routing that prevents execution")
            suggestions.append("Review rule salience or priority")
        
        elif status == "low_usage":
            suggestions.append(f"Rule '{rule_name}' has low execution count")
            suggestions.append("Consider whether rule is still needed")
            suggestions.append("Review rule conditions to see if they're too restrictive")
        
        elif status == "stale":
            suggestions.append(f"Rule '{rule_name}' hasn't fired in a long time")
            suggestions.append("Consider deprecating or removing if no longer relevant")
        
        return suggestions
    
    def get_rule_health_summary(self, spec: RulebookSpec) -> Dict[str, Any]:
        """
        Get overall health summary for rulebook.
        
        Args:
            spec: Rulebook specification
        
        Returns:
            Dictionary with health metrics
        """
        reports = self.analyze_rulebook(spec)
        
        total_rules = len([n for n in spec.nodes if n.kind == "rule"])
        dead_rules = len([r for r in reports if r.status in ["never_fired", "unreachable", "always_skipped"]])
        stale_rules = len([r for r in reports if r.status == "stale"])
        low_usage_rules = len([r for r in reports if r.status == "low_usage"])
        
        return {
            "total_rules": total_rules,
            "dead_rules": dead_rules,
            "stale_rules": stale_rules,
            "low_usage_rules": low_usage_rules,
            "active_rules": total_rules - dead_rules - stale_rules - low_usage_rules,
            "dead_rule_percentage": (dead_rules / total_rules * 100) if total_rules > 0 else 0.0,
            "reports": reports,
        }

