"""Impact analysis - analyze the impact of rules on outcomes."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from rulesmith.io.decision_result import DecisionResult, FiredRule


@dataclass
class RuleImpact:
    """Impact analysis for a rule."""
    
    rule_id: str
    rule_name: str
    execution_count: int = 0
    impact_score: float = 0.0  # 0.0 to 1.0
    outcome_changes: int = 0  # How many times rule changed the outcome
    downstream_effects: List[str] = field(default_factory=list)  # Rules that depend on this rule
    correlation_with_outcome: float = 0.0  # Correlation with positive outcomes
    average_output_value: float = 0.0
    output_variance: float = 0.0


@dataclass
class ImpactReport:
    """Impact analysis report."""
    
    rule_impacts: List[RuleImpact]
    total_executions: int
    outcome_field: str
    summary: Dict[str, Any]


class ImpactAnalyzer:
    """Analyzes the impact of rules on outcomes."""
    
    def __init__(self, outcome_field: str = "result"):
        """
        Initialize impact analyzer.
        
        Args:
            outcome_field: Field name to use as outcome (default: "result")
        """
        self.outcome_field = outcome_field
        self.execution_history: List[DecisionResult] = []
    
    def add_execution(self, result: DecisionResult) -> None:
        """Add an execution result to history."""
        self.execution_history.append(result)
    
    def analyze_impact(
        self,
        execution_history: Optional[List[DecisionResult]] = None,
        outcome_field: Optional[str] = None,
    ) -> ImpactReport:
        """
        Analyze impact of rules on outcomes.
        
        Args:
            execution_history: Optional execution history (uses instance history if None)
            outcome_field: Optional outcome field name (uses instance field if None)
        
        Returns:
            ImpactReport with rule impacts
        """
        history = execution_history or self.execution_history
        outcome_field = outcome_field or self.outcome_field
        
        if not history:
            return ImpactReport(
                rule_impacts=[],
                total_executions=0,
                outcome_field=outcome_field,
                summary={},
            )
        
        # Build rule impact data
        rule_data: Dict[str, Dict[str, Any]] = {}
        total_executions = len(history)
        
        for result in history:
            # Extract outcome
            outcome = self._extract_outcome(result, outcome_field)
            
            # Track each fired rule
            for fired_rule in result.fired:
                rule_id = fired_rule.id
                if rule_id not in rule_data:
                    rule_data[rule_id] = {
                        "rule_name": fired_rule.name,
                        "execution_count": 0,
                        "outcomes": [],
                        "outputs": [],
                        "outcome_changes": 0,
                    }
                
                rule_data[rule_id]["execution_count"] += 1
                rule_data[rule_id]["outcomes"].append(outcome)
                
                # Extract output values
                outputs = fired_rule.outputs
                if outcome_field in outputs:
                    rule_data[rule_id]["outputs"].append(outputs[outcome_field])
                else:
                    # Try to extract any numeric output
                    for value in outputs.values():
                        if isinstance(value, (int, float)):
                            rule_data[rule_id]["outputs"].append(value)
                            break
        
        # Calculate impacts
        rule_impacts = []
        for rule_id, data in rule_data.items():
            impact = self._calculate_rule_impact(rule_id, data, history, outcome_field)
            rule_impacts.append(impact)
        
        # Sort by impact score
        rule_impacts.sort(key=lambda x: x.impact_score, reverse=True)
        
        # Calculate summary
        summary = self._calculate_summary(rule_impacts, history, outcome_field)
        
        return ImpactReport(
            rule_impacts=rule_impacts,
            total_executions=total_executions,
            outcome_field=outcome_field,
            summary=summary,
        )
    
    def _extract_outcome(self, result: DecisionResult, outcome_field: str) -> Any:
        """Extract outcome from DecisionResult."""
        if isinstance(result.value, dict):
            return result.value.get(outcome_field)
        return None
    
    def _calculate_rule_impact(
        self,
        rule_id: str,
        data: Dict[str, Any],
        history: List[DecisionResult],
        outcome_field: str,
    ) -> RuleImpact:
        """Calculate impact metrics for a rule."""
        execution_count = data["execution_count"]
        outcomes = data["outcomes"]
        outputs = data["outputs"]
        
        # Calculate correlation with positive outcomes
        # Positive outcome = True, "approved", "accept", etc.
        positive_outcomes = sum(
            1 for o in outcomes
            if o is True or (isinstance(o, str) and o.lower() in ["approved", "accept", "yes", "pass"])
        )
        correlation = positive_outcomes / len(outcomes) if outcomes else 0.0
        
        # Calculate average output value
        avg_output = sum(outputs) / len(outputs) if outputs else 0.0
        
        # Calculate variance
        variance = 0.0
        if outputs and len(outputs) > 1:
            variance = sum((x - avg_output) ** 2 for x in outputs) / len(outputs)
        
        # Calculate impact score (combination of factors)
        # Impact = execution_frequency * correlation * output_significance
        execution_frequency = execution_count / len(history) if history else 0.0
        output_significance = abs(avg_output) if isinstance(avg_output, (int, float)) else 1.0
        
        impact_score = execution_frequency * correlation * output_significance
        
        # Count outcome changes (how many times rule changed the outcome)
        outcome_changes = 0
        for result in history:
            # Check if this rule was the deciding factor
            fired_ids = {fr.id for fr in result.fired}
            if rule_id in fired_ids:
                # Check if removing this rule would change outcome
                # (Simplified - in practice, would need to simulate)
                outcome_changes += 1
        
        return RuleImpact(
            rule_id=rule_id,
            rule_name=data["rule_name"],
            execution_count=execution_count,
            impact_score=impact_score,
            outcome_changes=outcome_changes,
            downstream_effects=[],  # Would need DAG analysis
            correlation_with_outcome=correlation,
            average_output_value=avg_output,
            output_variance=variance,
        )
    
    def _calculate_summary(
        self,
        rule_impacts: List[RuleImpact],
        history: List[DecisionResult],
        outcome_field: str,
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not rule_impacts:
            return {}
        
        # Top impactful rules
        top_rules = rule_impacts[:5]
        
        # Overall outcome distribution
        outcomes = []
        for result in history:
            outcome = self._extract_outcome(result, outcome_field)
            if outcome is not None:
                outcomes.append(outcome)
        
        positive_count = sum(
            1 for o in outcomes
            if o is True or (isinstance(o, str) and o.lower() in ["approved", "accept", "yes", "pass"])
        )
        
        return {
            "total_rules": len(rule_impacts),
            "top_impactful_rules": [
                {
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "impact_score": r.impact_score,
                }
                for r in top_rules
            ],
            "positive_outcome_rate": positive_count / len(outcomes) if outcomes else 0.0,
            "average_impact_score": sum(r.impact_score for r in rule_impacts) / len(rule_impacts),
        }
    
    def analyze_rule_dependency(
        self,
        rule_id: str,
        execution_history: Optional[List[DecisionResult]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze dependencies for a specific rule.
        
        Args:
            rule_id: Rule ID to analyze
            execution_history: Optional execution history
        
        Returns:
            Dependency analysis
        """
        history = execution_history or self.execution_history
        
        # Find rules that fire together with this rule
        co_occurrences: Dict[str, int] = {}
        
        for result in history:
            fired_ids = {fr.id for fr in result.fired}
            if rule_id in fired_ids:
                for other_id in fired_ids:
                    if other_id != rule_id:
                        co_occurrences[other_id] = co_occurrences.get(other_id, 0) + 1
        
        # Sort by co-occurrence frequency
        sorted_co = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "rule_id": rule_id,
            "co_occurring_rules": [
                {"rule_id": rid, "frequency": freq}
                for rid, freq in sorted_co[:10]  # Top 10
            ],
            "total_co_occurrences": sum(co_occurrences.values()),
        }

