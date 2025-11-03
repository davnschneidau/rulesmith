"""A/B testing outcomes library - enhanced tracking and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from rulesmith.ab.metrics import ABMetricsCollector, ArmMetrics, ComparativeMetrics
from rulesmith.ab.lineage import ABLineageTracker, ForkLineage
from rulesmith.io.decision_result import DecisionResult


@dataclass
class ABOutcome:
    """Structured outcome for an A/B test execution."""
    
    fork_name: str
    arm_name: str
    execution_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    outcome_value: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestReport:
    """Comprehensive A/B test report."""
    
    fork_name: str
    test_period_start: datetime
    test_period_end: datetime
    arm_metrics: Dict[str, ArmMetrics]
    comparative_metrics: List[ComparativeMetrics]
    winner: Optional[str] = None
    confidence: float = 0.0  # 0.0 to 1.0
    recommendation: str = ""
    outcomes: List[ABOutcome] = field(default_factory=list)


class ABOutcomesLibrary:
    """Library for tracking and analyzing A/B test outcomes."""
    
    def __init__(self):
        """Initialize A/B outcomes library."""
        self.outcomes: List[ABOutcome] = []
        self.metrics_collector = ABMetricsCollector()
        self.lineage_tracker = ABLineageTracker()
    
    def record_outcome(
        self,
        fork_name: str,
        arm_name: str,
        outcome_value: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        costs: Optional[Dict[str, float]] = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ABOutcome:
        """
        Record an A/B test outcome.
        
        Args:
            fork_name: Name of the fork
            arm_name: Name of the arm
            outcome_value: Outcome value
            metrics: Optional metrics dictionary
            costs: Optional costs dictionary
            latency_ms: Execution latency
            error: Optional error message
            execution_id: Optional execution ID
            metadata: Optional metadata
        
        Returns:
            ABOutcome object
        """
        outcome = ABOutcome(
            fork_name=fork_name,
            arm_name=arm_name,
            execution_id=execution_id,
            outcome_value=outcome_value,
            metrics=metrics or {},
            costs=costs or {},
            latency_ms=latency_ms,
            error=error,
            metadata=metadata or {},
        )
        
        self.outcomes.append(outcome)
        
        # Record in metrics collector
        self.metrics_collector.record_arm_execution(
            arm_name=arm_name,
            outcome=outcome_value,
            latency_ms=latency_ms,
            cost=sum(costs.values()) if costs else 0.0,
            error=error,
        )
        
        return outcome
    
    def record_from_decision_result(
        self,
        result: DecisionResult,
        fork_name: str,
        arm_name: str,
        execution_id: Optional[str] = None,
    ) -> ABOutcome:
        """
        Record outcome from a DecisionResult.
        
        Args:
            result: DecisionResult object
            fork_name: Name of the fork
            arm_name: Name of the arm
            execution_id: Optional execution ID
        
        Returns:
            ABOutcome object
        """
        # Extract outcome value
        outcome_value = None
        if isinstance(result.value, dict):
            outcome_value = result.value.get("result") or result.value.get("outcome")
        
        return self.record_outcome(
            fork_name=fork_name,
            arm_name=arm_name,
            outcome_value=outcome_value,
            metrics=result.metrics,
            costs=result.costs,
            latency_ms=result.get_total_duration_ms(),
            execution_id=execution_id or result.trace_uri,
        )
    
    def get_outcomes(
        self,
        fork_name: Optional[str] = None,
        arm_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ABOutcome]:
        """
        Get outcomes with optional filtering.
        
        Args:
            fork_name: Optional fork name filter
            arm_name: Optional arm name filter
            start_time: Optional start time filter
            end_time: Optional end time filter
        
        Returns:
            List of ABOutcome objects
        """
        filtered = self.outcomes
        
        if fork_name:
            filtered = [o for o in filtered if o.fork_name == fork_name]
        
        if arm_name:
            filtered = [o for o in filtered if o.arm_name == arm_name]
        
        if start_time:
            filtered = [o for o in filtered if o.timestamp >= start_time]
        
        if end_time:
            filtered = [o for o in filtered if o.timestamp <= end_time]
        
        return filtered
    
    def generate_report(
        self,
        fork_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_sample_size: int = 100,
    ) -> ABTestReport:
        """
        Generate comprehensive A/B test report.
        
        Args:
            fork_name: Name of the fork
            start_time: Optional start time
            end_time: Optional end time
            min_sample_size: Minimum sample size for statistical significance
        
        Returns:
            ABTestReport object
        """
        # Get outcomes for this fork
        outcomes = self.get_outcomes(fork_name=fork_name, start_time=start_time, end_time=end_time)
        
        if not outcomes:
            return ABTestReport(
                fork_name=fork_name,
                test_period_start=start_time or datetime.utcnow(),
                test_period_end=end_time or datetime.utcnow(),
                arm_metrics={},
                comparative_metrics=[],
                recommendation="No outcomes recorded",
            )
        
        # Get arm names
        arm_names = list(set(o.arm_name for o in outcomes))
        
        # Get metrics from collector
        arm_metrics_dict = {}
        for arm_name in arm_names:
            metrics = self.metrics_collector.get_arm_metrics(arm_name)
            if metrics:
                arm_metrics_dict[arm_name] = metrics
        
        # Calculate comparative metrics
        comparative_metrics = []
        if len(arm_names) >= 2:
            # Compare all pairs
            for i in range(len(arm_names)):
                for j in range(i + 1, len(arm_names)):
                    arm_a = arm_names[i]
                    arm_b = arm_names[j]
                    comp = self.metrics_collector.compare_arms(arm_a, arm_b)
                    if comp:
                        comparative_metrics.append(comp)
        
        # Determine winner
        winner = None
        confidence = 0.0
        if comparative_metrics:
            # Find best arm based on conversion rate
            best_arm = max(arm_metrics_dict.items(), key=lambda x: x[1].conversion_rate if x[1] else 0.0)
            winner = best_arm[0]
            
            # Calculate confidence from statistical significance
            if comparative_metrics:
                best_comp = max(comparative_metrics, key=lambda x: x.statistical_significance or 0.0)
                if best_comp.statistical_significance:
                    confidence = 1.0 - best_comp.statistical_significance
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            arm_metrics_dict,
            comparative_metrics,
            winner,
            confidence,
            min_sample_size,
        )
        
        return ABTestReport(
            fork_name=fork_name,
            test_period_start=start_time or min(o.timestamp for o in outcomes),
            test_period_end=end_time or max(o.timestamp for o in outcomes),
            arm_metrics=arm_metrics_dict,
            comparative_metrics=comparative_metrics,
            winner=winner,
            confidence=confidence,
            recommendation=recommendation,
            outcomes=outcomes,
        )
    
    def _generate_recommendation(
        self,
        arm_metrics: Dict[str, ArmMetrics],
        comparative_metrics: List[ComparativeMetrics],
        winner: Optional[str],
        confidence: float,
        min_sample_size: int,
    ) -> str:
        """Generate recommendation based on analysis."""
        if not arm_metrics:
            return "No data available"
        
        # Check sample sizes
        small_samples = [
            arm for arm, metrics in arm_metrics.items()
            if metrics.sample_size < min_sample_size
        ]
        
        if small_samples:
            return f"Insufficient sample size for {', '.join(small_samples)}. Continue test."
        
        if winner and confidence > 0.95:
            return f"Confident winner: {winner} (confidence: {confidence:.2%})"
        elif winner and confidence > 0.80:
            return f"Likely winner: {winner} (confidence: {confidence:.2%}). Consider more data."
        elif winner:
            return f"Potential winner: {winner} (confidence: {confidence:.2%}). Continue test."
        else:
            return "No clear winner. Continue test or consider additional variants."
    
    def export_outcomes(
        self,
        fork_name: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """
        Export outcomes to string format.
        
        Args:
            fork_name: Optional fork name filter
            format: Export format ("json" or "csv")
        
        Returns:
            Exported data as string
        """
        outcomes = self.get_outcomes(fork_name=fork_name)
        
        if format == "json":
            import json
            data = [
                {
                    "fork_name": o.fork_name,
                    "arm_name": o.arm_name,
                    "timestamp": o.timestamp.isoformat(),
                    "outcome_value": o.outcome_value,
                    "metrics": o.metrics,
                    "costs": o.costs,
                    "latency_ms": o.latency_ms,
                    "error": o.error,
                }
                for o in outcomes
            ]
            return json.dumps(data, indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                "fork_name", "arm_name", "timestamp", "outcome_value",
                "latency_ms", "error",
            ])
            
            # Data
            for o in outcomes:
                writer.writerow([
                    o.fork_name,
                    o.arm_name,
                    o.timestamp.isoformat(),
                    str(o.outcome_value),
                    o.latency_ms,
                    o.error or "",
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")

